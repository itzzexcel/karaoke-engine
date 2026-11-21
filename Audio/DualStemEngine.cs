using KaraokeEngine.Ai;
using KaraokeEngine.Utils;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace KaraokeEngine.Audio;

internal sealed class TrackBuffer
{
    public required string TrackId { get; init; }
    public required string Title { get; init; }
    public required short[] OriginalPcm16 { get; init; }
    public required short[] InstrumentalPcm16 { get; init; }
    public int TotalFrames => OriginalPcm16.Length / 2;

    // 0 = pending/DSP, 1 = separating, 2 = AI ready
    public byte[] ChunkState { get; set; } = Array.Empty<byte>();
    public int ChunkFrames { get; set; } = 261120;
    public int StepFrames { get; set; } = 252300;
    public string SeparatedForMode { get; set; } = "";
    public bool FromCache { get; set; }
    public bool SavedInstToCache { get; set; }
    public string ModelHash8 { get; set; } = "";
}

internal sealed record EngineStatusSnapshot(
    bool Enabled,
    string State,
    string Provider,
    string Mode,
    float VocalGain,
    float MasterVolume,
    string TrackId,
    int SeparatedPct,
    bool FromCache,
    int CacheEntries,
    double CacheMb,
    bool KaraokeModelInstalled,
    bool InstrumentalModelInstalled,
    float DownloadProgress,
    double LastChunkMs,
    string? Error
);

internal sealed class DualStemEngine : IWaveProvider, IDisposable
{
    private const float Inv32768 = 1f / 32768f;

    [DllImport("avrt.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr AvSetMmThreadCharacteristicsW(string taskName, ref uint taskIndex);

    [DllImport("avrt.dll", SetLastError = true)]
    private static extern bool AvRevertMmThreadCharacteristics(IntPtr avrtHandle);

    private readonly OnnxStemSeparator _separator = new();
    private DspVocalSuppressor _dsp = new(TidalStreamFetcher.TargetSampleRate);
    private readonly TidalStreamFetcher _fetcher = new();
    private readonly TidalSessionMuter _muter = new();
    private readonly StemDiskCache _cache = new();

    private readonly object _trackLock = new();

    private volatile TrackBuffer? _activeTrack;
    private TrackBuffer? _preloadedTrack;
    private CancellationTokenSource? _loadCts;
    private CancellationTokenSource? _preloadCts;
    private CancellationTokenSource? _modeSwitchCts;
    private CancellationTokenSource? _disableStopCts;
    private Thread? _workerThread;
    private readonly CancellationTokenSource _shutdownCts = new();
    private readonly SemaphoreSlim _workerWake = new(0, 1);

    private WasapiOut? _wasapiOut;
    private WaveFormat _waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(
        TidalStreamFetcher.TargetSampleRate,
        TidalStreamFetcher.TargetChannels
    );
    private volatile int _outputSampleRate = TidalStreamFetcher.TargetSampleRate;
    private int _audioThreadId;
    private IntPtr _avrtHandle = IntPtr.Zero;

    // Runtime configuration / state
    private volatile bool _enabled;
    private volatile bool _playing;
    private volatile string _mode = "karaoke"; // karaoke | instrumental | dsp
    private volatile bool _useGpu = true;
    private volatile float _targetVocalGain = 0.15f;
    private volatile float _smoothedVocalGain = 0.15f;
    private volatile float _targetMasterVolume = 1.0f;
    private volatile float _smoothedMasterVolume = 1.0f;
    private volatile int _syncOffsetMs;

    // Live Tidal clock tracker for seamless zero-gap handoff after initial chunk separates
    private string _latestTidalTrackId = "";
    private double _latestTidalTimeSec;
    private DateTime _latestTidalTimeUtc = DateTime.UtcNow;
    private double _lastSyncTimeSec = -1.0;
    private int _largeDriftCount;

    // Audio thread playhead & smoothing (owned by WASAPI callback; cross-thread ops go via _audioCmdLock)
    private double _playFramePos;
    private double _rateMultiplier = 1.0;
    private double _filteredDriftSec = 0.0;
    private float _aiCrossfade = 0f;
    private float _seekFadeGain = 1f;
    private float _playFadeGain = 0f;
    private float _handoffGain = 1f;
    private volatile float _targetHandoffGain = 1f;
    private volatile float _handoffRampSec = 0.40f;
    private volatile bool _disablingFadeOut;
    private float _vocalDuckFramesRemaining;
    private bool _wasInAiMode;
    private DateTime _lastDisabledUtc = DateTime.UtcNow;
    private DateTime _lastStatusEmitUtc = DateTime.MinValue;
    private DateTime _resumeGraceUntilUtc = DateTime.MinValue;

    private readonly object _audioCmdLock = new();
    private volatile bool _hasPendingAudioCmd;
    private double _pendingSeekFrame = -1.0;
    private double _pendingPauseSnapFrame = -1.0;
    private bool _pendingSeekFade;
    private bool _pendingDspReset;
    private bool _pendingStartInAi;
    private bool _pendingHandoffFade;
    private float? _pendingHandoffStart;
    private float _pendingHandoffTarget = 1f;
    private float _pendingHandoffSec = 0.40f;
    private bool _pendingVocalDuckTransition;
    private bool _pendingCancelVocalDuck;

    // Progress / diagnostics
    private volatile string _state = "idle";
    private volatile string _loadingTrackId = "";
    private volatile int _loadingSeparatedPct;
    private volatile float _downloadProgress = -1f;
    private double _lastChunkMs;
    private string? _lastError;
    private readonly SemaphoreSlim _downloadLock = new(1, 1);

    public WaveFormat WaveFormat => _waveFormat;

    public event Action? StatusChanged;

    public DualStemEngine()
    {
        _workerThread = new Thread(SeparationWorkerThreadMain)
        {
            IsBackground = true,
            Priority = ThreadPriority.BelowNormal,
            Name = "KaraokeSeparationWorker",
        };
        _workerThread.Start();
    }

    public EngineStatusSnapshot GetSnapshot()
    {
        var track = _activeTrack;

        int sepPct = 0;
        if (track != null && track.ChunkState.Length > 0)
        {
            int ready = 0;
            for (int i = 0; i < track.ChunkState.Length; i++)
            {
                if (track.ChunkState[i] == 2) ready++;
            }
            sepPct = (int)Math.Round(100.0 * ready / track.ChunkState.Length);
        }
        else if (!string.IsNullOrEmpty(_loadingTrackId))
        {
            sepPct = _loadingSeparatedPct;
        }

        string provider = _mode == "dsp"
            ? "DSP"
            : (track != null && track.FromCache && !_separator.IsLoaded
                ? "Cached"
                : (_separator.IsLoaded ? _separator.ActiveProvider : "DSP"));
        (int cacheEntries, double cacheMb) = _cache.GetStats();

        return new EngineStatusSnapshot(
            Enabled: _enabled,
            State: _state,
            Provider: provider,
            Mode: _mode,
            VocalGain: _targetVocalGain,
            MasterVolume: _targetMasterVolume,
            TrackId: track?.TrackId ?? _loadingTrackId,
            SeparatedPct: sepPct,
            FromCache: track?.FromCache ?? false,
            CacheEntries: cacheEntries,
            CacheMb: cacheMb,
            KaraokeModelInstalled: ModelCatalog.IsModelInstalled("karaoke"),
            InstrumentalModelInstalled: ModelCatalog.IsModelInstalled("instrumental"),
            DownloadProgress: _downloadProgress,
            LastChunkMs: Math.Round(_lastChunkMs, 1),
            Error: _lastError
        );
    }

    private void QueueAudioSeek(double targetFrame, bool fadeIn, bool resetDsp, bool startInAi = false)
    {
        lock (_audioCmdLock)
        {
            _pendingSeekFrame = targetFrame;
            _pendingPauseSnapFrame = -1.0;
            if (fadeIn) _pendingSeekFade = true;
            if (resetDsp) _pendingDspReset = true;
            if (startInAi) _pendingStartInAi = true;
            _hasPendingAudioCmd = true;
        }
    }

    private void QueueHandoffFade(float? startGain, float targetGain, float durationSec, bool vocalDuckMelt = false)
    {
        lock (_audioCmdLock)
        {
            _pendingHandoffFade = true;
            _pendingHandoffStart = startGain;
            _pendingHandoffTarget = targetGain;
            _pendingHandoffSec = MathF.Max(0.02f, durationSec);
            _targetHandoffGain = targetGain;
            _handoffRampSec = MathF.Max(0.02f, durationSec);
            if (vocalDuckMelt)
            {
                _pendingVocalDuckTransition = true;
                _pendingCancelVocalDuck = false;
            }
            _hasPendingAudioCmd = true;
        }
    }

    private void QueuePauseSnap(double targetFrame)
    {
        lock (_audioCmdLock)
        {
            if (_pendingSeekFrame < 0.0)
            {
                _pendingPauseSnapFrame = targetFrame;
                _hasPendingAudioCmd = true;
            }
        }
    }

    private void ClearPendingAudioCmds()
    {
        lock (_audioCmdLock)
        {
            _pendingSeekFrame = -1.0;
            _pendingPauseSnapFrame = -1.0;
            _pendingSeekFade = false;
            _pendingDspReset = false;
            _pendingStartInAi = false;
            _pendingHandoffFade = false;
            _pendingVocalDuckTransition = false;
            _pendingCancelVocalDuck = false;
            _hasPendingAudioCmd = false;
        }
    }

    private static float SoftLimit(float x)
    {
        float ax = MathF.Abs(x);
        if (ax <= 0.95f) return x;
        return MathF.CopySign(0.95f + 0.05f * MathF.Tanh((ax - 0.95f) * 20f), x);
    }

    private static float CubicSample(short[] pcm, int f0, int totalFrames, int ch, float t)
    {
        int fm1 = f0 > 0 ? f0 - 1 : 0;
        int fp1 = f0 + 1 < totalFrames ? f0 + 1 : totalFrames - 1;
        int fp2 = f0 + 2 < totalFrames ? f0 + 2 : totalFrames - 1;

        float y0 = pcm[fm1 * 2 + ch] * Inv32768;
        float y1 = pcm[f0 * 2 + ch] * Inv32768;
        float y2 = pcm[fp1 * 2 + ch] * Inv32768;
        float y3 = pcm[fp2 * 2 + ch] * Inv32768;

        float c0 = y1;
        float c1 = 0.5f * (y2 - y0);
        float c2 = y0 - 2.5f * y1 + 2f * y2 - 0.5f * y3;
        float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);
        return ((c3 * t + c2) * t + c1) * t + c0;
    }

    public void ClearCache()
    {
        _cache.ClearAll();
        lock (_trackLock)
        {
            if (_activeTrack != null) _activeTrack.SavedInstToCache = false;
            if (_preloadedTrack != null) _preloadedTrack.SavedInstToCache = false;
        }
        EmitStatus(force: true);
    }

    public void SetEnabled(bool enabled)
    {
        if (_enabled == enabled && !_disablingFadeOut) return;
        _enabled = enabled;
        _lastError = null;

        _disableStopCts?.Cancel();
        _disableStopCts?.Dispose();
        _disableStopCts = null;

        if (enabled)
        {
            bool wasFadingOut = _disablingFadeOut;
            _disablingFadeOut = false;

            var track = _activeTrack;
            if (track != null)
            {
                double liveSec = string.Equals(_latestTidalTrackId, track.TrackId, StringComparison.Ordinal)
                    ? (_latestTidalTimeSec + (_playing ? (DateTime.UtcNow - _latestTidalTimeUtc).TotalSeconds : 0.0))
                    : 0.0;
                double liveFrame = Math.Clamp((liveSec + _syncOffsetMs * 0.001) * TidalStreamFetcher.TargetSampleRate, 0.0, Math.Max(0, track.TotalFrames - 1));
                int startChunk = track.StepFrames > 0
                    ? Math.Clamp((int)(liveFrame / track.StepFrames), 0, Math.Max(0, track.ChunkState.Length - 1))
                    : 0;
                bool startInAi = _mode != "dsp" && startChunk < track.ChunkState.Length && track.ChunkState[startChunk] == 2;

                if (_wasapiOut != null && wasFadingOut)
                {
                    QueueHandoffFade(null, 1f, 0.65f, vocalDuckMelt: _playing);
                }
                else
                {
                    ClearPendingAudioCmds();
                    _playFramePos = liveFrame;
                    _rateMultiplier = 1.0;
                    _filteredDriftSec = 0.0;
                    _playFadeGain = 0f;
                    _seekFadeGain = 1f;
                    _handoffGain = 0f;
                    _targetHandoffGain = 1f;
                    _handoffRampSec = 0.65f;
                    _wasInAiMode = startInAi;
                    _aiCrossfade = startInAi ? 1f : 0f;
                    if (_playing)
                    {
                        _smoothedVocalGain = 1.0f;
                        _vocalDuckFramesRemaining = 1.15f * _outputSampleRate;
                    }
                    EnsureOutputStarted();
                }

                if (_playing) _muter.FadeToAsync(0f, 650);
                else _muter.SetMuted(true);
            }
            WakeWorker();
            UpdateState();
        }
        else
        {
            _lastDisabledUtc = DateTime.UtcNow;
            if (_wasapiOut != null && _playing && _activeTrack != null)
            {
                _disablingFadeOut = true;
                QueueHandoffFade(null, 0f, 0.32f, vocalDuckMelt: false);
                _muter.FadeToAsync(1.0f, 320);

                var cts = new CancellationTokenSource();
                _disableStopCts = cts;
                CancellationToken ct = cts.Token;
                _ = Task.Run(async () =>
                {
                    try
                    {
                        await Task.Delay(360, ct).ConfigureAwait(false);
                        if (_enabled || ct.IsCancellationRequested) return;
                        _disablingFadeOut = false;
                        _muter.EnsureUnmuted();
                        StopOutput();
                        _state = "idle";
                        EmitStatus(force: true);
                    }
                    catch (OperationCanceledException) { }
                });
            }
            else
            {
                _disablingFadeOut = false;
                _muter.EnsureUnmuted();
                StopOutput();
                _state = "idle";
            }
        }

        EmitStatus(force: true);
    }

    public void UpdateConfig(float? vocalGain, float? masterVolume, string? mode, bool? useGpu, int? syncOffsetMs, int? cacheMaxMb = null)
    {
        bool modeChanged = false;
        bool gpuChanged = false;

        if (vocalGain.HasValue)
        {
            _targetVocalGain = Math.Clamp(vocalGain.Value, 0f, 1f);
            lock (_audioCmdLock)
            {
                _pendingCancelVocalDuck = true;
                _hasPendingAudioCmd = true;
            }
        }

        if (masterVolume.HasValue)
            _targetMasterVolume = Math.Clamp(masterVolume.Value, 0f, 1f);

        if (cacheMaxMb.HasValue)
            _cache.SetMaxSizeMb(cacheMaxMb.Value);

        if (!string.IsNullOrWhiteSpace(mode))
        {
            string norm = mode.Trim().ToLowerInvariant();
            if (norm is "karaoke" or "instrumental" or "dsp" && norm != _mode)
            {
                _mode = norm;
                modeChanged = true;
            }
        }

        if (useGpu.HasValue && useGpu.Value != _useGpu)
        {
            _useGpu = useGpu.Value;
            if (_enabled && _mode != "dsp") gpuChanged = true;
        }

        if (syncOffsetMs.HasValue)
            _syncOffsetMs = Math.Clamp(syncOffsetMs.Value, -500, 500);

        if (modeChanged || gpuChanged)
        {
            _modeSwitchCts?.Cancel();
            _modeSwitchCts?.Dispose();
            _modeSwitchCts = null;

            bool handled = false;
            TrackBuffer? curActive;
            lock (_trackLock) { curActive = _activeTrack; }

            if (modeChanged && !gpuChanged && (_mode == "karaoke" || _mode == "instrumental") && curActive != null && ModelCatalog.IsModelInstalled(_mode))
            {
                string hash8 = ModelCatalog.GetModelHash8(_mode);
                short[]? cachedInst = _cache.TryLoadInstrumental(curActive.TrackId, _mode, hash8);
                if (cachedInst != null && cachedInst.Length == curActive.OriginalPcm16.Length)
                {
                    byte[] filled = new byte[Math.Max(1, curActive.ChunkState.Length)];
                    Array.Fill(filled, (byte)2);
                    var swapped = new TrackBuffer
                    {
                        TrackId = curActive.TrackId,
                        Title = curActive.Title,
                        OriginalPcm16 = curActive.OriginalPcm16,
                        InstrumentalPcm16 = cachedInst,
                        ChunkFrames = curActive.ChunkFrames,
                        StepFrames = curActive.StepFrames,
                        ChunkState = filled,
                        SeparatedForMode = _mode,
                        FromCache = true,
                        SavedInstToCache = true,
                        ModelHash8 = hash8,
                    };
                    lock (_trackLock)
                    {
                        _activeTrack = swapped;
                        if (_preloadedTrack != null)
                        {
                            short[]? preCachedInst = _cache.TryLoadInstrumental(_preloadedTrack.TrackId, _mode, hash8);
                            if (preCachedInst != null && preCachedInst.Length == _preloadedTrack.OriginalPcm16.Length)
                            {
                                byte[] preFilled = new byte[Math.Max(1, _preloadedTrack.ChunkState.Length)];
                                Array.Fill(preFilled, (byte)2);
                                _preloadedTrack = new TrackBuffer
                                {
                                    TrackId = _preloadedTrack.TrackId,
                                    Title = _preloadedTrack.Title,
                                    OriginalPcm16 = _preloadedTrack.OriginalPcm16,
                                    InstrumentalPcm16 = preCachedInst,
                                    ChunkFrames = _preloadedTrack.ChunkFrames,
                                    StepFrames = _preloadedTrack.StepFrames,
                                    ChunkState = preFilled,
                                    SeparatedForMode = _mode,
                                    FromCache = true,
                                    SavedInstToCache = true,
                                    ModelHash8 = hash8,
                                };
                            }
                            else
                            {
                                Array.Clear(_preloadedTrack.ChunkState, 0, _preloadedTrack.ChunkState.Length);
                                _preloadedTrack.SeparatedForMode = _mode;
                                _preloadedTrack.FromCache = false;
                                _preloadedTrack.SavedInstToCache = false;
                            }
                        }
                    }
                    QueueHandoffFade(0.35f, 1f, 0.16f);
                    Logger.Info($"Switched '{curActive.Title}' to mode '{_mode}' instantly from disk cache.");
                    handled = true;
                }
                else if (_enabled && _wasapiOut != null)
                {
                    // Smooth uncached mode transition: keep current audio running while pre-separating the full track in background
                    var switchCts = new CancellationTokenSource();
                    _modeSwitchCts = switchCts;
                    string targetMode = _mode;
                    string targetTrackId = curActive.TrackId;
                    _loadingTrackId = targetTrackId;
                    _loadingSeparatedPct = 0;
                    _ = Task.Run(() => SwitchModeUncachedAsync(curActive, targetMode, targetTrackId, switchCts.Token), switchCts.Token);
                    handled = true;
                }
            }

            if (!handled)
            {
                ResetCurrentTrackChunks();
            }
            WakeWorker();
        }

        UpdateState();
        EmitStatus(force: true);
    }

    private async Task SwitchModeUncachedAsync(TrackBuffer sourceTrack, string targetMode, string targetTrackId, CancellationToken ct)
    {
        try
        {
            if (!await _separator.EnsureLoadedAsync(targetMode, _useGpu, ct).ConfigureAwait(false))
                return;
            if (ct.IsCancellationRequested || _mode != targetMode || _activeTrack?.TrackId != targetTrackId)
                return;

            int chunkFrames = _separator.GetExactChunkFrames();
            int stepFrames = Math.Max(1024, chunkFrames - 8820);
            int numChunks = Math.Max(1, (int)Math.Ceiling(sourceTrack.TotalFrames / (double)stepFrames));
            var staging = new TrackBuffer
            {
                TrackId = sourceTrack.TrackId,
                Title = sourceTrack.Title,
                OriginalPcm16 = sourceTrack.OriginalPcm16,
                InstrumentalPcm16 = new short[sourceTrack.OriginalPcm16.Length],
                ChunkFrames = chunkFrames,
                StepFrames = stepFrames,
                ChunkState = new byte[numChunks],
                SeparatedForMode = targetMode,
                ModelHash8 = _separator.LoadedModelHash8,
            };

            int curChunk = Math.Clamp((int)(Volatile.Read(ref _playFramePos) / stepFrames), 0, numChunks - 1);
            for (int offset = 0; offset < numChunks; offset++)
            {
                if (ct.IsCancellationRequested || _mode != targetMode || _activeTrack?.TrackId != targetTrackId)
                    return;
                int targetIdx = (curChunk + offset) % numChunks;
                if (staging.ChunkState[targetIdx] != 2)
                {
                    SeparateSingleChunk(staging, targetIdx);
                }
                int readyNow = 0;
                for (int i = 0; i < numChunks; i++)
                {
                    if (staging.ChunkState[i] == 2) readyNow++;
                }
                _loadingSeparatedPct = (int)Math.Round(100.0 * readyNow / Math.Max(1, numChunks));
                EmitStatus(force: false);
                await Task.Delay(12, ct).ConfigureAwait(false);
            }

            if (ct.IsCancellationRequested || _mode != targetMode || _activeTrack?.TrackId != targetTrackId)
                return;

            staging.FromCache = true;
            staging.SavedInstToCache = true;
            _cache.SaveInstrumentalBackground(staging.TrackId, targetMode, staging.ModelHash8, staging.InstrumentalPcm16);

            lock (_trackLock)
            {
                if (_activeTrack?.TrackId != targetTrackId || _mode != targetMode) return;
                _activeTrack = staging;
                if (_preloadedTrack != null)
                {
                    Array.Clear(_preloadedTrack.ChunkState, 0, _preloadedTrack.ChunkState.Length);
                    _preloadedTrack.SeparatedForMode = targetMode;
                    _preloadedTrack.FromCache = false;
                    _preloadedTrack.SavedInstToCache = false;
                }
            }

            _loadingTrackId = "";
            _loadingSeparatedPct = 0;
            QueueHandoffFade(0.30f, 1f, 0.28f);
            WakeWorker();
            UpdateState();
            EmitStatus(force: true);
        }
        catch (OperationCanceledException)
        {
            if (_loadingTrackId == targetTrackId)
            {
                _loadingTrackId = "";
                _loadingSeparatedPct = 0;
            }
        }
        catch (Exception ex)
        {
            _loadingTrackId = "";
            _loadingSeparatedPct = 0;
            Logger.Warn($"Smooth mode switch fallback ({targetMode}): {ex.Message}");
        }
    }

    public void HandleSeek(string? trackId, double targetTimeSec, bool playing)
    {
        _playing = playing;
        if (!string.IsNullOrEmpty(trackId))
        {
            _latestTidalTrackId = trackId;
        }
        _latestTidalTimeSec = targetTimeSec;
        _latestTidalTimeUtc = DateTime.UtcNow;
        _lastSyncTimeSec = targetTimeSec;
        _largeDriftCount = 0;

        if (!_enabled) return;

        var track = _activeTrack;
        if (track == null) return;
        if (!string.IsNullOrEmpty(trackId) && track.TrackId != trackId) return;

        double targetSec = Math.Max(0.0, targetTimeSec + (_syncOffsetMs * 0.001));
        double targetFrame = Math.Clamp(targetSec * TidalStreamFetcher.TargetSampleRate, 0.0, Math.Max(0, track.TotalFrames - 1));
        _rateMultiplier = 1.0;
        _filteredDriftSec = 0.0;
        QueueAudioSeek(targetFrame, fadeIn: true, resetDsp: true);
        WakeWorker();
    }

    public void HandleSync(string? trackId, double currentTimeSec, bool playing, float? masterVolume)
    {
        bool wasPlaying = _playing;
        if (masterVolume.HasValue)
            _targetMasterVolume = Math.Clamp(masterVolume.Value, 0f, 1f);

        if (!string.IsNullOrEmpty(trackId))
        {
            _latestTidalTrackId = trackId;
        }
        _latestTidalTimeSec = currentTimeSec;
        _latestTidalTimeUtc = DateTime.UtcNow;

        if (!_enabled)
        {
            _playing = playing;
            return;
        }

        var track = _activeTrack;
        if (track == null)
        {
            _playing = playing;
            return;
        }
        if (!string.IsNullOrEmpty(trackId) && track.TrackId != trackId)
        {
            _playing = playing;
            return;
        }

        double targetSec = Math.Max(0.0, currentTimeSec + (_syncOffsetMs * 0.001));
        double targetFrame = Math.Clamp(targetSec * TidalStreamFetcher.TargetSampleRate, 0.0, Math.Max(0, track.TotalFrames - 1));

        if (!playing)
        {
            _playing = false;
            _rateMultiplier = 1.0;
            _filteredDriftSec = 0.0;
            _largeDriftCount = 0;
            double curPos = Volatile.Read(ref _playFramePos);
            // Snap playhead silently once pause fade reaches 0f so resume starts at the exact Tidal pause timestamp
            if (Math.Abs(targetFrame - curPos) > 0.04 * TidalStreamFetcher.TargetSampleRate)
            {
                QueuePauseSnap(targetFrame);
            }
            if (wasPlaying)
            {
                Logger.Info($"[TRANSPORT] Paused @ {curPos / TidalStreamFetcher.TargetSampleRate:F2}s (tidal={currentTimeSec:F2}s)");
            }
            return;
        }

        if (!wasPlaying)
        {
            _rateMultiplier = 1.0;
            _filteredDriftSec = 0.0;
            _largeDriftCount = 0;
            _lastSyncTimeSec = currentTimeSec;
            _resumeGraceUntilUtc = DateTime.UtcNow.AddMilliseconds(1200);
            _muter.ForceMuteBurstAsync();

            double curPos = Volatile.Read(ref _playFramePos);
            double resumeDriftSec = (targetFrame - curPos) / TidalStreamFetcher.TargetSampleRate;
            Logger.Info($"[TRANSPORT] Resumed @ {curPos / TidalStreamFetcher.TargetSampleRate:F2}s (tidal={currentTimeSec:F2}s, drift={resumeDriftSec:F3}s)");
            if (Math.Abs(resumeDriftSec) > 0.04)
            {
                int resChunk = track.StepFrames > 0
                    ? Math.Clamp((int)(targetFrame / track.StepFrames), 0, Math.Max(0, track.ChunkState.Length - 1))
                    : 0;
                bool startInAi = _mode != "dsp" && resChunk < track.ChunkState.Length && track.ChunkState[resChunk] == 2;
                QueueAudioSeek(targetFrame, fadeIn: true, resetDsp: !startInAi, startInAi: startInAi);
            }
            _playing = true;
            WakeWorker();
            return;
        }

        _playing = true;

        _muter.EnforceIfMutedAsync();
        if (DateTime.UtcNow < _resumeGraceUntilUtc)
        {
            _rateMultiplier = 1.0;
            _filteredDriftSec = 0.0;
            _largeDriftCount = 0;
            return;
        }

        // Ignore duplicate staircase currentTimeSec samples from Tidal IPC so we never compute artificial negative drift
        if (Math.Abs(currentTimeSec - _lastSyncTimeSec) < 0.001)
        {
            return;
        }
        _lastSyncTimeSec = currentTimeSec;

        double driftSec = (targetFrame - Volatile.Read(ref _playFramePos)) / TidalStreamFetcher.TargetSampleRate;
        _filteredDriftSec = _filteredDriftSec * 0.65 + driftSec * 0.35;

        // Require 3 consecutive fresh sync ticks > 2.0s desync before hard snapping (explicit seeks already use HandleSeek)
        if (Math.Abs(driftSec) > 2.0)
        {
            _largeDriftCount++;
            if (_largeDriftCount >= 3)
            {
                _largeDriftCount = 0;
                _rateMultiplier = 1.0;
                _filteredDriftSec = 0.0;
                Logger.Info($"[SYNC-SNAP] Large sustained desync ({driftSec:F3}s) -> snapping to {targetSec:F2}s");
                QueueAudioSeek(targetFrame, fadeIn: true, resetDsp: true);
                WakeWorker();
            }
        }
        else
        {
            _largeDriftCount = 0;
            if (Math.Abs(_filteredDriftSec) >= 0.14)
            {
                double nudge = Math.Clamp(_filteredDriftSec * 0.012, -0.0018, 0.0018);
                _rateMultiplier = 1.0 + nudge;
            }
            else
            {
                _rateMultiplier = 1.0;
            }
        }
    }

    public void LoadTrack(
        string trackId,
        string title,
        string manifestMimeType,
        string[] urls,
        string? encryptionType,
        string? keyId,
        double startTimeSec,
        bool playing,
        bool preload)
    {
        if (preload)
        {
            lock (_trackLock)
            {
                if (_activeTrack?.TrackId == trackId || _preloadedTrack?.TrackId == trackId)
                    return;
            }
            _preloadCts?.Cancel();
            _preloadCts?.Dispose();
            _preloadCts = new CancellationTokenSource();
            var pct = _preloadCts.Token;
            _ = Task.Run(() => FetchAndPrepareTrackAsync(trackId, title, manifestMimeType, urls, encryptionType, keyId, 0, playing, isPreload: true, preloadedPartial: null, pct), pct);
            return;
        }

        _playing = playing;
        _latestTidalTrackId = trackId;
        _latestTidalTimeSec = startTimeSec;
        _latestTidalTimeUtc = DateTime.UtcNow;
        _lastSyncTimeSec = startTimeSec;
        _largeDriftCount = 0;

        if (_loadingTrackId == trackId)
        {
            return;
        }

        // Fast promote preloaded track ONLY if AI is 100% separated (or track is from cache / DSP mode)
        TrackBuffer? promoted = null;
        TrackBuffer? preloadedOrig = null;
        lock (_trackLock)
        {
            if (_activeTrack?.TrackId == trackId)
            {
                HandleSync(trackId, startTimeSec, playing, null);
                return;
            }
            if (_preloadedTrack?.TrackId == trackId)
            {
                var candidate = _preloadedTrack;
                _preloadedTrack = null;
                bool fullyReady = _mode == "dsp" || candidate.FromCache || (candidate.SeparatedForMode == _mode && IsTrackFullySeparated(candidate));
                if (fullyReady)
                {
                    promoted = candidate;
                    double promotedFrame = Math.Clamp((startTimeSec + _syncOffsetMs * 0.001) * TidalStreamFetcher.TargetSampleRate, 0.0, Math.Max(0, promoted.TotalFrames - 1));
                    int promotedChunk = promoted.StepFrames > 0
                        ? Math.Clamp((int)(promotedFrame / promoted.StepFrames), 0, Math.Max(0, promoted.ChunkState.Length - 1))
                        : 0;
                    bool startInAi = _mode != "dsp" && promotedChunk < promoted.ChunkState.Length && promoted.ChunkState[promotedChunk] == 2;

                    _rateMultiplier = 1.0;
                    _filteredDriftSec = 0.0;
                    if (_wasapiOut != null)
                    {
                        _activeTrack = promoted;
                        QueueAudioSeek(promotedFrame, fadeIn: true, resetDsp: !startInAi, startInAi: startInAi);
                    }
                    else
                    {
                        ClearPendingAudioCmds();
                        _playFramePos = promotedFrame;
                        _seekFadeGain = 0f;
                        _wasInAiMode = startInAi;
                        _aiCrossfade = startInAi ? 1f : 0f;
                        _dsp.Reset();
                        _activeTrack = promoted;
                    }
                }
                else
                {
                    preloadedOrig = candidate;
                }
            }
        }

        if (promoted != null)
        {
            Logger.Info($"Promoted 100% ready preloaded track '{title}' ({trackId}) cleanly.");
            if (_enabled)
            {
                if (_wasapiOut != null)
                {
                    QueueHandoffFade(0f, 1f, 0.08f, vocalDuckMelt: false);
                }
                else
                {
                    _handoffGain = 0f;
                    _targetHandoffGain = 1f;
                    _handoffRampSec = 0.08f;
                    EnsureOutputStarted();
                }
                if (_muter.IsMuted)
                {
                    _muter.SetMuted(true);
                    if (playing) _muter.ForceMuteBurstAsync();
                }
                else if (playing)
                {
                    _muter.FadeToAsync(0f, 120);
                }
                else
                {
                    _muter.SetMuted(true);
                }
            }
            WakeWorker();
            UpdateState();
            EmitStatus(force: true);
            return;
        }

        _loadCts?.Cancel();
        _loadCts?.Dispose();
        _loadCts = new CancellationTokenSource();
        var ct = _loadCts.Token;

        _loadingTrackId = trackId;
        _loadingSeparatedPct = 0;
        _lastError = null;
        string currentMode = _mode;
        string currentHash8 = (currentMode != "dsp" && ModelCatalog.IsModelInstalled(currentMode))
            ? ModelCatalog.GetModelHash8(currentMode)
            : "";
        bool hasInstantStemCache = currentMode == "dsp"
            ? _cache.HasOriginal(trackId)
            : (!string.IsNullOrEmpty(currentHash8) && _cache.HasOriginal(trackId) && _cache.HasInstrumental(trackId, currentMode, currentHash8));

        if (!hasInstantStemCache)
        {
            _state = "loading";
            // Silence previous track in karaoke-engine and unmute TIDALPlayer immediately so the user hears the new song at 0.00s while we prepare!
            lock (_trackLock)
            {
                _activeTrack = null;
            }
            if (_enabled)
            {
                QueueHandoffFade(0f, 0f, 0.02f);
                _muter.EnsureUnmuted();
            }
        }
        else if (_enabled)
        {
            // Both original and AI stem are already cached on disk (~20ms load); keep TIDALPlayer muted for zero-flicker transition
            _muter.SetMuted(true);
        }
        EmitStatus(force: true);

        _ = Task.Run(() => FetchAndPrepareTrackAsync(trackId, title, manifestMimeType, urls, encryptionType, keyId, startTimeSec, playing, isPreload: false, preloadedOrig, ct), ct);
    }

    private async Task FetchAndPrepareTrackAsync(
        string trackId,
        string title,
        string manifestMimeType,
        string[] urls,
        string? encryptionType,
        string? keyId,
        double startTimeSec,
        bool playing,
        bool isPreload,
        TrackBuffer? preloadedPartial,
        CancellationToken ct)
    {
        try
        {
            if (isPreload)
            {
                await Task.Yield();
            }

            short[]? origPcm16 = preloadedPartial?.OriginalPcm16 ?? _cache.TryLoadOriginal(trackId);
            if (origPcm16 != null)
            {
                Logger.Info($"Cache HIT original '{title}' ({trackId}) in memory-ready PCM16.");
            }
            else
            {
                origPcm16 = await _fetcher.FetchAndDecode44kStereoPcm16Async(manifestMimeType, urls, encryptionType, keyId, ct);
                ct.ThrowIfCancellationRequested();
                _cache.SaveOriginalBackground(trackId, origPcm16);
            }

            string desiredMode = _mode;
            int chunkFrames = _separator.IsLoaded ? _separator.GetExactChunkFrames() : 261120;
            int stepFrames = Math.Max(1024, chunkFrames - 8820);
            int totalFrames = origPcm16.Length / 2;
            int numChunks = Math.Max(1, (int)Math.Ceiling(totalFrames / (double)stepFrames));

            TrackBuffer track;
            string modelHash8 = (desiredMode != "dsp" && ModelCatalog.IsModelInstalled(desiredMode))
                ? ModelCatalog.GetModelHash8(desiredMode)
                : "";
            short[]? cachedInst = !string.IsNullOrEmpty(modelHash8)
                ? _cache.TryLoadInstrumental(trackId, desiredMode, modelHash8)
                : null;

            if (cachedInst != null && cachedInst.Length == origPcm16.Length)
            {
                byte[] allReady = new byte[numChunks];
                Array.Fill(allReady, (byte)2);
                track = new TrackBuffer
                {
                    TrackId = trackId,
                    Title = title,
                    OriginalPcm16 = origPcm16,
                    InstrumentalPcm16 = cachedInst,
                    ChunkFrames = chunkFrames,
                    StepFrames = stepFrames,
                    ChunkState = allReady,
                    SeparatedForMode = desiredMode,
                    FromCache = true,
                    SavedInstToCache = true,
                    ModelHash8 = modelHash8,
                };
                Logger.Info($"Cache HIT instrumental '{title}' ({desiredMode}/#{modelHash8}) -> instant AI ready!");
            }
            else
            {
                if (preloadedPartial != null &&
                    preloadedPartial.SeparatedForMode == desiredMode &&
                    preloadedPartial.InstrumentalPcm16.Length == origPcm16.Length &&
                    preloadedPartial.ChunkState.Length == numChunks)
                {
                    track = preloadedPartial;
                }
                else
                {
                    track = new TrackBuffer
                    {
                        TrackId = trackId,
                        Title = title,
                        OriginalPcm16 = origPcm16,
                        InstrumentalPcm16 = new short[origPcm16.Length],
                        ChunkFrames = chunkFrames,
                        StepFrames = stepFrames,
                        ChunkState = new byte[numChunks],
                        SeparatedForMode = desiredMode,
                        ModelHash8 = modelHash8,
                    };
                }

                // Pre-separate the ENTIRE track in background while TIDALPlayer plays cleanly so karaoke-engine NEVER runs heavy GPU inference while outputting audio!
                if (!isPreload && _enabled && desiredMode != "dsp")
                {
                    if (!ModelCatalog.IsModelInstalled(desiredMode))
                    {
                        await StartModelDownloadAsync(desiredMode);
                    }
                }

                if (!isPreload && _enabled && desiredMode != "dsp" && ModelCatalog.IsModelInstalled(desiredMode))
                {
                    if (await _separator.EnsureLoadedAsync(desiredMode, _useGpu, ct))
                    {
                        chunkFrames = _separator.GetExactChunkFrames();
                        stepFrames = Math.Max(1024, chunkFrames - 8820);
                        EnsureChunkLayout(track, chunkFrames, stepFrames, desiredMode, _separator.LoadedModelHash8);

                        var sepSw = Stopwatch.StartNew();
                        int totalChunks = track.ChunkState.Length;
                        while (true)
                        {
                            ct.ThrowIfCancellationRequested();
                            double curTidalSec = string.Equals(_latestTidalTrackId, trackId, StringComparison.Ordinal)
                                ? (_latestTidalTimeSec + (_playing ? (DateTime.UtcNow - _latestTidalTimeUtc).TotalSeconds : 0.0))
                                : startTimeSec;
                            double curFrame = Math.Max(0.0, curTidalSec * TidalStreamFetcher.TargetSampleRate);
                            int liveChunkIdx = Math.Clamp((int)(curFrame / stepFrames), 0, Math.Max(0, totalChunks - 1));

                            int nextMissing = -1;
                            for (int offset = 0; offset < totalChunks; offset++)
                            {
                                int candidate = (liveChunkIdx + offset) % totalChunks;
                                if (track.ChunkState[candidate] != 2)
                                {
                                    nextMissing = candidate;
                                    break;
                                }
                            }

                            if (nextMissing < 0)
                            {
                                break; // 100% of track is separated!
                            }

                            SeparateSingleChunk(track, nextMissing);
                            int readyNow = 0;
                            for (int i = 0; i < totalChunks; i++)
                            {
                                if (track.ChunkState[i] == 2) readyNow++;
                            }
                            _loadingSeparatedPct = (int)Math.Round(100.0 * readyNow / Math.Max(1, totalChunks));
                            EmitStatus(force: false);

                            // Yield a 10ms breather between chunks so Windows DWM/Tidal audio thread never experiences GPU scheduling jitter
                            await Task.Delay(10, ct).ConfigureAwait(false);
                        }

                        sepSw.Stop();
                        track.FromCache = true;
                        track.SavedInstToCache = true;
                        _cache.SaveInstrumentalBackground(trackId, desiredMode, _separator.LoadedModelHash8, track.InstrumentalPcm16);
                        Logger.Info($"Prepared 100% of '{title}' ({totalChunks} chunks in {sepSw.Elapsed.TotalSeconds:F2}s) -> handing off with smooth fade!");
                    }
                }
            }

            if (isPreload)
            {
                lock (_trackLock)
                {
                    _preloadedTrack = track;
                }
                WakeWorker();
                return;
            }

            double finalStartSec = string.Equals(_latestTidalTrackId, trackId, StringComparison.Ordinal)
                ? (_latestTidalTimeSec + (_playing ? (DateTime.UtcNow - _latestTidalTimeUtc).TotalSeconds : 0.0))
                : startTimeSec;

            double startFrameClamped = Math.Clamp((finalStartSec + _syncOffsetMs * 0.001) * TidalStreamFetcher.TargetSampleRate, 0.0, Math.Max(0, track.TotalFrames - 1));
            int startChunk = track.StepFrames > 0
                ? Math.Clamp((int)(startFrameClamped / track.StepFrames), 0, Math.Max(0, track.ChunkState.Length - 1))
                : 0;
            bool startInAi = desiredMode != "dsp" && startChunk < track.ChunkState.Length && track.ChunkState[startChunk] == 2;
            bool needGradualCrossfade = (!track.FromCache || !_muter.IsMuted) && _playing && _enabled;
            float fadeSec = needGradualCrossfade ? 0.65f : 0.06f;

            lock (_trackLock)
            {
                _rateMultiplier = 1.0;
                _filteredDriftSec = 0.0;
                if (_wasapiOut != null)
                {
                    _activeTrack = track;
                    QueueAudioSeek(startFrameClamped, fadeIn: true, resetDsp: !startInAi, startInAi: startInAi);
                    QueueHandoffFade(0f, 1f, fadeSec, vocalDuckMelt: needGradualCrossfade);
                }
                else
                {
                    ClearPendingAudioCmds();
                    _playFramePos = startFrameClamped;
                    _seekFadeGain = 0f;
                    _handoffGain = 0f;
                    _targetHandoffGain = 1f;
                    _handoffRampSec = fadeSec;
                    if (needGradualCrossfade)
                    {
                        _smoothedVocalGain = 1.0f;
                        _vocalDuckFramesRemaining = 1.15f * _outputSampleRate;
                    }
                    _wasInAiMode = startInAi;
                    _aiCrossfade = startInAi ? 1f : 0f;
                    _dsp.Reset();
                    _activeTrack = track;
                }
            }

            _loadingTrackId = "";
            _loadingSeparatedPct = 0;
            if (_enabled)
            {
                EnsureOutputStarted();
                if (needGradualCrossfade)
                {
                    _muter.FadeToAsync(0f, 650);
                }
                else
                {
                    _muter.SetMuted(true);
                    if (_playing) _muter.ForceMuteBurstAsync();
                }
            }
            WakeWorker();
            UpdateState();
            EmitStatus(force: true);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            if (!isPreload)
            {
                _loadingTrackId = "";
                _loadingSeparatedPct = 0;
                _lastError = ex.Message;
                _state = "error";
                _muter.EnsureUnmuted();
                Logger.Error($"LoadTrack failed ({trackId}): {ex.Message}");
                EmitStatus(force: true);
            }
        }
    }

    public async Task StartModelDownloadAsync(string mode)
    {
        if (!await _downloadLock.WaitAsync(0)) return;
        try
        {
            _lastError = null;
            await ModelCatalog.DownloadModelAsync(mode, progress =>
            {
                _downloadProgress = progress;
                EmitStatus(force: false);
            }, _shutdownCts.Token);

            _downloadProgress = -1f;
            ResetCurrentTrackChunks();
            WakeWorker();
            UpdateState();
            EmitStatus(force: true);
        }
        catch (Exception ex)
        {
            _downloadProgress = -1f;
            _lastError = $"Model download failed: {ex.Message}";
            Logger.Error(_lastError);
            EmitStatus(force: true);
        }
        finally
        {
            _downloadLock.Release();
        }
    }

    private void SeparateSingleChunk(TrackBuffer track, int chunkIdx)
    {
        int chunkFrames = track.ChunkFrames;
        int startFrame = chunkIdx * track.StepFrames;
        int validFrames = Math.Min(chunkFrames, Math.Max(0, track.TotalFrames - startFrame));
        if (validFrames <= 0)
        {
            track.ChunkState[chunkIdx] = 2;
            return;
        }

        int overlapFrames = Math.Min(8820, validFrames);
        bool blendLeftEdge = chunkIdx > 0 && track.ChunkState[chunkIdx - 1] == 2 && overlapFrames > 0;
        short[]? savedLeftOverlap = null;
        if (blendLeftEdge)
        {
            savedLeftOverlap = new short[overlapFrames * 2];
            Array.Copy(track.InstrumentalPcm16, startFrame * 2, savedLeftOverlap, 0, savedLeftOverlap.Length);
        }

        track.ChunkState[chunkIdx] = 1;
        var sw = Stopwatch.StartNew();
        bool ok = _separator.SeparateChunk(
            track.OriginalPcm16.AsSpan(startFrame * 2, validFrames * 2),
            validFrames,
            track.InstrumentalPcm16.AsSpan(startFrame * 2, validFrames * 2)
        );
        sw.Stop();
        _lastChunkMs = sw.Elapsed.TotalMilliseconds;

        if (ok && blendLeftEdge && savedLeftOverlap != null)
        {
            short[] instDest = track.InstrumentalPcm16;
            for (int f = 0; f < overlapFrames; f++)
            {
                float alpha = f / (float)overlapFrames;
                float inv = 1f - alpha;
                int idx = (startFrame + f) * 2;
                instDest[idx] = (short)Math.Clamp((int)MathF.Round(savedLeftOverlap[f * 2] * inv + instDest[idx] * alpha), -32768, 32767);
                instDest[idx + 1] = (short)Math.Clamp((int)MathF.Round(savedLeftOverlap[f * 2 + 1] * inv + instDest[idx + 1] * alpha), -32768, 32767);
            }
        }

        track.ChunkState[chunkIdx] = ok ? (byte)2 : (byte)0;
    }

    private void ResetCurrentTrackChunks()
    {
        lock (_trackLock)
        {
            if (_activeTrack != null)
            {
                Array.Clear(_activeTrack.ChunkState, 0, _activeTrack.ChunkState.Length);
                _activeTrack.SeparatedForMode = _mode;
                _activeTrack.FromCache = false;
                _activeTrack.SavedInstToCache = false;
                _activeTrack.ModelHash8 = "";
            }
            if (_preloadedTrack != null)
            {
                Array.Clear(_preloadedTrack.ChunkState, 0, _preloadedTrack.ChunkState.Length);
                _preloadedTrack.SeparatedForMode = _mode;
                _preloadedTrack.FromCache = false;
                _preloadedTrack.SavedInstToCache = false;
                _preloadedTrack.ModelHash8 = "";
            }
        }
    }

    private void SeparationWorkerThreadMain()
    {
        CancellationToken ct = _shutdownCts.Token;
        short[] chunkOutScratch = Array.Empty<short>();

        while (!ct.IsCancellationRequested)
        {
            try
            {
                if (!_enabled && _separator.IsLoaded && (DateTime.UtcNow - _lastDisabledUtc).TotalSeconds > 600)
                {
                    _separator.Unload();
                    EmitStatus(force: true);
                }

                if (!_enabled || _mode == "dsp")
                {
                    _workerWake.Wait(1500, ct);
                    continue;
                }

                string desiredMode = _mode;
                bool desiredGpu = _useGpu;

                // If active track came from disk cache and is already 100%, only load ONNX if we need to pre-separate _preloadedTrack
                TrackBuffer? activeSnap;
                TrackBuffer? preloadSnap;
                lock (_trackLock)
                {
                    activeSnap = _activeTrack;
                    preloadSnap = _preloadedTrack;
                }

                bool activeNeedsWork = activeSnap != null && NeedsSeparation(activeSnap, desiredMode);
                bool preloadNeedsWork = preloadSnap != null && NeedsSeparation(preloadSnap, desiredMode);

                if (!activeNeedsWork && !preloadNeedsWork)
                {
                    UpdateState();
                    _workerWake.Wait(600, ct);
                    continue;
                }

                if (!_separator.EnsureLoadedAsync(desiredMode, desiredGpu, ct).GetAwaiter().GetResult())
                {
                    if (!ModelCatalog.IsModelInstalled(desiredMode))
                    {
                        StartModelDownloadAsync(desiredMode).GetAwaiter().GetResult();
                    }
                    if (!_separator.EnsureLoadedAsync(desiredMode, desiredGpu, ct).GetAwaiter().GetResult())
                    {
                        UpdateState();
                        _workerWake.Wait(1500, ct);
                        continue;
                    }
                }

                TrackBuffer? targetTrack = null;
                int targetChunkIdx = -1;
                bool workingOnPreload = false;
                int lookaheadReady = 0;
                int chunkFrames = _separator.GetExactChunkFrames();
                int stepFrames = Math.Max(1024, chunkFrames - 8820);
                string loadedHash8 = _separator.LoadedModelHash8;

                lock (_trackLock)
                {
                    if (_activeTrack != null)
                    {
                        EnsureChunkLayout(_activeTrack, chunkFrames, stepFrames, desiredMode, loadedHash8);
                        int totalChunks = _activeTrack.ChunkState.Length;
                        int curChunk = totalChunks > 0
                            ? Math.Clamp((int)(Volatile.Read(ref _playFramePos) / stepFrames), 0, totalChunks - 1)
                            : 0;

                        for (int offset = 0; offset < totalChunks; offset++)
                        {
                            int candidate = (curChunk + offset) % totalChunks;
                            byte st = _activeTrack.ChunkState[candidate];
                            if (st == 2 && offset == lookaheadReady) lookaheadReady++;
                            if (st == 0)
                            {
                                targetTrack = _activeTrack;
                                targetChunkIdx = candidate;
                                targetTrack.ChunkState[candidate] = 1;
                                break;
                            }
                        }

                        if (targetTrack == null && IsTrackFullySeparated(_activeTrack) && !_activeTrack.SavedInstToCache && desiredMode != "dsp")
                        {
                            _activeTrack.SavedInstToCache = true;
                            _cache.SaveInstrumentalBackground(_activeTrack.TrackId, desiredMode, loadedHash8, _activeTrack.InstrumentalPcm16);
                        }
                    }

                    if (targetTrack == null && _preloadedTrack != null)
                    {
                        EnsureChunkLayout(_preloadedTrack, chunkFrames, stepFrames, desiredMode, loadedHash8);
                        int preChunks = _preloadedTrack.ChunkState.Length;
                        for (int c = 0; c < preChunks; c++)
                        {
                            if (_preloadedTrack.ChunkState[c] == 0)
                            {
                                targetTrack = _preloadedTrack;
                                targetChunkIdx = c;
                                targetTrack.ChunkState[c] = 1;
                                workingOnPreload = true;
                                break;
                            }
                        }

                        if (targetTrack == null && IsTrackFullySeparated(_preloadedTrack) && !_preloadedTrack.SavedInstToCache && desiredMode != "dsp")
                        {
                            _preloadedTrack.SavedInstToCache = true;
                            _cache.SaveInstrumentalBackground(_preloadedTrack.TrackId, desiredMode, loadedHash8, _preloadedTrack.InstrumentalPcm16);
                        }
                    }
                }

                if (targetTrack == null || targetChunkIdx < 0)
                {
                    UpdateState();
                    EmitStatus(force: false);
                    _workerWake.Wait(600, ct);
                    continue;
                }

                int startFrame = targetChunkIdx * targetTrack.StepFrames;
                int validFrames = Math.Min(chunkFrames, Math.Max(0, targetTrack.TotalFrames - startFrame));
                if (chunkOutScratch.Length < chunkFrames * 2)
                {
                    chunkOutScratch = new short[chunkFrames * 2];
                }

                var sw = Stopwatch.StartNew();
                bool ok = _separator.SeparateChunk(
                    targetTrack.OriginalPcm16.AsSpan(startFrame * 2, validFrames * 2),
                    validFrames,
                    chunkOutScratch.AsSpan(0, validFrames * 2)
                );
                sw.Stop();
                _lastChunkMs = sw.Elapsed.TotalMilliseconds;

                if (ok)
                {
                    int overlapFrames = Math.Min(8820, validFrames);
                    bool blendLeftEdge = targetChunkIdx > 0 && targetTrack.ChunkState[targetChunkIdx - 1] == 2;
                    short[] instDest = targetTrack.InstrumentalPcm16;

                    if (blendLeftEdge && overlapFrames > 0)
                    {
                        for (int f = 0; f < overlapFrames; f++)
                        {
                            float alpha = f / (float)overlapFrames;
                            float inv = 1f - alpha;
                            int idx = (startFrame + f) * 2;
                            instDest[idx] = (short)Math.Clamp((int)MathF.Round(instDest[idx] * inv + chunkOutScratch[f * 2] * alpha), -32768, 32767);
                            instDest[idx + 1] = (short)Math.Clamp((int)MathF.Round(instDest[idx + 1] * inv + chunkOutScratch[f * 2 + 1] * alpha), -32768, 32767);
                        }
                        Array.Copy(chunkOutScratch, overlapFrames * 2, instDest, (startFrame + overlapFrames) * 2, (validFrames - overlapFrames) * 2);
                    }
                    else
                    {
                        Array.Copy(chunkOutScratch, 0, instDest, startFrame * 2, validFrames * 2);
                    }

                    targetTrack.ChunkState[targetChunkIdx] = 2;

                    if (IsTrackFullySeparated(targetTrack) && !targetTrack.SavedInstToCache && desiredMode != "dsp")
                    {
                        targetTrack.SavedInstToCache = true;
                        _cache.SaveInstrumentalBackground(targetTrack.TrackId, desiredMode, loadedHash8, targetTrack.InstrumentalPcm16);
                    }
                }
                else
                {
                    targetTrack.ChunkState[targetChunkIdx] = 0;
                }

                string prevState = _state;
                UpdateState();
                var snap = GetSnapshot();
                bool important = _state != prevState || snap.SeparatedPct == 100;
                EmitStatus(force: important);

                 // Adaptive pacing: yield a generous 35ms slice between background chunks so active WASAPI audio never stutters
                if (workingOnPreload)
                {
                    ct.WaitHandle.WaitOne(40);
                }
                else
                {
                    ct.WaitHandle.WaitOne(20);
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                Logger.Error($"Separation worker error: {ex.Message}");
                _lastError = ex.Message;
                try { ct.WaitHandle.WaitOne(500); } catch { break; }
            }
        }
    }

    private static bool NeedsSeparation(TrackBuffer track, string desiredMode)
    {
        if (desiredMode == "dsp") return false;
        if (track.SeparatedForMode != desiredMode) return true;
        return !IsTrackFullySeparated(track);
    }

    private static bool IsTrackFullySeparated(TrackBuffer track)
    {
        byte[] st = track.ChunkState;
        if (st.Length == 0) return false;
        for (int i = 0; i < st.Length; i++)
        {
            if (st[i] != 2) return false;
        }
        return true;
    }

    private static void EnsureChunkLayout(TrackBuffer track, int chunkFrames, int stepFrames, string desiredMode, string modelHash8)
    {
        if (track.FromCache &&
            track.SeparatedForMode == desiredMode &&
            (string.IsNullOrEmpty(track.ModelHash8) || string.IsNullOrEmpty(modelHash8) || track.ModelHash8 == modelHash8) &&
            IsTrackFullySeparated(track))
        {
            return;
        }

        int expectedChunks = Math.Max(1, (int)Math.Ceiling(track.TotalFrames / (double)stepFrames));
        if (track.ChunkFrames != chunkFrames ||
            track.StepFrames != stepFrames ||
            track.ChunkState.Length != expectedChunks ||
            track.SeparatedForMode != desiredMode)
        {
            track.ChunkFrames = chunkFrames;
            track.StepFrames = stepFrames;
            track.ChunkState = new byte[expectedChunks];
            track.SeparatedForMode = desiredMode;
            track.FromCache = false;
            track.SavedInstToCache = false;
            track.ModelHash8 = modelHash8;
        }
    }

    private void UpdateState()
    {
        if (!_enabled) { _state = "idle"; return; }
        if (!string.IsNullOrEmpty(_loadingTrackId)) { _state = "loading"; return; }

        var track = _activeTrack;
        if (track == null) { _state = "idle"; return; }
        if (_mode == "dsp" || (!_separator.IsLoaded && !track.FromCache)) { _state = "dsp_bridge"; return; }

        int curChunk = track.StepFrames > 0
            ? Math.Clamp((int)(Volatile.Read(ref _playFramePos) / track.StepFrames), 0, Math.Max(0, track.ChunkState.Length - 1))
            : 0;
        _state = (curChunk < track.ChunkState.Length && track.ChunkState[curChunk] == 2) ? "ai_active" : "dsp_bridge";
    }

    private void EmitStatus(bool force)
    {
        var now = DateTime.UtcNow;
        if (!force && (now - _lastStatusEmitUtc).TotalMilliseconds < 400) return;
        _lastStatusEmitUtc = now;
        StatusChanged?.Invoke();
    }

    private void WakeWorker()
    {
        try
        {
            if (_workerWake.CurrentCount == 0) _workerWake.Release();
        }
        catch { }
    }

    private void EnsureOutputStarted()
    {
        if (_wasapiOut != null) return;
        try
        {
            int devRate = TidalStreamFetcher.TargetSampleRate;
            try
            {
                using var enumerator = new MMDeviceEnumerator();
                using var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
                int r = device.AudioClient.MixFormat.SampleRate;
                if (r >= 22050 && r <= 192000) devRate = r;
            }
            catch (Exception ex)
            {
                Logger.Warn($"Could not query native audio sample rate, defaulting to 44100Hz: {ex.Message}");
            }

            if (devRate != _outputSampleRate || _waveFormat.SampleRate != devRate)
            {
                _outputSampleRate = devRate;
                _waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(devRate, TidalStreamFetcher.TargetChannels);
                _dsp = new DspVocalSuppressor(devRate);
            }

            var output = new WasapiOut(AudioClientShareMode.Shared, useEventSync: true, latency: 240);
            output.Init(this);
            output.Play();
            _wasapiOut = output;
            Logger.Info($"Opened WASAPI shared output @ {devRate}Hz 2ch (240ms event-sync buffer, native rate).");
        }
        catch (Exception ex)
        {
            Logger.Error($"Failed opening WASAPI shared output: {ex.Message}");
            _lastError = ex.Message;
        }
    }

    private void StopOutput()
    {
        var output = _wasapiOut;
        _wasapiOut = null;
        if (output != null)
        {
            try { output.Stop(); } catch { }
            try { output.Dispose(); } catch { }
        }
        if (_avrtHandle != IntPtr.Zero)
        {
            try { AvRevertMmThreadCharacteristics(_avrtHandle); } catch { }
            _avrtHandle = IntPtr.Zero;
        }
        _audioThreadId = 0;
        _playFadeGain = 0f;
        _handoffGain = 1f;
        _targetHandoffGain = 1f;
        _vocalDuckFramesRemaining = 0f;
        _disablingFadeOut = false;
        ClearPendingAudioCmds();
    }

    private float[] _mixScratch = Array.Empty<float>();

    public int Read(byte[] byteBuffer, int byteOffset, int byteCount)
    {
        int floatCount = byteCount / 4;
        if (floatCount <= 0)
        {
            Array.Clear(byteBuffer, byteOffset, byteCount);
            return byteCount;
        }
        if (_mixScratch.Length < floatCount)
        {
            _mixScratch = new float[floatCount];
        }
        ReadSamples(_mixScratch, 0, floatCount);
        Buffer.BlockCopy(_mixScratch, 0, byteBuffer, byteOffset, floatCount * 4);
        int rem = byteCount - (floatCount * 4);
        if (rem > 0)
        {
            Array.Clear(byteBuffer, byteOffset + floatCount * 4, rem);
        }
        return byteCount;
    }

    public int ReadSamples(float[] buffer, int offset, int count)
    {
        int tid = Environment.CurrentManagedThreadId;
        if (_audioThreadId != tid)
        {
            _audioThreadId = tid;
            try
            {
                Thread.CurrentThread.Priority = ThreadPriority.Highest;
                if (_avrtHandle == IntPtr.Zero)
                {
                    uint taskIdx = 0;
                    _avrtHandle = AvSetMmThreadCharacteristicsW("Pro Audio", ref taskIdx);
                }
            }
            catch { }
        }

        int stereoFrames = count / 2;
        int sr = Math.Max(22050, _outputSampleRate);
        var track = _activeTrack;
        bool wantPlay = (_enabled || _disablingFadeOut) && _playing && track != null && track.TotalFrames >= 2;

        if (_hasPendingAudioCmd)
        {
            lock (_audioCmdLock)
            {
                if (_pendingSeekFrame >= 0.0)
                {
                    Volatile.Write(ref _playFramePos, _pendingSeekFrame);
                    _pendingSeekFrame = -1.0;
                    _pendingPauseSnapFrame = -1.0;
                }
                if (_pendingSeekFade)
                {
                    _seekFadeGain = 0f;
                    _pendingSeekFade = false;
                }
                if (_pendingStartInAi)
                {
                    _aiCrossfade = 1f;
                    _wasInAiMode = true;
                    _pendingStartInAi = false;
                }
                if (_pendingDspReset)
                {
                    if (!_wasInAiMode)
                    {
                        _dsp.Reset();
                    }
                    _pendingDspReset = false;
                }
                if (_pendingHandoffFade)
                {
                    if (_pendingHandoffStart.HasValue) _handoffGain = _pendingHandoffStart.Value;
                    _targetHandoffGain = _pendingHandoffTarget;
                    _handoffRampSec = _pendingHandoffSec;
                    _pendingHandoffFade = false;
                }
                if (_pendingCancelVocalDuck)
                {
                    _vocalDuckFramesRemaining = 0f;
                    _pendingVocalDuckTransition = false;
                    _pendingCancelVocalDuck = false;
                }
                else if (_pendingVocalDuckTransition)
                {
                    _smoothedVocalGain = 1.0f;
                    _vocalDuckFramesRemaining = 1.15f * sr;
                    _pendingVocalDuckTransition = false;
                }
                if (_pendingPauseSnapFrame >= 0.0 && !wantPlay && _playFadeGain <= 0.0001f)
                {
                    Volatile.Write(ref _playFramePos, _pendingPauseSnapFrame);
                    _pendingPauseSnapFrame = -1.0;
                }
                _hasPendingAudioCmd = _pendingPauseSnapFrame >= 0.0;
            }
        }

        if (!wantPlay && _playFadeGain <= 0.0001f)
        {
            _playFadeGain = 0f;
            Array.Clear(buffer, offset, count);
            return count;
        }

        if (track == null || track.TotalFrames < 2)
        {
            _playFadeGain = 0f;
            Array.Clear(buffer, offset, count);
            return count;
        }

        short[] orig = track.OriginalPcm16;
        short[] inst = track.InstrumentalPcm16;
        byte[] chunkStates = track.ChunkState;
        int totalFrames = track.TotalFrames;
        int chunkFrames = track.ChunkFrames;
        int stepFrames = Math.Max(1, track.StepFrames);
        bool allowAi = _mode != "dsp" && (_separator.IsLoaded || track.FromCache);

        float invSr = 1f / sr;
        float smoothStep = 55f * invSr;
        double duckSmoothStep = 1.6 * invSr;
        float crossStep = 28f * invSr;
        float seekFadeStep = 110f * invSr;
        float playFadeOutStep = invSr / 0.025f; // ~25 ms cosine pause fade
        float playFadeInStep = invSr / 0.025f;  // ~25 ms cosine resume fade
        float handoffStep = invSr / MathF.Max(0.02f, _handoffRampSec);
        double frameStep = ((double)TidalStreamFetcher.TargetSampleRate / sr) * _rateMultiplier;

        for (int i = 0; i < stereoFrames; i++)
        {
            if (wantPlay)
            {
                if (_playFadeGain < 1f) _playFadeGain = MathF.Min(1f, _playFadeGain + playFadeInStep);
            }
            else
            {
                _playFadeGain = MathF.Max(0f, _playFadeGain - playFadeOutStep);
                if (_playFadeGain <= 0.0001f)
                {
                    _playFadeGain = 0f;
                    int writtenSamples = i * 2;
                    Array.Clear(buffer, offset + writtenSamples, count - writtenSamples);
                    break;
                }
            }

            if (_handoffGain < _targetHandoffGain)
            {
                _handoffGain = MathF.Min(_targetHandoffGain, _handoffGain + handoffStep);
            }
            else if (_handoffGain > _targetHandoffGain)
            {
                _handoffGain = MathF.Max(_targetHandoffGain, _handoffGain - handoffStep);
            }

            if (_disablingFadeOut)
            {
                _smoothedVocalGain += (1.0f - _smoothedVocalGain) * (float)duckSmoothStep;
            }
            else if (_vocalDuckFramesRemaining > 0f)
            {
                _vocalDuckFramesRemaining -= 1f;
                _smoothedVocalGain += (_targetVocalGain - _smoothedVocalGain) * (float)duckSmoothStep;
            }
            else
            {
                _smoothedVocalGain += (_targetVocalGain - _smoothedVocalGain) * smoothStep;
            }

            _smoothedMasterVolume += (_targetMasterVolume - _smoothedMasterVolume) * smoothStep;
            if (_seekFadeGain < 1f) _seekFadeGain = MathF.Min(1f, _seekFadeGain + seekFadeStep);

            int f0 = (int)_playFramePos;
            if (f0 >= totalFrames - 1)
            {
                buffer[offset + i * 2] = 0f;
                buffer[offset + i * 2 + 1] = 0f;
                continue;
            }

            float frac = (float)(_playFramePos - f0);
            float origL = CubicSample(orig, f0, totalFrames, 0, frac);
            float origR = CubicSample(orig, f0, totalFrames, 1, frac);

            int chunkIdx = chunkStates.Length > 0 ? Math.Clamp(f0 / stepFrames, 0, chunkStates.Length - 1) : 0;
            bool aiReady = allowAi && chunkStates.Length > 0 && (
                chunkStates[chunkIdx] == 2 ||
                (chunkIdx > 0 && chunkStates[chunkIdx - 1] == 2 && f0 < (chunkIdx - 1) * stepFrames + chunkFrames)
            );
            _aiCrossfade = aiReady
                ? MathF.Min(1f, _aiCrossfade + crossStep)
                : MathF.Max(0f, _aiCrossfade - crossStep);

            float outL, outR;
            float vocalGain = _smoothedVocalGain;

            if (_mode == "dsp")
            {
                if (_wasInAiMode)
                {
                    _dsp.Reset();
                    _wasInAiMode = false;
                }
                _dsp.ProcessSample(origL, origR, out float dspOrigL, out float dspOrigR, out float dspInstL, out float dspInstR);
                outL = dspInstL + vocalGain * (dspOrigL - dspInstL);
                outR = dspInstR + vocalGain * (dspOrigR - dspInstR);
            }
            else if (aiReady && _aiCrossfade >= 0.999f)
            {
                // Steady AI path: zero FFT work inside real-time audio callback!
                _wasInAiMode = true;
                float aiInstL = CubicSample(inst, f0, totalFrames, 0, frac);
                float aiInstR = CubicSample(inst, f0, totalFrames, 1, frac);

                outL = aiInstL + vocalGain * (origL - aiInstL);
                outR = aiInstR + vocalGain * (origR - aiInstR);
            }
            else if (aiReady && _aiCrossfade > 0.001f)
            {
                // Smooth cosine fade-in from original full mix to AI stem (never metallic DSP bridge!)
                _wasInAiMode = true;
                float aiInstL = CubicSample(inst, f0, totalFrames, 0, frac);
                float aiInstR = CubicSample(inst, f0, totalFrames, 1, frac);
                float aiMixL = aiInstL + vocalGain * (origL - aiInstL);
                float aiMixR = aiInstR + vocalGain * (origR - aiInstR);

                float alpha = 0.5f - 0.5f * MathF.Cos(_aiCrossfade * MathF.PI);
                outL = origL * (1f - alpha) + aiMixL * alpha;
                outR = origR * (1f - alpha) + aiMixR * alpha;
            }
            else
            {
                // AI chunk not ready yet -> play clean original audio while AI catches up, then crossfade in!
                _wasInAiMode = false;
                outL = origL;
                outR = origR;
            }

            float playEnv = _playFadeGain >= 0.9999f
                ? 1f
                : (_playFadeGain <= 0.0001f ? 0f : 0.5f - 0.5f * MathF.Cos(_playFadeGain * MathF.PI));
            float handoffEnv = _handoffGain >= 0.9999f
                ? 1f
                : (_handoffGain <= 0.0001f ? 0f : 0.5f - 0.5f * MathF.Cos(_handoffGain * MathF.PI));
            float gain = _smoothedMasterVolume * _seekFadeGain * playEnv * handoffEnv;
            buffer[offset + i * 2] = SoftLimit(outL * gain);
            buffer[offset + i * 2 + 1] = SoftLimit(outR * gain);

            Volatile.Write(ref _playFramePos, _playFramePos + frameStep);
        }

        if (_hasPendingAudioCmd && !wantPlay && _playFadeGain <= 0.0001f)
        {
            lock (_audioCmdLock)
            {
                if (_pendingPauseSnapFrame >= 0.0)
                {
                    Volatile.Write(ref _playFramePos, _pendingPauseSnapFrame);
                    _pendingPauseSnapFrame = -1.0;
                }
                _hasPendingAudioCmd = _pendingSeekFrame >= 0.0;
            }
        }

        return count;
    }

    public void Dispose()
    {
        try { _shutdownCts.Cancel(); } catch { }
        try { _loadCts?.Cancel(); } catch { }
        try { _preloadCts?.Cancel(); } catch { }
        try { _modeSwitchCts?.Cancel(); } catch { }
        try { _disableStopCts?.Cancel(); } catch { }
        WakeWorker();
        _muter.EnsureUnmuted();
        StopOutput();
        try { _workerThread?.Join(1500); } catch { }
        _separator.Dispose();
        _muter.Dispose();
        _shutdownCts.Dispose();
    }
}
