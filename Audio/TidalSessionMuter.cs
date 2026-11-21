using KaraokeEngine.Utils;
using NAudio.CoreAudioApi;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace KaraokeEngine.Audio;

/// <summary>
/// Controls TIDALPlayer/TIDAL Windows WASAPI session volume & mute state
/// with smooth cosine crossfades and guaranteed unmute restoration on disable/exit.
/// </summary>
internal sealed class TidalSessionMuter : IDisposable
{
    private readonly object _lock = new();
    private bool _shouldMute;
    private volatile bool _isFading;
    private float _currentVolume = 1.0f;
    private float _targetVolume = 1.0f;
    private DateTime _lastEnforceUtc = DateTime.MinValue;
    private CancellationTokenSource? _fadeCts;

    public bool IsMuted
    {
        get { lock (_lock) return _shouldMute; }
    }

    public void FadeToAsync(float targetVolume, int durationMs)
    {
        targetVolume = Math.Clamp(targetVolume, 0f, 1f);
        CancellationTokenSource cts;
        float startVol;

        lock (_lock)
        {
            _fadeCts?.Cancel();
            _fadeCts?.Dispose();
            _fadeCts = null;

            _targetVolume = targetVolume;
            _shouldMute = targetVolume <= 0.001f;
            _lastEnforceUtc = DateTime.UtcNow;
            startVol = _currentVolume;

            if (durationMs <= 16 || MathF.Abs(targetVolume - startVol) < 0.01f)
            {
                _isFading = false;
                _currentVolume = targetVolume;
                ApplySessionGainLocked(targetVolume, _shouldMute);
                return;
            }

            cts = new CancellationTokenSource();
            _fadeCts = cts;
            _isFading = true;
            // Ensure unmuted immediately when ramping volume gradually (clickless at startVol)
            ApplySessionGainLocked(startVol, mute: false);
        }

        CancellationToken ct = cts.Token;
        _ = Task.Run(async () =>
        {
            const int stepMs = 16;
            int totalSteps = Math.Max(2, durationMs / stepMs);
            try
            {
                for (int step = 1; step <= totalSteps; step++)
                {
                    if (ct.IsCancellationRequested) return;
                    await Task.Delay(stepMs, ct).ConfigureAwait(false);

                    float t = Math.Clamp(step / (float)totalSteps, 0f, 1f);
                    float ease = 0.5f - 0.5f * MathF.Cos(t * MathF.PI);
                    float vol = startVol + (targetVolume - startVol) * ease;

                    lock (_lock)
                    {
                        if (ct.IsCancellationRequested) return;
                        _currentVolume = vol;
                        ApplySessionGainLocked(vol, mute: false);
                    }
                }

                lock (_lock)
                {
                    if (ct.IsCancellationRequested) return;
                    _currentVolume = targetVolume;
                    ApplySessionGainLocked(targetVolume, targetVolume <= 0.001f);
                }
            }
            catch (OperationCanceledException) { }
            finally
            {
                lock (_lock)
                {
                    if (ReferenceEquals(_fadeCts, cts))
                    {
                        _isFading = false;
                        _fadeCts = null;
                    }
                }
                cts.Dispose();
            }
        });
    }

    public void SetMuted(bool mute)
    {
        lock (_lock)
        {
            _fadeCts?.Cancel();
            _fadeCts?.Dispose();
            _fadeCts = null;
            _isFading = false;

            _shouldMute = mute;
            _targetVolume = mute ? 0f : 1f;
            _currentVolume = _targetVolume;
            _lastEnforceUtc = DateTime.UtcNow;
            ApplySessionGainLocked(_currentVolume, mute);
        }
    }

    public void EnforceIfMuted()
    {
        lock (_lock)
        {
            if (!_shouldMute || _isFading) return;
            var now = DateTime.UtcNow;
            if ((now - _lastEnforceUtc).TotalSeconds < 1.5) return;
            _lastEnforceUtc = now;
            ApplySessionGainLocked(0f, mute: true);
        }
    }

    public void EnforceIfMutedAsync()
    {
        lock (_lock)
        {
            if (!_shouldMute || _isFading) return;
            var now = DateTime.UtcNow;
            if ((now - _lastEnforceUtc).TotalSeconds < 1.5) return;
            _lastEnforceUtc = now;
        }

        _ = Task.Run(() =>
        {
            lock (_lock)
            {
                if (_shouldMute && !_isFading) ApplySessionGainLocked(0f, mute: true);
            }
        });
    }

    public void ForceMuteBurstAsync()
    {
        lock (_lock)
        {
            if (!_shouldMute) return;
            _fadeCts?.Cancel();
            _fadeCts?.Dispose();
            _fadeCts = null;
            _isFading = false;
            _currentVolume = 0f;
            _targetVolume = 0f;
            _lastEnforceUtc = DateTime.UtcNow;
        }

        _ = Task.Run(async () =>
        {
            lock (_lock)
            {
                if (!_shouldMute || _isFading) return;
                ApplySessionGainLocked(0f, mute: true);
            }

            await Task.Delay(120).ConfigureAwait(false);
            lock (_lock)
            {
                if (!_shouldMute || _isFading) return;
                ApplySessionGainLocked(0f, mute: true);
            }

            await Task.Delay(280).ConfigureAwait(false);
            lock (_lock)
            {
                if (!_shouldMute || _isFading) return;
                ApplySessionGainLocked(0f, mute: true);
            }
        });
    }

    public void EnsureUnmuted()
    {
        lock (_lock)
        {
            _fadeCts?.Cancel();
            _fadeCts?.Dispose();
            _fadeCts = null;
            _isFading = false;

            if (_shouldMute || _currentVolume < 0.99f)
            {
                Logger.Info("Restoring TIDALPlayer WASAPI session unmute.");
            }
            _shouldMute = false;
            _targetVolume = 1.0f;
            _currentVolume = 1.0f;
            ApplySessionGainLocked(1.0f, mute: false);
        }
    }

    private static void ApplySessionGainLocked(float volume, bool mute)
    {
        volume = Math.Clamp(volume, 0f, 1f);
        try
        {
            var targetPids = new HashSet<uint>();
            foreach (var procName in new[] { "TIDALPlayer", "TIDAL" })
            {
                foreach (var p in Process.GetProcessesByName(procName))
                {
                    try { targetPids.Add((uint)p.Id); }
                    finally { p.Dispose(); }
                }
            }

            if (targetPids.Count == 0) return;

            using var enumerator = new MMDeviceEnumerator();
            using var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
            var sessions = device.AudioSessionManager.Sessions;

            for (int i = 0; i < sessions.Count; i++)
            {
                var session = sessions[i];
                uint pid = session.GetProcessID;
                if (targetPids.Contains(pid))
                {
                    var vol = session.SimpleAudioVolume;
                    if (mute)
                    {
                        if (!vol.Mute) vol.Mute = true;
                        if (MathF.Abs(vol.Volume - volume) > 0.005f) vol.Volume = volume;
                    }
                    else
                    {
                        if (MathF.Abs(vol.Volume - volume) > 0.005f) vol.Volume = volume;
                        if (vol.Mute) vol.Mute = false;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Logger.Warn($"TidalSessionMuter error (vol={volume:F2}, mute={mute}): {ex.Message}");
        }
    }

    public void Dispose()
    {
        EnsureUnmuted();
    }
}
