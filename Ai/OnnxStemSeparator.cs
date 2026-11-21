using KaraokeEngine.Audio;
using KaraokeEngine.Utils;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace KaraokeEngine.Ai;

internal sealed class OnnxStemSeparator : IDisposable
{
    private const float Inv32768 = 1f / 32768f;
    private readonly object _sync = new();
    private readonly SemaphoreSlim _loadGate = new(1, 1);
    private InferenceSession? _session;
    private MdxModelSpec? _spec;

    private sealed class StftWorkerSlot
    {
        public readonly MixedRadixFft Fft;
        public readonly float[] WinFrame;
        public readonly float[] FftRe;
        public readonly float[] FftIm;
        public readonly float[] IfftRe;
        public readonly float[] IfftIm;
        public readonly float[] IfftTime;
        public readonly float[] OlaL;
        public readonly float[] OlaR;

        public StftWorkerSlot(int nFft, int paddedLen)
        {
            Fft = new MixedRadixFft(nFft);
            WinFrame = new float[nFft];
            FftRe = new float[nFft];
            FftIm = new float[nFft];
            IfftRe = new float[nFft];
            IfftIm = new float[nFft];
            IfftTime = new float[nFft];
            OlaL = new float[paddedLen];
            OlaR = new float[paddedLen];
        }
    }

    private float[]? _window;

    private string _loadedMode = "";
    private bool _loadedWithGpu;
    private bool _ioIsFp16;
    private string _inputName = "input";
    private string _outputName = "output";

    private int _dimF = 2048;
    private int _dimT = 256;
    private int _nFft = 5120;
    private int _hopLength = 1024;

    // Preallocated STFT / Zero-Copy OrtValue / iSTFT working buffers (zero per-chunk allocations)
    private float[] _inputTensorBuf = Array.Empty<float>();
    private float[] _outputTensorBuf = Array.Empty<float>();
    private Float16[] _inputFp16Buf = Array.Empty<Float16>();
    private Float16[] _outputFp16Buf = Array.Empty<Float16>();
    private float[] _padL = Array.Empty<float>();
    private float[] _padR = Array.Empty<float>();
    private float[] _olaOutL = Array.Empty<float>();
    private float[] _olaOutR = Array.Empty<float>();
    private float[] _olaInvNorm = Array.Empty<float>();
    private StftWorkerSlot[] _workers = Array.Empty<StftWorkerSlot>();

    private RunOptions? _runOptions;
    private OrtValue? _inputOrtValue;
    private OrtValue? _outputOrtValue;
    private string[] _inputNames = Array.Empty<string>();
    private string[] _outputNames = Array.Empty<string>();
    private OrtValue[] _inputValues = Array.Empty<OrtValue>();
    private OrtValue[] _outputValues = Array.Empty<OrtValue>();

    public bool IsLoaded
    {
        get { lock (_sync) return _session != null; }
    }

    public bool IsFp16 { get; private set; }
    public string ActiveProvider { get; private set; } = "DSP";
    public string LoadedMode => _loadedMode;
    public string LoadedModelHash8 { get; private set; } = "";

    public async Task<bool> EnsureLoadedAsync(string mode, bool useGpu, CancellationToken ct)
    {
        string normalizedMode = string.Equals(mode, "instrumental", StringComparison.OrdinalIgnoreCase)
            ? "instrumental"
            : "karaoke";

        lock (_sync)
        {
            if (_session != null && _loadedMode == normalizedMode && _loadedWithGpu == useGpu)
            {
                return true;
            }
        }

        await _loadGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            lock (_sync)
            {
                if (_session != null && _loadedMode == normalizedMode && _loadedWithGpu == useGpu)
                {
                    return true;
                }
            }

            if (!ModelCatalog.IsModelInstalled(normalizedMode))
            {
                return false;
            }

            string path = ModelCatalog.GetModelPath(normalizedMode);
            MdxModelSpec spec = await ModelCatalog.ResolveSpecAsync(path, normalizedMode, ct).ConfigureAwait(false);
            string hash8 = ModelCatalog.GetModelHash8(normalizedMode);

            InferenceSession? session = null;
            string providerName = "CPU";
            bool loadedFp16File = false;

            if (useGpu)
            {
                bool preferFp16 = string.Equals(Environment.GetEnvironmentVariable("KARAOKE_USE_FP16"), "1", StringComparison.Ordinal);
                if (preferFp16 && ModelCatalog.HasFp16Model(normalizedMode))
                {
                    string fp16Path = ModelCatalog.GetFp16ModelPath(normalizedMode);
                    try
                    {
                        using var dmlOpts = CreateDmlSessionOptions();
                        session = new InferenceSession(fp16Path, dmlOpts);
                        providerName = "DirectML";
                        loadedFp16File = true;
                    }
                    catch (Exception ex)
                    {
                        Logger.Warn($"DirectML FP16 initialization failed, retrying canonical FP32 on DirectML: {ex.Message}");
                        session?.Dispose();
                        session = null;
                        loadedFp16File = false;
                    }
                }

                if (session == null)
                {
                    try
                    {
                        using var dmlOpts = CreateDmlSessionOptions();
                        session = new InferenceSession(path, dmlOpts);
                        providerName = "DirectML";
                    }
                    catch (Exception ex)
                    {
                        Logger.Warn($"DirectML initialization failed, falling back to CPU: {ex.Message}");
                        session?.Dispose();
                        session = null;
                    }
                }
            }

            if (session == null)
            {
                using var cpuOpts = new SessionOptions
                {
                    EnableMemoryPattern = true,
                    ExecutionMode = ExecutionMode.ORT_PARALLEL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
                };
                session = new InferenceSession(path, cpuOpts);
                providerName = "CPU";
                loadedFp16File = false;
            }

            lock (_sync)
            {
                DisposeOrtBindingsLocked();
                _session?.Dispose();
                _session = session;
                _spec = spec;
                _loadedMode = normalizedMode;
                _loadedWithGpu = useGpu;
                LoadedModelHash8 = hash8;

                var inputMeta = session.InputMetadata.First();
                _inputName = inputMeta.Key;
                _outputName = session.OutputMetadata.Keys.FirstOrDefault() ?? "output";
                _ioIsFp16 = inputMeta.Value.ElementDataType == TensorElementType.Float16;
                IsFp16 = _ioIsFp16 || loadedFp16File;
                ActiveProvider = providerName == "DirectML" && IsFp16 ? "DirectML FP16" : providerName;

                int[] dims = inputMeta.Value.Dimensions;
                _dimF = dims.Length == 4 && dims[2] > 0 ? dims[2] : spec.DimF;
                _dimT = dims.Length == 4 && dims[3] > 0 ? dims[3] : spec.DimT;
                _nFft = spec.NFft;
                _hopLength = spec.HopLength;

                _window = MixedRadixFft.CreatePeriodicHannWindow(_nFft);
                AllocateWorkspaceLocked();
                Logger.Info($"Loaded ONNX '{normalizedMode}' ({ActiveProvider}, #{hash8}) -> shape=[1,4,{_dimF},{_dimT}], n_fft={_nFft}, stem={spec.PrimaryStem}, comp={spec.Compensate}");
                return true;
            }
        }
        finally
        {
            _loadGate.Release();
        }
    }

    private static SessionOptions CreateDmlSessionOptions()
    {
        var dmlOpts = new SessionOptions
        {
            EnableMemoryPattern = false,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
        };
        dmlOpts.AppendExecutionProvider_DML(0);
        return dmlOpts;
    }

    private void AllocateWorkspaceLocked()
    {
        DisposeOrtBindingsLocked();

        int nSamples = (_dimT - 1) * _hopLength;
        int paddedLen = nSamples + _nFft;
        int tensorLen = 4 * _dimF * _dimT;

        _padL = new float[paddedLen];
        _padR = new float[paddedLen];
        _olaOutL = new float[paddedLen];
        _olaOutR = new float[paddedLen];
        _olaInvNorm = new float[paddedLen];

        // Precompute constant periodic Hann OLA normalization curve once per model load
        float[] normAccum = new float[paddedLen];
        for (int t = 0; t < _dimT; t++)
        {
            int off = t * _hopLength;
            for (int n = 0; n < _nFft; n++)
            {
                float w = _window![n];
                normAccum[off + n] += w * w;
            }
        }
        for (int i = 0; i < paddedLen; i++)
        {
            _olaInvNorm[i] = 1f / MathF.Max(1e-7f, normAccum[i]);
        }

        int workerCount = Math.Clamp(Environment.ProcessorCount / 4, 2, 2);
        _workers = new StftWorkerSlot[workerCount];
        for (int w = 0; w < workerCount; w++)
        {
            _workers[w] = new StftWorkerSlot(_nFft, paddedLen);
        }

        long[] shape = { 1L, 4L, _dimF, _dimT };
        _runOptions = new RunOptions();
        if (_ioIsFp16)
        {
            _inputTensorBuf = Array.Empty<float>();
            _outputTensorBuf = Array.Empty<float>();
            _inputFp16Buf = new Float16[tensorLen];
            _outputFp16Buf = new Float16[tensorLen];
            _inputOrtValue = OrtValue.CreateTensorValueFromMemory(_inputFp16Buf, shape);
            _outputOrtValue = OrtValue.CreateTensorValueFromMemory(_outputFp16Buf, shape);
        }
        else
        {
            _inputFp16Buf = Array.Empty<Float16>();
            _outputFp16Buf = Array.Empty<Float16>();
            _inputTensorBuf = new float[tensorLen];
            _outputTensorBuf = new float[tensorLen];
            _inputOrtValue = OrtValue.CreateTensorValueFromMemory(_inputTensorBuf, shape);
            _outputOrtValue = OrtValue.CreateTensorValueFromMemory(_outputTensorBuf, shape);
        }

        _inputNames = new[] { _inputName };
        _outputNames = new[] { _outputName };
        _inputValues = new[] { _inputOrtValue };
        _outputValues = new[] { _outputOrtValue };
    }

    public int GetExactChunkFrames()
    {
        lock (_sync) return (_dimT - 1) * _hopLength;
    }

    public bool SeparateChunk(
        ReadOnlySpan<short> stereoPcm16Interleaved,
        int validStereoFrames,
        Span<short> instrumentalPcm16Out)
    {
        lock (_sync)
        {
            if (_session == null || _window == null || _spec == null || _workers.Length == 0)
                return false;

            int nSamples = (_dimT - 1) * _hopLength;
            int halfFft = _nFft / 2;
            int paddedLen = nSamples + _nFft;
            int framesToCopy = Math.Min(validStereoFrames, nSamples);

            Array.Clear(_padL, 0, paddedLen);
            Array.Clear(_padR, 0, paddedLen);

            for (int i = 0; i < framesToCopy; i++)
            {
                _padL[halfFft + i] = stereoPcm16Interleaved[i * 2] * Inv32768;
                _padR[halfFft + i] = stereoPcm16Interleaved[i * 2 + 1] * Inv32768;
            }

            RunCoreLocked(framesToCopy);

            float comp = _spec.Compensate;
            bool primaryIsVocals = string.Equals(_spec.PrimaryStem, "Vocals", StringComparison.OrdinalIgnoreCase);

            for (int i = 0; i < framesToCopy; i++)
            {
                int pIdx = halfFft + i;
                float scale = _olaInvNorm[pIdx] * comp;
                float stemL = _olaOutL[pIdx] * scale;
                float stemR = _olaOutR[pIdx] * scale;

                float origL = stereoPcm16Interleaved[i * 2] * Inv32768;
                float origR = stereoPcm16Interleaved[i * 2 + 1] * Inv32768;

                float instL = primaryIsVocals ? (origL - stemL) : stemL;
                float instR = primaryIsVocals ? (origR - stemR) : stemR;

                instrumentalPcm16Out[i * 2] = (short)Math.Clamp((int)MathF.Round(instL * 32767f), -32768, 32767);
                instrumentalPcm16Out[i * 2 + 1] = (short)Math.Clamp((int)MathF.Round(instR * 32767f), -32768, 32767);
            }

            return true;
        }
    }

    public bool SeparateChunk(
        ReadOnlySpan<float> stereoInterleaved,
        int validStereoFrames,
        Span<float> instrumentalInterleavedOut)
    {
        lock (_sync)
        {
            if (_session == null || _window == null || _spec == null || _workers.Length == 0)
                return false;

            int nSamples = (_dimT - 1) * _hopLength;
            int halfFft = _nFft / 2;
            int paddedLen = nSamples + _nFft;
            int framesToCopy = Math.Min(validStereoFrames, nSamples);

            Array.Clear(_padL, 0, paddedLen);
            Array.Clear(_padR, 0, paddedLen);

            for (int i = 0; i < framesToCopy; i++)
            {
                _padL[halfFft + i] = stereoInterleaved[i * 2];
                _padR[halfFft + i] = stereoInterleaved[i * 2 + 1];
            }

            RunCoreLocked(framesToCopy);

            float comp = _spec.Compensate;
            bool primaryIsVocals = string.Equals(_spec.PrimaryStem, "Vocals", StringComparison.OrdinalIgnoreCase);

            for (int i = 0; i < framesToCopy; i++)
            {
                int pIdx = halfFft + i;
                float scale = _olaInvNorm[pIdx] * comp;
                float stemL = _olaOutL[pIdx] * scale;
                float stemR = _olaOutR[pIdx] * scale;

                if (primaryIsVocals)
                {
                    instrumentalInterleavedOut[i * 2] = stereoInterleaved[i * 2] - stemL;
                    instrumentalInterleavedOut[i * 2 + 1] = stereoInterleaved[i * 2 + 1] - stemR;
                }
                else
                {
                    instrumentalInterleavedOut[i * 2] = stemL;
                    instrumentalInterleavedOut[i * 2 + 1] = stemR;
                }
            }

            return true;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Float16 FloatToFp16(float v)
        => new(BitConverter.HalfToUInt16Bits((Half)Math.Clamp(v, -65504f, 65504f)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Fp16ToFloat(Float16 h)
    {
        float v = (float)BitConverter.UInt16BitsToHalf(h.value);
        return float.IsFinite(v) ? v : 0f;
    }

    private void RunCoreLocked(int framesToCopy)
    {
        int nSamples = (_dimT - 1) * _hopLength;
        int halfFft = _nFft / 2;
        int paddedLen = nSamples + _nFft;

        if (framesToCopy > 1)
        {
            for (int i = 0; i < halfFft; i++)
            {
                int mirrorL = Math.Min(framesToCopy - 1, halfFft - i);
                _padL[i] = _padL[halfFft + mirrorL];
                _padR[i] = _padR[halfFft + mirrorL];

                int tailIdx = halfFft + nSamples + i;
                int mirrorR = Math.Max(0, nSamples - 2 - i);
                _padL[tailIdx] = _padL[halfFft + mirrorR];
                _padR[tailIdx] = _padR[halfFft + mirrorR];
            }
        }

        int ftStride = _dimF * _dimT;
        int ch0Base = 0;
        int ch1Base = ftStride;
        int ch2Base = 2 * ftStride;
        int ch3Base = 3 * ftStride;

        int workerCount = _workers.Length;
        var parallelOpts = new ParallelOptions { MaxDegreeOfParallelism = workerCount };
        float[] window = _window!;
        float[] padL = _padL;
        float[] padR = _padR;
        bool ioIsFp16 = _ioIsFp16;
        float[] inBuf = _inputTensorBuf;
        float[] outBuf = _outputTensorBuf;
        Float16[] inFp16 = _inputFp16Buf;
        Float16[] outFp16 = _outputFp16Buf;

        // Parallel Forward STFT across disjoint t-slices
        Parallel.For(0, workerCount, parallelOpts, w =>
        {
            int tStart = (w * _dimT) / workerCount;
            int tEnd = ((w + 1) * _dimT) / workerCount;
            var slot = _workers[w];

            for (int t = tStart; t < tEnd; t++)
            {
                int offset = t * _hopLength;

                for (int n = 0; n < _nFft; n++) slot.WinFrame[n] = padL[offset + n] * window[n];
                slot.Fft.Forward(slot.WinFrame, slot.FftRe, slot.FftIm);
                if (ioIsFp16)
                {
                    for (int f = 0; f < _dimF; f++)
                    {
                        int cell = f * _dimT + t;
                        inFp16[ch0Base + cell] = FloatToFp16(slot.FftRe[f]);
                        inFp16[ch1Base + cell] = FloatToFp16(slot.FftIm[f]);
                    }
                }
                else
                {
                    for (int f = 0; f < _dimF; f++)
                    {
                        int cell = f * _dimT + t;
                        inBuf[ch0Base + cell] = slot.FftRe[f];
                        inBuf[ch1Base + cell] = slot.FftIm[f];
                    }
                }

                for (int n = 0; n < _nFft; n++) slot.WinFrame[n] = padR[offset + n] * window[n];
                slot.Fft.Forward(slot.WinFrame, slot.FftRe, slot.FftIm);
                if (ioIsFp16)
                {
                    for (int f = 0; f < _dimF; f++)
                    {
                        int cell = f * _dimT + t;
                        inFp16[ch2Base + cell] = FloatToFp16(slot.FftRe[f]);
                        inFp16[ch3Base + cell] = FloatToFp16(slot.FftIm[f]);
                    }
                }
                else
                {
                    for (int f = 0; f < _dimF; f++)
                    {
                        int cell = f * _dimT + t;
                        inBuf[ch2Base + cell] = slot.FftRe[f];
                        inBuf[ch3Base + cell] = slot.FftIm[f];
                    }
                }
            }
        });

        // Zero-allocation DirectML GPU inference directly between prebound OrtValues
        _session!.Run(_runOptions!, _inputNames, _inputValues, _outputNames, _outputValues);

        // Parallel Inverse STFT into disjoint worker OLA scratch buffers
        Parallel.For(0, workerCount, parallelOpts, w =>
        {
            int tStart = (w * _dimT) / workerCount;
            int tEnd = ((w + 1) * _dimT) / workerCount;
            if (tEnd <= tStart) return;

            var slot = _workers[w];
            int rangeStart = tStart * _hopLength;
            int rangeLen = ((tEnd - 1) * _hopLength + _nFft) - rangeStart;
            Array.Clear(slot.OlaL, rangeStart, rangeLen);
            Array.Clear(slot.OlaR, rangeStart, rangeLen);

            for (int t = tStart; t < tEnd; t++)
            {
                Array.Clear(slot.IfftRe, 0, _nFft);
                Array.Clear(slot.IfftIm, 0, _nFft);
                if (ioIsFp16)
                {
                    for (int f = 0; f < _dimF; f++)
                    {
                        int cell = f * _dimT + t;
                        float re = Fp16ToFloat(outFp16[ch0Base + cell]);
                        float im = Fp16ToFloat(outFp16[ch1Base + cell]);
                        slot.IfftRe[f] = re;
                        slot.IfftIm[f] = im;
                        if (f > 0 && f < halfFft)
                        {
                            int mir = _nFft - f;
                            slot.IfftRe[mir] = re;
                            slot.IfftIm[mir] = -im;
                        }
                    }
                }
                else
                {
                    for (int f = 0; f < _dimF; f++)
                    {
                        int cell = f * _dimT + t;
                        float re = outBuf[ch0Base + cell];
                        float im = outBuf[ch1Base + cell];
                        slot.IfftRe[f] = re;
                        slot.IfftIm[f] = im;
                        if (f > 0 && f < halfFft)
                        {
                            int mir = _nFft - f;
                            slot.IfftRe[mir] = re;
                            slot.IfftIm[mir] = -im;
                        }
                    }
                }
                slot.Fft.Inverse(slot.IfftRe, slot.IfftIm, slot.IfftTime);

                int offset = t * _hopLength;
                for (int n = 0; n < _nFft; n++)
                {
                    slot.OlaL[offset + n] += slot.IfftTime[n] * window[n];
                }

                Array.Clear(slot.IfftRe, 0, _nFft);
                Array.Clear(slot.IfftIm, 0, _nFft);
                if (ioIsFp16)
                {
                    for (int f = 0; f < _dimF; f++)
                    {
                        int cell = f * _dimT + t;
                        float re = Fp16ToFloat(outFp16[ch2Base + cell]);
                        float im = Fp16ToFloat(outFp16[ch3Base + cell]);
                        slot.IfftRe[f] = re;
                        slot.IfftIm[f] = im;
                        if (f > 0 && f < halfFft)
                        {
                            int mir = _nFft - f;
                            slot.IfftRe[mir] = re;
                            slot.IfftIm[mir] = -im;
                        }
                    }
                }
                else
                {
                    for (int f = 0; f < _dimF; f++)
                    {
                        int cell = f * _dimT + t;
                        float re = outBuf[ch2Base + cell];
                        float im = outBuf[ch3Base + cell];
                        slot.IfftRe[f] = re;
                        slot.IfftIm[f] = im;
                        if (f > 0 && f < halfFft)
                        {
                            int mir = _nFft - f;
                            slot.IfftRe[mir] = re;
                            slot.IfftIm[mir] = -im;
                        }
                    }
                }
                slot.Fft.Inverse(slot.IfftRe, slot.IfftIm, slot.IfftTime);

                for (int n = 0; n < _nFft; n++)
                {
                    slot.OlaR[offset + n] += slot.IfftTime[n] * window[n];
                }
            }
        });

        // Fast linear OLA reduction over each worker's active span
        Array.Clear(_olaOutL, 0, paddedLen);
        Array.Clear(_olaOutR, 0, paddedLen);
        for (int w = 0; w < workerCount; w++)
        {
            int tStart = (w * _dimT) / workerCount;
            int tEnd = ((w + 1) * _dimT) / workerCount;
            if (tEnd <= tStart) continue;

            var slot = _workers[w];
            int rangeStart = tStart * _hopLength;
            int rangeEnd = (tEnd - 1) * _hopLength + _nFft;
            float[] srcL = slot.OlaL;
            float[] srcR = slot.OlaR;
            for (int idx = rangeStart; idx < rangeEnd; idx++)
            {
                _olaOutL[idx] += srcL[idx];
                _olaOutR[idx] += srcR[idx];
            }
        }
    }

    private void DisposeOrtBindingsLocked()
    {
        try { _inputOrtValue?.Dispose(); } catch { }
        try { _outputOrtValue?.Dispose(); } catch { }
        try { _runOptions?.Dispose(); } catch { }
        _inputOrtValue = null;
        _outputOrtValue = null;
        _runOptions = null;
        _inputValues = Array.Empty<OrtValue>();
        _outputValues = Array.Empty<OrtValue>();
    }

    public void Unload()
    {
        lock (_sync)
        {
            DisposeOrtBindingsLocked();
            _session?.Dispose();
            _session = null;
            _loadedMode = "";
            LoadedModelHash8 = "";
            IsFp16 = false;
            _ioIsFp16 = false;
            ActiveProvider = "DSP";
            _inputTensorBuf = Array.Empty<float>();
            _outputTensorBuf = Array.Empty<float>();
            _inputFp16Buf = Array.Empty<Float16>();
            _outputFp16Buf = Array.Empty<Float16>();
            _padL = Array.Empty<float>();
            _padR = Array.Empty<float>();
            _olaOutL = Array.Empty<float>();
            _olaOutR = Array.Empty<float>();
            _olaInvNorm = Array.Empty<float>();
            _workers = Array.Empty<StftWorkerSlot>();
        }
        Logger.Info("Unloaded ONNX session & freed VRAM/workspace.");
    }

    public void Dispose()
    {
        Unload();
        try { _loadGate.Dispose(); } catch { }
    }
}
