using System;

namespace KaraokeEngine.Audio;

/// <summary>
/// Ultra-low-latency streaming stereo STFT Mid-Side vocal suppressor.
/// Keeps sub-bass (&lt; 180 Hz) and cymbal air (&gt; 7.5 kHz) untouched while estimating
/// the center-panned vocal stem vs backing instrumental for instant Fast DSP mode and seek bridging.
/// </summary>
internal sealed class DspVocalSuppressor
{
    public const int FftSize = 2048;
    public const int HopSize = 256;

    private readonly MixedRadixFft _fft = new(FftSize);
    private readonly float[] _window = MixedRadixFft.CreatePeriodicHannWindow(FftSize);
    private readonly float _olaNorm;

    // Rolling input capture buffer per channel
    private readonly float[] _inL = new float[FftSize];
    private readonly float[] _inR = new float[FftSize];

    // Overlap-add accumulators for instrumental estimate & delayed original signal
    private readonly float[] _outInstL = new float[FftSize];
    private readonly float[] _outInstR = new float[FftSize];
    private readonly float[] _delayOrigL = new float[FftSize];
    private readonly float[] _delayOrigR = new float[FftSize];
    private int _hopFill;

    // Scratch FFT buffers
    private readonly float[] _winL = new float[FftSize];
    private readonly float[] _winR = new float[FftSize];
    private readonly float[] _reL = new float[FftSize];
    private readonly float[] _imL = new float[FftSize];
    private readonly float[] _reR = new float[FftSize];
    private readonly float[] _imR = new float[FftSize];
    private readonly float[] _timeL = new float[FftSize];
    private readonly float[] _timeR = new float[FftSize];

    private readonly int _minVocalBin;
    private readonly int _maxVocalBin;
    private readonly int _lowRollEndBin;
    private readonly int _highRollStartBin;

    public DspVocalSuppressor(int sampleRate = 44100)
    {
        float sumSq = 0f;
        for (int i = 0; i < FftSize; i++) sumSq += _window[i] * _window[i];
        _olaNorm = HopSize / MathF.Max(1e-6f, sumSq);

        float binHz = sampleRate / (float)FftSize;
        _minVocalBin = Math.Max(1, (int)(160f / binHz));
        _lowRollEndBin = Math.Max(_minVocalBin + 1, (int)(260f / binHz));
        _highRollStartBin = (int)(6200f / binHz);
        _maxVocalBin = Math.Min(FftSize / 2, (int)(7800f / binHz));
    }

    public void Reset()
    {
        Array.Clear(_inL, 0, _inL.Length);
        Array.Clear(_inR, 0, _inR.Length);
        Array.Clear(_outInstL, 0, _outInstL.Length);
        Array.Clear(_outInstR, 0, _outInstR.Length);
        Array.Clear(_delayOrigL, 0, _delayOrigL.Length);
        Array.Clear(_delayOrigR, 0, _delayOrigR.Length);
        _hopFill = 0;
    }

    /// <summary>
    /// Processes one stereo sample pair and returns the delayed matched original sample and estimated instrumental sample.
    /// Keeping matched OLA latency guarantees smooth zero-phase blending with vocalGain.
    /// </summary>
    public void ProcessSample(
        float left, float right,
        out float origOutL, out float origOutR,
        out float instOutL, out float instOutR)
    {
        int readIdx = _hopFill;
        instOutL = _outInstL[readIdx];
        instOutR = _outInstR[readIdx];
        origOutL = _delayOrigL[readIdx];
        origOutR = _delayOrigR[readIdx];

        int writePos = FftSize - HopSize + _hopFill;
        _inL[writePos] = left;
        _inR[writePos] = right;
        _hopFill++;

        if (_hopFill >= HopSize)
        {
            ProcessHop();
            _hopFill = 0;
        }
    }

    private void ProcessHop()
    {
        for (int i = 0; i < FftSize; i++)
        {
            float w = _window[i];
            _winL[i] = _inL[i] * w;
            _winR[i] = _inR[i] * w;
        }

        _fft.Forward(_winL, _reL, _imL);
        _fft.Forward(_winR, _reR, _imR);

        int half = FftSize / 2;
        for (int k = 0; k <= half; k++)
        {
            if (k >= _minVocalBin && k <= _maxVocalBin)
            {
                float lr = _reL[k], li = _imL[k];
                float rr = _reR[k], ri = _imR[k];

                // Mid = (L + R) / 2, Side = (L - R) / 2
                float midR = 0.5f * (lr + rr);
                float midI = 0.5f * (li + ri);
                float sideR = 0.5f * (lr - rr);
                float sideI = 0.5f * (li - ri);

                float midMagSq = midR * midR + midI * midI;
                float sideMagSq = sideR * sideR + sideI * sideI;
                float magL = MathF.Sqrt(lr * lr + li * li);
                float magR = MathF.Sqrt(rr * rr + ri * ri);

                // Pan-centrality & inter-channel magnitude balance (centered vocals have L ~= R & Mid >> Side)
                float coherence = midMagSq / (midMagSq + sideMagSq + 1e-12f);
                float panMatch = (2f * magL * magR) / (magL * magL + magR * magR + 1e-12f);
                float centrality = Math.Clamp((coherence * panMatch - 0.55f) / 0.45f, 0f, 1f);

                float bandEnv = 1f;
                if (k < _lowRollEndBin)
                    bandEnv = (k - _minVocalBin) / (float)(_lowRollEndBin - _minVocalBin);
                else if (k > _highRollStartBin)
                    bandEnv = (_maxVocalBin - k) / (float)Math.Max(1, _maxVocalBin - _highRollStartBin);

                // Smooth soft mask retaining ~12% center body so stereo drums & bass harmonics survive naturally
                float vocalMask = centrality * bandEnv * 0.88f;
                float keepMid = 1f - vocalMask;

                // Preserve Side stereo field completely, attenuate only centered vocal Mid
                float newMidR = midR * keepMid;
                float newMidI = midI * keepMid;

                _reL[k] = newMidR + sideR;
                _imL[k] = newMidI + sideI;
                _reR[k] = newMidR - sideR;
                _imR[k] = newMidI - sideI;

                if (k > 0 && k < half)
                {
                    int mirror = FftSize - k;
                    _reL[mirror] = _reL[k];
                    _imL[mirror] = -_imL[k];
                    _reR[mirror] = _reR[k];
                    _imR[mirror] = -_imR[k];
                }
            }
        }

        _fft.Inverse(_reL, _imL, _timeL);
        _fft.Inverse(_reR, _imR, _timeR);

        // Shift overlap-add output buffers by HopSize
        int keep = FftSize - HopSize;
        Array.Copy(_inL, HopSize, _inL, 0, keep);
        Array.Copy(_inR, HopSize, _inR, 0, keep);
        Array.Copy(_outInstL, HopSize, _outInstL, 0, keep);
        Array.Copy(_outInstR, HopSize, _outInstR, 0, keep);
        Array.Copy(_delayOrigL, HopSize, _delayOrigL, 0, keep);
        Array.Copy(_delayOrigR, HopSize, _delayOrigR, 0, keep);

        Array.Clear(_outInstL, keep, HopSize);
        Array.Clear(_outInstR, keep, HopSize);
        Array.Clear(_delayOrigL, keep, HopSize);
        Array.Clear(_delayOrigR, keep, HopSize);

        for (int i = 0; i < FftSize; i++)
        {
            float scale = _window[i] * _olaNorm;
            _outInstL[i] += _timeL[i] * scale;
            _outInstR[i] += _timeR[i] * scale;

            // Matched windowed OLA on raw input guarantees phase-aligned original for lossless vocalGain blending
            _delayOrigL[i] += _winL[i] * _olaNorm;
            _delayOrigR[i] += _winR[i] * _olaNorm;
        }
    }
}
