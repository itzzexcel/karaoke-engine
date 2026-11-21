using System;
using System.Collections.Generic;

namespace KaraokeEngine.Audio;

/// <summary>
/// High-precision iterative Stockham mixed-radix complex FFT (supports factors 2, 3, 4, 5)
/// covering all UVR MDX-Net FFT lengths (4096, 5120, 6144, 7680, 8192) with PyTorch-exact STFT/iSTFT scaling.
/// </summary>
internal sealed class MixedRadixFft
{
    public int Length { get; }
    private readonly int[] _radices;
    private readonly int[] _pValues;
    private readonly float[][] _fwdTwCos;
    private readonly float[][] _fwdTwSin;
    private readonly float[][] _invTwCos;
    private readonly float[][] _invTwSin;

    // Reusable scratch ping-pong buffers (instance-owned, not thread-safe across concurrent callers)
    private readonly float[] _bufARe;
    private readonly float[] _bufAIm;
    private readonly float[] _bufBRe;
    private readonly float[] _bufBIm;

    private static readonly float Sin120 = MathF.Sqrt(3f) * 0.5f;
    private static readonly float Cos72 = MathF.Cos(2f * MathF.PI / 5f);
    private static readonly float Sin72 = MathF.Sin(2f * MathF.PI / 5f);
    private static readonly float Cos144 = MathF.Cos(4f * MathF.PI / 5f);
    private static readonly float Sin144 = MathF.Sin(4f * MathF.PI / 5f);

    public MixedRadixFft(int length)
    {
        if (length < 2) throw new ArgumentOutOfRangeException(nameof(length));
        Length = length;

        var rads = new List<int>();
        int rem = length;
        while (rem % 4 == 0) { rads.Add(4); rem /= 4; }
        while (rem % 2 == 0) { rads.Add(2); rem /= 2; }
        while (rem % 3 == 0) { rads.Add(3); rem /= 3; }
        while (rem % 5 == 0) { rads.Add(5); rem /= 5; }
        if (rem != 1)
        {
            throw new ArgumentException($"Unsupported FFT size {length}: only products of 2, 3, 5 are supported.", nameof(length));
        }

        _radices = rads.ToArray();
        _pValues = new int[_radices.Length];
        _fwdTwCos = new float[_radices.Length][];
        _fwdTwSin = new float[_radices.Length][];
        _invTwCos = new float[_radices.Length][];
        _invTwSin = new float[_radices.Length][];

        int p = 1;
        for (int stage = 0; stage < _radices.Length; stage++)
        {
            int r = _radices[stage];
            _pValues[stage] = p;
            int span = r * p;
            int tableLen = p * r;
            var fCos = new float[tableLen];
            var fSin = new float[tableLen];
            var iCos = new float[tableLen];
            var iSin = new float[tableLen];

            for (int k = 0; k < p; k++)
            {
                for (int m = 0; m < r; m++)
                {
                    double angle = -2.0 * Math.PI * k * m / span;
                    float c = (float)Math.Cos(angle);
                    float s = (float)Math.Sin(angle);
                    int idx = k * r + m;
                    fCos[idx] = c;
                    fSin[idx] = s;
                    iCos[idx] = c;
                    iSin[idx] = -s;
                }
            }

            _fwdTwCos[stage] = fCos;
            _fwdTwSin[stage] = fSin;
            _invTwCos[stage] = iCos;
            _invTwSin[stage] = iSin;
            p *= r;
        }

        _bufARe = new float[length];
        _bufAIm = new float[length];
        _bufBRe = new float[length];
        _bufBIm = new float[length];
    }

    public static float[] CreatePeriodicHannWindow(int length)
    {
        var win = new float[length];
        float scale = 2f * MathF.PI / length;
        for (int n = 0; n < length; n++)
        {
            win[n] = 0.5f * (1f - MathF.Cos(scale * n));
        }
        return win;
    }

    public void Forward(ReadOnlySpan<float> realInput, Span<float> realOutput, Span<float> imagOutput)
    {
        for (int i = 0; i < Length; i++)
        {
            _bufARe[i] = realInput[i];
            _bufAIm[i] = 0f;
        }

        ExecuteStages(inverse: false, out float[] finalRe, out float[] finalIm);
        finalRe.AsSpan().CopyTo(realOutput);
        finalIm.AsSpan().CopyTo(imagOutput);
    }

    public void Inverse(ReadOnlySpan<float> realInput, ReadOnlySpan<float> imagInput, Span<float> realOutput)
    {
        realInput.CopyTo(_bufARe);
        imagInput.CopyTo(_bufAIm);

        ExecuteStages(inverse: true, out float[] finalRe, out _);
        float invN = 1f / Length;
        for (int i = 0; i < Length; i++)
        {
            realOutput[i] = finalRe[i] * invN;
        }
    }

    private void ExecuteStages(bool inverse, out float[] outRe, out float[] outIm)
    {
        float[] srcRe = _bufARe;
        float[] srcIm = _bufAIm;
        float[] dstRe = _bufBRe;
        float[] dstIm = _bufBIm;

        float sign = inverse ? 1f : -1f;
        var twCosArr = inverse ? _invTwCos : _fwdTwCos;
        var twSinArr = inverse ? _invTwSin : _fwdTwSin;

        for (int stage = 0; stage < _radices.Length; stage++)
        {
            int r = _radices[stage];
            int p = _pValues[stage];
            int qPrime = Length / (r * p);
            float[] twCos = twCosArr[stage];
            float[] twSin = twSinArr[stage];

            if (r == 2) StageRadix2(srcRe, srcIm, dstRe, dstIm, p, qPrime, twCos, twSin);
            else if (r == 4) StageRadix4(srcRe, srcIm, dstRe, dstIm, p, qPrime, sign, twCos, twSin);
            else if (r == 3) StageRadix3(srcRe, srcIm, dstRe, dstIm, p, qPrime, sign, twCos, twSin);
            else StageRadix5(srcRe, srcIm, dstRe, dstIm, p, qPrime, sign, twCos, twSin);

            (srcRe, dstRe) = (dstRe, srcRe);
            (srcIm, dstIm) = (dstIm, srcIm);
        }

        outRe = srcRe;
        outIm = srcIm;
    }

    private static void StageRadix2(
        float[] sRe, float[] sIm, float[] dRe, float[] dIm,
        int p, int qPrime, float[] twCos, float[] twSin)
    {
        int stride = 2 * qPrime;
        for (int k = 0; k < p; k++)
        {
            int sBase = k * stride;
            int dBase0 = k * qPrime;
            int dBase1 = (k + p) * qPrime;
            float c1 = twCos[k * 2 + 1];
            float sn1 = twSin[k * 2 + 1];

            for (int j1 = 0; j1 < qPrime; j1++)
            {
                int i0 = sBase + j1;
                int i1 = i0 + qPrime;

                float x0r = sRe[i0], x0i = sIm[i0];
                float a1r = sRe[i1], a1i = sIm[i1];
                float x1r = a1r * c1 - a1i * sn1;
                float x1i = a1r * sn1 + a1i * c1;

                dRe[dBase0 + j1] = x0r + x1r;
                dIm[dBase0 + j1] = x0i + x1i;
                dRe[dBase1 + j1] = x0r - x1r;
                dIm[dBase1 + j1] = x0i - x1i;
            }
        }
    }

    private static void StageRadix4(
        float[] sRe, float[] sIm, float[] dRe, float[] dIm,
        int p, int qPrime, float sign, float[] twCos, float[] twSin)
    {
        int stride = 4 * qPrime;
        for (int k = 0; k < p; k++)
        {
            int sBase = k * stride;
            int twBase = k * 4;
            float c1 = twCos[twBase + 1], sn1 = twSin[twBase + 1];
            float c2 = twCos[twBase + 2], sn2 = twSin[twBase + 2];
            float c3 = twCos[twBase + 3], sn3 = twSin[twBase + 3];

            int d0 = k * qPrime;
            int d1 = (k + p) * qPrime;
            int d2 = (k + 2 * p) * qPrime;
            int d3 = (k + 3 * p) * qPrime;

            for (int j1 = 0; j1 < qPrime; j1++)
            {
                int idx0 = sBase + j1;
                int idx1 = idx0 + qPrime;
                int idx2 = idx1 + qPrime;
                int idx3 = idx2 + qPrime;

                float u0r = sRe[idx0], u0i = sIm[idx0];

                float a1r = sRe[idx1], a1i = sIm[idx1];
                float u1r = a1r * c1 - a1i * sn1;
                float u1i = a1r * sn1 + a1i * c1;

                float a2r = sRe[idx2], a2i = sIm[idx2];
                float u2r = a2r * c2 - a2i * sn2;
                float u2i = a2r * sn2 + a2i * c2;

                float a3r = sRe[idx3], a3i = sIm[idx3];
                float u3r = a3r * c3 - a3i * sn3;
                float u3i = a3r * sn3 + a3i * c3;

                float t0r = u0r + u2r, t0i = u0i + u2i;
                float t1r = u0r - u2r, t1i = u0i - u2i;
                float t2r = u1r + u3r, t2i = u1i + u3i;
                // (u1 - u3) * (sign * i): (-y * sign, x * sign)
                float diff13r = u1r - u3r;
                float diff13i = u1i - u3i;
                float rotR = -diff13i * sign;
                float rotI = diff13r * sign;

                dRe[d0 + j1] = t0r + t2r;
                dIm[d0 + j1] = t0i + t2i;
                dRe[d1 + j1] = t1r + rotR;
                dIm[d1 + j1] = t1i + rotI;
                dRe[d2 + j1] = t0r - t2r;
                dIm[d2 + j1] = t0i - t2i;
                dRe[d3 + j1] = t1r - rotR;
                dIm[d3 + j1] = t1i - rotI;
            }
        }
    }

    private static void StageRadix3(
        float[] sRe, float[] sIm, float[] dRe, float[] dIm,
        int p, int qPrime, float sign, float[] twCos, float[] twSin)
    {
        int stride = 3 * qPrime;
        float sin120s = sign * Sin120;
        for (int k = 0; k < p; k++)
        {
            int sBase = k * stride;
            int twBase = k * 3;
            float c1 = twCos[twBase + 1], sn1 = twSin[twBase + 1];
            float c2 = twCos[twBase + 2], sn2 = twSin[twBase + 2];

            int d0 = k * qPrime;
            int d1 = (k + p) * qPrime;
            int d2 = (k + 2 * p) * qPrime;

            for (int j1 = 0; j1 < qPrime; j1++)
            {
                int idx0 = sBase + j1;
                int idx1 = idx0 + qPrime;
                int idx2 = idx1 + qPrime;

                float u0r = sRe[idx0], u0i = sIm[idx0];

                float a1r = sRe[idx1], a1i = sIm[idx1];
                float u1r = a1r * c1 - a1i * sn1;
                float u1i = a1r * sn1 + a1i * c1;

                float a2r = sRe[idx2], a2i = sIm[idx2];
                float u2r = a2r * c2 - a2i * sn2;
                float u2i = a2r * sn2 + a2i * c2;

                float t1r = u1r + u2r, t1i = u1i + u2i;
                float t2r = u0r - 0.5f * t1r, t2i = u0i - 0.5f * t1i;
                float t3r = sin120s * (u1r - u2r), t3i = sin120s * (u1i - u2i);

                dRe[d0 + j1] = u0r + t1r;
                dIm[d0 + j1] = u0i + t1i;
                dRe[d1 + j1] = t2r - t3i;
                dIm[d1 + j1] = t2i + t3r;
                dRe[d2 + j1] = t2r + t3i;
                dIm[d2 + j1] = t2i - t3r;
            }
        }
    }

    private static void StageRadix5(
        float[] sRe, float[] sIm, float[] dRe, float[] dIm,
        int p, int qPrime, float sign, float[] twCos, float[] twSin)
    {
        int stride = 5 * qPrime;
        float s72 = sign * Sin72;
        float s144 = sign * Sin144;

        for (int k = 0; k < p; k++)
        {
            int sBase = k * stride;
            int twBase = k * 5;
            float c1 = twCos[twBase + 1], sn1 = twSin[twBase + 1];
            float c2 = twCos[twBase + 2], sn2 = twSin[twBase + 2];
            float c3 = twCos[twBase + 3], sn3 = twSin[twBase + 3];
            float c4 = twCos[twBase + 4], sn4 = twSin[twBase + 4];

            int d0 = k * qPrime;
            int d1 = (k + p) * qPrime;
            int d2 = (k + 2 * p) * qPrime;
            int d3 = (k + 3 * p) * qPrime;
            int d4 = (k + 4 * p) * qPrime;

            for (int j1 = 0; j1 < qPrime; j1++)
            {
                int idx0 = sBase + j1;
                int idx1 = idx0 + qPrime;
                int idx2 = idx1 + qPrime;
                int idx3 = idx2 + qPrime;
                int idx4 = idx3 + qPrime;

                float u0r = sRe[idx0], u0i = sIm[idx0];

                float a1r = sRe[idx1], a1i = sIm[idx1];
                float u1r = a1r * c1 - a1i * sn1;
                float u1i = a1r * sn1 + a1i * c1;

                float a2r = sRe[idx2], a2i = sIm[idx2];
                float u2r = a2r * c2 - a2i * sn2;
                float u2i = a2r * sn2 + a2i * c2;

                float a3r = sRe[idx3], a3i = sIm[idx3];
                float u3r = a3r * c3 - a3i * sn3;
                float u3i = a3r * sn3 + a3i * c3;

                float a4r = sRe[idx4], a4i = sIm[idx4];
                float u4r = a4r * c4 - a4i * sn4;
                float u4i = a4r * sn4 + a4i * c4;

                float s14r = u1r + u4r, s14i = u1i + u4i;
                float d14r = u1r - u4r, d14i = u1i - u4i;
                float s23r = u2r + u3r, s23i = u2i + u3i;
                float d23r = u2r - u3r, d23i = u2i - u3i;

                dRe[d0 + j1] = u0r + s14r + s23r;
                dIm[d0 + j1] = u0i + s14i + s23i;

                float m1r = u0r + Cos72 * s14r + Cos144 * s23r;
                float m1i = u0i + Cos72 * s14i + Cos144 * s23i;
                float n1r = s72 * d14r + s144 * d23r;
                float n1i = s72 * d14i + s144 * d23i;

                dRe[d1 + j1] = m1r - n1i;
                dIm[d1 + j1] = m1i + n1r;
                dRe[d4 + j1] = m1r + n1i;
                dIm[d4 + j1] = m1i - n1r;

                float m2r = u0r + Cos144 * s14r + Cos72 * s23r;
                float m2i = u0i + Cos144 * s14i + Cos72 * s23i;
                float n2r = s144 * d14r - s72 * d23r;
                float n2i = s144 * d14i - s72 * d23i;

                dRe[d2 + j1] = m2r - n2i;
                dIm[d2 + j1] = m2i + n2r;
                dRe[d3 + j1] = m2r + n2i;
                dIm[d3 + j1] = m2i - n2r;
            }
        }
    }
}
