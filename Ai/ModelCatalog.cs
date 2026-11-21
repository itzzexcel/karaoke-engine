using KaraokeEngine.Utils;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace KaraokeEngine.Ai;

internal sealed record MdxModelSpec(
    int DimF,
    int DimT,
    int NFft,
    int HopLength,
    float Compensate,
    string PrimaryStem,
    bool IsKaraoke
);

internal static class ModelCatalog
{
    public const string KaraokeFile = "UVR_MDXNET_KARA_2.onnx";
    public const string InstrumentalFile = "UVR-MDX-NET-Inst_HQ_3.onnx";

    private const string KaraokeUrl = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx";
    private const string InstrumentalUrl = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx";
    private const string MetadataUrl = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data_new.json";

    public static string ModelsDirectory => Path.Combine(Logger.AppDirectory, "models");
    private static string CachedMetaPath => Path.Combine(Logger.AppDirectory, "model_data_new.json");

    // Verified UVR tail-10MB MD5 hashes -> architecture hyper-parameters
    private static readonly Dictionary<string, MdxModelSpec> BuiltInSpecs = new(StringComparer.OrdinalIgnoreCase)
    {
        ["2f5501189a2f6db6349916fabe8c90de"] = new(2048, 256, 6144, 1024, 1.035f, "Vocals", true),
        ["1d64a6d2c30f709b8c9b4ce1366d96ee"] = new(2048, 256, 5120, 1024, 1.065f, "Instrumental", true),
        ["2cdd429caac38f0194b133884160f2c6"] = new(3072, 256, 7680, 1024, 1.045f, "Instrumental", false),
        ["488b3e6f8bd3717d9d7c428476be2d75"] = new(3072, 256, 7680, 1024, 1.035f, "Instrumental", false),
        ["ceed671467c1f64ebdfac8a2490d0d52"] = new(3072, 256, 7680, 1024, 1.035f, "Instrumental", false),
        ["d2a1376f310e4f7fa37fb9b5774eb701"] = new(3072, 256, 7680, 1024, 1.035f, "Instrumental", false),
        ["b06327a00d5e5fbc7d96e1781bbdb596"] = new(3072, 256, 6144, 1024, 1.035f, "Instrumental", false),
        ["94ff780b977d3ca07c7a343dab2e25dd"] = new(3072, 256, 6144, 1024, 1.039f, "Instrumental", false),
        ["55657dd70583b0fedfba5f67df11d711"] = new(3072, 256, 6144, 1024, 1.022f, "Instrumental", false),
        ["b6bccda408a436db8500083ef3491e8b"] = new(3072, 256, 7680, 1024, 1.020f, "Instrumental", false),
        ["b78da4afc6512f98e4756f5977f5c6b9"] = new(3072, 256, 7680, 1024, 1.021f, "Instrumental", false),
        ["e5572e58abf111f80d8241d2e44e7fa4"] = new(3072, 256, 7680, 1024, 1.028f, "Instrumental", false),
        ["203f2a3955221b64df85a41af87cf8f0"] = new(3072, 256, 6144, 1024, 1.035f, "Instrumental", false),
        ["291c2049608edb52648b96e27eb80e95"] = new(3072, 256, 6144, 1024, 1.035f, "Instrumental", false),
        ["ead8d05dab12ec571d67549b3aab03fc"] = new(3072, 256, 6144, 1024, 1.035f, "Instrumental", false),
        ["cc63408db3d80b4d85b0287d1d7c9632"] = new(3072, 256, 6144, 1024, 1.033f, "Instrumental", false),
    };

    private static readonly HttpClient _http = new() { Timeout = TimeSpan.FromMinutes(10) };
    private static readonly ConcurrentDictionary<string, (long Length, long LastWriteTicks, string Hash)> _hashCache = new(StringComparer.OrdinalIgnoreCase);

    public static string GetModelPath(string mode)
    {
        string fileName = string.Equals(mode, "instrumental", StringComparison.OrdinalIgnoreCase) ? InstrumentalFile : KaraokeFile;
        // 1. Check portable / bundled models folder next to karaoke-engine.exe first
        try
        {
            string bundledPath = Path.Combine(AppContext.BaseDirectory, "models", fileName);
            if (File.Exists(bundledPath) && new FileInfo(bundledPath).Length > 1_000_000)
            {
                return bundledPath;
            }
        }
        catch { }

        // 2. Fall back to user-writable %LOCALAPPDATA%\karaoke-engine\models
        return Path.Combine(ModelsDirectory, fileName);
    }

    public static string GetFp16ModelPath(string mode)
    {
        string path = GetModelPath(mode);
        return path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)
            ? path[..^5] + ".fp16.onnx"
            : path + ".fp16.onnx";
    }

    public static bool HasFp16Model(string mode)
    {
        string fp16Path = GetFp16ModelPath(mode);
        return File.Exists(fp16Path) && new FileInfo(fp16Path).Length > 1_000_000;
    }

    public static bool IsModelInstalled(string mode)
    {
        string path = GetModelPath(mode);
        return File.Exists(path) && new FileInfo(path).Length > 1_000_000;
    }

    public static string GetModelHash8(string mode)
    {
        if (!IsModelInstalled(mode)) return "none";
        try
        {
            string hash = ComputeUvrTailMd5(GetModelPath(mode));
            return hash.Length >= 8 ? hash[..8] : hash;
        }
        catch
        {
            return "none";
        }
    }

    public static string ComputeUvrTailMd5(string filePath)
    {
        var fi = new FileInfo(filePath);
        long ticks = fi.LastWriteTimeUtc.Ticks;
        if (_hashCache.TryGetValue(filePath, out var cached) &&
            cached.Length == fi.Length &&
            cached.LastWriteTicks == ticks)
        {
            return cached.Hash;
        }

        using var fs = File.OpenRead(filePath);
        // UVR hashes the final 10,000 KiB (10000 * 1024 = 10,240,000 bytes) of the model file
        const long tailBytes = 10000L * 1024L;
        if (fs.Length > tailBytes)
        {
            fs.Seek(-tailBytes, SeekOrigin.End);
        }
        using var md5 = MD5.Create();
        byte[] hash = md5.ComputeHash(fs);
        string hex = Convert.ToHexStringLower(hash);
        _hashCache[filePath] = (fi.Length, ticks, hex);
        return hex;
    }

    public static async Task<MdxModelSpec> ResolveSpecAsync(string filePath, string mode, CancellationToken ct)
    {
        string hash = ComputeUvrTailMd5(filePath);
        if (BuiltInSpecs.TryGetValue(hash, out var spec))
        {
            Logger.Info($"Model hash {hash} matched built-in spec: n_fft={spec.NFft}, dim_f={spec.DimF}, dim_t={spec.DimT}, stem={spec.PrimaryStem}, comp={spec.Compensate}");
            return spec;
        }

        var remoteSpec = await TryLookupRemoteMetadataAsync(hash, ct);
        if (remoteSpec != null)
        {
            Logger.Info($"Model hash {hash} matched UVR metadata: n_fft={remoteSpec.NFft}, dim_f={remoteSpec.DimF}, dim_t={remoteSpec.DimT}, stem={remoteSpec.PrimaryStem}");
            return remoteSpec;
        }

        Logger.Warn($"Model hash {hash} not in table; using fallback defaults for mode '{mode}'.");
        return string.Equals(mode, "instrumental", StringComparison.OrdinalIgnoreCase)
            ? new MdxModelSpec(3072, 256, 6144, 1024, 1.035f, "Instrumental", false)
            : new MdxModelSpec(2048, 256, 5120, 1024, 1.065f, "Instrumental", true);
    }

    private static async Task<MdxModelSpec?> TryLookupRemoteMetadataAsync(string hash, CancellationToken ct)
    {
        try
        {
            string? json = null;
            if (File.Exists(CachedMetaPath) && (DateTime.UtcNow - File.GetLastWriteTimeUtc(CachedMetaPath)).TotalDays < 14)
            {
                json = await File.ReadAllTextAsync(CachedMetaPath, ct);
            }
            else
            {
                json = await _http.GetStringAsync(MetadataUrl, ct);
                Directory.CreateDirectory(Path.GetDirectoryName(CachedMetaPath)!);
                await File.WriteAllTextAsync(CachedMetaPath, json, ct);
            }

            using var doc = JsonDocument.Parse(json);
            if (!doc.RootElement.TryGetProperty(hash, out var entry)) return null;

            int dimF = entry.TryGetProperty("mdx_dim_f_set", out var df) ? df.GetInt32() : 2048;
            int dimTExp = entry.TryGetProperty("mdx_dim_t_set", out var dt) ? dt.GetInt32() : 8;
            int dimT = 1 << Math.Clamp(dimTExp, 6, 10);
            int nFft = entry.TryGetProperty("mdx_n_fft_scale_set", out var nf) ? nf.GetInt32() : 6144;
            float comp = entry.TryGetProperty("compensate", out var cp) ? cp.GetSingle() : 1.035f;
            string stem = entry.TryGetProperty("primary_stem", out var ps) ? (ps.GetString() ?? "Vocals") : "Vocals";
            bool isKara = entry.TryGetProperty("is_karaoke", out var ik) && ik.GetBoolean();

            return new MdxModelSpec(dimF, dimT, nFft, 1024, comp, stem, isKara);
        }
        catch (Exception ex)
        {
            Logger.Warn($"Remote UVR model metadata lookup failed: {ex.Message}");
            return null;
        }
    }

    public static async Task<string> DownloadModelAsync(
        string mode,
        Action<float> onProgress,
        CancellationToken ct)
    {
        Directory.CreateDirectory(ModelsDirectory);
        bool instMode = string.Equals(mode, "instrumental", StringComparison.OrdinalIgnoreCase);
        string url = instMode ? InstrumentalUrl : KaraokeUrl;
        string destPath = GetModelPath(mode);
        string partPath = destPath + ".part";

        Logger.Info($"Downloading AI model '{Path.GetFileName(destPath)}' from {url}");
        onProgress(0f);

        using var req = new HttpRequestMessage(HttpMethod.Get, url);
        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
        resp.EnsureSuccessStatusCode();

        long totalBytes = resp.Content.Headers.ContentLength ?? -1L;
        await using var netStream = await resp.Content.ReadAsStreamAsync(ct);

        await using (var fs = new FileStream(partPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true))
        {
            byte[] buffer = new byte[81920];
            long totalRead = 0;
            int read;
            DateTime lastReport = DateTime.UtcNow;

            while ((read = await netStream.ReadAsync(buffer, ct)) > 0)
            {
                await fs.WriteAsync(buffer.AsMemory(0, read), ct);
                totalRead += read;
                if (totalBytes > 0 && (DateTime.UtcNow - lastReport).TotalMilliseconds >= 120)
                {
                    lastReport = DateTime.UtcNow;
                    onProgress(Math.Clamp((float)totalRead / totalBytes, 0f, 0.99f));
                }
            }
            await fs.FlushAsync(ct);
        }

        if (File.Exists(destPath)) File.Delete(destPath);
        File.Move(partPath, destPath);
        onProgress(1f);
        Logger.Info($"Model ready at {destPath}");
        return destPath;
    }
}
