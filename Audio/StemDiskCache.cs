using KaraokeEngine.Utils;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace KaraokeEngine.Audio;

/// <summary>
/// Persistent binary PCM16 disk cache for decoded Tidal originals and separated instrumental stems.
/// Zero-copy MemoryMarshal I/O + LRU eviction keeps repeat track loads and Karaoke/Inst switches instant.
/// </summary>
internal sealed class StemDiskCache
{
    private const uint Magic = 0x314B524B; // "KRK1"
    private const uint Version = 1;
    private const int HeaderBytes = 32;

    private readonly string _cacheDir;
    private readonly object _quotaLock = new();
    private readonly ConcurrentDictionary<string, byte> _inFlightWrites = new(StringComparer.OrdinalIgnoreCase);

    private long _maxBytes = 2048L * 1024 * 1024;
    private int _entryCount;
    private long _totalBytes;

    public StemDiskCache() : this(null) { }

    internal StemDiskCache(string? customDirectory)
    {
        _cacheDir = string.IsNullOrWhiteSpace(customDirectory)
            ? Path.Combine(Logger.AppDirectory, "cache")
            : customDirectory;
        InitializeDirectory();
    }

    public void SetMaxSizeMb(int mb)
    {
        long nextMax = Math.Clamp(mb, 256, 16384) * 1024L * 1024L;
        lock (_quotaLock)
        {
            _maxBytes = nextMax;
            EnforceQuotaLocked(protectedPath: null);
        }
    }

    public (int Entries, double TotalMb) GetStats()
    {
        lock (_quotaLock)
        {
            return (_entryCount, Math.Round(_totalBytes / (1024.0 * 1024.0), 1));
        }
    }

    public bool HasOriginal(string trackId) => File.Exists(GetOriginalPath(trackId));

    public bool HasInstrumental(string trackId, string mode, string modelHash8)
        => File.Exists(GetInstrumentalPath(trackId, mode, modelHash8));

    public short[]? TryLoadOriginal(string trackId)
        => TryLoadInternal(GetOriginalPath(trackId), $"{trackId}|orig");

    public short[]? TryLoadInstrumental(string trackId, string mode, string modelHash8)
        => TryLoadInternal(GetInstrumentalPath(trackId, mode, modelHash8), $"{trackId}|{NormalizeMode(mode)}|{modelHash8}");

    public void SaveOriginalBackground(string trackId, short[] pcm16)
        => QueueBackgroundSave(GetOriginalPath(trackId), $"{trackId}|orig", pcm16);

    public void SaveInstrumentalBackground(string trackId, string mode, string modelHash8, short[] pcm16)
        => QueueBackgroundSave(GetInstrumentalPath(trackId, mode, modelHash8), $"{trackId}|{NormalizeMode(mode)}|{modelHash8}", pcm16);

    public void ClearAll()
    {
        lock (_quotaLock)
        {
            try
            {
                if (Directory.Exists(_cacheDir))
                {
                    foreach (string file in Directory.EnumerateFiles(_cacheDir, "*.*"))
                    {
                        if (file.EndsWith(".krk", StringComparison.OrdinalIgnoreCase) ||
                            file.EndsWith(".tmp", StringComparison.OrdinalIgnoreCase))
                        {
                            try { File.Delete(file); } catch { }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Warn($"StemDiskCache ClearAll warning: {ex.Message}");
            }
            RefreshStatsLocked();
        }
        Logger.Info("Stem disk cache cleared.");
    }

    private void InitializeDirectory()
    {
        lock (_quotaLock)
        {
            try
            {
                Directory.CreateDirectory(_cacheDir);
                foreach (string tmp in Directory.EnumerateFiles(_cacheDir, "*.tmp"))
                {
                    try { File.Delete(tmp); } catch { }
                }
                RefreshStatsLocked();
            }
            catch (Exception ex)
            {
                Logger.Warn($"StemDiskCache init warning: {ex.Message}");
            }
        }
    }

    private void QueueBackgroundSave(string path, string tag, short[] pcm16)
    {
        if (pcm16 == null || pcm16.Length < 4 || (pcm16.Length & 1) != 0) return;
        if (!_inFlightWrites.TryAdd(path, 1)) return;

        _ = Task.Run(() =>
        {
            string tmpPath = $"{path}.{Guid.NewGuid():N}.tmp";
            try
            {
                Directory.CreateDirectory(_cacheDir);
                int totalFrames = pcm16.Length / 2;
                ulong tagHash = ComputeFnv1a64(tag);

                byte[] header = new byte[HeaderBytes];
                BitConverter.TryWriteBytes(header.AsSpan(0, 4), Magic);
                BitConverter.TryWriteBytes(header.AsSpan(4, 4), Version);
                BitConverter.TryWriteBytes(header.AsSpan(8, 4), TidalStreamFetcher.TargetSampleRate);
                BitConverter.TryWriteBytes(header.AsSpan(12, 4), TidalStreamFetcher.TargetChannels);
                BitConverter.TryWriteBytes(header.AsSpan(16, 4), totalFrames);
                BitConverter.TryWriteBytes(header.AsSpan(20, 4), 0);
                BitConverter.TryWriteBytes(header.AsSpan(24, 8), tagHash);

                using (var fs = new FileStream(tmpPath, FileMode.Create, FileAccess.Write, FileShare.None, 131072, FileOptions.SequentialScan))
                {
                    fs.Write(header, 0, HeaderBytes);
                    fs.Write(MemoryMarshal.AsBytes(pcm16.AsSpan()));
                    fs.Flush(flushToDisk: false);
                }

                File.Move(tmpPath, path, overwrite: true);
                try { File.SetLastWriteTimeUtc(path, DateTime.UtcNow); } catch { }

                lock (_quotaLock)
                {
                    EnforceQuotaLocked(protectedPath: path);
                }
            }
            catch (Exception ex)
            {
                Logger.Warn($"StemDiskCache save failed ({Path.GetFileName(path)}): {ex.Message}");
                try { if (File.Exists(tmpPath)) File.Delete(tmpPath); } catch { }
            }
            finally
            {
                _inFlightWrites.TryRemove(path, out _);
            }
        });
    }

    private short[]? TryLoadInternal(string path, string expectedTag)
    {
        if (!File.Exists(path)) return null;

        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 131072, FileOptions.SequentialScan);
            if (fs.Length < HeaderBytes)
            {
                fs.Dispose();
                TryDeleteCorrupt(path);
                return null;
            }

            Span<byte> header = stackalloc byte[HeaderBytes];
            fs.ReadExactly(header);

            uint magic = BitConverter.ToUInt32(header[0..4]);
            uint ver = BitConverter.ToUInt32(header[4..8]);
            int sampleRate = BitConverter.ToInt32(header[8..12]);
            int channels = BitConverter.ToInt32(header[12..16]);
            int totalFrames = BitConverter.ToInt32(header[16..20]);
            ulong tagHash = BitConverter.ToUInt64(header[24..32]);

            long expectedPayloadBytes = (long)totalFrames * channels * sizeof(short);
            if (magic != Magic ||
                ver != Version ||
                sampleRate != TidalStreamFetcher.TargetSampleRate ||
                channels != TidalStreamFetcher.TargetChannels ||
                totalFrames <= 0 ||
                totalFrames > 100_000_000 ||
                tagHash != ComputeFnv1a64(expectedTag) ||
                fs.Length != HeaderBytes + expectedPayloadBytes)
            {
                fs.Dispose();
                TryDeleteCorrupt(path);
                return null;
            }

            var pcm16 = new short[totalFrames * channels];
            fs.ReadExactly(MemoryMarshal.AsBytes(pcm16.AsSpan()));
            fs.Dispose();

            try { File.SetLastWriteTimeUtc(path, DateTime.UtcNow); } catch { }
            return pcm16;
        }
        catch (Exception ex)
        {
            Logger.Warn($"StemDiskCache load failed ({Path.GetFileName(path)}): {ex.Message}");
            TryDeleteCorrupt(path);
            return null;
        }
    }

    private void EnforceQuotaLocked(string? protectedPath)
    {
        try
        {
            var dir = new DirectoryInfo(_cacheDir);
            if (!dir.Exists)
            {
                _entryCount = 0;
                _totalBytes = 0;
                return;
            }

            List<FileInfo> files = dir.GetFiles("*.krk")
                .OrderBy(f => f.LastWriteTimeUtc)
                .ToList();

            long total = 0;
            foreach (var f in files) total += f.Length;

            if (total > _maxBytes)
            {
                long targetBytes = (long)(_maxBytes * 0.85);
                foreach (var f in files)
                {
                    if (total <= targetBytes) break;
                    if (!string.IsNullOrEmpty(protectedPath) &&
                        string.Equals(f.FullName, protectedPath, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }
                    long len = f.Length;
                    try
                    {
                        f.Delete();
                        total -= len;
                    }
                    catch { }
                }
            }

            RefreshStatsLocked();
        }
        catch (Exception ex)
        {
            Logger.Warn($"StemDiskCache quota check warning: {ex.Message}");
        }
    }

    private void RefreshStatsLocked()
    {
        try
        {
            var dir = new DirectoryInfo(_cacheDir);
            if (!dir.Exists)
            {
                _entryCount = 0;
                _totalBytes = 0;
                return;
            }

            int count = 0;
            long bytes = 0;
            foreach (var f in dir.EnumerateFiles("*.krk"))
            {
                count++;
                bytes += f.Length;
            }
            _entryCount = count;
            _totalBytes = bytes;
        }
        catch
        {
            _entryCount = 0;
            _totalBytes = 0;
        }
    }

    private static void TryDeleteCorrupt(string path)
    {
        try
        {
            if (File.Exists(path)) File.Delete(path);
        }
        catch { }
    }

    private string GetOriginalPath(string trackId)
        => Path.Combine(_cacheDir, $"orig_{SanitizeKey(trackId)}.krk");

    private string GetInstrumentalPath(string trackId, string mode, string modelHash8)
    {
        string normMode = NormalizeMode(mode);
        string safeHash = SanitizeKey(string.IsNullOrWhiteSpace(modelHash8) ? "none" : modelHash8);
        return Path.Combine(_cacheDir, $"inst_{SanitizeKey(trackId)}_{normMode}_{safeHash}.krk");
    }

    private static string NormalizeMode(string mode)
        => string.Equals(mode, "instrumental", StringComparison.OrdinalIgnoreCase) ? "instrumental" : "karaoke";

    private static string SanitizeKey(string raw)
    {
        if (string.IsNullOrEmpty(raw)) return "unknown";
        var sb = new StringBuilder(Math.Min(raw.Length, 48));
        bool clean = true;
        foreach (char c in raw)
        {
            if (char.IsLetterOrDigit(c) || c == '-' || c == '_')
            {
                if (sb.Length < 40) sb.Append(c);
            }
            else
            {
                clean = false;
            }
        }
        if (sb.Length == 0 || !clean || raw.Length > 40)
        {
            uint shortHash = (uint)ComputeFnv1a64(raw);
            if (sb.Length == 0) return $"t_{shortHash:x8}";
            return $"{sb}_{shortHash:x8}";
        }
        return sb.ToString();
    }

    private static ulong ComputeFnv1a64(string text)
    {
        ulong hash = 14695981039346656037UL;
        foreach (char ch in text)
        {
            hash ^= (byte)(ch & 0xFF);
            hash *= 1099511628211UL;
            hash ^= (byte)((ch >> 8) & 0xFF);
            hash *= 1099511628211UL;
        }
        return hash;
    }
}
