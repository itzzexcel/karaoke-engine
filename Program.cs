using KaraokeEngine.Ai;
using KaraokeEngine.Audio;
using KaraokeEngine.Server;
using KaraokeEngine.Utils;
using Microsoft.Win32;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace KaraokeEngine;

internal static class Program
{
    private const int DefaultPort = 5445;
    private const string RunRegistryKey = @"Software\Microsoft\Windows\CurrentVersion\Run";
    private const string RunValueName = "KaraokeEngine";

    static async Task<int> Main(string[] args)
    {
        bool showConsole = Array.Exists(args, a => a.Equals("--show-console", StringComparison.OrdinalIgnoreCase));
        bool selfTest = Array.Exists(args, a => a.Equals("--self-test", StringComparison.OrdinalIgnoreCase));
        bool installStartup = Array.Exists(args, a => a.Equals("--install-startup", StringComparison.OrdinalIgnoreCase));
        bool uninstallStartup = Array.Exists(args, a => a.Equals("--uninstall-startup", StringComparison.OrdinalIgnoreCase));
        bool prefetchModels = Array.Exists(args, a => a.Equals("--prefetch-models", StringComparison.OrdinalIgnoreCase));

        Logger.ConfigureConsole(showConsole || selfTest || installStartup || uninstallStartup || prefetchModels);

        if (installStartup)
        {
            return ConfigureWindowsStartup(enable: true);
        }
        if (uninstallStartup)
        {
            return ConfigureWindowsStartup(enable: false);
        }
        if (prefetchModels)
        {
            return await PrefetchAllModelsAsync();
        }
        if (selfTest)
        {
            return await RunSelfTestAsync();
        }

        // Auto-register Windows startup on first interactive launch unless --no-startup is passed
        if (!Array.Exists(args, a => a.Equals("--no-startup", StringComparison.OrdinalIgnoreCase)))
        {
            EnsureFirstRunStartupRegistered();
        }

        using var singleton = new Mutex(true, @"Local\karaoke_engine_singleton_mutex", out bool createdNew);
        if (!createdNew)
        {
            Logger.Info("Another karaoke-engine instance is already running; exiting cleanly.");
            return 0;
        }

        using var engine = new DualStemEngine();
        using var server = new KaraokeServer(engine, DefaultPort);
        using var cts = new CancellationTokenSource();

        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
        {
            Logger.Error($"Unhandled exception: {e.ExceptionObject}");
            engine.Dispose();
        };
        TaskScheduler.UnobservedTaskException += (_, e) =>
        {
            Logger.Warn($"Unobserved task exception: {e.Exception.Message}");
            e.SetObserved();
        };
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try { cts.Cancel(); } catch { }
            engine.Dispose();
        };
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            try { cts.Cancel(); } catch { }
        };

        Logger.Info($"karaoke-engine starting on ws://127.0.0.1:{DefaultPort}/control");
        await server.RunAsync(cts.Token);
        return 0;
    }

    private static void EnsureFirstRunStartupRegistered()
    {
        try
        {
            string marker = Path.Combine(Logger.AppDirectory, ".startup_registered");
            if (File.Exists(marker)) return;
            ConfigureWindowsStartup(enable: true);
            Directory.CreateDirectory(Logger.AppDirectory);
            File.WriteAllText(marker, DateTime.UtcNow.ToString("O"));
        }
        catch { }
    }

    private static int ConfigureWindowsStartup(bool enable)
    {
        try
        {
            string exePath = Environment.ProcessPath ?? Process.GetCurrentProcess().MainModule?.FileName ?? "";
            if (string.IsNullOrEmpty(exePath)) return 1;

            using var key = Registry.CurrentUser.OpenSubKey(RunRegistryKey, writable: true);
            if (key == null) return 1;

            if (enable)
            {
                key.SetValue(RunValueName, $"\"{exePath}\"");
                Logger.Info($"Registered karaoke-engine in Windows startup: \"{exePath}\"");
            }
            else
            {
                key.DeleteValue(RunValueName, throwOnMissingValue: false);
                Logger.Info("Removed karaoke-engine from Windows startup.");
            }
            return 0;
        }
        catch (Exception ex)
        {
            Logger.Error($"Failed to configure Windows startup: {ex.Message}");
            return 1;
        }
    }

    private static async Task<int> PrefetchAllModelsAsync()
    {
        foreach (string mode in new[] { "karaoke", "instrumental" })
        {
            if (ModelCatalog.IsModelInstalled(mode))
            {
                Console.WriteLine($"[MODEL:{mode}] Already installed at {ModelCatalog.GetModelPath(mode)}");
                continue;
            }
            Console.WriteLine($"[MODEL:{mode}] Downloading...");
            await ModelCatalog.DownloadModelAsync(mode, p =>
            {
                Console.Write($"\r[MODEL:{mode}] {Math.Round(p * 100)}%   ");
            }, CancellationToken.None);
            Console.WriteLine();
        }
        return 0;
    }

    private static async Task<int> RunSelfTestAsync()
    {
        Console.WriteLine("=== karaoke-engine self-test ===");

        // 1. Validate MixedRadixFft accuracy across all MDX sizes (5120, 6144, 7680)
        int[] fftSizes = { 2048, 5120, 6144, 7680 };
        var rng = new Random(42);
        foreach (int n in fftSizes)
        {
            var fft = new MixedRadixFft(n);
            float[] x = new float[n];
            float[] re = new float[n];
            float[] im = new float[n];
            float[] recon = new float[n];
            for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            fft.Forward(x, re, im);
            fft.Inverse(re, im, recon);

            float maxErr = 0f;
            for (int i = 0; i < n; i++) maxErr = MathF.Max(maxErr, MathF.Abs(x[i] - recon[i]));
            Console.WriteLine($"[FFT] N={n}: round-trip maxErr={maxErr:E3}");
            if (maxErr > 1e-4f)
            {
                Console.WriteLine($"FAIL: FFT N={n} error too large");
                return 1;
            }
        }

        // 2. Validate DspVocalSuppressor center-panned suppression
        var dsp = new DspVocalSuppressor(44100);
        double centerEnergyIn = 0, centerEnergyOut = 0;
        for (int i = 0; i < 8820; i++)
        {
            float vocal1k = 0.5f * MathF.Sin(2f * MathF.PI * 1000f * i / 44100f);
            dsp.ProcessSample(vocal1k, vocal1k, out _, out _, out float instL, out float instR);
            if (i > 4096)
            {
                centerEnergyIn += vocal1k * vocal1k;
                centerEnergyOut += 0.5 * (instL * instL + instR * instR);
            }
        }
        double attenDb = 10.0 * Math.Log10((centerEnergyOut + 1e-12) / (centerEnergyIn + 1e-12));
        Console.WriteLine($"[DSP] 1kHz center vocal attenuation={attenDb:F1} dB");
        if (attenDb > -8.0)
        {
            Console.WriteLine("FAIL: DSP center suppression insufficient");
            return 1;
        }

        // 3. Benchmark DirectML GPU separation (preferring FP16 when converted) for all installed models
        using var sep = new OnnxStemSeparator();
        foreach (string testMode in new[] { "karaoke", "instrumental" })
        {
            if (!ModelCatalog.IsModelInstalled(testMode))
            {
                Console.WriteLine($"[ONNX:{testMode}] Skipped (not installed at {ModelCatalog.GetModelPath(testMode)})");
                continue;
            }

            bool loaded = await sep.EnsureLoadedAsync(testMode, useGpu: true, CancellationToken.None);
            int frames = sep.GetExactChunkFrames();
            Console.WriteLine($"[ONNX:{testMode}] loaded={loaded} provider={sep.ActiveProvider} fp16={sep.IsFp16} chunkFrames={frames} ({frames / 44100.0:F2}s)");

            if (loaded)
            {
                float[] dummyStereo = new float[frames * 2];
                float[] outInst = new float[frames * 2];
                for (int i = 0; i < frames; i++)
                {
                    float s = 0.25f * MathF.Sin(2f * MathF.PI * 440f * i / 44100f);
                    dummyStereo[i * 2] = s;
                    dummyStereo[i * 2 + 1] = s;
                }

                // Warmup + 3 consecutive zero-LOH timed runs
                sep.SeparateChunk(dummyStereo, frames, outInst);
                int gc2Before = GC.CollectionCount(2);
                var sw = Stopwatch.StartNew();
                for (int r = 0; r < 3; r++)
                {
                    sep.SeparateChunk(dummyStereo, frames, outInst);
                }
                sw.Stop();
                int gc2Delta = GC.CollectionCount(2) - gc2Before;
                double avgMs = sw.Elapsed.TotalMilliseconds / 3.0;

                double rmsAccum = 0;
                for (int i = 0; i < outInst.Length; i++)
                {
                    float v = outInst[i];
                    if (float.IsNaN(v) || float.IsInfinity(v))
                    {
                        Console.WriteLine($"FAIL: ONNX:{testMode} output contains NaN/Inf at sample {i}");
                        return 1;
                    }
                    rmsAccum += v * v;
                }
                double rms = Math.Sqrt(rmsAccum / Math.Max(1, outInst.Length));
                Console.WriteLine($"[ONNX:{testMode}] 3x runs avg={avgMs:F1} ms ({(frames / 44100.0 * 1000.0) / Math.Max(1.0, avgMs):F1}x real-time), gc2Delta={gc2Delta}, rms={rms:F4}");
                if (gc2Delta != 0 || rms <= 1e-6)
                {
                    Console.WriteLine($"FAIL: Zero-GC or non-zero output check failed for {testMode}");
                    return 1;
                }
            }
        }

        // 4. Verify StemDiskCache binary round-trip in isolated temp directory
        string testCacheDir = Path.Combine(Path.GetTempPath(), $"krk_cache_test_{Guid.NewGuid():N}");
        try
        {
            var testCache = new StemDiskCache(testCacheDir);
            var origSample = new short[4096];
            var instSample = new short[4096];
            for (int i = 0; i < 4096; i++)
            {
                origSample[i] = (short)((i * 73) % 30000 - 15000);
                instSample[i] = (short)((i * 37) % 20000 - 10000);
            }
            testCache.SaveOriginalBackground("test_track_99", origSample);
            testCache.SaveInstrumentalBackground("test_track_99", "karaoke", "1d64a6d2", instSample);

            short[]? readOrig = null;
            short[]? readInst = null;
            for (int wait = 0; wait < 40; wait++)
            {
                readOrig ??= testCache.TryLoadOriginal("test_track_99");
                readInst ??= testCache.TryLoadInstrumental("test_track_99", "karaoke", "1d64a6d2");
                if (readOrig != null && readInst != null) break;
                await Task.Delay(25);
            }
            if (readOrig == null || readInst == null || !readOrig.AsSpan().SequenceEqual(origSample) || !readInst.AsSpan().SequenceEqual(instSample))
            {
                Console.WriteLine("FAIL: StemDiskCache round-trip verification failed");
                return 1;
            }
            Console.WriteLine("[CACHE] Binary KRK1 write/read bit-exact round-trip OK");
        }
        finally
        {
            try { if (Directory.Exists(testCacheDir)) Directory.Delete(testCacheDir, recursive: true); } catch { }
        }

        Console.WriteLine("SELF-TEST PASSED");
        return 0;
    }
}
