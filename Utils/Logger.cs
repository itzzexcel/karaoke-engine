using System;
using System.IO;
using System.Runtime.InteropServices;

namespace KaraokeEngine.Utils;

internal static class Logger
{
    private const long MaxLogBytes = 2 * 1024 * 1024;
    private static readonly object _lock = new();
    private static readonly string _appDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "karaoke-engine"
    );
    private static readonly string _logPath = Path.Combine(_appDir, "karaoke.log");
    private static bool _showConsole;

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetConsoleWindow();

    [DllImport("user32.dll")]
    private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

    public static string AppDirectory => _appDir;

    public static void ConfigureConsole(bool show)
    {
        _showConsole = show;
        try
        {
            var handle = GetConsoleWindow();
            if (handle != IntPtr.Zero)
            {
                ShowWindow(handle, show ? 5 : 0);
            }
        }
        catch { }
    }

    public static void Info(string message) => Write("INFO", message);
    public static void Warn(string message) => Write("WARN", message);
    public static void Error(string message) => Write("ERROR", message);

    private static void Write(string level, string message)
    {
        string line = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}] [{level}] {message}";
        lock (_lock)
        {
            try
            {
                Directory.CreateDirectory(_appDir);
                if (File.Exists(_logPath))
                {
                    var info = new FileInfo(_logPath);
                    if (info.Length >= MaxLogBytes)
                    {
                        string backup = _logPath + ".old";
                        if (File.Exists(backup)) File.Delete(backup);
                        File.Move(_logPath, backup);
                    }
                }
                File.AppendAllText(_logPath, line + Environment.NewLine);
            }
            catch { }

            if (_showConsole)
            {
                try { Console.WriteLine(line); } catch { }
            }
        }
    }
}
