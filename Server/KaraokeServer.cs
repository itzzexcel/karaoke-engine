using KaraokeEngine.Audio;
using KaraokeEngine.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace KaraokeEngine.Server;

internal sealed class KaraokeServer : IDisposable
{
    private sealed class ClientConnection : IDisposable
    {
        public readonly WebSocket Socket;
        public readonly SemaphoreSlim SendLock = new(1, 1);
        private int _disposed;

        public ClientConnection(WebSocket socket) => Socket = socket;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            try { Socket.Dispose(); } catch { }
            try { SendLock.Dispose(); } catch { }
        }
    }

    private readonly DualStemEngine _engine;
    private readonly int _port;
    private readonly object _clientsLock = new();
    private readonly List<ClientConnection> _clients = new();
    private readonly CancellationTokenSource _cts = new();
    private HttpListener? _listener;
    private CancellationTokenSource? _disconnectGraceCts;

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    public KaraokeServer(DualStemEngine engine, int port = 5445)
    {
        _engine = engine;
        _port = port;
        _engine.StatusChanged += OnEngineStatusChanged;
    }

    public async Task RunAsync(CancellationToken externalCt)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(_cts.Token, externalCt);
        CancellationToken ct = linked.Token;

        _ = Task.Run(() => PeriodicStatusPumpAsync(ct), ct);

        int backoffMs = 500;
        while (!ct.IsCancellationRequested)
        {
            HttpListener? listener = null;
            try
            {
                listener = new HttpListener();
                listener.Prefixes.Add($"http://127.0.0.1:{_port}/");
                listener.Prefixes.Add($"http://localhost:{_port}/");
                listener.Start();
                _listener = listener;
                backoffMs = 500;
                Logger.Info($"Karaoke server listening @ ws://127.0.0.1:{_port}/control & http://127.0.0.1:{_port}/health");

                while (!ct.IsCancellationRequested)
                {
                    var context = await listener.GetContextAsync().WaitAsync(ct);
                    if (context.Request.IsWebSocketRequest)
                    {
                        _ = Task.Run(() => HandleWebSocketClientAsync(context, ct), ct);
                    }
                    else
                    {
                        HandleHttpRequest(context);
                    }
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                Logger.Error($"HTTP/WS listener error (retrying in {backoffMs}ms): {ex.Message}");
            }
            finally
            {
                if (listener != null)
                {
                    try { listener.Stop(); } catch { }
                    try { listener.Close(); } catch { }
                }
                _listener = null;
            }

            if (!ct.IsCancellationRequested)
            {
                try { await Task.Delay(backoffMs, ct); } catch { break; }
                backoffMs = Math.Min(5000, backoffMs * 2);
            }
        }
    }

    private void HandleHttpRequest(HttpListenerContext ctx)
    {
        try
        {
            ctx.Response.Headers.Add("Access-Control-Allow-Origin", "*");
            ctx.Response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            ctx.Response.Headers.Add("Access-Control-Allow-Headers", "Content-Type");

            if (ctx.Request.HttpMethod == "OPTIONS")
            {
                ctx.Response.StatusCode = 204;
                ctx.Response.Close();
                return;
            }

            string path = ctx.Request.Url?.AbsolutePath ?? "/";
            if (path.Equals("/health", StringComparison.OrdinalIgnoreCase))
            {
                var payload = JsonSerializer.Serialize(_engine.GetSnapshot(), _jsonOptions);
                byte[] bytes = Encoding.UTF8.GetBytes(payload);
                ctx.Response.StatusCode = 200;
                ctx.Response.ContentType = "application/json";
                ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
                ctx.Response.Close();
                return;
            }

            ctx.Response.StatusCode = 404;
            ctx.Response.Close();
        }
        catch { }
    }

    private async Task HandleWebSocketClientAsync(HttpListenerContext ctx, CancellationToken ct)
    {
        ClientConnection? conn = null;
        try
        {
            var wsCtx = await ctx.AcceptWebSocketAsync(null);
            conn = new ClientConnection(wsCtx.WebSocket);
            List<ClientConnection> staleClients = new();
            lock (_clientsLock)
            {
                if (_clients.Count > 0)
                {
                    staleClients.AddRange(_clients);
                    _clients.Clear();
                }
                _clients.Add(conn);
                _disconnectGraceCts?.Cancel();
                _disconnectGraceCts?.Dispose();
                _disconnectGraceCts = null;
            }
            if (staleClients.Count > 0)
            {
                Logger.Info($"Evicting {staleClients.Count} stale /control WebSocket session(s).");
                foreach (var old in staleClients)
                {
                    try { old.Dispose(); } catch { }
                }
            }
            Logger.Info("Tidal plugin connected to /control WebSocket.");

            await SendStatusToClientAsync(conn, ct);

            byte[] buffer = new byte[64 * 1024];
            var ms = new MemoryStream();

            while (conn.Socket.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                var res = await conn.Socket.ReceiveAsync(new ArraySegment<byte>(buffer), ct);
                if (res.MessageType == WebSocketMessageType.Close)
                {
                    await conn.Socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
                    break;
                }

                if (res.MessageType == WebSocketMessageType.Text)
                {
                    ms.Write(buffer, 0, res.Count);
                    if (res.EndOfMessage)
                    {
                        string msg = Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length);
                        ms.SetLength(0);
                        DispatchMessage(msg);
                    }
                }
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { }
        catch (Exception ex)
        {
            Logger.Info($"WebSocket session ended: {ex.Message}");
        }
        finally
        {
            if (conn != null)
            {
                bool noClientsLeft;
                lock (_clientsLock)
                {
                    _clients.Remove(conn);
                    noClientsLeft = _clients.Count == 0;
                    if (noClientsLeft && !ct.IsCancellationRequested)
                    {
                        _disconnectGraceCts?.Cancel();
                        _disconnectGraceCts?.Dispose();
                        _disconnectGraceCts = new CancellationTokenSource();
                        var graceToken = _disconnectGraceCts.Token;
                        _ = Task.Run(async () =>
                        {
                            try
                            {
                                await Task.Delay(2500, graceToken);
                                lock (_clientsLock)
                                {
                                    if (_clients.Count > 0) return;
                                }
                                _engine.SetEnabled(false);
                            }
                            catch (OperationCanceledException) { }
                        }, CancellationToken.None);
                    }
                }
                conn.Dispose();
            }
        }
    }

    private void DispatchMessage(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            string type = root.TryGetProperty("type", out var t) ? (t.GetString() ?? "") : "";

            switch (type)
            {
                case "enable":
                {
                    bool en = root.TryGetProperty("enabled", out var e) && e.GetBoolean();
                    _engine.SetEnabled(en);
                    break;
                }
                case "config":
                {
                    float? vg = root.TryGetProperty("vocalGain", out var vgp) && vgp.ValueKind == JsonValueKind.Number ? vgp.GetSingle() : null;
                    float? mv = root.TryGetProperty("masterVolume", out var mvp) && mvp.ValueKind == JsonValueKind.Number ? mvp.GetSingle() : null;
                    string? mode = root.TryGetProperty("mode", out var mp) ? mp.GetString() : null;
                    bool? gpu = root.TryGetProperty("useGpu", out var gp) && (gp.ValueKind == JsonValueKind.True || gp.ValueKind == JsonValueKind.False) ? gp.GetBoolean() : null;
                    int? off = root.TryGetProperty("syncOffsetMs", out var op) && op.ValueKind == JsonValueKind.Number ? op.GetInt32() : null;
                    int? cacheMb = root.TryGetProperty("cacheMaxMb", out var cp) && cp.ValueKind == JsonValueKind.Number ? cp.GetInt32() : null;
                    _engine.UpdateConfig(vg, mv, mode, gpu, off, cacheMb);
                    break;
                }
                case "clear_cache":
                {
                    _engine.ClearCache();
                    break;
                }
                case "load_track":
                {
                    string trackId = root.TryGetProperty("trackId", out var tid) ? (tid.ToString() ?? "") : "";
                    string title = root.TryGetProperty("title", out var ttl) ? (ttl.GetString() ?? "") : "";
                    string mime = root.TryGetProperty("manifestMimeType", out var mm) ? (mm.GetString() ?? "application/vnd.tidal.bts") : "application/vnd.tidal.bts";
                    string? encType = root.TryGetProperty("encryptionType", out var et) ? et.GetString() : null;
                    string? keyId = root.TryGetProperty("keyId", out var ki) ? ki.GetString() : null;
                    double startTime = root.TryGetProperty("startTime", out var st) && st.ValueKind == JsonValueKind.Number ? st.GetDouble() : 0.0;
                    bool playing = root.TryGetProperty("playing", out var pl) && pl.GetBoolean();
                    bool preload = root.TryGetProperty("preload", out var pr) && pr.GetBoolean();

                    var urls = new List<string>();
                    if (root.TryGetProperty("urls", out var ua) && ua.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var item in ua.EnumerateArray())
                        {
                            string? u = item.GetString();
                            if (!string.IsNullOrEmpty(u)) urls.Add(u);
                        }
                    }

                    if (!string.IsNullOrEmpty(trackId) && urls.Count > 0)
                    {
                        _engine.LoadTrack(trackId, title, mime, urls.ToArray(), encType, keyId, startTime, playing, preload);
                    }
                    break;
                }
                case "seek":
                {
                    string? trackId = root.TryGetProperty("trackId", out var tid) ? tid.ToString() : null;
                    double curTime = root.TryGetProperty("currentTime", out var ctProp) && ctProp.ValueKind == JsonValueKind.Number ? ctProp.GetDouble() : 0.0;
                    bool playing = root.TryGetProperty("playing", out var plProp) && plProp.GetBoolean();
                    _engine.HandleSeek(trackId, curTime, playing);
                    break;
                }
                case "sync":
                {
                    string? trackId = root.TryGetProperty("trackId", out var tid) ? tid.ToString() : null;
                    double curTime = root.TryGetProperty("currentTime", out var ctProp) && ctProp.ValueKind == JsonValueKind.Number ? ctProp.GetDouble() : 0.0;
                    bool playing = root.TryGetProperty("playing", out var plProp) && plProp.GetBoolean();
                    float? mv = root.TryGetProperty("masterVolume", out var mvProp) && mvProp.ValueKind == JsonValueKind.Number ? mvProp.GetSingle() : null;
                    _engine.HandleSync(trackId, curTime, playing, mv);
                    break;
                }
                case "download_model":
                {
                    string model = root.TryGetProperty("model", out var m) ? (m.GetString() ?? "karaoke") : "karaoke";
                    _ = _engine.StartModelDownloadAsync(model);
                    break;
                }
            }
        }
        catch (Exception ex)
        {
            Logger.Warn($"Invalid WS command: {ex.Message}");
        }
    }

    private void OnEngineStatusChanged()
    {
        _ = BroadcastStatusAsync();
    }

    private async Task PeriodicStatusPumpAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(500, ct);
                if (_engine.GetSnapshot().Enabled)
                {
                    await BroadcastStatusAsync();
                }
            }
            catch (OperationCanceledException) { break; }
            catch { }
        }
    }

    private async Task SendStatusToClientAsync(ClientConnection conn, CancellationToken ct)
    {
        if (conn.Socket.State != WebSocketState.Open) return;
        bool acquired = false;
        try { acquired = conn.SendLock.Wait(0); } catch (ObjectDisposedException) { return; }
        if (!acquired) return;

        try
        {
            byte[] bytes = Encoding.UTF8.GetBytes(BuildStatusJson());
            using var sendCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            sendCts.CancelAfter(1000);
            await conn.Socket.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, sendCts.Token);
        }
        catch { }
        finally
        {
            try { conn.SendLock.Release(); } catch (ObjectDisposedException) { }
        }
    }

    private async Task BroadcastStatusAsync()
    {
        List<ClientConnection> snap;
        lock (_clientsLock)
        {
            if (_clients.Count == 0) return;
            snap = new List<ClientConnection>(_clients);
        }

        byte[] bytes = Encoding.UTF8.GetBytes(BuildStatusJson());
        var seg = new ArraySegment<byte>(bytes);

        foreach (var conn in snap)
        {
            if (conn.Socket.State != WebSocketState.Open) continue;
            bool acquired = false;
            try { acquired = conn.SendLock.Wait(0); } catch (ObjectDisposedException) { continue; }
            if (!acquired) continue; // Drop overlapping status frame instead of aborting WebSocket

            try
            {
                using var cts = new CancellationTokenSource(1000);
                await conn.Socket.SendAsync(seg, WebSocketMessageType.Text, true, cts.Token);
            }
            catch { }
            finally
            {
                try { conn.SendLock.Release(); } catch (ObjectDisposedException) { }
            }
        }
    }

    private string BuildStatusJson()
    {
        var s = _engine.GetSnapshot();
        return JsonSerializer.Serialize(new
        {
            type = "status",
            enabled = s.Enabled,
            state = s.State,
            provider = s.Provider,
            mode = s.Mode,
            vocalGain = s.VocalGain,
            masterVolume = s.MasterVolume,
            trackId = s.TrackId,
            separatedPct = s.SeparatedPct,
            fromCache = s.FromCache,
            cacheEntries = s.CacheEntries,
            cacheMb = s.CacheMb,
            modelInstalled = new
            {
                karaoke = s.KaraokeModelInstalled,
                instrumental = s.InstrumentalModelInstalled,
            },
            downloadProgress = s.DownloadProgress,
            lastChunkMs = s.LastChunkMs,
            error = s.Error,
        }, _jsonOptions);
    }

    public void Dispose()
    {
        _engine.StatusChanged -= OnEngineStatusChanged;
        try { _disconnectGraceCts?.Cancel(); } catch { }
        try { _cts.Cancel(); } catch { }
        try { _listener?.Stop(); } catch { }
        try { _listener?.Close(); } catch { }
        _disconnectGraceCts?.Dispose();
        _cts.Dispose();
    }
}
