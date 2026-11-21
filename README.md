# Karaoke Engine (`karaoke-engine`)

Real-time **Apple Music Sing**-style AI vocal separation & vocal level control backend for **Tidal Desktop (TidaLuna)** on Windows.

## Highlights

- **DirectML GPU Acceleration (Zero CUDA Setup)**: Runs ONNX MDX-NET vocal separation on any DirectX 12 GPU (NVIDIA, AMD, Intel) via `Microsoft.ML.OnnxRuntime.DirectML`.
- **Two Studio AI Models + Zero-Latency DSP Mode**:
  - **Karaoke (`UVR_MDXNET_KARA_2.onnx`)**: Removes lead vocals while preserving backing harmonies (`~50x` real-time on RTX 4060).
  - **Instrumental (`UVR-MDX-NET-Inst_HQ_3.onnx`)**: Removes all vocals for a pure instrumental stem (`~38x` real-time on RTX 4060).
  - **DSP**: Instant zero-VRAM Mid-Side spectral vocal suppression.
- **Smooth Gradual Crossfades & Vocal Melt**: Keeps original Tidal audio playing cleanly while preparing uncached tracks, then performs a 650 ms cosine crossfade and a 1.15 s gradual vocal fade-out into the AI stem.
- **Persistent Zero-Copy Disk Cache (`KRK1`)**: Caches separated PCM16 stems in `%LOCALAPPDATA%\karaoke-engine\cache\` (configurable LRU quota, default `2 GB`) so replayed songs switch in `< 20 ms` with `0%` GPU load.
- **Silent Background Daemon + Auto-Startup**: Runs invisibly in the background on `ws://127.0.0.1:5445/control` and automatically registers in Windows Startup (`HKCU\...\Run`) on first launch.

---

## Quick Start (End Users)

1. Download **`karaoke-engine-win-x64.zip`** (Lite — auto-downloads AI models on first use) or **`karaoke-engine-win-x64-full-models.zip`** (Full Offline — includes both `.onnx` models pre-bundled) from [**GitHub Releases**](https://github.com/itzzexcel/karaoke-engine/releases/latest).
2. Extract the folder anywhere and double-click **`karaoke-engine.exe`**.
   - It runs silently in the background, registers itself to start with Windows automatically, and listens on `ws://127.0.0.1:5445/control`.
3. Open **Tidal Desktop** with the `@itzzexcel/karaoke` TidaLuna plugin enabled, click the microphone icon in the player bar, and adjust the **Level (`0% – 100%`)**.

---

## Command-Line Flags

| Flag | Description |
| :--- | :--- |
| *(no flags)* | Runs silently in the background and registers Windows auto-startup on first run |
| `--show-console` | Keeps the console window visible with live log output |
| `--no-startup` | Runs without registering in Windows Startup |
| `--install-startup` | Registers `karaoke-engine.exe` in `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` and exits |
| `--uninstall-startup` | Removes `karaoke-engine.exe` from Windows Startup and exits |
| `--prefetch-models` | Downloads both `UVR_MDXNET_KARA_2.onnx` and `UVR-MDX-NET-Inst_HQ_3.onnx` into `%LOCALAPPDATA%\karaoke-engine\models` |
| `--self-test` | Runs automated FFT, DSP, DirectML ONNX zero-GC benchmark, and `KRK1` disk cache verification |

---

## Building & Packaging Releases

Run the included PowerShell script to build a **self-contained Windows x64 release** (no `.NET 9` installation required on user machines):

```powershell
# Lite ZIP (~35 MB, downloads AI models on demand) + Full ZIP (~155 MB, includes both ONNX models)
.\publish-release.ps1 -IncludeModels
```

Output archives are written to:
- `dist\karaoke-engine-win-x64.zip`
- `dist\karaoke-engine-win-x64-full-models.zip`
