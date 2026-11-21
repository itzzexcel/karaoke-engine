param(
    [switch]$IncludeModels
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $Root "dist\karaoke-engine-win-x64"
$ZipLite = Join-Path $Root "dist\karaoke-engine-win-x64.zip"
$ZipFull = Join-Path $Root "dist\karaoke-engine-win-x64-full-models.zip"

Write-Host "=== Publishing karaoke-engine (Self-Contained win-x64 Release) ===" -ForegroundColor Cyan
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }

dotnet publish "$Root\karaoke-engine.csproj" `
    -c Release `
    -r win-x64 `
    --self-contained true `
    -p:DebugType=None `
    -p:DebugSymbols=false `
    -o $OutDir

if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed" }

# Remove unnecessary .pdb / .xml files if any came from dependencies
Get-ChildItem $OutDir -Include *.pdb, *.xml -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue

if (Test-Path $ZipLite) { Remove-Item $ZipLite -Force }
Compress-Archive -Path "$OutDir\*" -DestinationPath $ZipLite -CompressionLevel Optimal
$LiteMb = [Math]::Round((Get-Item $ZipLite).Length / 1MB, 1)
Write-Host "[OK] Created Lite Release ZIP ($LiteMb MB, downloads AI models on demand): $ZipLite" -ForegroundColor Green

if ($IncludeModels) {
    $LocalModels = Join-Path $env:LOCALAPPDATA "karaoke-engine\models"
    $BundledModels = Join-Path $OutDir "models"
    New-Item -ItemType Directory -Path $BundledModels -Force | Out-Null

    foreach ($file in @("UVR_MDXNET_KARA_2.onnx", "UVR-MDX-NET-Inst_HQ_3.onnx")) {
        $src = Join-Path $LocalModels $file
        if (Test-Path $src) {
            Copy-Item $src -Destination (Join-Path $BundledModels $file) -Force
            Write-Host "  + Bundled $file"
        } else {
            Write-Warning "  ! Missing local model $src (run karaoke-engine.exe --prefetch-models first)"
        }
    }

    if (Test-Path $ZipFull) { Remove-Item $ZipFull -Force }
    Compress-Archive -Path "$OutDir\*" -DestinationPath $ZipFull -CompressionLevel Optimal
    $FullMb = [Math]::Round((Get-Item $ZipFull).Length / 1MB, 1)
    Write-Host "[OK] Created Full Offline Release ZIP ($FullMb MB, includes Karaoke + Inst AI models): $ZipFull" -ForegroundColor Green
}
