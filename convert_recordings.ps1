<#
convert_recordings.ps1 — batch-convert iPhone .m4a keystroke recordings to WAV.

Straight format conversion only: no trimming, no cropping. Each file is decoded
in full and resampled to 48 kHz so the whole pool shares one sample rate
(features.py asserts this). Crop the record/stop finger-taps off the ends of each
recording yourself before dropping the .m4a files in.

Recordings are organised by session, one subfolder per recording day, and the
session structure is MIRRORED from unconverted_raw into converted_wavs:

    unconverted_raw\day_1\a.m4a   ->   converted_wavs\day_1\a.wav

Usage (from the project root):
    ./convert_recordings.ps1 -Session day_2        # unconverted_raw\day_2 -> converted_wavs\day_2
    ./convert_recordings.ps1                       # every session subfolder found
    ./convert_recordings.ps1 -InDir "some\folder"  # a flat folder, no session subdivision

Output name = input basename + .wav  (so a.m4a -> a.wav, space.m4a -> space.wav).
#>
param(
    [string]$InDir      = "unconverted_raw",
    [string]$OutDir     = "converted_wavs",
    [string]$Session    = "",
    [int]$SampleRate    = 48000
)

$ErrorActionPreference = "Stop"

# --- locate ffmpeg (PATH first, then the winget install location) ---
function Resolve-Tool([string]$name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallback = Join-Path $env:LOCALAPPDATA `
        "Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\$name.exe"
    if (Test-Path $fallback) { return $fallback }
    throw "$name not found on PATH or at the winget fallback location. Install ffmpeg or add it to PATH."
}
$ffmpeg = Resolve-Tool "ffmpeg"

if (-not (Test-Path $InDir))  { throw "Input folder not found: $InDir" }

# Work out which session subfolders to convert. A named -Session does just that
# one; otherwise every subfolder holding .m4a files, falling back to a flat
# layout when the input folder holds the recordings directly.
if ($Session) {
    $sessions = @($Session)
} else {
    $sessions = @(Get-ChildItem -Path $InDir -Directory |
                  Where-Object { (Get-ChildItem -Path $_.FullName -Filter *.m4a).Count -gt 0 } |
                  ForEach-Object { $_.Name })
    if ($sessions.Count -eq 0) { $sessions = @("") }   # flat layout
}

$total = 0
foreach ($s in $sessions) {
    $srcDir = if ($s) { Join-Path $InDir $s }  else { $InDir }
    $dstDir = if ($s) { Join-Path $OutDir $s } else { $OutDir }

    if (-not (Test-Path $srcDir)) { throw "Session folder not found: $srcDir" }
    if (-not (Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir | Out-Null }

    $files = Get-ChildItem -Path $srcDir -Filter *.m4a
    if ($files.Count -eq 0) { throw "No .m4a files found in $srcDir" }

    Write-Host "session '$s': $($files.Count) file(s)  $srcDir -> $dstDir"
    foreach ($f in $files) {
        $outPath = Join-Path $dstDir ($f.BaseName + ".wav")
        & $ffmpeg -y -i $f.FullName -ar $SampleRate $outPath
        Write-Host "  $($f.Name) -> $($f.BaseName).wav"
    }
    $total += $files.Count
}

Write-Host "done: converted $total file(s) into $OutDir\ at $SampleRate Hz."
