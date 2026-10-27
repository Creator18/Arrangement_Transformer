# Run the backend with a song-to-piano arranger, outside the preview pane (the desktop app stops preview
# servers on its own; a Terminal-panel tab keeps running).  Same settings as the backend-* configs in
# .claude/launch.json.
#   & .\serve.ps1                                              arr_scratch at 0.9
#   & .\serve.ps1 -Arranger app/ml/spectral/checkpoints/arr_attack_lora/best.pt
#   & .\serve.ps1 -Arranger app/ml/spectral/checkpoints/joint_s5/best.pt -Switches singer,drums
#   & .\serve.ps1 -Arranger app/ml/spectral/checkpoints/base20m/best.pt -Solo app/ml/spectral/checkpoints/solo_det/best.pt
#       (solo-voice fill and remove, app/services/solo_fill.py; -SoloRule floor+voice for the voice filter;
#        -SoloPlace octave -SoloRule octave: fill only in octave bands the normal pass leaves empty)
#   & .\serve.ps1 -Arranger app/ml/spectral/checkpoints/base20m/best.pt -Threshold 0.8 -Cleanup app/ml/spectral/checkpoints/cleanup1/best.pt
#       (keep / drop each note, app/services/cleanup.py; -Keep 0.3 keeps more)
param(
    [string]$Arranger = "app/ml/spectral/checkpoints/arr_scratch/best.pt",
    [string]$Threshold = "0.9",
    [string]$Switches = "",
    [string]$Solo = "",
    [string]$SoloRule = "floor",
    [string]$SoloPlace = "floor",
    [string]$Cleanup = "",
    [string]$Keep = "0.4"
)
Set-Location $PSScriptRoot
# FluidSynth library for the solo fill's band (pyfluidsynth looks for libfluidsynth-3.dll on PATH)
$fs = "C:\Users\HP\fluidsynth\fluidsynth-v2.6.1-win10-x64-cpp11\bin"
if (Test-Path $fs) { $env:PATH = "$fs;$env:PATH" }
$env:ARRANGEMENT_PIANO_MODE = "spectral"
$env:ARRANGEMENT_PIANO_ONLY = "1"
$env:ARRANGEMENT_SONG_ARRANGER = $Arranger
$env:ARRANGEMENT_SONG_ARRANGER_THR = $Threshold
$env:ARRANGEMENT_SONG_ARRANGER_FRAME_THR = "0.5"
$env:ARRANGEMENT_FORCE_SONG = "1"
$env:ARRANGEMENT_SPECTRAL_ONSET_THR = "0.85"
if ($Switches) { $env:ARRANGEMENT_JOINT_SWITCHES = $Switches }
if ($Solo) { $env:ARRANGEMENT_SOLO_DETECTOR = $Solo; $env:ARRANGEMENT_SOLO_RULE = $SoloRule; $env:ARRANGEMENT_SOLO_PLACE = $SoloPlace }
else { Remove-Item Env:ARRANGEMENT_SOLO_DETECTOR, Env:ARRANGEMENT_SOLO_RULE, Env:ARRANGEMENT_SOLO_PLACE -ErrorAction SilentlyContinue }   # a previous -Solo run in this shell must not linger
if ($Cleanup) { $env:ARRANGEMENT_CLEANUP = $Cleanup; $env:ARRANGEMENT_CLEANUP_KEEP = $Keep }
else { Remove-Item Env:ARRANGEMENT_CLEANUP, Env:ARRANGEMENT_CLEANUP_KEEP -ErrorAction SilentlyContinue }
Write-Host "backend: arranger $Arranger at $Threshold$(if ($Solo) { " + solo fill ($Solo, $SoloPlace placement, $SoloRule)" })$(if ($Cleanup) { " + clean-up ($Cleanup, keep $Keep)" })"
& .\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
