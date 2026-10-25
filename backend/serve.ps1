# Run the backend with a song-to-piano arranger, outside the preview pane (the desktop app stops preview
# servers on its own; a Terminal-panel tab keeps running).  Same settings as the backend-* configs in
# .claude/launch.json.
#   & .\serve.ps1                                              arr_scratch at 0.9
#   & .\serve.ps1 -Arranger app/ml/spectral/checkpoints/arr_attack_lora/best.pt
#   & .\serve.ps1 -Arranger app/ml/spectral/checkpoints/joint_s5/best.pt -Switches singer,drums
param(
    [string]$Arranger = "app/ml/spectral/checkpoints/arr_scratch/best.pt",
    [string]$Threshold = "0.9",
    [string]$Switches = ""
)
Set-Location $PSScriptRoot
$env:ARRANGEMENT_PIANO_MODE = "spectral"
$env:ARRANGEMENT_PIANO_ONLY = "1"
$env:ARRANGEMENT_SONG_ARRANGER = $Arranger
$env:ARRANGEMENT_SONG_ARRANGER_THR = $Threshold
$env:ARRANGEMENT_SONG_ARRANGER_FRAME_THR = "0.5"
$env:ARRANGEMENT_FORCE_SONG = "1"
$env:ARRANGEMENT_SPECTRAL_ONSET_THR = "0.85"
if ($Switches) { $env:ARRANGEMENT_JOINT_SWITCHES = $Switches }
Write-Host "backend: arranger $Arranger at $Threshold"
& .\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
