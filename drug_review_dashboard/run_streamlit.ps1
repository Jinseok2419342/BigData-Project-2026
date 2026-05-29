param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$logPath = Join-Path $PSScriptRoot "streamlit.run.log"
$streamlitPath = Join-Path $PSScriptRoot "..\venv\Scripts\streamlit.exe"

if (Test-Path -LiteralPath $streamlitPath) {
    & $streamlitPath run app.py --server.port $Port --server.headless true --browser.gatherUsageStats false *> $logPath
}
else {
    streamlit run app.py --server.port $Port --server.headless true --browser.gatherUsageStats false *> $logPath
}
