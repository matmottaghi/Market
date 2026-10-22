param([switch]$Install, [int]$Port = 8501)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed.' }
    $Install = $true
}
if ($Install) {
    & $pythonPath -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network/DNS and retry with -Install.' }
}
& $pythonPath -c 'import streamlit, torch, pandas, plotly'
if ($LASTEXITCODE -ne 0) { throw 'Dependencies missing. Run .\Start-Market.ps1 -Install' }
& $pythonPath -m market_dss init
if ($LASTEXITCODE -ne 0) { throw 'Storage initialization failed.' }
& $pythonPath -m streamlit run app.py --server.port $Port --server.address 127.0.0.1
