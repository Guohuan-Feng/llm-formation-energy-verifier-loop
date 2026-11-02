$ErrorActionPreference='Stop'
$h2Root='C:\Users\28908\Codex\fung-research\v2-diagnosis'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONHASHSEED='0'
$env:PYTHONPATH=$h2Root
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:TEMP="$h2Root\work\h-data\H2\cache\tmp"
$env:TMP=$env:TEMP
& "$h2Root\work\p1-env\Scripts\python.exe" -B -c "from pathlib import Path; p=Path(r'$h2Root').resolve(); paths=[p,p/'work/h-data/H2',p/'work/h-data/H2/benchmark_stdout.log',p/'work/h-data/H2/benchmark_stderr.log']; assert all('onedrive' not in str(x.resolve()).lower() for x in paths)"
if($LASTEXITCODE -ne 0){throw 'Resolved H2 path preflight failed'}
$h2Process=Start-Process -FilePath "$h2Root\work\p1-env\Scripts\python.exe" -ArgumentList @('-B',"$h2Root\h2_benchmark.py",'benchmark') -WorkingDirectory "$h2Root\work\h-data\H2" -WindowStyle Hidden -RedirectStandardOutput "$h2Root\work\h-data\H2\benchmark_stdout.log" -RedirectStandardError "$h2Root\work\h-data\H2\benchmark_stderr.log" -PassThru
$h2Process.WaitForExit()
$h2State=(Get-Content -LiteralPath "$h2Root\H2_formal_dataset\metrics.json" -Raw | ConvertFrom-Json).status
if($h2State -eq 'benchmark_complete'){exit 0}
if($h2State -like 'stopped_budget*'){exit 75}
exit 1
