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
& "$h2Root\work\p1-env\Scripts\python.exe" -B -c "from pathlib import Path; p=Path(r'$h2Root').resolve(); paths=[p,p/'work/h-data/H2',p/'work/h-data/H2/repeat_stdout.log',p/'work/h-data/H2/repeat_stderr.log']; assert all('onedrive' not in str(x.resolve()).lower() for x in paths)"
if($LASTEXITCODE -ne 0){throw 'Resolved H2 output path preflight failed'}
$h2Budget=Get-Content -LiteralPath "$h2Root\work\h-data\H2\budget.json" -Raw | ConvertFrom-Json
$h2RemainingMs=[int][Math]::Max(1,(86400-[double]$h2Budget.spent_seconds-30)*1000)
$h2Process=Start-Process -FilePath "$h2Root\work\p1-env\Scripts\python.exe" -ArgumentList @('-B',"$h2Root\h2_protocol.py",'repeat') -WorkingDirectory "$h2Root\work\h-data\H2" -WindowStyle Hidden -RedirectStandardOutput "$h2Root\work\h-data\H2\repeat_stdout.log" -RedirectStandardError "$h2Root\work\h-data\H2\repeat_stderr.log" -PassThru
$h2ProcessHandle=$h2Process.Handle
if(-not $h2Process.WaitForExit($h2RemainingMs)){
    & taskkill.exe /PID $h2Process.Id /T /F | Out-Null
    & "$h2Root\work\p1-env\Scripts\python.exe" -B -c "import h2_common as c;c.setup();b=c.read(c.DATA/'budget.json');b['spent_seconds']=c.BUDGET_SECONDS;b.pop('active_phase',None);b['phases'].append({'name':'serial_repeat_protocol1','status':'stopped_hard_budget','completed_utc':c.now()});c.write(c.DATA/'budget.json',b);c.metrics_update(status='stopped_budget',phase='repeat_hard_budget_guard',stop_reason='Total24h watchdog stopped owned repeat process tree; saved prior results retained.')"
    exit 75
}
$h2Process.Refresh()
$h2ChildExitCode=$h2Process.ExitCode
if($null -eq $h2ChildExitCode){throw 'Child process exit code unavailable'}
if($h2ChildExitCode -ne 0){exit $h2ChildExitCode}
$h2State=(Get-Content -LiteralPath "$h2Root\H2_formal_dataset\metrics.json" -Raw | ConvertFrom-Json).status
if($h2State -eq 'serial_repeat_complete'){exit 0}
if($h2State -like 'stopped_budget*'){exit 75}
exit 1
