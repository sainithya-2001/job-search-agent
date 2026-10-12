# Daily job-agent cycle, run by Windows Task Scheduler (task "JobAgentDaily", weekdays 8:00 AM).
# Runs `main.py run` with the project's .venv Python and writes everything it prints to
# logs\scheduled_YYYYMMDD_HHMMSS.log. The run itself also writes logs\run_*.jsonl, logs\run_*_summary.md,
# and output\digest_YYYY-MM-DD.md.
#   -DryRun   plan only (`main.py run --dry-run`): no LLM calls, no JSearch requests, no DB writes.
param([switch]$DryRun)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$env:PYTHONUTF8 = '1'                 # the run prints non-ASCII characters; keep the log readable
$env:PYTHONIOENCODING = 'utf-8'

$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ('scheduled_{0:yyyyMMdd_HHmmss}.log' -f (Get-Date))
$python = Join-Path $root '.venv\Scripts\python.exe'
$runArgs = if ($DryRun) { "-u main.py run --dry-run" } else { "-u main.py run" }   # -u: write to the log as it runs

"[$(Get-Date -Format s)] start: $python $runArgs" | Out-File -FilePath $log -Encoding utf8
# cmd.exe does the redirect so Python's stderr lands in the log as plain text (Windows PowerShell 5.1 would
# wrap each stderr line in an error record).
cmd.exe /c "`"$python`" $runArgs >> `"$log`" 2>&1"
$code = $LASTEXITCODE
"[$(Get-Date -Format s)] exit code: $code" | Out-File -FilePath $log -Encoding utf8 -Append
exit $code
