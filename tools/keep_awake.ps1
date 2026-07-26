param([int]$WatchPid)

# Keeps the machine awake while a long training run is in flight.
# Non-elevated: SetThreadExecutionState only affects the calling thread's request,
# no admin rights or power-plan changes required. The request is released when
# the watched process exits, so this never leaves the machine permanently awake.

Add-Type -Name Power -Namespace Win32 -MemberDefinition @"
[DllImport("kernel32.dll", SetLastError=true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@

$ES_CONTINUOUS      = [uint32]0x80000000
$ES_SYSTEM_REQUIRED = [uint32]0x00000001

[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)

while (Get-Process -Id $WatchPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
