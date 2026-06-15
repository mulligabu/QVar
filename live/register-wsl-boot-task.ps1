# Registers a Windows Task Scheduler task that boots the WSL 'Ubuntu' distro at system
# startup -- no interactive login required -- and holds it alive. Once WSL is up, systemd
# (systemd=true) auto-starts crypto-engines.service, which launches the tmux 'crypto'
# session (main + conditional engines + dashboard).
#
# RUN THIS ONCE, FROM AN ELEVATED (Administrator) PowerShell:
#   powershell -ExecutionPolicy Bypass -File \\wsl$\Ubuntu\home\user\crypto_v2\live\register-wsl-boot-task.ps1
# or copy it to the Windows side first, then run it elevated.
#
# Principal: S4U logon for main\ramij = "run whether user is logged on or not" WITHOUT
# storing a password, triggered at boot. Action: wsl.exe ... sleep infinity = a keepalive
# that prevents WSL2's idle auto-shutdown so the engines keep running headless.

$ErrorActionPreference = "Stop"

# --- must be elevated ---
$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    Write-Error "This must be run from an ELEVATED (Administrator) PowerShell."
    exit 1
}

$TaskName = "WSL-crypto-boot"
$User     = "main\ramij"
$Distro   = "Ubuntu"
$Wsl      = "$env:WINDIR\System32\wsl.exe"

# Boot the distro (-> systemd -> crypto-engines.service) and hold it alive with sleep.
$Action = New-ScheduledTaskAction -Execute $Wsl `
    -Argument "-d $Distro -u user --exec /usr/bin/sleep infinity"

$Trigger = New-ScheduledTaskTrigger -AtStartup

$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType S4U -RunLevel Highest

# No execution time limit (keepalive runs forever); survive battery; start when ready.
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
    -Principal $Principal -Settings $Settings -Force `
    -Description "Boot WSL Ubuntu at startup (no login) and keep it alive so crypto_v2 engines auto-run."

Write-Host "Registered task '$TaskName'. Test now without rebooting:  Start-ScheduledTask -TaskName $TaskName"
