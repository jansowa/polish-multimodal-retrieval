param([string]$Unit = 'codex-wiki-recovery-20261031.service')
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class WikiRecoveryPower {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern uint SetThreadExecutionState(uint flags);
}
'@
try {
    [void][WikiRecoveryPower]::SetThreadExecutionState([uint32]2147483649)
    while ($true) {
        $queueState = & wsl.exe -e systemctl --user is-active $Unit 2>$null
        if ($queueState -notin @('active', 'activating', 'reloading')) { break }
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][WikiRecoveryPower]::SetThreadExecutionState([uint32]2147483648)
}
