param([string]$TaskName = 'Codex-WikiRecovery-20261031')
$ErrorActionPreference = 'Stop'
$recoveryRepo = Split-Path (Split-Path (Split-Path (Split-Path $PSCommandPath -Parent) -Parent) -Parent) -Parent
$recoveryLogRoot = 'C:\Users\jsowa\benchmark-translations\data-ood-20261030\A\wiki-recovery-20261031\logs'
$recoveryStatus = Join-Path $recoveryRepo 'reports\data_study_20261030\wiki_recovery_20261031\STATUS.json'
$recoverySupervisorLog = Join-Path $recoveryLogRoot 'windows_supervisor.log'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class WikiRecoveryPersistentPower {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern uint SetThreadExecutionState(uint flags);
}
'@
try {
    $previousPowerState = [WikiRecoveryPersistentPower]::SetThreadExecutionState([uint32]2147483649)
    if ($previousPowerState -eq 0) { throw 'SetThreadExecutionState failed' }
    for ($recoveryAttempt = 1; $recoveryAttempt -le 5; $recoveryAttempt++) {
        if (Test-Path -LiteralPath $recoveryStatus) {
            $recoveryState = Get-Content -LiteralPath $recoveryStatus -Raw | ConvertFrom-Json
            if ($recoveryState.status -eq 'complete') {
                Add-Content -LiteralPath $recoverySupervisorLog -Value "$(Get-Date -Format o) COMPLETE"
                exit 0
            }
        }
        Add-Content -LiteralPath $recoverySupervisorLog -Value "$(Get-Date -Format o) START attempt=$recoveryAttempt"
        # This is a long-running WSL execution, not a transient systemd user
        # service. Its login remains alive for the entire queue lifetime.
        $recoveryArguments = @('-e','/home/jsowa/.venvs/pl-mmb-benchmark/bin/python','-u',
            '/mnt/d/work/translate-vision-embedder-benchmark/pl-multimodal-benchmark-translator/pl-multimodal-benchmark-translator/experiments/data_study_20261030/wiki_recovery_20261031/overnight_queue.py')
        $recoveryWorker = Start-Process -FilePath 'wsl.exe' -ArgumentList $recoveryArguments -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $recoveryLogRoot "windows_worker_$recoveryAttempt.out.log") `
            -RedirectStandardError (Join-Path $recoveryLogRoot "windows_worker_$recoveryAttempt.err.log")
        Add-Content -LiteralPath $recoverySupervisorLog -Value "$(Get-Date -Format o) WSL_PID=$($recoveryWorker.Id)"
        $recoveryWorker.WaitForExit()
        Add-Content -LiteralPath $recoverySupervisorLog -Value "$(Get-Date -Format o) EXIT code=$($recoveryWorker.ExitCode)"
        if ($recoveryWorker.ExitCode -eq 0) { exit 0 }
        Start-Sleep -Seconds 60
    }
    throw 'Queue failed after five supervised attempts; inspect logs'
}
finally {
    [void][WikiRecoveryPersistentPower]::SetThreadExecutionState([uint32]2147483648)
}
