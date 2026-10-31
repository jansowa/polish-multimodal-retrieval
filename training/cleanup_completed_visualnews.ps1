param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$studyRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'runs/tinyclip-visualnews-v1'))
$projectRoot = Split-Path -Parent $PSScriptRoot
$reportRoot = Join-Path $projectRoot 'reports'
$verified = Get-Content -Raw -LiteralPath (Join-Path $reportRoot 'tinyclip_visualnews_verification.json') | ConvertFrom-Json
$results = (Get-Content -Raw -LiteralPath (Join-Path $reportRoot 'tinyclip_visualnews.json') | ConvertFrom-Json).results
$workerStatus = Get-Content -Raw -LiteralPath (Join-Path $studyRoot 'status.json') | ConvertFrom-Json
if ($verified.status -ne 'passed' -or $workerStatus.status -ne 'complete') { throw 'Training and checkpoint verification must both finish first.' }
$plan = @()
foreach ($result in $results) {
    if ($result.reused_reference) { continue }
    $check = $verified.checks.PSObject.Properties[$result.experiment].Value
    if (-not $check.checkpoint_verified -or -not $check.primary_and_dev_reproduced -or -not $check.source_verified) {
        throw "Checkpoint not verified: $($result.experiment)"
    }
    $runFolder = [IO.Path]::GetFullPath((Join-Path $studyRoot $result.experiment))
    if (-not $runFolder.StartsWith($studyRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Resolved target escaped the VisualNews study directory.'
    }
    $best = Join-Path $runFolder 'best.pt'
    if ((Get-FileHash -LiteralPath $best -Algorithm SHA256).Hash.ToLowerInvariant() -ne $result.checkpoint_sha256) {
        throw "Best checkpoint changed: $best"
    }
    $last = Join-Path $runFolder 'last.pt'
    if (Test-Path -LiteralPath $last) {
        $item = Get-Item -LiteralPath $last
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing reparse point: $last" }
        $plan += [pscustomobject]@{ path=$item.FullName; bytes=$item.Length;
            sha256=(Get-FileHash -LiteralPath $last -Algorithm SHA256).Hash.ToLowerInvariant();
            best=$best; best_sha256=$result.checkpoint_sha256;
            reason='Completed adaptation run; selected checkpoint, both dev metrics, loss and source hashes verified; optimizer/RNG resume state no longer needed' }
    }
}
if (-not $plan.Count) { Write-Output 'No remaining completed VisualNews resume checkpoints.'; return }
$recoverable = ($plan | Measure-Object -Property bytes -Sum).Sum
$audit = [ordered]@{ status='planned'; created_at=[DateTime]::UtcNow.ToString('o'); remove=$plan;
    recoverable_bytes=$recoverable; best_checkpoints_retained=$true; immutable_inputs_untouched=$true }
if ($Apply) {
    $audit.free_bytes_before = (Get-PSDrive D).Free
    foreach ($entry in $plan) {
        if ((Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw "Resume checkpoint changed during cleanup: $($entry.path)"
        }
        Remove-Item -LiteralPath $entry.path
    }
    foreach ($entry in $plan) {
        if ((Get-FileHash -LiteralPath $entry.best -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.best_sha256) {
            throw "Selected checkpoint changed: $($entry.best)"
        }
    }
    $audit.status = 'complete'
    $audit.free_bytes_after = (Get-PSDrive D).Free
    $audit.observed_free_increase_bytes = $audit.free_bytes_after - $audit.free_bytes_before
    $audit | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reportRoot 'training_storage_cleanup_visualnews.json') -Encoding UTF8
}
[pscustomobject]@{ status=$audit.status; files=$plan.Count; recoverable_GiB=[Math]::Round($recoverable / 1GB,3) }
