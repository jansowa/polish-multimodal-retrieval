param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$trainingRoot = (Resolve-Path -LiteralPath (Join-Path $workspaceRoot 'training')).Path
$reportPath = Join-Path $workspaceRoot 'reports/training_storage_cleanup.json'

function Resolve-TrainingFile([string]$path) {
    $resolved = (Resolve-Path -LiteralPath $path).Path
    if (-not $resolved.StartsWith($trainingRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escapes training: $resolved"
    }
    if ((Get-Item -LiteralPath $resolved).PSIsContainer) { throw "Expected file: $resolved" }
    return $resolved
}

function Get-Sha([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$remove = @()
foreach ($series in @('tinyclip-optimization','tinyclip-generalization','tinyclip-frozen-replication')) {
    $directory = Join-Path $trainingRoot "runs/$series"
    foreach ($file in Get-ChildItem -LiteralPath $directory -File -Filter 'last.pt' -Recurse) {
        $path = Resolve-TrainingFile $file.FullName
        $resultPath = Join-Path $file.DirectoryName 'result.json'
        $bestPath = Join-Path $file.DirectoryName 'best.pt'
        if (-not (Test-Path -LiteralPath $resultPath) -or -not (Test-Path -LiteralPath $bestPath)) { continue }
        $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $bestPath = Resolve-TrainingFile $bestPath
        if ((Get-Sha $bestPath) -ne $result.checkpoint_sha256) { throw "Selected checkpoint differs: $bestPath" }
        $remove += [pscustomobject]@{path=$path;bytes=$file.Length;sha256=(Get-Sha $path);reason='Optimizer/RNG resume state of a completed run; selected best checkpoint verified and retained';best=$bestPath;best_sha256=$result.checkpoint_sha256}
    }
}

$links = @()
$groups = @(
    @('runs/p0-e2/evaluation_examples.json','runs/vision-four/reference/artifacts/evaluation_examples.json'),
    @('runs/vision-four/common/evaluation_compact.json','runs/clean-v1/common/evaluation_compact.json','runs/vision-commercial-five/common/evaluation_compact.json','runs/tinyclip-adaptation/finalist_evaluation/common/evaluation_compact.json')
)
foreach ($group in $groups) {
    $canonical = Resolve-TrainingFile (Join-Path $trainingRoot $group[0])
    $sha = Get-Sha $canonical
    foreach ($relative in $group[1..($group.Count-1)]) {
        $duplicate = Resolve-TrainingFile (Join-Path $trainingRoot $relative)
        if ((Get-Sha $duplicate) -ne $sha) { throw "Files differ: $duplicate" }
        $links += [pscustomobject]@{path=$duplicate;canonical=$canonical;bytes=(Get-Item -LiteralPath $duplicate).Length;sha256=$sha;reason='Byte-identical immutable evaluation snapshot; keep both paths via NTFS hardlink'}
    }
}

$freeBefore = (Get-PSDrive -Name D).Free
$value = [ordered]@{status='planned';created_at=[DateTime]::UtcNow.ToString('o');remove=$remove;hardlink=$links;recoverable_bytes=($remove | Measure-Object bytes -Sum).Sum+($links | Measure-Object bytes -Sum).Sum;active_visualnews_tree_untouched=$true;free_bytes_before=$freeBefore}
if ($Apply) {
    foreach ($item in $links) {
        $duplicate = Resolve-TrainingFile $item.path
        $canonical = Resolve-TrainingFile $item.canonical
        $backup = $duplicate + '.dedup-backup'
        if (Test-Path -LiteralPath $backup) { throw "Backup already exists: $backup" }
        # Checked absolute paths stay within training; use native PowerShell end to end.
        if (-not $backup.StartsWith($trainingRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Backup escapes training' }
        Move-Item -LiteralPath $duplicate -Destination $backup
        try {
            New-Item -ItemType HardLink -Path $duplicate -Target $canonical | Out-Null
            if ((Get-Sha $duplicate) -ne $item.sha256) { throw 'Hardlink content verification failed' }
        } catch {
            if (Test-Path -LiteralPath $duplicate) { Remove-Item -LiteralPath $duplicate }
            Move-Item -LiteralPath $backup -Destination $duplicate
            throw
        }
        $checkedBackup = Resolve-TrainingFile $backup
        Remove-Item -LiteralPath $checkedBackup
    }
    foreach ($item in $remove) {
        $checked = Resolve-TrainingFile $item.path
        if ((Get-Sha $checked) -ne $item.sha256 -or (Get-Sha $item.best) -ne $item.best_sha256) { throw 'File changed after plan' }
        Remove-Item -LiteralPath $checked
    }
    $value.status = 'complete'
    $value.free_bytes_after = (Get-PSDrive -Name D).Free
    $value.observed_free_increase_bytes = $value.free_bytes_after - $freeBefore
}
$value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
[pscustomobject]@{status=$value.status;resume_files=$remove.Count;deduplicated_files=$links.Count;recoverable_GiB=[Math]::Round($value.recoverable_bytes/1GB,3);free_GiB=[Math]::Round(((Get-PSDrive D).Free)/1GB,3)} | ConvertTo-Json
