param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$trainingRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$reportRoot = Join-Path $projectRoot 'reports'
$auditPath = Join-Path $reportRoot 'training_cache_compression.json'
if ($Apply -and (Test-Path -LiteralPath $auditPath)) {
    $existingAudit = Get-Content -Raw -LiteralPath $auditPath | ConvertFrom-Json
    if ($existingAudit.status -eq 'complete') { Write-Output 'Compression already completed; original savings audit retained.'; return }
}
$verification = Get-Content -Raw -LiteralPath (Join-Path $reportRoot 'tinyclip_visualnews_verification.json') | ConvertFrom-Json
$worker = Get-Content -Raw -LiteralPath (Join-Path $trainingRoot 'runs/tinyclip-visualnews-v1/status.json') | ConvertFrom-Json
if ($verification.status -ne 'passed' -or $worker.status -ne 'complete') {
    throw 'Wait for training and selected-checkpoint verification to finish.'
}
$volume = Get-Volume -DriveLetter D
if ($volume.FileSystem -ne 'NTFS' -or $volume.AllocationUnitSize -gt 4096) { throw 'NTFS with <=4KiB clusters required.' }
if (-not ('TrainingCacheSpace' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class TrainingCacheSpace {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    private static extern uint GetCompressedFileSizeW(string path, out uint high);
    public static long Bytes(string path) {
        uint high;
        uint low=GetCompressedFileSizeW(path,out high);
        if (low==uint.MaxValue) {
            int error=Marshal.GetLastWin32Error();
            if (error!=0) throw new Win32Exception(error);
        }
        return (long)(((ulong)high<<32)|low);
    }
}
'@
}
$targets = @('runs/tinyclip-adaptation/cache/prefix.npy',
    'runs/tinyclip-generalization/pixels/native_rgb.npy',
    'runs/tinyclip-visualnews-v1/data/native_rgb.npy')
$entries = @()
foreach ($relative in $targets) {
    $target = [IO.Path]::GetFullPath((Join-Path $trainingRoot $relative))
    if (-not $target.StartsWith($trainingRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Cache target escaped training directory.'
    }
    $item = Get-Item -LiteralPath $target
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Only regular cache files supported: $target"
    }
    $before = [TrainingCacheSpace]::Bytes($target)
    if (-not $Apply) {
        [pscustomobject]@{ path=$target; logical_GiB=[Math]::Round($item.Length/1GB,3);
            physical_GiB=[Math]::Round($before/1GB,3); attributes=$item.Attributes.ToString() }
        continue
    }
    Write-Output "Checking SHA before lossless NTFS compression: $relative"
    $sha = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    & compact.exe /C /Q $target
    if ($LASTEXITCODE -ne 0) { throw "NTFS compression failed for $target" }
    $after = [TrainingCacheSpace]::Bytes($target)
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha) {
        throw "Logical file content changed: $target"
    }
    $entries += [pscustomobject]@{ path=$target; logical_bytes=$item.Length; physical_bytes_before=$before;
        physical_bytes_after=$after; recovered_bytes=($before-$after); sha256=$sha; content_unchanged=$true }
    [ordered]@{ status='compressing'; compression='Standard transparent NTFS compression, compact /C; no WOF/EXE mode';
        entries=$entries; recovered_bytes=($entries | Measure-Object -Property recovered_bytes -Sum).Sum;
        all_paths_retained=$true; logical_file_content_unchanged=$true; reversible=$true } |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $auditPath -Encoding UTF8
    Write-Output ('Content verified; recovered {0:N3} GiB: {1}' -f (($before-$after)/1GB),$relative)
}
if ($Apply) {
    $audit = Get-Content -Raw -LiteralPath $auditPath | ConvertFrom-Json
    $audit.status = 'complete'
    $audit | Add-Member -NotePropertyName completed_at -NotePropertyValue ([DateTime]::UtcNow.ToString('o'))
    $audit | Add-Member -NotePropertyName free_bytes_after -NotePropertyValue ((Get-PSDrive D).Free)
    $audit | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $auditPath -Encoding UTF8
    [pscustomobject]@{ status='complete'; recovered_GiB=[Math]::Round($audit.recovered_bytes/1GB,3);
        free_GiB=[Math]::Round($audit.free_bytes_after/1GB,3) }
}
