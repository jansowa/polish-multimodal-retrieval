param([switch]$Apply,[ValidateSet('adaptation','generalization','visualnews')][string]$Kind='adaptation')
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$training = [IO.Path]::GetFullPath($PSScriptRoot)
$source = [IO.Path]::GetFullPath((Join-Path $training 'runs/tinyclip-adaptation/cache'))
$backup = [IO.Path]::GetFullPath((Join-Path $training 'runs/tinyclip-adaptation/cache.pre_c_relocation_20261023'))
$destination = 'C:\Users\jsowa\benchmark-translations\series-2026-10-23\relocated\tinyclip-adaptation-cache'
$expectedCount=3
if($Kind -eq 'generalization'){
    $source=[IO.Path]::GetFullPath((Join-Path $training 'runs/tinyclip-generalization/pixels'))
    $backup=[IO.Path]::GetFullPath((Join-Path $training 'runs/tinyclip-generalization/pixels.pre_c_relocation_20261023'))
    $destination='C:\Users\jsowa\benchmark-translations\series-2026-10-23\relocated\tinyclip-generalization-pixels'
    $expectedCount=2
}
if($Kind -eq 'visualnews'){
    $source=[IO.Path]::GetFullPath((Join-Path $training 'runs/tinyclip-visualnews-v1/data'))
    $backup=[IO.Path]::GetFullPath((Join-Path $training 'runs/tinyclip-visualnews-v1/data.pre_c_relocation_20261023'))
    $destination='C:\Users\jsowa\benchmark-translations\series-2026-10-23\relocated\visualnews-frozen-data'
    $expectedCount=11
}
$report = Join-Path $workspace 'reports/series_2026-10-23'
$resultPath = Join-Path $report 'storage_relocation_result.json'
if($Kind -eq 'generalization'){$resultPath=Join-Path $report 'storage_relocation_generalization_result.json'}
if($Kind -eq 'visualnews'){$resultPath=Join-Path $report 'storage_relocation_visualnews_result.json'}
if(Test-Path -LiteralPath $resultPath){
    $result = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if($result.status -eq 'complete') { Write-Output 'Relocation already complete; retained audit.'; return }
}
foreach($path in @($source,$backup)){
    if(-not $path.StartsWith($training + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){
        throw "Target escapes named training directory: $path"
    }
}
if((Get-Item -LiteralPath $source).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Source is already a reparse point'}
if(Test-Path -LiteralPath $backup){throw 'Backup exists; inspect before proceeding'}
$files = @(Get-ChildItem -LiteralPath $source -File)
if((Get-ChildItem -LiteralPath $source -Directory).Count -ne 0 -or $files.Count -ne $expectedCount){throw 'Unexpected cache contents'}
function Get-Sha([string]$path){(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
$entries = @($files | ForEach-Object { [pscustomobject]@{name=$_.Name;bytes=$_.Length;sha256=(Get-Sha $_.FullName)} })
$bytes = ($entries | Measure-Object -Property bytes -Sum).Sum
if((Get-PSDrive C).Free - $bytes -lt 4GB){throw 'C reserve would fall below 4 GiB'}
$decision = [ordered]@{
    status='registered_before_relocation';authorization='User requested continued work and use of free C space';
    source=$source;destination=$destination;preserved_original_path='NTFS junction';
    files=$entries;logical_bytes=$bytes;frozen_snapshots_moved=$false;pipeline_paths_touched=$false;
    reserve_bytes=4GB;free_C_before=(Get-PSDrive C).Free;free_D_before=(Get-PSDrive D).Free;
    verification='SHA each file before/copy/junction; WSL SHA of prefix.npy; remove checked backup only after verification'
}
if($Kind -eq 'visualnews'){
    $decision.frozen_snapshots_moved=$true
    $decision.frozen_snapshot_bytes_and_original_paths_preserved=$true
}
$decisionPath = Join-Path $report 'storage_relocation_decision.json'
if($Kind -eq 'generalization'){$decisionPath=Join-Path $report 'storage_relocation_generalization_decision.json'}
if($Kind -eq 'visualnews'){$decisionPath=Join-Path $report 'storage_relocation_visualnews_decision.json'}
if(Test-Path -LiteralPath $decisionPath){
    $prior=Get-Content -LiteralPath $decisionPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if($prior.source -ne $source -or $prior.destination -ne $destination){throw 'Different relocation already registered'}
    foreach($entry in $entries){if(($prior.files | Where-Object name -eq $entry.name).sha256 -ne $entry.sha256){throw 'Cache changed since decision'}}
}else{$decision | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $decisionPath -Encoding UTF8}
if(-not $Apply){$decision | ConvertTo-Json -Depth 6;return}
if(Test-Path -LiteralPath $destination){throw 'Destination already exists; inspect instead of overwriting'}
New-Item -ItemType Directory -Path $destination | Out-Null
foreach($entry in $entries){
    $from=Join-Path $source $entry.name;$to=Join-Path $destination $entry.name
    Copy-Item -LiteralPath $from -Destination $to
    if((Get-Sha $to) -ne $entry.sha256 -or (Get-Sha $from) -ne $entry.sha256){throw 'Copy SHA mismatch; originals retained'}
}
Move-Item -LiteralPath $source -Destination $backup
try{
    New-Item -ItemType Junction -Path $source -Target $destination | Out-Null
    foreach($entry in $entries){if((Get-Sha (Join-Path $source $entry.name)) -ne $entry.sha256){throw 'Junction SHA mismatch'}}
    $checkFile=($entries | Sort-Object bytes -Descending | Select-Object -First 1)
    $linuxPrefix='/mnt/d'+($source.Substring(2).Replace('\','/'))+'/'+$checkFile.name
    $wslHash = & wsl -e sha256sum $linuxPrefix
    if($LASTEXITCODE -ne 0 -or -not $wslHash.StartsWith($checkFile.sha256)){throw 'WSL junction read failed'}
}catch{
    if(Test-Path -LiteralPath $source){[IO.Directory]::Delete($source)}
    Move-Item -LiteralPath $backup -Destination $source
    throw
}
# Verify the resolved absolute deletion target, its scope and regular-directory type.
$checkedBackup=(Resolve-Path -LiteralPath $backup).Path
if($checkedBackup -ne $backup -or -not $checkedBackup.StartsWith($training + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){
    throw 'Backup deletion target failed scope check'
}
if((Get-Item -LiteralPath $checkedBackup).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Backup must be a regular directory'}
foreach($entry in $entries){if((Get-Sha (Join-Path $checkedBackup $entry.name)) -ne $entry.sha256){throw 'Backup changed before removal'}}
Remove-Item -LiteralPath $checkedBackup -Recurse
$decision.status='complete'
$decision.free_C_after=(Get-PSDrive C).Free;$decision.free_D_after=(Get-PSDrive D).Free
$decision.wsl_prefix_SHA_verified=$true;$decision.files_SHA_unchanged=$true
$decision | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding UTF8
$decision | ConvertTo-Json -Depth 8
