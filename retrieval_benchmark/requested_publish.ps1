param([Parameter(Mandatory=$true)][string]$RepoRoot)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $RepoRoot
$runDir = Join-Path $RepoRoot 'training/runs/f_series/leaderboard-requested-20261101'
$mutexFile = Join-Path $runDir 'windows-publisher.lock'
$lockHandle = [System.IO.File]::Open($mutexFile, 'OpenOrCreate', 'ReadWrite', 'None')
$published = ''
function Invoke-GitChecked {
    param([string[]]$GitArguments)
    # Windows PowerShell treats successful Git progress on stderr as ErrorRecord.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & git @GitArguments 2>&1
        $gitExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    if ($gitExit -ne 0) { throw "git $($GitArguments -join ' ') failed: $output" }
    return $output
}
try {
    while ($true) {
        $requestPath = Join-Path $runDir 'PUBLISH_REQUEST.json'
        if (Test-Path -LiteralPath $requestPath) {
            $request = Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
            if ($request.results_sha256 -ne $published) {
                try {
                    $results = Join-Path $RepoRoot 'retrieval_benchmark/space/results.json'
                    $actual = (Get-FileHash -LiteralPath $results -Algorithm SHA256).Hash.ToLowerInvariant()
                    if ($actual -ne $request.results_sha256) { throw 'Leaderboard changed during publication; retry next poll' }
                    $staged = @(Invoke-GitChecked -GitArguments @('diff','--cached','--name-only'))
                    if (@($staged | Where-Object { $_ -and $_ -ne 'retrieval_benchmark/space/results.json' }).Count) {
                        throw 'Another session has staged changes; retry without committing them'
                    }
                    Invoke-GitChecked -GitArguments @('add','--','retrieval_benchmark/space/results.json')
                    $staged = @(Invoke-GitChecked -GitArguments @('diff','--cached','--name-only'))
                    if ($staged.Count) {
                        Invoke-GitChecked -GitArguments @('commit','-m','[S04-LEADERBOARD] Add completed reference leaderboard scores')
                    }
                    Invoke-GitChecked -GitArguments @('push','origin','HEAD:main')
                    $published = $request.results_sha256
                    $commit = (Invoke-GitChecked -GitArguments @('rev-parse','HEAD')) -join ''
                    @{ status='pushed'; commit=$commit; results_sha256=$published; time=[DateTime]::UtcNow.ToString('o'); pid=$PID } |
                        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'PUBLISH_STATUS.json') -Encoding UTF8
                } catch {
                    @{ status='retrying'; error=$_.Exception.Message; time=[DateTime]::UtcNow.ToString('o'); pid=$PID } |
                        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'PUBLISH_STATUS.json') -Encoding UTF8
                }
            }
        }
        $statePath = Join-Path $runDir 'STATE.json'
        if ((Test-Path -LiteralPath $statePath) -and (Test-Path -LiteralPath $requestPath)) {
            $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
            $request = Get-Content -LiteralPath $requestPath -Raw | ConvertFrom-Json
            if ($state.status -in @('complete','partial') -and $published -eq $request.results_sha256) { break }
        }
        Start-Sleep -Seconds 15
    }
} finally { $lockHandle.Dispose() }
