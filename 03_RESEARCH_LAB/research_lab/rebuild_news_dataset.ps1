param(
    [string]$RawFile = "data/forex_factory_cache.csv",
    [string]$CleanFile = "data/news_eurusd_m15_validated.csv",
    [string]$AuditFile = "data/news_eurusd_m15_audit.csv",
    [string]$SummaryFile = "data/news_eurusd_m15_summary.json"
)

$ErrorActionPreference = "Stop"

$nyTz = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$utcTz = [System.TimeZoneInfo]::Utc

$fixedSchedule = @{
    "non-farm employment change" = "08:30"
    "unemployment rate" = "08:30"
    "cpi y/y" = "08:30"
    "cpi m/m" = "08:30"
    "core cpi m/m" = "08:30"
    "retail sales m/m" = "08:30"
    "core retail sales m/m" = "08:30"
    "adp non-farm employment change" = "08:15"
    "ism manufacturing pmi" = "10:00"
    "ism services pmi" = "10:00"
    "advance gdp q/q" = "08:30"
    "prelim gdp q/q" = "08:30"
    "final gdp q/q" = "08:30"
    "gdp q/q" = "08:30"
    "ppi m/m" = "08:30"
    "ppi y/y" = "08:30"
    "core ppi m/m" = "08:30"
    "fomc meeting minutes" = "14:00"
    "fomc statement" = "14:00"
    "fed announcement" = "14:00"
    "federal funds rate" = "14:00"
    "fomc press conference" = "14:30"
    "main refinancing rate" = "07:45"
    "ecb press conference" = "08:30"
}

function Normalize-EventName {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return "" }
    return (($Value.Trim().ToLower() -replace "\s+", " "))
}

function Convert-RawToUtcAndNy {
    param([string]$OriginalTimestamp)
    $raw = [datetimeoffset]::Parse($OriginalTimestamp, [System.Globalization.CultureInfo]::InvariantCulture)
    $utc = $raw.ToUniversalTime()
    $ny = [System.TimeZoneInfo]::ConvertTime($raw, $nyTz)
    return @{
        utc = $utc
        ny = $ny
    }
}

function Convert-FixedNySchedule {
    param(
        [string]$OriginalTimestamp,
        [string]$ExpectedTimeNy
    )
    $raw = [datetimeoffset]::Parse($OriginalTimestamp, [System.Globalization.CultureInfo]::InvariantCulture)
    $localDate = $raw.Date.ToString("yyyy-MM-dd")
    $dt = [datetime]::ParseExact("$localDate $ExpectedTimeNy", "yyyy-MM-dd HH:mm", [System.Globalization.CultureInfo]::InvariantCulture)
    $offset = $nyTz.GetUtcOffset($dt)
    $ny = [datetimeoffset]::new($dt, $offset)
    $utc = $ny.ToUniversalTime()
    return @{
        utc = $utc
        ny = $ny
    }
}

$cleanRows = Import-Csv $CleanFile
foreach ($row in $cleanRows) {
    $normalized = Normalize-EventName $row.event_name_normalized
    if ($row.validation_status -eq "approved_fixed_schedule" -and $fixedSchedule.ContainsKey($normalized)) {
        $converted = Convert-FixedNySchedule -OriginalTimestamp $row.timestamp_original -ExpectedTimeNy $fixedSchedule[$normalized]
        $row.timestamp_utc = $converted.utc.ToString("o")
        $row.timestamp_ny = $converted.ny.ToString("o")
    }
    elseif ($row.validation_status -eq "approved_raw_schedule") {
        $converted = Convert-RawToUtcAndNy -OriginalTimestamp $row.timestamp_original
        $row.timestamp_utc = $converted.utc.ToString("o")
        $row.timestamp_ny = $converted.ny.ToString("o")
    }
}
$cleanRows | Export-Csv $CleanFile -NoTypeInformation -Encoding UTF8

$auditRows = Import-Csv $AuditFile
foreach ($row in $auditRows) {
    $normalized = Normalize-EventName $row.event_name_normalized
    if ($row.validation_status -eq "approved_fixed_schedule" -and $fixedSchedule.ContainsKey($normalized)) {
        $converted = Convert-FixedNySchedule -OriginalTimestamp $row.timestamp_original -ExpectedTimeNy $fixedSchedule[$normalized]
        $row.timestamp_utc = $converted.utc.ToString("o")
        $row.timestamp_ny = $converted.ny.ToString("o")
    }
    elseif ($row.validation_status -eq "approved_raw_schedule") {
        $converted = Convert-RawToUtcAndNy -OriginalTimestamp $row.timestamp_original
        $row.timestamp_utc = $converted.utc.ToString("o")
        $row.timestamp_ny = $converted.ny.ToString("o")
    }
}
$auditRows | Export-Csv $AuditFile -NoTypeInformation -Encoding UTF8

$clean = Import-Csv $CleanFile
$audit = Import-Csv $AuditFile
$summary = [ordered]@{
    raw_source_path = $RawFile
    clean_dataset_path = $CleanFile
    audit_dataset_path = $AuditFile
    raw_rows = $audit.Count
    normalized_rows = $audit.Count
    approved_rows = ($clean | Measure-Object).Count
    rejected_rows = ($audit | Where-Object { $_.validation_status -notlike "approved*" } | Measure-Object).Count
    duplicate_rows_removed = ($audit | Where-Object { $_.validation_status -eq "rejected_duplicate" } | Measure-Object).Count
    approved_raw_schedule = ($clean | Where-Object { $_.validation_status -eq "approved_raw_schedule" } | Measure-Object).Count
    approved_fixed_schedule = ($clean | Where-Object { $_.validation_status -eq "approved_fixed_schedule" } | Measure-Object).Count
    source_approved = $false
    module_verdict = "REJECTED_DISABLED"
}
($summary | ConvertTo-Json -Depth 5) | Set-Content $SummaryFile -Encoding UTF8
