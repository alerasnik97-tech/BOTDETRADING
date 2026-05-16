#Requires -Version 5.1
# Solo lectura: lista archivos/carpetas en DEST que no existen en SOURCE (misma ruta relativa).
$ErrorActionPreference = 'Stop'

$SourceRoot = 'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo ANTIGRABITY'
$DestRoot   = 'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo CURSOR'
$ReportPath = Join-Path $DestRoot '_audit_dest_extras_report.txt'

function Get-ExtendedPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { return $Path }
    $p = $Path.TrimEnd('\')
    if ($p.StartsWith('\\?\', [StringComparison]::Ordinal)) { return $p }
    if ($p.StartsWith('\\', [StringComparison]::Ordinal)) {
        if ($p.StartsWith('\\?\UNC\', [StringComparison]::OrdinalIgnoreCase)) { return $p }
        $unc = $p.TrimStart('\')
        return '\\?\UNC\' + $unc
    }
    return '\\?\' + $p
}

function Join-Extended([string]$Root, [string]$Relative) {
    $r = $Root.TrimEnd('\')
    if ([string]::IsNullOrWhiteSpace($Relative)) { return $r }
    return $r + '\' + ($Relative -replace '/', '\')
}

function Test-ReparsePoint([string]$Path) {
    try {
        $ep = Get-ExtendedPath $Path
        $di = New-Object System.IO.DirectoryInfo($ep)
        if ($di.Exists) {
            return ($di.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
        }
        $fi = New-Object System.IO.FileInfo($ep)
        if ($fi.Exists) {
            return ($fi.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
        }
    } catch { }
    return $false
}

function SourceHasDirectory([string]$Rel) {
    $fullSrc = Join-Extended $script:srcLong $Rel
    return [System.IO.Directory]::Exists($fullSrc)
}

function SourceHasFile([string]$Rel) {
    $fullSrc = Join-Extended $script:srcLong $Rel
    return [System.IO.File]::Exists($fullSrc)
}

$extraDirs    = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
$extraFiles   = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
$reparseInDest = New-Object System.Collections.Generic.List[string]
$scanErrors   = New-Object System.Collections.Generic.List[string]

$srcLong = Get-ExtendedPath $SourceRoot
$dstLong = Get-ExtendedPath $DestRoot

function Walk-Dest([string]$RelativePath) {
    $rel = if ([string]::IsNullOrEmpty($RelativePath)) { '' } else { $RelativePath }
    $fullDst = Join-Extended $dstLong $rel

    try {
        if (Test-ReparsePoint $fullDst) {
            if ($rel) { [void]$reparseInDest.Add($rel) }
            return
        }
    } catch {
        $scanErrors.Add("Reparse check dest: $rel :: $($_.Exception.Message)") | Out-Null
        return
    }

    if (-not [System.IO.Directory]::Exists($fullDst)) { return }

    if ($rel -and -not (SourceHasDirectory $rel)) {
        [void]$extraDirs.Add($rel)
    }

    try {
        $entries = [System.IO.Directory]::EnumerateFileSystemEntries($fullDst)
    } catch {
        $scanErrors.Add("Enumerate dest failed: $rel :: $($_.Exception.Message)") | Out-Null
        return
    }

    foreach ($entry in $entries) {
        $name = Split-Path $entry -Leaf
        $childRel = if ($rel) { "$rel\$name" } else { $name }

        try {
            if (Test-ReparsePoint $entry) {
                $reparseInDest.Add($childRel) | Out-Null
                continue
            }
        } catch {
            $scanErrors.Add("Reparse check dest child: $childRel :: $($_.Exception.Message)") | Out-Null
            continue
        }

        if ([System.IO.Directory]::Exists($entry)) {
            Walk-Dest $childRel
        } elseif ([System.IO.File]::Exists($entry)) {
            if (-not (SourceHasFile $childRel)) {
                [void]$extraFiles.Add($childRel)
            }
        }
    }
}

Walk-Dest ''

$sortedExtraDirs  = @($extraDirs | Sort-Object)
$sortedExtraFiles = @($extraFiles | Sort-Object)

$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("=== EXTRAS EN DESTINO (no estan en origen) ===")
[void]$sb.AppendLine("Origen: $SourceRoot")
[void]$sb.AppendLine("Destino: $DestRoot")
[void]$sb.AppendLine("Generado: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
[void]$sb.AppendLine("Rutas extendidas \\?\ para MAX_PATH")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("A. Carpetas extra (conteo): $($sortedExtraDirs.Count)")
foreach ($d in $sortedExtraDirs) { [void]$sb.AppendLine("  $d") }
[void]$sb.AppendLine("")
[void]$sb.AppendLine("B. Archivos extra (conteo): $($sortedExtraFiles.Count)")
foreach ($f in $sortedExtraFiles) { [void]$sb.AppendLine("  $f") }
[void]$sb.AppendLine("")
[void]$sb.AppendLine("C. Reparse points / junctions / symlinks en destino (no recursados bajo ellos):")
if ($reparseInDest.Count -eq 0) { [void]$sb.AppendLine("  (ninguno reportado en este barrido)") }
else { $reparseInDest | Sort-Object | ForEach-Object { [void]$sb.AppendLine("  $_") } }
[void]$sb.AppendLine("")
[void]$sb.AppendLine("D. Errores de escaneo:")
if ($scanErrors.Count -eq 0) { [void]$sb.AppendLine("  (ninguno)") }
else { $scanErrors | ForEach-Object { [void]$sb.AppendLine("  $_") } }

$text = $sb.ToString()
[System.IO.File]::WriteAllText($ReportPath, $text, [System.Text.UTF8Encoding]::new($false))
Write-Output $text
Write-Output ""
Write-Output "Reporte: $ReportPath"
