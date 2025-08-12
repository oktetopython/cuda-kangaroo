# PowerShell script to fix UTF-8 BOM encoding issues
# This script converts files to ANSI encoding to avoid C4819 warnings

$files = @(
    "SECPK1\IntMod.cpp",
    "main.cpp",
    "Network.cpp",
    "Backup.cpp",
    "bitcoin_puzzle_130_test.cpp",
    "bl_algorithm_verification.cpp",
    "debug_point_structure.cpp",
    "ec_verification_test.cpp",
    "optimizations\phase4\generate_bl_real_ec_table.cpp",
    "optimizations\phase4\bl_precompute_table_v2.h",
    "optimizations\phase4\bl_real_ec_generator.h",
    "optimizations\phase4\bl_online_solver_v2.h",
    "puzzle135_bl_generator.cpp",
    "puzzle135_challenge.cpp",
    "bitcoin_puzzle_solver.h",
    "test_kangaroo_bl_integration.cpp",
    "test_minimal_load.cpp",
    "test_puzzle135_system.cpp",
    "test_real_ec_solver.cpp",
    "test_simple_solver.cpp",
    "test_small_puzzle.cpp"
)

Write-Host "Fixing encoding issues by converting to ANSI..."

foreach ($file in $files) {
    $fullPath = Join-Path $PSScriptRoot $file
    if (Test-Path $fullPath) {
        Write-Host "Processing: $file"

        try {
            # Read content and convert to ANSI (Default system encoding)
            $content = Get-Content $fullPath -Raw -Encoding UTF8
            if ($content) {
                # Save as ANSI (Default system encoding)
                $content | Out-File -FilePath $fullPath -Encoding Default -NoNewline
                Write-Host "  - Converted to ANSI: $file"
            }
        } catch {
            Write-Host "  - Error processing $file : $($_.Exception.Message)"
        }
    } else {
        Write-Host "  - File not found: $file"
    }
}

Write-Host "Encoding fix completed!"
