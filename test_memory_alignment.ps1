# 测试内存对齐修复的PowerShell脚本

Write-Host "=== Kangaroo Memory Alignment Fix Test ===" -ForegroundColor Green

# 设置CUDA调试环境
$env:CUDA_LAUNCH_BLOCKING = "1"
$env:CUDA_DEVICE_DEBUG = "1"

# 编译项目
Write-Host "Building project..." -ForegroundColor Yellow
cmake --build build --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Build successful!" -ForegroundColor Green

# 运行测试
Write-Host "Running memory alignment test..." -ForegroundColor Yellow
$output = & .\build\Release\kangaroo.exe -t 0 -gpu in.txt 2>&1

# 检查关键输出
$alignmentFixed = $output | Select-String "Memory aligned to.*bytes"
$kernelLaunched = $output | Select-String "Per-SM kernel.*completed"
$noAlignmentErrors = -not ($output | Select-String "misaligned address")

Write-Host "`n=== Test Results ===" -ForegroundColor Cyan

if ($alignmentFixed) {
    Write-Host "✅ Memory alignment fix applied" -ForegroundColor Green
    $alignmentFixed | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
} else {
    Write-Host "❌ Memory alignment fix not detected" -ForegroundColor Red
}

if ($kernelLaunched) {
    Write-Host "✅ Per-SM kernel executed successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Per-SM kernel execution failed" -ForegroundColor Red
}

if ($noAlignmentErrors) {
    Write-Host "✅ No alignment errors detected" -ForegroundColor Green
} else {
    Write-Host "❌ Alignment errors still present" -ForegroundColor Red
    $output | Select-String "misaligned address" | ForEach-Object { 
        Write-Host "   $_" -ForegroundColor Red 
    }
}

# 显示完整输出（前50行）
Write-Host "`n=== Full Output (first 50 lines) ===" -ForegroundColor Cyan
$output | Select-Object -First 50 | ForEach-Object { Write-Host $_ -ForegroundColor Gray }

# 总结
if ($alignmentFixed -and $kernelLaunched -and $noAlignmentErrors) {
    Write-Host "`n🎉 Memory alignment fix successful!" -ForegroundColor Green
    Write-Host "Per-SM kernel is now working correctly." -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Memory alignment fix needs further investigation." -ForegroundColor Yellow
    Write-Host "Check the output above for specific issues." -ForegroundColor Yellow
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Green