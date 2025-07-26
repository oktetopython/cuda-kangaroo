# 多次运行Bernstein算法并统计结果
# PowerShell脚本用于实现多次运行取平均

param(
    [int]$runs = 10,  # 默认运行10次
    [string]$testFile = "test_puzzle24.txt"
)

Write-Host "🚀 开始多次运行Bernstein算法统计分析" -ForegroundColor Green
Write-Host "运行次数: $runs" -ForegroundColor Yellow
Write-Host "测试文件: $testFile" -ForegroundColor Yellow
Write-Host "期望结果: 0xdc2a04 (14428676)" -ForegroundColor Yellow
Write-Host "=" * 60

$results = @()
$expectedValue = 0xdc2a04
$totalDifference = 0
$successCount = 0

for ($i = 1; $i -le $runs; $i++) {
    Write-Host "运行 $i/$runs..." -ForegroundColor Cyan
    
    # 运行程序并捕获输出
    $output = & .\kangaroo.exe -test 23 15 4 $testFile 2>&1
    
    # 从输出中提取结果
    $resultLine = $output | Select-String "单次运行结果: 0x([0-9a-fA-F]+)"
    if ($resultLine) {
        $hexResult = $resultLine.Matches[0].Groups[1].Value
        $decResult = [Convert]::ToInt64($hexResult, 16)
        
        $difference = [Math]::Abs($decResult - $expectedValue)
        $accuracy = (1 - ($difference / $expectedValue)) * 100
        
        $results += @{
            Run = $i
            HexResult = "0x$hexResult"
            DecResult = $decResult
            Difference = $difference
            Accuracy = $accuracy
        }
        
        $totalDifference += $difference
        $successCount++
        
        Write-Host "  结果: 0x$hexResult ($decResult)" -ForegroundColor White
        Write-Host "  差值: $difference, 精度: $($accuracy.ToString('F2'))%" -ForegroundColor $(if ($accuracy -gt 99) { "Green" } else { "Yellow" })
    } else {
        Write-Host "  ❌ 未能提取结果" -ForegroundColor Red
    }
    
    Write-Host ""
}

# 统计分析
if ($successCount -gt 0) {
    Write-Host "📊 统计分析结果" -ForegroundColor Green
    Write-Host "=" * 60
    
    # 计算平均值
    $decResults = $results | ForEach-Object { $_.DecResult }
    $avgDecResult = ($decResults | Measure-Object -Average).Average
    $avgHexResult = "0x{0:X}" -f [int64]$avgDecResult
    $avgDifference = [Math]::Abs($avgDecResult - $expectedValue)
    $avgAccuracy = (1 - ($avgDifference / $expectedValue)) * 100
    
    Write-Host "成功运行次数: $successCount/$runs" -ForegroundColor White
    Write-Host "平均结果: $avgHexResult ($([int64]$avgDecResult))" -ForegroundColor White
    Write-Host "期望结果: 0xdc2a04 ($expectedValue)" -ForegroundColor White
    Write-Host "平均差值: $([int64]$avgDifference)" -ForegroundColor White
    Write-Host "平均精度: $($avgAccuracy.ToString('F2'))%" -ForegroundColor $(if ($avgAccuracy -gt 99.5) { "Green" } else { "Yellow" })
    
    # 找出最佳结果
    $bestResult = $results | Sort-Object Difference | Select-Object -First 1
    Write-Host ""
    Write-Host "🏆 最佳单次结果:" -ForegroundColor Green
    Write-Host "  运行 $($bestResult.Run): $($bestResult.HexResult) ($($bestResult.DecResult))" -ForegroundColor White
    Write-Host "  差值: $($bestResult.Difference), 精度: $($bestResult.Accuracy.ToString('F2'))%" -ForegroundColor Green
    
    # 精度分布
    $highAccuracy = ($results | Where-Object { $_.Accuracy -gt 99 }).Count
    $veryHighAccuracy = ($results | Where-Object { $_.Accuracy -gt 99.5 }).Count
    
    Write-Host ""
    Write-Host "📈 精度分布:" -ForegroundColor Green
    Write-Host "  >99%精度: $highAccuracy/$successCount ($([int](($highAccuracy/$successCount)*100))%)" -ForegroundColor White
    Write-Host "  >99.5%精度: $veryHighAccuracy/$successCount ($([int](($veryHighAccuracy/$successCount)*100))%)" -ForegroundColor White
    
} else {
    Write-Host "❌ 没有成功的运行结果" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 多次运行统计完成！" -ForegroundColor Green
