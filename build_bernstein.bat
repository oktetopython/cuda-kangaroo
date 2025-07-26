@echo off
echo === 编译Bernstein预计算表生成器 ===

:: 设置编译器路径
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set MSVC_PATH=F:\visio\VC\Tools\MSVC\14.44.35207

:: 编译命令
g++ -std=c++17 -O3 ^
    -I. ^
    -DWIN64 ^
    bernstein_table_generator.cpp ^
    SECPK1/SECP256K1.cpp ^
    SECPK1/Int.cpp ^
    SECPK1/IntMod.cpp ^
    SECPK1/Point.cpp ^
    SECPK1/IntGroup.cpp ^
    SECPK1/Random.cpp ^
    -o bernstein_generator.exe

if %ERRORLEVEL% EQU 0 (
    echo === 编译成功 ===
    echo 运行: bernstein_generator.exe
) else (
    echo === 编译失败 ===
)

pause
