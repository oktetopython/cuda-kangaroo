@echo off
title Bitcoin Puzzle 135 - Specific Component Builder (Windows)
color 0B

if "%1"=="" goto usage

echo.
echo 🔨 BITCOIN PUZZLE 135 - SPECIFIC COMPONENT BUILDER
echo ==================================================
echo.

REM Check if build directory exists
if not exist "build" (
    echo 📁 Creating build directory...
    mkdir build
)

cd build

REM Configure CMake if not already done
if not exist "CMakeCache.txt" (
    echo 🔧 Configuring CMake...
    cmake .. -DCMAKE_BUILD_TYPE=Release
    if %ERRORLEVEL% neq 0 (
        echo ❌ CMake configuration failed!
        pause
        exit /b 1
    )
)

REM Build specific component based on parameter
if /i "%1"=="kangaroo" goto build_kangaroo
if /i "%1"=="puzzle135" goto build_puzzle135
if /i "%1"=="generator" goto build_generator
if /i "%1"=="tests" goto build_tests
if /i "%1"=="benchmark" goto build_benchmark
if /i "%1"=="all" goto build_all
goto invalid_option

:build_kangaroo
echo 🚀 Building Main Kangaroo Program...
echo ===================================
cmake --build . --config Release --target kangaroo
if %ERRORLEVEL% neq 0 (
    echo ❌ Kangaroo build failed!
    pause
    exit /b 1
)
echo ✅ Kangaroo built successfully!
echo 📍 Location: .\build\Release\kangaroo.exe
goto end

:build_puzzle135
echo 🎯 Building Bitcoin Puzzle 135 Components...
echo ============================================
echo 🔨 Building Puzzle135 Challenge...
cmake --build . --config Release --target puzzle135_challenge
if %ERRORLEVEL% neq 0 (
    echo ❌ Puzzle135 Challenge build failed!
    pause
    exit /b 1
)
echo ✅ Puzzle135 Challenge built successfully!

echo 🔨 Building Puzzle135 Generator...
cmake --build . --config Release --target puzzle135_bl_generator
if %ERRORLEVEL% neq 0 (
    echo ❌ Puzzle135 Generator build failed!
    pause
    exit /b 1
)
echo ✅ Puzzle135 Generator built successfully!

echo 📍 Locations:
echo   ├── .\build\Release\puzzle135_challenge.exe
echo   └── .\build\Release\puzzle135_bl_generator.exe
goto end

:build_generator
echo 📊 Building Table Generators...
echo ===============================
echo 🔨 Building Puzzle135 Generator...
cmake --build . --config Release --target puzzle135_bl_generator
if %ERRORLEVEL% neq 0 (
    echo ❌ Puzzle135 Generator build failed!
    pause
    exit /b 1
)
echo ✅ Puzzle135 Generator built successfully!

echo 🔨 Building Real EC Generator...
cmake --build . --config Release --target generate_bl_real_ec_table
if %ERRORLEVEL% neq 0 (
    echo ❌ Real EC Generator build failed!
    pause
    exit /b 1
)
echo ✅ Real EC Generator built successfully!

echo 📍 Locations:
echo   ├── .\build\Release\puzzle135_bl_generator.exe
echo   └── .\build\Release\generate_bl_real_ec_table.exe
goto end

:build_tests
echo 🧪 Building Test Programs...
echo ============================
echo 🔨 Building System Test...
cmake --build . --config Release --target test_puzzle135_system
if %ERRORLEVEL% neq 0 (
    echo ❌ System Test build failed!
    pause
    exit /b 1
)
echo ✅ System Test built successfully!

echo 🔨 Building Small Puzzle Test...
cmake --build . --config Release --target test_small_puzzle
if %ERRORLEVEL% neq 0 (
    echo ❌ Small Puzzle Test build failed!
    pause
    exit /b 1
)
echo ✅ Small Puzzle Test built successfully!

echo 📍 Locations:
echo   ├── .\build\Release\test_puzzle135_system.exe
echo   └── .\build\Release\test_small_puzzle.exe
goto end

:build_benchmark
echo 📈 Building Performance Benchmark...
echo ====================================
cmake --build . --config Release --target performance_benchmark
if %ERRORLEVEL% neq 0 (
    echo ❌ Performance Benchmark build failed!
    pause
    exit /b 1
)
echo ✅ Performance Benchmark built successfully!
echo 📍 Location: .\build\Release\performance_benchmark.exe
goto end

:build_all
echo 🏗️ Building All Components...
echo =============================
call ..\build_all_windows.bat
goto end

:usage
echo.
echo 📖 USAGE: build_specific_windows.bat [component]
echo.
echo 🎯 Available Components:
echo   kangaroo    - Main GPU-accelerated Kangaroo solver
echo   puzzle135   - Bitcoin Puzzle 135 challenge components
echo   generator   - Precompute table generators
echo   tests       - System verification and test programs
echo   benchmark   - Performance benchmark program
echo   all         - Build all components
echo.
echo 💡 Examples:
echo   build_specific_windows.bat kangaroo
echo   build_specific_windows.bat puzzle135
echo   build_specific_windows.bat tests
echo.
pause
exit /b 1

:invalid_option
echo ❌ Invalid option: %1
echo.
echo 🎯 Valid options: kangaroo, puzzle135, generator, tests, benchmark, all
echo.
pause
exit /b 1

:end
cd ..
echo.
echo 🎉 Build Complete!
echo =================
echo.
echo 🚀 Quick Commands:
echo   Test system:    .\build\Release\test_puzzle135_system.exe
echo   Generate table: .\build\Release\puzzle135_bl_generator.exe 40 8192 table.bin
echo   Start challenge:.\build\Release\puzzle135_challenge.exe table.bin 1000000000
echo.
echo 📚 For detailed usage, see COMPLETE_USER_GUIDE.md
echo.
pause
