@echo off
title Bitcoin Puzzle 135 System - Windows Build Script
color 0A

echo.
echo 🚀 BITCOIN PUZZLE 135 SYSTEM - WINDOWS BUILD SCRIPT
echo ====================================================
echo.
echo 🔧 Building all components for Windows...
echo.

REM Check if build directory exists
if not exist "build" (
    echo 📁 Creating build directory...
    mkdir build
)

cd build

echo.
echo 🔨 Step 1: Configuring CMake...
echo ================================
cmake .. -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
    echo ❌ CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo 🔨 Step 2: Building Main Kangaroo Program...
echo =============================================
cmake --build . --config Release --target kangaroo
if %ERRORLEVEL% neq 0 (
    echo ❌ Kangaroo build failed!
    pause
    exit /b 1
)
echo ✅ Kangaroo built successfully!

echo.
echo 🔨 Step 3: Building Bitcoin Puzzle 135 Challenge...
echo ===================================================
cmake --build . --config Release --target puzzle135_challenge
if %ERRORLEVEL% neq 0 (
    echo ❌ Puzzle135 Challenge build failed!
    pause
    exit /b 1
)
echo ✅ Puzzle135 Challenge built successfully!

echo.
echo 🔨 Step 4: Building Puzzle135 Table Generator...
echo ================================================
cmake --build . --config Release --target puzzle135_bl_generator
if %ERRORLEVEL% neq 0 (
    echo ❌ Puzzle135 Generator build failed!
    pause
    exit /b 1
)
echo ✅ Puzzle135 Generator built successfully!

echo.
echo 🔨 Step 5: Building Real EC Table Generator...
echo ==============================================
cmake --build . --config Release --target generate_bl_real_ec_table
if %ERRORLEVEL% neq 0 (
    echo ❌ Real EC Generator build failed!
    pause
    exit /b 1
)
echo ✅ Real EC Generator built successfully!

echo.
echo 🔨 Step 6: Building System Tests...
echo ===================================
cmake --build . --config Release --target test_puzzle135_system
if %ERRORLEVEL% neq 0 (
    echo ❌ System Test build failed!
    pause
    exit /b 1
)
echo ✅ System Test built successfully!

cmake --build . --config Release --target test_small_puzzle
if %ERRORLEVEL% neq 0 (
    echo ❌ Small Puzzle Test build failed!
    pause
    exit /b 1
)
echo ✅ Small Puzzle Test built successfully!

echo.
echo 🔨 Step 7: Building Performance Benchmark...
echo ============================================
cmake --build . --config Release --target performance_benchmark
if %ERRORLEVEL% neq 0 (
    echo ❌ Performance Benchmark build failed!
    pause
    exit /b 1
)
echo ✅ Performance Benchmark built successfully!

echo.
echo 🎉 BUILD COMPLETE! All components built successfully!
echo =====================================================
echo.
echo 📦 Built Programs:
echo ├── kangaroo.exe                   - Main GPU-accelerated solver
echo ├── puzzle135_challenge.exe        - Bitcoin Puzzle 135 challenge
echo ├── puzzle135_bl_generator.exe     - Puzzle 135 table generator
echo ├── generate_bl_real_ec_table.exe  - Real EC table generator
echo ├── test_puzzle135_system.exe      - System verification
echo ├── test_small_puzzle.exe          - Small-scale algorithm test
echo └── performance_benchmark.exe      - Performance benchmark
echo.
echo 🚀 Quick Start:
echo 1. Run system test: .\Release\test_puzzle135_system.exe
echo 2. Generate table:  .\Release\puzzle135_bl_generator.exe 40 8192 table.bin
echo 3. Start challenge: .\Release\puzzle135_challenge.exe table.bin 1000000000
echo.
echo 📚 For detailed usage, see COMPLETE_USER_GUIDE.md
echo.

cd ..

echo 🔧 Creating convenience scripts...
echo.

REM Create quick test script
echo @echo off > quick_test.bat
echo echo 🧪 Running Quick System Test... >> quick_test.bat
echo .\build\Release\test_puzzle135_system.exe >> quick_test.bat
echo pause >> quick_test.bat

REM Create table generation script
echo @echo off > generate_table.bat
echo echo 📊 Generating Precompute Table... >> generate_table.bat
echo echo Usage: generate_table.bat [L] [T] [output_file] >> generate_table.bat
echo echo Example: generate_table.bat 40 8192 puzzle135_table.bin >> generate_table.bat
echo if "%%1"=="" goto usage >> generate_table.bat
echo if "%%2"=="" goto usage >> generate_table.bat
echo if "%%3"=="" goto usage >> generate_table.bat
echo .\build\Release\puzzle135_bl_generator.exe %%1 %%2 %%3 >> generate_table.bat
echo goto end >> generate_table.bat
echo :usage >> generate_table.bat
echo echo Please provide: L T output_file >> generate_table.bat
echo echo Example: generate_table.bat 40 8192 puzzle135_table.bin >> generate_table.bat
echo :end >> generate_table.bat
echo pause >> generate_table.bat

REM Create challenge script
echo @echo off > start_challenge.bat
echo echo 🎯 Starting Bitcoin Puzzle 135 Challenge... >> start_challenge.bat
echo echo Usage: start_challenge.bat [table_file] [max_steps] >> start_challenge.bat
echo echo Example: start_challenge.bat puzzle135_table.bin 1000000000 >> start_challenge.bat
echo if "%%1"=="" goto usage >> start_challenge.bat
echo if "%%2"=="" goto usage >> start_challenge.bat
echo .\build\Release\puzzle135_challenge.exe %%1 %%2 >> start_challenge.bat
echo goto end >> start_challenge.bat
echo :usage >> start_challenge.bat
echo echo Please provide: table_file max_steps >> start_challenge.bat
echo echo Example: start_challenge.bat puzzle135_table.bin 1000000000 >> start_challenge.bat
echo :end >> start_challenge.bat
echo pause >> start_challenge.bat

echo ✅ Convenience scripts created:
echo ├── quick_test.bat      - Quick system verification
echo ├── generate_table.bat  - Generate precompute tables
echo └── start_challenge.bat - Start Bitcoin Puzzle challenge
echo.

echo 🎉 WINDOWS BUILD COMPLETE!
echo ==========================
echo All programs and scripts are ready to use!
echo.
pause
