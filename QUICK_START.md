# Bitcoin Puzzle 135 快速启动指南

## 🚀 一键启动 (已编译环境)

### 1. 检查环境
```bash
# 检查GPU
nvidia-smi

# 检查程序
./kangaroo.exe -h
```

### 2. 生成任务配置
```bash
# 生成32个子区间 (约2^125每个)
python puzzle135_production_prefilter.py 55 32

# 生成GPU任务配置
python scripts/make_kangaroo_jobs.py puzzle135_subintervals.json
```

### 3. 启动求解
```bash
# 启动监控 (新终端)
python scripts/monitor_puzzle135.py

# 启动单个测试任务
powershell -ExecutionPolicy Bypass -File start_test_job.ps1

# 或启动全部32个任务
start_puzzle135.bat
```

## 📊 监控面板说明

```
任务名称                  GPU  状态        进度    速度        ETA
puzzle135_interval_0     GPU0 running     0.1%    350.2 MK/s  1000y
puzzle135_interval_1     GPU0 not_started 0.0%    -           -

📈 总体统计:
   总任务数: 32
   平均进度: 0.1%
   总速度: 350.2 MK/s    ← 这里显示实际速度
   活跃任务: 1
   running: 1
   not_started: 31
```

## ⚡ 性能参考

| GPU | 推荐参数 | 预期速度 | 内存使用 |
|-----|----------|----------|----------|
| RTX 2080 Ti | `-d 11 -g 64,64` | 350-450 MK/s | 1.3GB |
| RTX 3080 | `-d 11 -g 68,64` | 500-650 MK/s | 1.5GB |
| RTX 4090 | `-d 11 -g 128,64` | 800-1200 MK/s | 2.0GB |

## 🔧 快速故障排除

### 问题: 监控显示0速度
**解决**: 检查日志文件
```bash
# 查看日志
tail -f logs/puzzle135_interval_0.log

# 如果日志为空，手动启动任务
build/Release/kangaroo.exe -t 0 -gpu -d 11 -g 64,64 configs/puzzle135_interval_0.txt
```

### 问题: GPU内存不足
**解决**: 降低参数
```bash
# 原参数: -d 11 -g 64,64
# 改为: -d 12 -g 32,32  (减少内存使用)
```

### 问题: 编译失败
**解决**: 重新编译
```bash
# Windows
cmake -B build -S .
cmake --build build --config Release
copy build\Release\kangaroo.exe .

# Linux  
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cp build/kangaroo .
```

## 📁 重要文件位置

```
项目根目录/
├── kangaroo.exe                    # 主程序
├── puzzle135_subintervals.json     # 子区间配置
├── jobs.json                       # GPU任务配置
├── configs/                        # 任务配置文件
│   ├── puzzle135_interval_0.txt
│   └── ...
├── logs/                           # 日志文件
│   ├── puzzle135_interval_0.log
│   └── ...
├── work/                           # 工作状态文件
│   ├── puzzle135_interval_0.work
│   └── ...
└── results/                        # 结果文件
    ├── puzzle135_interval_0.result
    └── ...
```

## 🎯 成功标志

✅ **正常运行**:
- 监控显示 `running` 状态
- 速度显示 `350+ MK/s`
- 日志显示进度更新
- GPU使用率 > 90%

❌ **异常情况**:
- 速度显示 `0.0 MK/s`
- 状态显示 `error`
- 日志文件为空
- GPU使用率 < 50%

## 🏆 私钥发现

当找到私钥时，程序会在日志中显示：
```
Private Key Found!
Key: 0x[64位十六进制私钥]
```

同时结果文件会包含完整信息。

---

**提示**: 如遇问题，请查看完整的 `COMPLETE_USAGE_GUIDE.md` 获取详细说明。
