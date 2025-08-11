# 🚀 CUDA-BSGS-Kangaroo 快速参考卡

## 📦 程序清单

| 程序 | 功能 | 用法示例 |
|------|------|----------|
| **kangaroo.exe** | 主求解器 | `kangaroo.exe -t 0 -gpu -d 7 -g 234,128 config.txt` |
| **generate_bl_real_ec_table.exe** | 预计算表生成 | `generate_bl_real_ec_table.exe 20 1024` |
| **performance_benchmark.exe** | 性能测试 | `performance_benchmark.exe` |
| **bl_algorithm_verification.exe** | 算法验证 | `bl_algorithm_verification.exe` |
| **test_simple_solver.exe** | 简单测试 | `test_simple_solver.exe` |
| **puzzle135_challenge.exe** | Puzzle 135专用 | `puzzle135_challenge.exe` |

---

## ⚡ 快速开始

### 1. 编译
```bash
# Windows
cd build && cmake --build . --config Release

# Linux  
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

### 2. 基本测试
```bash
# 算法验证
build\Release\bl_algorithm_verification.exe

# 性能测试
build\Release\performance_benchmark.exe

# 生成测试表
build\Release\generate_bl_real_ec_table.exe 20 1024
```

### 3. 运行主程序
```bash
# 创建配置文件
echo 8000000000000000 > test_config.txt
echo 10000000000000000 >> test_config.txt  
echo 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 >> test_config.txt

# 运行
build\Release\kangaroo.exe -t 0 -gpu -d 7 -g 128,64 test_config.txt
```

---

## 🔧 常用参数

### kangaroo.exe 核心参数
```bash
-t 0              # CPU线程数（0=仅GPU）
-gpu              # 启用GPU
-d 6|7|8          # DP大小（6=快但易overflow，8=慢但稳定）
-g 128,64         # GPU网格（保守）
-g 234,128        # GPU网格（平衡）
-g 512,256        # GPU网格（激进）
-w work.dat       # 保存工作文件
-wi 300           # 每5分钟保存
```

### 预计算表参数
```bash
generate_bl_real_ec_table.exe L_bits T
# L_bits: 搜索区间（20=小，30=中，40=大）
# T: 表大小（1024=小，4096=中，8192=大）
```

---

## 📊 性能指标

### 正常运行状态
```
[98.52 MK/s][GPU 98.52 MK/s][Count 2^27.87][Dead 0][02s][2.3/5.7MB]
```

### 关键指标
- **MK/s**: 性能指标，目标 > 60
- **Dead**: 死亡数，应保持 < 100
- **Memory**: 内存使用，监控增长

---

## 🚨 故障排除

### GPU Buffer Overflow
**症状**: `WARNING: distinguished points lost`
**解决**:
```bash
# 方案1: 增加DP大小
-d 7  # 从6改为7

# 方案2: 减少GPU线程  
-g 128,64  # 减少网格大小
```

### 性能下降
**症状**: MK/s降到0.00
**解决**:
- 重启程序
- 检查内存使用
- 调整参数

### 编译错误
**症状**: CUDA相关错误
**解决**:
- 安装CUDA Toolkit 12.0+
- 检查GPU计算能力 ≥ 5.2

---

## 🎯 Bitcoin Puzzle 135

### 目标信息
- **地址**: `16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v`
- **范围**: `4000000000000000000000000000000000` - `7fffffffffffffffffffffffffffffffff`
- **奖励**: ~32 BTC

### 推荐配置
```bash
# 配置文件 puzzle135.txt
4000000000000000000000000000000000
7fffffffffffffffffffffffffffffffff  
02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16

# 运行命令
kangaroo.exe -t 0 -gpu -d 8 -g 234,128 -w puzzle135.dat -wi 300 puzzle135.txt
```

---

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| **UNIFIED_USAGE_GUIDE.md** | 完整使用说明 |
| **README.md** | 项目概述 |
| **COMPLETE_USER_GUIDE.md** | 详细用户指南 |
| **PARAMETER_REFERENCE.md** | 参数参考 |
| **QUICK_REFERENCE.md** | 本快速参考 |

---

## ⚠️ 重要提醒

1. **真实挑战**: Bitcoin Puzzle是真实的加密货币挑战
2. **安全第一**: 发现私钥请立即安全保存
3. **资源消耗**: 大规模计算需要大量GPU资源
4. **备份数据**: 运行前请备份重要数据
5. **法律合规**: 请遵守当地法律法规

---

**🎯 准备好了吗？开始你的ECDLP挑战！**
