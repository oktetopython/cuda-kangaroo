```markdown
# Kangaroo 魔改 AI-Agent 指导原则  

---

## 1  目标
把 **Kangaroo v2.1**（2018 代码）升级为 **现代全栈 GPU 方案**，  
支持 **sm_52 → sm_90 全系 NVIDIA GPU**，**Windows 10/11 + Linux 20.04/22.04**，  
并 **突破官方 125-bit 上限 → 135-bit**。

---

## 2  AI-Agent 零重复原则
- **先搜后写**：每次生成函数前，搜索 `kang_*`；相似度 ≥ 90 % 直接复用  
- **规格先行**：先写 `docs/README.md` 再写代码  
- **逐层验证**：每 200 行 diff 必须跑一次 `ctest`  
- **双平台 CI**：Windows + Linux GitHub Actions 全绿才合并

---

## 3  技术规格（写入 docs/README.md）
| 维度 | 要求 | 可验证指标 |
|---|---|---|
| **最大区间** | **≥ 135-bit** | 跑 135-bit 已知解 |
| **GPU 架构** | **sm_52 → sm_90** | `nvcc --list-gpus` |
| **OS 兼容** | **Win 10/11 + Ubuntu 20.04/22.04** | CI 双矩阵 |
| **编译器** | **CUDA 12.2 + GCC 12 / MSVC 2022** | CI 绿 |
| **内存上限** | **≤ 24 GB/卡** | `nvidia-smi` 监控 |
| **API** | **C++17** | `clang-tidy` 零警告 |

---

## 4  零重复规则（强制）
```markdown
## Kangaroo Zero-Duplicate Rule
- 任何函数先搜索 `kang_*`；相似度 ≥ 90 % 直接复用  
- 新增函数必须带 `// NEW: <原因>` 注释  
- PR 行数 ≤ 200；CI 绿灯后才可合并  
- Windows/Linux 双平台 CI 必须同时通过
```

---

## 5  现代化改造清单（逐条可落）

| 序号 | 模块 | 原版痛点 | 魔改动作 | 代码文件 |
|---|---|---|---|---|
| 1 | **数据位宽** | 256-bit 溢出 | **512-bit 全通路** | `src/kang_uint512.cu` |
| 2 | **哈希表** | 32-bit 索引 | **128-bit 分片** | `src/kang_hash128.cu` |
| 3 | **DP 掩码** | 手动 `-d` | **自适应 DP** | `src/kang_dp_auto.cu` |
| 4 | **GPU 内核** | 单 kernel | **per-SM 分块** | `src/kang_block.cu` |
| 5 | **编译链** | CUDA 10.2 | **CUDA 12.2 + 全架构** | `CMakeLists.txt` |
| 6 | **CI/CD** | 无 | **GitHub Actions 双矩阵** | `.github/workflows/ci.yml` |
| 7 | **文档** | 无 | **docs/README.md** | `docs/README.md` |

---

## 6  逐日任务（AI-Agent TODO）

| 日期 | 任务 | 交付物 | 检查 |
|---|---|---|---|
| D1 | 环境扫描 + 规格撰写 | `docs/README.md` | README 审核通过 |
| D2 | uint512_t & 哈希表 | `src/kang_uint512.cu` + `kang_hash128.cu` | 单元测试绿 |
| D3 | per-SM kernel | `src/kang_block.cu` | CI 绿 |
| D4 | 自适应 DP | `src/kang_dp_auto.cu` | CI 绿 |
| D5 | CMake 全架构 | `CMakeLists.txt` | Windows+Linux CI 绿 |
| D6 | GitHub Actions 双矩阵 | `.github/workflows/ci.yml` | 全绿 |
| D7 | 135-bit 已知解冒烟测试 | `test/135bit_known.cpp` | 结果正确 |

---

## 7  一键部署

```bash
git clone https://github.com/oktetopython/kangaroo-ext.git
cd kangaroo-ext
git checkout v3.0.0
make gpu=1 sm=52,60,70,75,80,86,89,90 all
./kangaroo -m bsgs -f puzzle135.txt -b 135 -gpu -t 0
```

---

## 8  结束确认

- [ ] `git tag v3.0.0` 包含所有魔改  
- [ ] `make test` 在 **Windows + Linux + sm_52 → sm_90** 全绿  
- [ ] 135-bit 已知解 **< 5 天** 验证通过
```