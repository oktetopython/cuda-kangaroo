# 🚀 AI Agent 快速参考卡片
## *CUDA-BSGS-Kangaroo 项目*

---

## 🔄 四步必做流程
```
1. Context7        → 收集技术资料
2. Sequential      → 结构化分析  
3. Feedback        → 用户确认
4. Memory          → 存储经验
```

---

## 🛠️ MCP服务速查

### 信息收集类
- `resolve-library-id_Context_7` - 技术文档检索
- `get-library-docs_Context_7` - 获取最佳实践
- `codebase-retrieval` - 代码搜索分析
- `git-commit-retrieval` - 历史变更查询

### 分析决策类  
- `sequentialthinking_Sequential_thinking` - 问题分解
- `models_taskmaster-ai` - 项目管理

### 交互反馈类
- `interactive_feedback_mcp-feedback-enhanced` - 用户确认
- `view_tasklist` - 任务状态查看

### 执行操作类
- `str-replace-editor` - 代码编辑
- `launch-process` - 编译构建
- `diagnostics` - 问题诊断

### 存储记录类
- `create_entities_Memory` - 知识存储
- `add_observations_Memory` - 经验记录

---

## 📊 质量标准速查

### 现代C++要求
```cpp
✅ auto ptr = std::make_unique<Type[]>(size);
❌ Type* ptr = new Type[size]; delete[] ptr;

✅ RAII自动管理
❌ 手动malloc/free

✅ 异常安全
❌ 裸指针传递
```

### 测试要求
- 编译: 零错误零警告
- 单元测试: ≥80%覆盖率  
- 集成测试: 100%关键路径
- 性能测试: 指标达标

---

## 🔒 安全边界速查

### 高风险 (必须确认)
- 删除文件/目录
- 修改CMakeLists.txt
- 系统级命令
- GPU内核修改

### 中风险 (建议确认)  
- 大规模重构
- 版本升级
- 性能优化
- 新算法实现

### 低风险 (可自动)
- 代码格式化
- 注释添加
- 文档更新
- 单元测试

---

## ⚡ 常用命令

### 编译验证
```bash
# 清理构建
cmake --build . --target clean

# 编译项目  
cmake --build . --config Release --target kangaroo

# 功能验证
.\Release\kangaroo.exe -v
```

### 问题诊断
```bash
# 查看编译错误
cmake --build . 2>&1 | grep -i error

# 内存检查 (Linux)
valgrind --tool=memcheck ./kangaroo

# 性能分析
perf record ./kangaroo
```

---

## 🎯 项目KPI

### 安全指标
- 漏洞修复率: 100%
- 新漏洞引入: 0个
- 安全测试覆盖: 100%

### 质量指标  
- 编译成功率: 100%
- 功能回归: 0个
- 性能保持: ±5%

### 效率指标
- 修复时间: <2小时/漏洞
- 代码重用: ≥80%
- 自动化: ≥70%

---

## 🚨 紧急处理

### 编译失败
1. 检查最近修改
2. git diff查看变更
3. 逐步回滚
4. Sequential分析原因

### 性能下降
1. 性能分析定位
2. 对比历史数据  
3. 检查低效算法
4. 优化关键路径

### 安全问题
1. 停止相关功能
2. 评估影响范围
3. 制定修复方案
4. 实施并验证

---

## 📋 检查清单

### 代码质量
- [ ] new/delete → 智能指针
- [ ] malloc/free → 现代分配
- [ ] 异常安全实现
- [ ] 内存泄漏检查
- [ ] 编译零警告
- [ ] 测试覆盖≥80%

### 安全检查
- [ ] 输入验证完整
- [ ] 缓冲区溢出防护
- [ ] 整数溢出检查  
- [ ] 加密实现安全
- [ ] 内存清理彻底
- [ ] 错误信息安全

---

## 🔗 相关文件

- `ai-agent-rules-v2.0.md` - 完整规则框架
- `ModernMemoryManager.h` - 内存管理基础设施
- `extensions.json` - VSCode扩展配置

---

**快速参考 v2.0** | **更新**: 2025-01-12 | **项目**: CUDA-BSGS-Kangaroo
