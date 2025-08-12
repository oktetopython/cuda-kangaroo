# 🏗️ AI Agent 工作规则框架 v2.0
## *CUDA-BSGS-Kangaroo 项目指导规则*

---

## 📋 框架概述

本框架为CUDA-BSGS-Kangaroo项目建立了完善的AI Agent工作标准，确保开发过程中的一致性和高质量执行。

### 🎯 核心理念
```
Plan Before Action → Execute with Quality → Verify and Learn → Document and Share
```

---

## 🔄 I. 核心工作流程 (强制执行)

### 1.1 四步必做流程 (不可跳过)

| 步骤 | MCP服务 | 目标 | 输出 |
|------|---------|------|------|
| **Step 1: Context7** | `resolve-library-id_Context_7`<br>`get-library-docs_Context_7` | 收集最新技术资料和最佳实践 | 技术文档、代码示例、最佳实践 |
| **Step 2: Sequential Thinking** | `sequentialthinking_Sequential_thinking` | 结构化分析和问题分解 | 思维链、解决方案假设、验证结果 |
| **Step 3: Interactive Feedback** | `interactive_feedback_mcp-feedback-enhanced` | 获取用户确认和反馈 | 用户选择、修正建议、执行许可 |
| **Step 4: Memory** | `create_entities_Memory`<br>`add_observations_Memory` | 存储关键信息和经验 | 知识图谱更新、经验积累 |

### 1.2 扩展工作流程

#### 阶段A: 需求分析与规划
```
用户请求 → Context7技术调研 → Sequential Thinking问题分析 → 
代码检索现状理解 → TaskMaster AI任务规划 → Interactive Feedback方案确认 → Memory规划存储
```

#### 阶段B: 实施与验证
```
任务执行 → 文件操作代码修改 → 进程管理编译测试 → 诊断服务问题检测 → 
测试验证 → Memory成功记录 (如失败则Sequential Thinking问题分析后重试)
```

---

## 🛠️ II. MCP服务使用标准

### 2.1 服务分类与使用原则

#### 🔍 信息收集类
- **Context7**: 技术调研、最佳实践查找 - 必须获取最新文档，验证信息准确性
- **codebase-retrieval**: 代码理解、现状分析 - 全面搜索相关代码，理解架构模式
- **git-commit-retrieval**: 历史分析、变更理解 - 查找相似修改，学习历史经验
- **web-search/web-fetch**: 补充信息、验证资料 - 交叉验证信息，确保时效性

#### 🧠 分析决策类
- **Sequential Thinking**: 复杂问题分解、方案设计 - 逻辑清晰，步骤完整，可验证
- **TaskMaster AI**: 项目管理、任务规划 - 任务粒度适中，依赖关系明确

#### 💬 交互反馈类
- **Interactive Feedback**: 方案确认、选择决策 - 选项清晰，风险说明，等待确认
- **任务管理服务**: 进度跟踪、状态更新 - 及时更新，状态准确

#### ⚡ 执行操作类
- **文件操作**: 代码编辑、文档创建 - 增量修改，保持兼容性
- **进程管理**: 编译构建、测试执行 - 错误处理，结果验证
- **浏览器自动化**: 网页测试、UI验证 - 用户场景覆盖，异常处理

#### 💾 存储记录类
- **Memory**: 知识存储、经验积累 - 结构化存储，关联建立
- **记忆服务**: 重要信息记录 - 简洁准确，便于检索

### 2.2 服务组合模式

#### 模式1: 技术调研组合
```
Context7 → Sequential Thinking → Interactive Feedback → Memory
```

#### 模式2: 代码修改组合
```
codebase-retrieval → Sequential Thinking → 文件操作 → 进程管理 → 诊断服务 → Memory
```

#### 模式3: 项目管理组合
```
TaskMaster AI → 任务管理服务 → Interactive Feedback → Memory
```

---

## 📊 III. 质量标准体系

### 3.1 代码质量标准 (CUDA-BSGS-Kangaroo专用)

#### 现代C++标准 (强制要求)
```cpp
// ✅ 推荐：现代C++ RAII
auto resource = std::make_unique<Resource>();
auto array = std::make_unique<Int[]>(size);

// ❌ 禁止：C风格内存管理
Resource* resource = new Resource();
Int* array = new Int[size];
delete resource;
delete[] array;
```

#### 通用编程原则
- **RAII**: 资源自动管理，异常安全 - 编译测试，内存检查
- **DRY**: 避免代码重复，提取公共逻辑 - 代码审查，重复度检测
- **SOLID**: 单一职责，开闭原则等 - 架构审查，设计模式检查
- **类型安全**: 强类型，编译时检查 - 编译器警告，静态分析

### 3.2 测试和验证要求

#### 测试金字塔
```
         /\
        /E2E\      ← 30% (用户流程测试)
       /------\
      /集成测试 \   ← 40% (模块间交互)
     /----------\
    /  单元测试  \ ← 30% (函数级测试)
   /--------------\
```

#### 质量门禁
- **编译**: 零错误，零警告
- **静态分析**: 规范符合率100%
- **单元测试**: 覆盖率≥80%
- **集成测试**: 关键路径100%
- **性能测试**: 指标达标

### 3.3 错误处理和回滚机制

#### 错误处理层次
```cpp
// Level 1: 预防性检查
if (!input.isValid()) {
    throw std::invalid_argument("Invalid input");
}

// Level 2: 异常安全保证
try {
    auto result = riskyOperation();
    return result;
} catch (const std::exception& e) {
    logger.error("Operation failed: {}", e.what());
    throw; // 重新抛出
}

// Level 3: 系统级恢复 (RAII)
class TransactionGuard {
    ~TransactionGuard() {
        if (!committed_) rollback();
    }
};
```

---

## 🔒 IV. 安全操作边界

### 4.1 需要用户确认的操作

#### 高风险操作 (必须确认)
- 删除文件或目录
- 修改关键配置文件 (CMakeLists.txt, .vcxproj)
- 执行系统级命令
- 修改GPU内核代码
- 更改加密算法实现

#### 中风险操作 (建议确认)
- 大规模代码重构
- CUDA版本升级
- 性能优化修改
- 新算法实现

#### 低风险操作 (可自动执行)
- 代码格式化
- 注释添加
- 文档更新
- 单元测试编写

### 4.2 兼容性要求

#### API兼容性原则
```cpp
// ✅ 向后兼容的扩展
class GPUEngineV2 : public GPUEngine {
public:
    void newOptimization();
    void existingMethod() override;
    void existingMethod(NewParameter param); // 重载
};

// ❌ 破坏性变更
class GPUEngineV2 {
    // 删除了GPUEngine的方法 - 破坏兼容性
};
```

### 4.3 性能考虑原则

#### 性能优化层次
```
1. 算法优化 (最高优先级) - BSGS算法改进
   ↓
2. 数据结构优化 - 哈希表、内存布局
   ↓
3. CUDA优化 - 内核优化、内存合并
   ↓
4. 硬件优化 (最低优先级) - GPU架构特定
```

#### 性能监控指标
- **响应时间**: 密钥搜索 <10s (32位范围)
- **吞吐量**: >1M keys/sec
- **GPU利用率**: >80%
- **内存效率**: <4GB GPU内存

---

## 📈 V. 项目特定工作流程

### 5.1 CUDA-BSGS-Kangaroo 安全修复流程

#### 标准安全修复流程
```
1. 安全问题识别 → Context7安全最佳实践收集
2. Sequential Thinking漏洞分析 → 影响范围评估
3. codebase-retrieval相关代码搜索 → 修复方案设计
4. Interactive Feedback方案确认 → 用户批准
5. 文件操作增量修改 → 进程管理编译测试
6. 安全测试验证 → Memory经验记录
```

#### 安全修复质量检查清单
- [ ] 漏洞根本原因分析完成
- [ ] 修复方案不引入新漏洞
- [ ] 保持API向后兼容性
- [ ] 性能影响评估通过
- [ ] 安全测试覆盖完整
- [ ] 文档更新同步

### 5.2 代码现代化流程

#### C++现代化标准流程
```
1. 识别C风格代码 → Context7现代C++最佳实践
2. Sequential Thinking现代化方案 → RAII设计
3. 创建现代化基础设施 → ModernMemoryManager.h
4. 增量替换传统代码 → 智能指针化
5. 编译测试验证 → 功能回归测试
6. Memory现代化经验记录
```

---

## 🎯 VI. 成功指标和持续改进

### 6.1 项目KPI指标

#### 安全指标
- **漏洞修复率**: 100%
- **新漏洞引入**: 0个
- **安全测试覆盖**: 100%关键功能

#### 质量指标
- **编译成功率**: 100%
- **功能回归**: 0个
- **性能保持**: ±5%范围内

#### 效率指标
- **修复平均时间**: <2小时/漏洞
- **代码重用率**: ≥80%
- **自动化程度**: ≥70%

### 6.2 持续改进机制

#### 改进触发条件
- 安全漏洞重复出现
- 性能指标下降
- 编译失败率上升
- 用户反馈问题

#### 改进实施流程
1. **问题识别**: 指标监控和反馈收集
2. **根因分析**: Sequential Thinking深入分析
3. **方案设计**: 基于最佳实践制定改进方案
4. **试点验证**: 小范围验证效果
5. **全面推广**: 更新规则和流程
6. **效果评估**: 持续监控改进效果

---

## 📚 VII. 快速参考

### 7.1 常用命令速查

#### 编译验证
```bash
# 清理构建
cmake --build . --target clean

# 编译项目
cmake --build . --config Release --target kangaroo

# 功能验证
.\Release\kangaroo.exe -v
```

#### 代码质量检查
```bash
# 静态分析 (如果配置)
cppcheck --enable=all --std=c++17 .

# 内存检查 (Debug模式)
valgrind --tool=memcheck ./kangaroo
```

### 7.2 紧急情况处理

#### 编译失败
1. 检查最近修改的文件
2. 使用git diff查看变更
3. 逐步回滚到最后工作版本
4. Sequential Thinking分析失败原因

#### 性能下降
1. 使用性能分析工具定位瓶颈
2. 对比历史性能数据
3. 检查是否引入低效算法
4. 优化关键路径代码

#### 安全问题
1. 立即停止相关功能
2. 评估影响范围和严重程度
3. 制定紧急修复方案
4. 实施修复并验证

---

---

## 📋 VIII. 附录：详细实施指南

### 8.1 MCP服务配置模板

#### .ai-agent-rules/mcp-config.json
```json
{
  "framework": {
    "version": "2.0",
    "mandatory_steps": ["context7", "sequential_thinking", "interactive_feedback", "memory"],
    "quality_gates": ["compile", "test", "review"]
  },
  "mcp_services": {
    "context7": {
      "enabled": true,
      "timeout": 300,
      "max_tokens": 10000
    },
    "sequential_thinking": {
      "enabled": true,
      "max_thoughts": 10,
      "require_verification": true
    },
    "interactive_feedback": {
      "enabled": true,
      "timeout": 600,
      "require_confirmation": ["high_risk", "medium_risk"]
    },
    "memory": {
      "enabled": true,
      "auto_save": true,
      "knowledge_graph": true
    }
  }
}
```

### 8.2 代码模板和示例

#### 现代C++内存管理模板
```cpp
// 替换前：C风格内存管理
TH_PARAM* params = (TH_PARAM*)malloc(sizeof(TH_PARAM) * count);
// ... 使用 ...
free(params);

// 替换后：现代C++ RAII
auto params = std::make_unique<TH_PARAM[]>(count);
// 自动清理，无需手动释放
```

#### GPU内存RAII封装示例
```cpp
class CudaMemoryRAII {
private:
    void* ptr_;
    size_t size_;
public:
    CudaMemoryRAII(size_t size) : size_(size) {
        cudaMalloc(&ptr_, size);
    }
    ~CudaMemoryRAII() {
        if (ptr_) cudaFree(ptr_);
    }
    void* get() const { return ptr_; }
};
```

### 8.3 故障排除指南

#### 常见问题解决方案

**问题1: 编译错误 - 智能指针类型不匹配**
```
错误: cannot convert 'std::unique_ptr<Int[]>' to 'Int*'
解决: 使用 .get() 方法获取原始指针
示例: grp->Set(dx.get());
```

**问题2: GPU内存管理错误**
```
错误: CUDA memory access violation
解决: 检查GPU内存分配和释放顺序
示例: 确保在GPU操作完成后再释放内存
```

**问题3: 性能下降**
```
问题: 现代化后性能下降
解决: 检查是否引入不必要的拷贝操作
示例: 使用移动语义和引用传递
```

### 8.4 项目检查清单

#### 代码质量检查清单
- [ ] 所有new/delete替换为智能指针
- [ ] 所有malloc/free替换为现代分配
- [ ] 异常安全保证实现
- [ ] 内存泄漏检查通过
- [ ] 编译零警告
- [ ] 单元测试覆盖≥80%
- [ ] 性能回归测试通过
- [ ] 文档更新同步

#### 安全检查清单
- [ ] 输入验证完整
- [ ] 缓冲区溢出防护
- [ ] 整数溢出检查
- [ ] 加密实现安全
- [ ] 随机数生成安全
- [ ] 内存清理彻底
- [ ] 错误信息不泄露敏感信息

---

**版本**: v2.0
**更新日期**: 2025-01-12
**适用项目**: CUDA-BSGS-Kangaroo
**维护者**: AI Agent Framework Team

---

## 🔗 相关文档链接

- [ModernMemoryManager.h](../ModernMemoryManager.h) - 现代内存管理基础设施
- [项目构建指南](../README.md) - 项目编译和运行说明
- [安全修复记录](../SECURITY.md) - 历史安全问题和修复记录
- [性能优化指南](../PERFORMANCE.md) - 性能调优最佳实践

---

*本规则框架基于软件工程最佳实践和AI Agent工作经验制定，旨在确保CUDA-BSGS-Kangaroo项目的高质量开发和维护。*
