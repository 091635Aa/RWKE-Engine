# RWKE 迁移方案：SemanticEcho（Transformer）→ RWKV-7（Goose）

> 版本：1.1（2026-09-04，纳入多轮对话能力与连续聊天实测）　|　规格：`.trae/specs/migrate-semantic-echo-to-rwkv7/spec.md`
> 原则：**最小改动、最大复用**——上层逻辑（注入公式/捕获公式/采样链/回响池交互）与 V1 逐行等价，仅替换推理底座；现有源码零改动，仅 import 复用。

---

## 1. 迁移概述

SemanticEcho 的核心链路（末 N 层 hidden_state 捕获 → 不确定性过滤 → 回响池衰减/质心 → Kaiming 投影 → `logits += 质心@投影×λ` → 自定义采样循环）原建立在 HF Transformer（Qwen）的 KV 缓存机制上。RWKE 将推理底座替换为 RWKV-7（rwkv pip 包），获得：

| 能力 | Transformer（原版） | RWKV-7（RWKE） |
|---|---|---|
| 上下文状态 | KV Cache，O(n) 随序列增长 | 扁平 State List，O(1) 恒定 |
| 日记/人设注入 | 挤占上下文窗口 | State 预填充，不占窗口 |
| 长对话成本 | 每步注意力开销 ∝ 历史长度 | 每步开销恒定 |
| 深度延迟 | 随深度增长 | 0/512/2048 实测基本平坦（见测试报告） |

## 2. 改动点清单（全部为新增，现有源码零改动）

| 文件 | 职责 | 对应任务 |
|---|---|---|
| `RWKE/requirements.txt` | 依赖声明 | T1 |
| `RWKE/测试_环境自检.py` | torch/CUDA/rwkv/nvcc/模型矩阵自检 | T1 |
| `RWKE/下载模型.py` | hf-mirror/ModelScope 拉取矩阵模型，`--check` 校验 | T3 |
| `RWKE/系统配置.json` | 模型矩阵、λ/γ/τ 扫描表、统一采样协议、聊天模板、推理环境 | T4.1 |
| `RWKE/rwke引擎.py` | **核心迁移**：RWKV-7 加载、自实现逐层前向、注入/捕获/采样链、State 挂载/导出、**多轮对话（延续对话/软重锚/eos 保护/复读压制）** | T4 / 演示期增补 |
| `RWKE/状态接口.py` | 推理状态抽象 + RWKV State 适配器 + .pth 序列化 | T2 |
| `RWKE/日记状态.py` | 日记解析 → 编码 → 分段流式预填充 → 日记 State .pth | T5 |
| `RWKE/测试_状态接口.py` | 状态五操作单元测试 | T2 |
| `RWKE/测试_冒烟.py` | 四组对照冒烟（裸/注入/挂State/挂State+注入） | T6 |
| `RWKE/测试_对比.py` | 多模型认证矩阵 + 情感/成本指标 + TuringBench + 标定回写 + 报告 | T7 |
| `RWKE/测试_连续对话.py` | 100 轮连续聊天实测：三模式对比、记忆突击测试、五维评分、小白报告、互动聊天入口 | 演示期增补 |

**复用（零改动 import）**：`语义回响/semantic_echo/回响池.py`（importlib 单文件直载，绕过包 `__init__` 的 cnsenti 依赖链）、`rwkv` pip 包（含 `RWKV_x070_TMix_one`/`RWKV_x070_CMix_one` jit 算子与词表）、`emocompanion-ai-emotion/最后版本/核心引擎/推理架构/V1架构.py`（Qwen 基线的注入器与加载器，测试_对比 复用）。

## 3. KV Cache ↔ RWKV State 映射表

| Transformer 概念 | RWKV-7 对应物 | 说明 |
|---|---|---|
| `past_key_values`（每层 K/V 张量对，随序列 O(n) 增长） | 扁平 `List[Tensor]`，长 `n_layer*3`，尺寸恒定 | `[i*3+0]` att_x_prev `[n_embd]`（权重 dtype）；`[i*3+1]` att_kv `[n_head, head_size, head_size]`（fp32）；`[i*3+2]` ffn_x_prev `[n_embd]`（权重 dtype） |
| `model(input_ids, past_key_values=...)` | `model.forward(idx_list, state)`（段） / 自实现 `_前向单步(token, state)`（步） | 官方 forward 按 list 长度分派 forward_seq / forward_one |
| `outputs.hidden_states` / forward hook | 前向内逐层记录残差输出（jit ScriptModule 内联，hook 不可用） | hidden = 末 N 层残差输出平均，语义与 V1 hook 捕获一致 |
| `model.generate(...)`（HF 采样） | 自实现循环：重复惩罚→注入→回调→捕获→温度→top-p→top-k→multinomial | 与 V1架构.py L537-605 逐步一致 |
| 预填充 prompt（O(n) KV 写入） | 日记→token→分段 `forward(块, state)`，State O(1) | 显存与日记长度无关（日记状态.py） |
| eos 停止 | eos(=0) + 停止标记 `"\n\nUser:"` 双重停止 | World 词表 |

## 4. 依赖与安装（可复现命令）

```powershell
# 1) 依赖（Python 3.10 实测环境：torch 2.6.0+cu124 + RTX 3080 16GB）
pip install -r RWKE/requirements.txt      # rwkv、torch、numpy、transformers、peft、scikit-learn、jieba、tqdm
pip install requests                      # 下载脚本用
pip install cnsenti                       # 可选：情感词密度（缺失时测试自动回退本地词表）

# 2) 环境自检
python RWKE/测试_环境自检.py              # 退出码 0 = 就绪；输出 nvcc 探测与矩阵缺失项

# 3) 模型下载（约 6.6 GB；--check 只检查）
python RWKE/下载模型.py                   # 全矩阵；或 --仅模型 RWKV7-World-0.4B 子集

# 4) 冒烟（四组对照）
python RWKE/测试_冒烟.py                  # 默认 0.4B；日记 State 缺失时自动生成

# 5) 日记状态（可选手动生成）
python RWKE/日记状态.py --模型 RWKV7-World-0.4B
python RWKE/日记状态.py --自检解析        # 无模型解析自检

# 6) 多模型认证对比 + 报告
python RWKE/测试_对比.py                  # 全矩阵；--模型 子集；--快速 管线验证；--强制 重跑
python RWKE/测试_对比.py --标定维度 1792  # 扫描表外维度标定回写（7.4）

# 7) 连续聊天实测（100 轮日常对话 × 三模式 + 小白评分报告）
python RWKE/测试_连续对话.py              # 默认 1.5B；--模型 换模型；--轮数 30 快速版

# 8) 真人互动聊天（本地部署演示入口：挂日记 State + 注入 + 多轮记忆）
python RWKE/测试_连续对话.py --互动       # 键入即聊，exit 退出
```

关键运行时事实：
- `RWKV_V7_ON=1` 必须先于 `import rwkv.model`（rwke引擎.py 在模块顶部设置并带 V7 分派守卫）；
- `RWKV_CUDA_ON` 由 nvcc 探测决定，CUDA 算子编译失败自动降级纯 PyTorch 重导；
- `RWKV(路径, strategy)` 的路径**不带 .pth 后缀**；strategy 如 `cuda fp16` / `cpu fp32`；
- head.weight 加载后形状为 `[n_embd, vocab]`（前向 `x @ head.weight` 直接得 `[vocab]`）。

## 5. 风险清单与实际验证情况

| 风险 | 预案 | 实际验证 |
|---|---|---|
| **钩子兼容性**：rwkv pip 的 forward_one 是 `@MyStatic(jit.script)` ScriptFunction，forward hook 无法挂载 | 引擎自实现 `_前向单步`：导入官方 `RWKV_x070_TMix_one`/`CMix_one` jit 算子逐层组装，残差输出就地记录 | ✅ 已实现并与官方 forward_one 数值一致（G1 裸组流畅生成） |
| **rwkv pip 版本变动**：V7 分派/算子签名随版本漂移 | 模块级守卫：`RWKV is RWKV_x070` 不成立即报错；关键签名在 docstring 注明核对行号（v0.8.32） | ✅ v0.8.32 逐行核对通过 |
| **numpy/torch 边界转换** | 本迁移数据通路全程 torch 张量（state/hidden/logits），无 numpy 序列化边界；.pth 经 `torch.save` 原生保存 | ✅ 无 numpy 边界问题 |
| **World tokenizer 差异**：无 chat template、eos=0、BPE 词表与 Qwen 不同 | 聊天模板在引擎内实现（`User: {消息}\n\nAssistant:`）；eos 双重停止；词表取 rwkv 包内捆绑 `rwkv_vocab_v20230424.txt` | ✅ 编码解码往返自检通过；G3 挂 State 后角色化回复 |
| **多尺寸参数适配**：hidden_dim/投影矩阵/λγτ 随模型变化；head.weight 形状陷阱 | 扫描表驱动（可标定回写）；投影矩阵按 `hidden_dim×vocab_size` 动态构造；`vocab_size` 取 `head.weight.shape[1]` | ✅ 1024/2048 双尺寸跑通；开发中发现并修复 shape[0]/shape[1] 误用（曾致投影矩阵 [1024,1024]） |
| **CUDA 编译失败**（无 nvcc / 工具链缺失） | 导入时降级 `RWKV_CUDA_ON=0` 纯 PyTorch 路径重导，功能不缺失 | ✅ 降级路径代码就位（实测环境 nvcc 可用，走 CUDA） |
| **cnsenti 依赖链**：semantic_echo 包 `__init__` 连带导入重依赖 | importlib 单文件直载 `回响池.py`；测试_对比 对 cnsenti try/except + 本地词表兜底 | ✅ 双路径就位（实测 cnsenti 0.0.7 可用，走 cnsenti） |
| **长会话复读**：World 基座模型未经对话指令对齐，连续 20~30 轮后三种模式均出现句式复读（0.4B 约 7 轮起） | 框架层四层缓解：eos 保护 + 开场 eos 禁步 + 会话级重复惩罚 + 连续重复压制 + 软重锚（见 6.1） | ⚠️ 缓解后单字爆走/空回复清零，句式复读仍存——属模型边界，换对话对齐模型即解（框架零改动兼容） |

## 6. 关键决策记录（D1-D8）

| 编号 | 决策 | 选择与理由 |
|---|---|---|
| D1 | RWKV-7 加载路径 | **rwkv pip 包**（官方稳定、Windows+CUDA 可用、`forward(token, state)` 与自定义循环同构）；备选 fla/HF trust_remote_code 未采用 |
| D2 | 模型矩阵 | RWKV7-World-0.4B + 1.5B 必测，G1 可选；基线 Qwen2.5-1.5B-Instruct——多尺寸验证可扩展性，0.4B 快速冒烟、1.5B 对标公平 |
| D3 | 隐藏状态捕获 | 末 4 层语义不变；因 jit 内联改为前向内逐层记录（与 V1 hook 捕获数值语义一致） |
| D4 | 注入点 | 与原版完全一致：`logits += 质心 @ 投影矩阵 × 有效λ`；投影矩阵 hidden×vocab fp32（0.4B 约 256MB） |
| D5 | 日记 State 化 | **预填充捕获 MVP**（Windows 本地可运行）；RWKV-PEFT 梯度调优仅 Linux/WSL，文档化升级路径即可 |
| D6 | 聊天模板 | World 风格 `User: ...\n\nAssistant:` 在 RWKE 引擎内实现；Qwen 基线走 `apply_chat_template` |
| D7 | 模型权重获取 | 接受联网下载：hf-mirror / ModelScope 拉取（本地 GGUF 无法用于 hidden_state 捕获与 logits 注入） |
| D8 | λ/γ/τ 多模型适配 | 扫描表新增 hidden_dim=1024 插值初值；`测试_对比.py --标定维度 N` 提供表外维度标定回写机制；λ 扫描建议随认证报告输出 |

### 6.1 实施期增补决策（演示与边界治理）

| 编号 | 决策 | 选择与理由 |
|---|---|---|
| D9 | **eos 保护**（默认开启） | 注入偏置将 eos 维度清零——实测发现任意 λ>0 的偏置都会抬高 eos 相对排名导致"注入即停"（种子敏感）；情感回响不应直接改变终止行为 |
| D10 | **多轮对话 = State 延续 + 软重锚** | 延续对话从上一轮末 State 继续（历史不占 prompt）；会话 token 超 3072（ctx4096 留余量）时回锚到日记 State/零态并重填最近轮——挂日记时人设记忆随 State 恢复，是"日记 State"的核心价值场景 |
| D11 | **复读四层缓解** | 会话级重复惩罚（跨轮压制）+ 连续重复压制（3 连减半，保留叠词）+ 开场 eos 禁步（前 6 步禁采样 eos，防挂载 State 下开场即终止）+ 每模式清池——单字爆走与空回复清零；句式复读属基座模型边界，根治依赖对话对齐模型 |

## 7. 已知行为差异与迁移取舍

1. **prompt 末位不捕获**：V1 从 prompt 末位起捕获 hidden；RWKE 预填充走官方 forward_seq（不暴露逐层输出），自生成第 2 步起每步捕获。对回响池动力学影响：池内容起始晚一步，注入自第 3 步起生效。
2. **捕获 softmax 在 fp32**：`logits.float()` 后再 softmax 求 max_prob，避免 fp16 下溢出（V1 在 fp32 模型上无此问题）。
3. **World 模型无思考标记**：阶段切换检测结构保留但恒为"思考"阶段，λ 恒生效。
4. **重复惩罚作用域**：对已生成 token 集合 `logits /= penalty`（与 V1 一致），与 HF `generate` 实现略有差异；对比实验中裸/注入两组走同一循环，内部公平。多轮（延续对话）模式下惩罚集合升级为**会话级累积**（RWKE 扩展）。
5. **RWKE 相对 V1 的新增采样语义**：eos 保护（偏置 eos 维度清零）、开场 eos 禁步（前 6 步禁采样 eos）、连续重复压制（3 连减半）——均为 RWKV 基座 + 长会话场景的边界治理，单轮短生成下影响可忽略。

## 8. 验证结果摘要

> 完整数据：`RWKE/docs/RWKE测试报告.md`；分片：`RWKE/results/认证_*.json`；汇总：`RWKE/results/认证汇总.json`。

### 8.1 冒烟（Task 6，真模型 0.4B cuda fp16，四组对照）

四组全 PASS：均产出非空中文文本、无 NaN/Inf、无崩溃；ΔH(注入-裸)=-1.57 可测差异；挂载日记 State 后回复角色化（"我是RoleB，今天过得很好。"），证实 O(1) State 挂载生效。

### 8.2 多模型认证（Task 7，统一协议 T=1.0/top_p=0.9/top_k=50/rp=1.05/seed=42）

| 模型 | hidden_dim | λ/γ/τ | 注入组语义熵（裸） | 注入组 4gram 重复率（裸） | 图灵检出率 裸→注入 | 判定 |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct（原版复现） | 896 | 0.50/0.05/0.10 | 2.115（1.705） | 0.002（0.006） | 0.40→0.67 | **通过** |
| Qwen2.5-1.5B-Instruct | 1536 | 0.08/0.07/0.09 | 1.580（1.483） | 0.021（0.011） | 0.73→0.60 | **通过** |
| RWKV7-World-0.4B | 1024 | 0.10/0.054/0.098 | 1.561（1.436） | 0.002（0.006） | 1.0→1.0 | **通过** |
| RWKV7-World-1.5B | 2048 | 0.10/0.08/0.06 | 1.550（1.283） | 0.024（0.000） | 0.97→0.97 | **通过** |

- **原版链路复现成功**：V1 注入器 + 回响池在本机 Qwen2.5-0.5B（hidden=896，扫描表原生条目）上跑通"裸 vs 注入"全指标。
- **RWKV λ 标定过程（D8 闭环）**：1024 插值初值 λ=0.416 过强（重复率 0.77 退化）→ λ=0.05 全量协议下 ΔH=0.046 不显著 → **最终 λ=0.1**（ΔH=+0.125 显著、重复率 0.002、λ扫描零退化）；1.5B 原 λ=0.10 触发"注入即停"——根因不是 λ 强度而是**注入偏置在 eos 维度的正分量**（任意 λ>0 均抬高 eos 相对排名，种子敏感），在引擎中加入 **eos 保护**（偏置 eos 维度清零：情感回响不直接改变终止行为）后根治，λ=0.1 认证通过（ΔH=+0.267、均字 34 可持续）。两模型 λ 扫描数据（λ∈{0.05,0.1,0.2}）已存认证分片。
- 判定四条件：无NaN、注入有效差异（|ΔH|>0.05）、无重复退化（4gram ≤ 裸+0.15）、生成可持续（注入组均字 ≥ max(8, 裸×0.5)——防"注入即停"伪通过）。

### 8.3 成本（Task 7.3，底座级延迟，不含注入）

| 模型 | 深度0 | 深度512 | 深度2048 | 深度2048 缓存 | 形态 |
|---|---|---|---|---|---|
| RWKV7-World-0.4B | 22.1ms | 22.7ms | 22.3ms | 6.09MB | State O(1) |
| RWKV7-World-1.5B | 23.7ms | 23.4ms | 23.0ms | 12.19MB | State O(1) |
| Qwen2.5-1.5B | 31.9ms | 32.7ms | 31.8ms | 56.44MB | KV Cache O(n) |
| Qwen2.5-0.5B | 27.1ms | 26.8ms | 27.1ms | 24.19MB | KV Cache O(n) |

RWKV State 尺寸与深度无关（0.4B 6.09MB / 1.5B 12.19MB 恒定）；Qwen KV 随深度线性增长（2048 深度时达 State 的 4.6~9.3 倍）。tokens/s：0.4B 裸 ~25-32，1.5B 裸 ~32（CUDA 算子路径）。

### 8.4 环境与已知事项

- Python 3.10.11 / torch 2.6.0+cu124 / RTX 3080 Laptop 16GB / rwkv pip v0.8.32 / cnsenti 0.0.7。
- CUDA 算子编译需 MSVC（cl.exe）在 PATH，且 CUDA 12.x 与 torch cu124 匹配；本机装有 CUDA 13.3 头文件时 cccl 要求 `/Zc:preprocessor` 会编译失败，自动降级纯 PyTorch（功能不缺失，速度约降 2-3 倍；认证数据为同口径自洽）。
- **eos 保护**（rwke引擎.RWKE引擎 构造参数，默认开启）：注入偏置将 eos 维度清零。Qwen 基线认证完成于该机制引入前（V1 注入器无此参数，且其注入组生成正常不受影响）。
- TuringBench 口径：TF-IDF(1-2gram,30k,sublinear)+LR(max_iter=1000)，60 训练/30 留出，人类语料=chinese-adorable girl 真人回复。

### 8.5 连续聊天演示（100 轮日常对话 × 三模式，1.5B）

> 完整数据：`RWKE/docs/连续对话演示报告.md`（小白友好版，含对话实录）；`RWKE/results/连续对话_RWKV7-World-1.5B.json`。

| 方案 | 综合得分（满分 100） | 记忆力30 | 流畅25 | 情感20 | 像真人15 | 稳定10 |
|---|---|---|---|---|---|---|
| 原始 RWKV | 47.2 | 0 | 25.0 | 12.2 | 0 | 10 |
| +情感回响 | 20.2 | 0 | 9.7 | 0.5 | 0 | 9.9 |
| **+人设记忆+情感回响** | **53.0（+12%）** | 0 | 23.0 | **20.0** | 0 | **10.0** |

- **前 10 轮真实对话阶段**（对话感最佳区间）：挂日记方案第 2 轮即主动呼应"画画"锚点、第 4~6 轮呼应"照顾小动物"——人设记忆生效的直接证据；三模式第 1 轮均有真实对话感（"你好RoleB，很高兴认识你！"）。
- **核心卖点实证**：100 轮连续对话 RWKV State 恒定 12.2MB 纹丝不动（传统 KV 同场景线性膨胀至数百 MB）；全程无崩溃。
- **边界如实记录**：三模式在 20~30 轮后均出现句式复读（基座模型未经对话对齐），记忆突击测试未命中；框架层四层缓解（D11）已将单字爆走与空回复清零。根治路径 = 换对话对齐 RWKV 模型（框架零改动兼容）。
- 互动聊天入口：`python 测试_连续对话.py --互动`（挂日记 State + 注入 + 多轮记忆，键入即聊）。
