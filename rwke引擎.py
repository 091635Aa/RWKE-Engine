# -*- coding: utf-8 -*-
"""
RWKE 推理引擎：SemanticEcho → RWKV-7 迁移核心
==============================================

将 V1架构.py 的"语义回响注入器 + 自定义生成循环"逐行迁移到 RWKV-7 (Goose) 后端。
上层逻辑（注入公式 / 捕获公式 / 采样链 / 回响池交互）与 V1 完全等价，
仅替换推理底座：HF Transformer（KV 缓存，O(n)）→ RWKV-7 State（O(1) 常量空间）。

对照迁移表
----------
| V1（HF Transformer）                    | RWKE（RWKV-7）                              |
|-----------------------------------------|---------------------------------------------|
| model(input_ids, past_key_values)       | 自实现 _前向单步（导入官方 TMix_one/CMix_one）|
| register_forward_hook 捕获末N层 hidden  | 前向内逐层记录（jit 内联无法挂钩子）          |
| past_key_values 随序列 O(n) 增长        | 扁平 State List 长 n_layer*3，O(1)           |
| hidden = 末N层 decoder layer 输出平均   | hidden = 末N层 CMix 残差后输出平均            |
| eos 停止                                | eos(=0) + 停止标记 "\n\nUser:" 双重停止       |

关键实现事实（rwkv pip v0.8.32 model.py 逐行核对）
--------------------------------------------------
- env ``RWKV_V7_ON=1`` 必须先于 ``import rwkv.model``（L1679 据此分派 RWKV_x070）；
- env ``RWKV_CUDA_ON`` 同样在导入时生效（决定 TMix_seq 走 CUDA 算子或纯 PyTorch），
  CUDA 扩展编译失败时本模块自动降级重导（纯 PyTorch 路径功能不缺失）；
- ``RWKV_x070_TMix_one`` / ``RWKV_x070_CMix_one`` 为模块级 @MyStatic(jit.script) 函数，
  导入后可直接调用（本引擎自实现前向即复用它们，与官方 forward_one 数值一致）；
- state 为扁平 List 长 n_layer*3，每层 3 张量：
  ``[i*3+0]`` att_x_prev [n_embd]（DTYPE）、
  ``[i*3+1]`` att_kv [n_head, head_size, head_size]（fp32）、
  ``[i*3+2]`` ffn_x_prev [n_embd]（DTYPE）；
- ``emb.weight`` 加载时已做 ln0 归一化（L272），前向直接查表无需再归一；
- ``head.weight`` 加载时已转置为 [n_embd, vocab]（前向 ``x @ head.weight`` 直接得 [vocab]）；
- ``RWKV(MODEL_NAME, strategy)`` 的 MODEL_NAME 不带 .pth 后缀（内部自行拼接）。

与 V1 的两处已知行为差异（迁移取舍，详见 docs/RWKE迁移方案.md）
----------------------------------------------------------------
1. prompt 预填充走官方 forward_seq（其不暴露逐层输出），故 prompt 末位
   的 hidden_state 不捕获；自生成第 2 步起每步捕获（V1 从 prompt 末位起捕获）。
2. 捕获时 softmax 在 fp32 上计算（``logits.float()``），避免 fp16 下 max_prob 溢出。
"""

import os

# ----------------------------------------------------------------------
# 环境变量必须在 import rwkv.model 之前设置（模块导入时决定 V7 分派与算子编译）
# ----------------------------------------------------------------------
import json
import math
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

_RWKE_DIR = Path(__file__).resolve().parent


def _读系统配置():
    配置路径 = _RWKE_DIR / "系统配置.json"
    try:
        with open(配置路径, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # 配置缺失不阻断导入，引擎构造时再报错
        warnings.warn(f"系统配置读取失败（{e}）：{配置路径}")
        return {}


_配置 = _读系统配置()
_推理环境 = _配置.get("推理环境", {})

os.environ.setdefault("RWKV_V7_ON", str(_推理环境.get("RWKV_V7_ON", "1")))


def _注入msvc环境() -> bool:
    """
    cl.exe 不在 PATH 时，按 vswhere 找到的 VS/BuildTools 注入 PATH/INCLUDE/LIB
    （等价 vcvars64.bat，但纯 Python 实现、不依赖 cmd）；返回 cl 是否可用。
    """
    if shutil.which("cl"):
        return True
    查找 = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not 查找.exists():
        return False
    根 = subprocess.run(
        [str(查找), "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        capture_output=True, text=True,
    ).stdout.strip()
    if not 根:
        return False
    msvc候选 = sorted((Path(根) / "VC" / "Tools" / "MSVC").glob("*"))
    sdk根 = Path(r"C:\Program Files (x86)\Windows Kits\10")
    sdk候选 = sorted((sdk根 / "Include").glob("10.*")) if (sdk根 / "Include").exists() else []
    if not msvc候选 or not sdk候选:
        return False
    msvc, 版本 = msvc候选[-1], sdk候选[-1].name
    os.environ["PATH"] = os.pathsep.join([str(msvc / "bin" / "Hostx64" / "x64"),
                                          os.environ.get("PATH", "")])
    os.environ["INCLUDE"] = os.pathsep.join([
        str(msvc / "include"),
        str(sdk根 / "Include" / 版本 / "ucrt"),
        str(sdk根 / "Include" / 版本 / "um"),
        str(sdk根 / "Include" / 版本 / "shared"),
    ])
    os.environ["LIB"] = os.pathsep.join([
        str(msvc / "lib" / "x64"),
        str(sdk根 / "Lib" / 版本 / "ucrt" / "x64"),
        str(sdk根 / "Lib" / 版本 / "um" / "x64"),
    ])
    return shutil.which("cl") is not None


def _自举cuda编译环境() -> Optional[str]:
    """
    让本机具备编译 rwkv 自定义 CUDA 算子的条件（仅在 RWKV_CUDA_ON=1 时调用）。

    四件事（全程不阻断导入，失败由调用方自动降级纯 PyTorch）：
    1. **MSVC**：cl.exe 不在 PATH 时按 vswhere 注入 PATH/INCLUDE/LIB，并把 nvcc 同级
       的 lib/x64 加进 LIB（链接 cublas.lib 用）；
    2. **CCCL 预处理器**：CUDA 12.9+ 自带 CCCL 对 MSVC 传统预处理器直接 #error，
       经 CL / NVCC_PREPEND_FLAGS 注入 /Zc:preprocessor（两者都是编译器自带机制）；
    3. **版本校验放行**：torch 编译期 CUDA 主版本与本机 nvcc 不一致时（如 torch
       cu124 配本机 nvcc 13.3）torch.utils.cpp_extension 会直接抛 RuntimeError；
       此处放行——目标架构（sm_86）与生成代码一致，实测编译/加载/运行均正常。
    4. **陈旧锁文件**：torch 的 FileBaton 以「锁文件是否存在」判定占用（不是 OS 锁），
       进程被强杀时 release() 不执行，残留锁会让后续 load() 在 wait() 里永久自旋
       （无超时）。此处只清 10 分钟以上的（实测首次编译全部算子约 40s，不会误删）。

    返回 None 表示自举成功，否则返回失败原因字符串（供日志）。
    """
    if os.name != "nt":
        return None
    try:
        if not os.environ.get("CL"):
            os.environ["CL"] = "/Zc:preprocessor"
        if not os.environ.get("NVCC_PREPEND_FLAGS"):
            os.environ["NVCC_PREPEND_FLAGS"] = "-Xcompiler /Zc:preprocessor"
        nvcc = shutil.which("nvcc")
        if not nvcc:
            return "PATH 中找不到 nvcc"
        if not _注入msvc环境():
            return "找不到 cl.exe（未安装 MSVC C++ BuildTools？）"
        cuda根 = Path(nvcc).parent.parent
        lib目录 = cuda根 / "lib" / "x64"
        if lib目录.exists() and str(lib目录) not in os.environ.get("LIB", ""):
            os.environ["LIB"] = os.pathsep.join([os.environ.get("LIB", ""), str(lib目录)])
        import torch.utils.cpp_extension as _ce

        # 显式给出目标架构：否则 torch 会对每个未知架构试编译探测，导入期可能卡很久
        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            import torch

            主版本, 次版本 = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{主版本}.{次版本}"

        # 清理陈旧锁文件（见 docstring 第 4 条）：rglob 仅扫 torch_extensions 根，目录数很少
        扩展根 = Path(_ce.get_default_build_root())
        if 扩展根.exists():
            截止 = time.time() - 600
            for 锁 in 扩展根.rglob("lock"):
                try:
                    if 锁.stat().st_mtime < 截止:
                        锁.unlink()
                except OSError:
                    pass

        _ce._check_cuda_version = lambda *a, **k: None  # 主版本不一致放行（见 docstring 第 3 条）
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _导入rwkv模型模块():
    """导入 rwkv.model：CUDA 算子按配置+nvcc 探测决定；编译失败自动降级纯 PyTorch。

    注意：RWKV_CUDA_ON=1 时 TMix 递推改走 CUDA 算子，**数值与纯 PyTorch 路径不同**
    （实测递推输出最大差 ~4e-2，超出本项目 fp16 跨核噪声 3e-2 容差）。故缺省不开；
    要开需在 系统配置.json 的 推理环境 里显式写 "RWKV_CUDA_ON": "1"。
    """
    import importlib

    if "rwkv.model" in sys.modules:
        return sys.modules["rwkv.model"]

    if "RWKV_CUDA_ON" not in os.environ:
        显式 = _推理环境.get("RWKV_CUDA_ON")
        if 显式 is not None:
            os.environ["RWKV_CUDA_ON"] = str(显式)
        elif _推理环境.get("RWKV_CUDA_ON_自动", True) and shutil.which("nvcc"):
            os.environ["RWKV_CUDA_ON"] = "1"
        else:
            os.environ["RWKV_CUDA_ON"] = "0"

    if os.environ.get("RWKV_CUDA_ON") == "1":
        失败原因 = _自举cuda编译环境()
        if 失败原因:
            warnings.warn(f"CUDA 编译环境自举失败（{失败原因}），将尝试直接编译；失败即降级纯 PyTorch")

    try:
        return importlib.import_module("rwkv.model")
    except Exception as e:
        if os.environ.get("RWKV_CUDA_ON") == "1":
            warnings.warn(
                f"RWKV-7 CUDA 算子编译失败（{e}），自动降级为纯 PyTorch 路径（功能不缺失，速度略降）"
            )
            os.environ["RWKV_CUDA_ON"] = "0"
            sys.modules.pop("rwkv.model", None)
            sys.modules.pop("rwkv", None)
            return importlib.import_module("rwkv.model")
        raise


_rwkv模型模块 = _导入rwkv模型模块()
import rwkv as _rwkv包  # noqa: E402  （须在环境变量设置之后）

if not hasattr(_rwkv模型模块, "RWKV_x070") or _rwkv模型模块.RWKV is not _rwkv模型模块.RWKV_x070:
    raise RuntimeError(
        "rwkv.model 已被其他代码以 RWKV_V7_ON!=1 先行导入，V7 分派失效。"
        "请在新进程中使用本模块，或确保 os.environ['RWKV_V7_ON']='1' 先于任何 rwkv 导入。"
    )

RWKV = _rwkv模型模块.RWKV
RWKV_x070_TMix_one = _rwkv模型模块.RWKV_x070_TMix_one
RWKV_x070_CMix_one = _rwkv模型模块.RWKV_x070_CMix_one
RWKV_x070_TMix_seq = _rwkv模型模块.RWKV_x070_TMix_seq
RWKV_x070_CMix_seq = _rwkv模型模块.RWKV_x070_CMix_seq

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402

from 状态接口 import RWKV状态, 加载状态  # noqa: E402


# ======================================================================
# 回响池直载（绕过 semantic_echo 包 __init__ 的重依赖链）
# ======================================================================

def _加载回响池类(路径: Optional[str] = None):
    """
    importlib 直接加载 回响池.py 单文件。

    原因：``from semantic_echo.回响池 import 语义回响池`` 会触发包 __init__，
    连带导入 采样处理器（依赖 transformers）与 情感过滤器（依赖 cnsenti）。
    回响池.py 本身仅依赖 torch/math/typing，单文件直载零额外依赖。
    """
    if 路径:
        候选路径 = [Path(路径)]
    else:
        # 候选顺序：02 目录内 → 01_Transformer架构（2026-09-16 架构整理后的归档位置）
        候选路径 = [
            _RWKE_DIR.parent / "语义回响" / "semantic_echo" / "回响池.py",
            _RWKE_DIR.parent.parent / "01_Transformer架构" / "语义回响" / "semantic_echo" / "回响池.py",
        ]
    池路径 = next((p for p in 候选路径 if p.exists()), None)
    if 池路径 is None:
        raise FileNotFoundError(
            "回响池源文件不存在，已尝试：\n  " + "\n  ".join(str(p) for p in 候选路径))
    spec = importlib.util.spec_from_file_location("rwke_echo_pool", str(池路径))
    模块 = importlib.util.module_from_spec(spec)
    sys.modules["rwke_echo_pool"] = 模块
    spec.loader.exec_module(模块)
    return 模块.语义回响池


import importlib.util  # noqa: E402

_语义回响池类 = _加载回响池类()


# ======================================================================
# TS 整环快速步进模块（提速②：把 N 层循环关进一次 TorchScript 调用，
# 消除逐层 Python↔TorchScript 边界穿越；投影 GPU 常驻为提速①见 _初始化投影）
# ======================================================================

class _RWKE_TS步进模块(torch.nn.Module):
    """
    整环单步前向（jit.script）：emb → N层(ln1→TMix_one→残差→ln2→CMix_one→残差)
    → ln_out → head，并按引擎捕获语义返回 hidden = 末K层残差流均值。

    逐算子镜像 rwkv.model 的 RWKV_x070_TMix_one / RWKV_x070_CMix_one（非DE；
    逐 token 路径为纯 PyTorch，不依赖 wkv CUDA 扩展）与引擎 _前向单步。

    TorchScript 约束的落法：
    - 不可闭包全局张量、不支持 f-string 字典键 → 权重经 Module 属性承载，
      每层 33 个参数扁平化为 List[Tensor]（b = i*33 整数索引）；
    - List[Tensor] 参数按值拷贝，槽位演进必须落到【新列表】函数式返回。

    语义要点（与官方逐位对齐）：
    - TMix 的 x 参数与 att_x_prev 状态槽存的都是 ln1 后的值；
    - CMix 的 d = x_prev − x（ln2 后的 x 与状态槽做差）；
    - v_first 仅层 0 赋值；kv 状态 fp32 [H, N, N]，输出前转回权重 dtype。
    """

    def __init__(
        self,
        层参数: List[torch.Tensor],
        emb: torch.Tensor,
        ln_out_w: torch.Tensor,
        ln_out_b: torch.Tensor,
        head: torch.Tensor,
        L: int,
        Hn: int,
        Nn: int,
        Cn: int,
        K: int,
    ) -> None:
        super().__init__()
        self.层参数 = 层参数
        self.emb = emb
        self.ln_out_w = ln_out_w
        self.ln_out_b = ln_out_b
        self.head = head
        self.L = L
        self.Hn = Hn
        self.Nn = Nn
        self.Cn = Cn
        self.K = K

    def forward(self, token张量: torch.Tensor, 状态列表: List[torch.Tensor]):
        x = self.emb[token张量]
        v_first = torch.empty_like(x)
        Hn = self.Hn
        Nn = self.Nn
        Cn = self.Cn
        L = self.L
        K = self.K
        残差列表: List[torch.Tensor] = []
        # TS 列表按值传递：演进落新列表并返回（调用方重绑），原地改不回传
        新状态: List[torch.Tensor] = []
        for t0 in 状态列表:
            新状态.append(t0)

        for i in range(L):
            b = i * 33
            lnx = torch.layer_norm(x, (Cn,), weight=self.层参数[b + 0], bias=self.层参数[b + 1])

            # ---- TMix_one（x 参数 = ln1 后的 lnx）----
            x0 = x
            xx0 = 新状态[i * 3 + 0] - lnx
            xr = lnx + xx0 * self.层参数[b + 2]
            xw = lnx + xx0 * self.层参数[b + 3]
            xk = lnx + xx0 * self.层参数[b + 4]
            xv = lnx + xx0 * self.层参数[b + 5]
            xa = lnx + xx0 * self.层参数[b + 6]
            xg = lnx + xx0 * self.层参数[b + 7]
            r = xr @ self.层参数[b + 22]
            w = torch.tanh(xw @ self.层参数[b + 9]) @ self.层参数[b + 10]
            k = xk @ self.层参数[b + 23]
            v = xv @ self.层参数[b + 24]
            a = torch.sigmoid(self.层参数[b + 11] + (xa @ self.层参数[b + 12]) @ self.层参数[b + 13])
            g = torch.sigmoid(xg @ self.层参数[b + 17]) @ self.层参数[b + 18]
            kk = torch.nn.functional.normalize(
                (k * self.层参数[b + 19]).view(Hn, Nn), dim=-1, p=2.0
            ).view(Hn * Nn)
            k = k * (1 + (a - 1) * self.层参数[b + 20])
            if i == 0:
                v_first = v
            else:
                v = v + (v_first - v) * torch.sigmoid(
                    self.层参数[b + 14] + (xv @ self.层参数[b + 15]) @ self.层参数[b + 16]
                )
            w = torch.exp(-0.606531 * torch.sigmoid((self.层参数[b + 8] + w).float()))

            kv = 新状态[i * 3 + 1]
            vk = v.view(Hn, Nn, 1) @ k.view(Hn, 1, Nn)
            ab = (-kk).view(Hn, Nn, 1) @ (kk * a).view(Hn, 1, Nn)
            kv = kv * w.view(Hn, 1, Nn) + kv @ ab.float() + vk.float()
            xx0 = kv.to(x0.dtype) @ r.view(Hn, Nn, 1)
            xx0 = torch.nn.functional.group_norm(
                xx0.view(1, Hn * Nn), num_groups=Hn,
                weight=self.层参数[b + 26], bias=self.层参数[b + 27], eps=64e-5,
            ).view(Hn * Nn)
            xx0 = xx0 + ((r * k * self.层参数[b + 21]).view(Hn, Nn).sum(dim=-1, keepdim=True)
                         * v.view(Hn, Nn)).view(Hn * Nn)
            xx = (xx0 * g) @ self.层参数[b + 25]
            新状态[i * 3 + 0] = lnx  # att_x_prev ← ln1 后的 x
            新状态[i * 3 + 1] = kv
            x = x0 + xx

            # ---- CMix_one（非DE）----
            xx1 = torch.layer_norm(x, (Cn,), weight=self.层参数[b + 28], bias=self.层参数[b + 29])
            d = 新状态[i * 3 + 2] - xx1
            kk2 = xx1 + d * self.层参数[b + 30]
            kk2 = torch.relu(kk2 @ self.层参数[b + 31]) ** 2
            out = kk2 @ self.层参数[b + 32]
            新状态[i * 3 + 2] = xx1
            x = x + out

            if i >= L - K:
                残差列表.append(x)

        hidden = torch.stack(残差列表).mean(dim=0)
        x = torch.layer_norm(x, (Cn,), weight=self.ln_out_w, bias=self.ln_out_b)
        logits = x @ self.head
        return logits, 新状态, hidden


# ======================================================================
# RWKE 推理引擎
# ======================================================================

class RWKE引擎:
    """
    RWKV-7 语义回响推理引擎。

    一次构造对应一个模型；对外能力：
    - ``生成(消息)``：单轮对话生成（注入/捕获/采样全链路）；
    - ``挂载日记状态(路径)``：加载预填充日记 State（O(1) 挂载，模型标识防错挂）；
    - ``导出状态(路径)``：把最近一次生成的末态存为 .pth（日记状态来源）；
    - ``logits_callback`` / ``轮次回调``：与 V1 兼容的实验观测挂点。

    Parameters
    ----------
    模型标识 : str
        系统配置.json 模型矩阵中的键，如 "RWKV7-World-0.4B"。
    模型路径 : Optional[str]
        覆盖配置中的路径（相对路径相对 RWKE 目录）；.pth 后缀可有可无。
    加载策略 : Optional[str]
        rwkv strategy 字符串，默认取配置"加载策略"（cuda fp16），失败自动回退
        配置"CPU回退策略"（cpu fp32）。
    启用注入 / 启用捕获 : bool
        回响层开关（对照 V1 同名参数；关闭后为裸 RWKV 生成）。
    λ / γ / τ : Optional[float]
        覆盖扫描表值：λ 注入强度、γ 回响池衰减、τ 不确定性入池阈值。
    投影种子 : Optional[int]
        覆盖配置"投影种子"（默认 42）。
    末N层捕获 : Optional[int]
        覆盖配置"末N层捕获"（默认 4）。
    回响池路径 : Optional[str]
        覆盖回响池.py 的查找路径。
    适配器路径 : Optional[str]
        LoRA 适配器 checkpoint（.pt，微调.py 产出）路径。提供时生成走"基座+适配器"
        路径（被适配投影取 W_eff = W + (alpha/rank)·BA，数值与合并权重路径一致），
        并校验 checkpoint 基座标识与本引擎一致（防错挂）；None 时不挂载，行为与
        未扩展前完全一致。
    eos保护 : bool
        True（默认）时注入偏置将 eos 维度清零：情感回响不直接改变终止行为。
        依据：RWKV7-World-1.5B 实测发现任意 λ>0 的注入均会抬高 eos 相对排名
        导致"注入即停"（种子敏感的临界行为），对 eos 维度去偏置后消除。
    """

    def __init__(
        self,
        模型标识: str,
        模型路径: Optional[str] = None,
        加载策略: Optional[str] = None,
        启用注入: bool = True,
        启用捕获: bool = True,
        λ: Optional[float] = None,
        γ: Optional[float] = None,
        τ: Optional[float] = None,
        投影种子: Optional[int] = None,
        末N层捕获: Optional[int] = None,
    回响池路径: Optional[str] = None,
    eos保护: bool = True,
    偏置归一: bool = False,
    注入衰减: bool = False,
    池容量: Optional[int] = None,
    适配器路径: Optional[str] = None,
    快速步进: bool = True,
) -> None:
        if not _配置:
            raise RuntimeError("系统配置.json 缺失或不可读，无法构造 RWKE引擎")

        矩阵 = _配置.get("模型矩阵", {})
        if 模型标识 not in 矩阵:
            raise KeyError(f"模型标识 '{模型标识}' 不在系统配置模型矩阵中：{list(矩阵)}")
        条目 = 矩阵[模型标识]
        if 条目.get("后端") != "rwkv7":
            raise ValueError(f"模型 '{模型标识}' 后端为 '{条目.get('后端')}'，本引擎仅支持 rwkv7")

        self.模型标识 = 模型标识
        self.配置 = _配置
        self.采样协议 = dict(_配置.get("采样协议", {}))
        self.聊天模板 = dict(_配置.get("聊天模板", {}))

        # ---- 多尺寸自适应：λ/γ/τ 按扫描表读取（可被构造参数覆盖）----
        self.hidden_dim = int(条目["hidden_dim"])
        扫描表 = _配置.get("扫描表", {})
        表项 = 扫描表.get(str(self.hidden_dim))
        if 表项 is None:
            raise KeyError(f"扫描表缺少 hidden_dim={self.hidden_dim} 的条目：{list(扫描表)}")
        self.λ强度 = float(λ) if λ is not None else float(表项[0])
        self.γ衰减 = float(γ) if γ is not None else float(表项[1])
        self.τ阈值 = float(τ) if τ is not None else float(表项[2])

        # ---- 模型加载（路径去 .pth 后缀；cuda fp16 失败回退 cpu fp32）----
        路径 = Path(模型路径) if 模型路径 else Path(条目["路径"])
        if not 路径.is_absolute():
            路径 = _RWKE_DIR / 路径
        rwkv路径 = str(路径.with_suffix("")) if 路径.suffix == ".pth" else str(路径)

        策略 = 加载策略 or _推理环境.get("加载策略", "cuda fp16")
        回退策略 = _推理环境.get("CPU回退策略", "cpu fp32")
        try:
            self.模型 = RWKV(rwkv路径, 策略)
            self.加载策略 = 策略
        except Exception as e:
            warnings.warn(f"策略 '{策略}' 加载失败（{e}），回退 '{回退策略}'")
            self.模型 = RWKV(rwkv路径, 回退策略)
            self.加载策略 = 回退策略

        # ---- 尺寸与设备信息（取自模型实例与权重，均为 forward_one 同款来源）----
        self.n_layer = self.模型.n_layer
        self.n_embd = self.模型.n_embd
        self.n_head = self.模型.n_head
        self.head_size = self.模型.head_size
        self.vocab_size = int(self.模型.z["head.weight"].shape[1])  # head 已转置为 [n_embd, vocab]
        self.设备 = self.模型.z["emb.weight"].device
        self.权重dtype = self.模型.z["emb.weight"].dtype

        # ---- LoRA 适配器挂载槽（可选；不传时缓存为 None，前向与纯基座数值路径逐位一致）----
        self.适配器标识: Optional[str] = None
        self._适配权重缓存: Optional[Dict[str, torch.Tensor]] = None
        self._权重版本 = 0  # 适配器挂载会自增，须先于 _挂载适配器 初始化
        if 适配器路径:
            self._挂载适配器(适配器路径)

        self.eos_token_id = 0  # RWKV World 词表：0 号为 <|rrhh|> 结束符
        self.eos保护 = bool(eos保护)
        # 前 N 步禁止采样 eos：挂载日记 State 下模型有"开场即终止"倾向（eos=0 无文本，
        # 采样到即表现为 1 字乱码回复），强制先开口再允许自然终止
        self.eos禁步数 = 6
        # 缓和注入变体（条纹治理）：
        # 偏置归一：偏置 RMS 归一到 0.5×λ——注入强度与池质心范数漂移解耦（强度有界，方向保留）
        # 注入衰减：单轮内前强后弱（进度调度，P6/PIS 同思路），步内系数 min 0.4
        # 池容量：限制回响池条目数（默认 1024），质心跟随近期情绪，防长会话质心僵化
        self.偏置归一 = bool(偏置归一)
        self.注入衰减 = bool(注入衰减)
        self.池容量 = int(池容量) if 池容量 is not None else None
        self._注入衰减系数 = 1.0

        # ---- 快速步进（提速②）：TS 整环单步，权重经 _权重 取值（自动含适配器 W_eff）；
        # 挂载/卸载适配器时 _权重版本 自增，首次步进惰性重建 TS 模块 ----
        self.快速步进 = bool(快速步进)
        self._ts模块 = None
        self._ts权重版本 = -1

        # ---- 提速③：会话级重复惩罚 GPU 缓冲——原实现每步对集合做 Python 逐 token
        # 索引除法（O(会话长度)，长会话数百 token ≈ 数十毫秒/步）；改为每轮一次 H2D
        # 重建 + 每步单次向量化索引除 + 新 token 标量写入 ----
        self._惩罚缓冲 = torch.zeros(8192, dtype=torch.int64, device=self.设备)
        self._惩罚数 = 0
        # ---- 提速④：TS 步进 token 输入缓冲（免每步 tensor 分配 + H2D）；
        # 必须 0 维——emb[0维张量] → [C]，[1]形会多出一维污染下游形状 ----
        self._步进token缓冲 = torch.zeros((), dtype=torch.int64, device=self.设备)
        # ---- 提速⑤：CUDA Graph 步进（消除 TS 逐 kernel launch 开销，实测再 3x）；
        # 状态进出：轮首 copy-in 一次 + 图内自续；hidden/logits 输出 clone 防图缓冲覆写；
        # 仅 CUDA 设备可用，CPU 回退路径自动走无图 TS ----
        self._图 = None
        self._图状态列表: Optional[List[torch.Tensor]] = None
        self._图logits缓冲: Optional[torch.Tensor] = None
        self._图hidden缓冲: Optional[torch.Tensor] = None
        self._图权重版本 = -1

        # ---- 提速⑥：回响池 GPU 镜像——池.添加() 每步 CPU 同步拷贝（2 次流水线泡）
        # + 池.计算质心() Python 逐向量累加 O(n)（推进清缓存 → 每步全量重算，O(n²)/轮）。
        # 改为：捕获/注入全程 GPU 镜像（向量化衰减质心 µs 级），池对象轮末批量同步一次 ----
        _池cap = int(self.池容量) if self.池容量 is not None else 1024
        self._池V = torch.zeros(_池cap, self.hidden_dim, dtype=torch.float32, device=self.设备)
        self._池W = torch.zeros(_池cap, dtype=torch.float32, device=self.设备)
        self._池TS = torch.zeros(_池cap, dtype=torch.int64, device=self.设备)
        self._池n = 0

        # ---- World tokenizer（词表由 rwkv pip 包内捆绑）----
        词表 = os.path.join(os.path.dirname(_rwkv包.__file__), "rwkv_vocab_v20230424.txt")
        self.分词器 = TRIE_TOKENIZER(词表)

        # ---- 回响池（γ 来自扫描表；池容量可限制条目数防质心僵化）----
        if self.池容量 is not None:
            self.回响池 = _语义回响池类(self.hidden_dim, self.池容量, decay_gamma=self.γ衰减)
        else:
            self.回响池 = _语义回响池类(self.hidden_dim, decay_gamma=self.γ衰减)

        # ---- 注入器状态（对照 V1架构.py L244-310）----
        self.启用注入 = 启用注入
        self.启用捕获 = 启用捕获
        self.末N层捕获 = int(末N层捕获) if 末N层捕获 is not None else int(
            self.采样协议.get("末N层捕获", 4)
        )
        self.思考标记对 = ("", "")  # World 模型无思考标记：全程单阶段，λ 恒生效
        self.思考阶段λ: Optional[float] = None
        self.正文阶段λ: float = 0.0
        self.当前阶段: str = "思考"
        self.当前hidden_state: Optional[torch.Tensor] = None

        self._初始化投影(int(投影种子) if 投影种子 is not None else int(self.采样协议.get("投影种子", 42)))

        # ---- 日记 State 挂载槽（已对齐设备/dtype 的张量组）----
        self._挂载state: Optional[List[torch.Tensor]] = None
        self._末state: Optional[List[torch.Tensor]] = None
        # ---- 架构级身份记忆（状态注入.py）：槽位锁定 + 每步回注，默认关闭 ----
        self._身份记忆 = None
        self.身份回注系数 = 0.92
        # ---- 输出链路（导向链.py）：情感/身份/复制 三路加性先验，默认关闭 ----
        self._导向链 = None
        self._本轮情感: Optional[str] = None
        # 会话级已生成 token 集合：延续对话模式下跨轮累积，供重复惩罚使用
        # （否则多轮对话每轮只惩罚本轮 token，模型易陷入跨轮复读循环）
        self._会话token集合: set = set()
        # 软重锚：World 模型训练窗口 ctx4096，超窗后 State 表征越界会导致退化复读。
        # 会话 token 数接近预算时，回到挂载 State（或零态）并重填最近轮，保证不越窗。
        self.上下文预算 = 3072
        self._会话token数 = 0
        self._最近轮token: List[int] = []

    # ──────────────────────────────────────────────
    # 投影矩阵初始化（V1架构.py L316-333 逐行等价）
    # ──────────────────────────────────────────────

    def _初始化投影(self, seed: int) -> None:
        """固定种子的随机投影矩阵 hidden_dim → vocab_size（Kaiming 缩放）。"""
        rng = torch.Generator()
        rng.manual_seed(seed)

        scale = math.sqrt(2.0 / self.hidden_dim)
        self.投影矩阵 = torch.randn(
            self.hidden_dim, self.vocab_size,
            generator=rng,
            dtype=torch.float32,
        ) * scale

        self.投影矩阵.requires_grad_(False)
        # 提速①：GPU 常驻副本——_注入偏置 每步 537MB（G1-1.5B）CPU→GPU 搬运曾占 ~65%
        # 生成耗时（PCIe 页面内存 ~220ms/步），一次上卡后注入只剩 ~2ms matmul
        self.投影矩阵GPU = self.投影矩阵.to(self.设备)

    # ──────────────────────────────────────────────
    # LoRA 适配器挂载（微调 Task 4：推理集成）
    # ──────────────────────────────────────────────

    def _挂载适配器(self, 适配器路径: str) -> None:
        """
        加载 LoRA checkpoint 并构建 W_eff 权重缓存（延迟导入 微调 模块防循环依赖：
        微调.py 顶部 import rwke引擎 复用本模块的模型工厂与 jit 算子句柄）。

        - checkpoint["基座标识"] 与本引擎 模型标识 不符即抛 ValueError（防错挂）；
        - A/B 快照经 微调.LoRA适配器.载入快照 写入：顺带校验 checkpoint 被适配键
          齐全（缺键 KeyError）与形状匹配（错形 copy_ 报错）；
        - W_eff 经 微调.计算有效权重缓存 现算（ΔW 走 fp64 matmul，与 合并适配器
          的 CPU 合并侧逐位一致），缓存张量对齐 self.设备 与 self.权重dtype
          （cuda fp16 策略下即 fp16，CPU 回退策略下为 fp32）。
        """
        import 微调  # 局部导入：与 微调.py 存在模块级相互引用，运行时按需加载

        p = Path(适配器路径)
        if not p.exists():
            raise FileNotFoundError(f"适配器 checkpoint 不存在：{p}")
        checkpoint = torch.load(str(p), map_location="cpu")
        if checkpoint.get("基座标识") != self.模型标识:
            raise ValueError(
                f"适配器 checkpoint 基座标识不符（防错挂）：文件为 '{checkpoint.get('基座标识')}'，"
                f"当前引擎为 '{self.模型标识}'"
            )
        适配器 = 微调.LoRA适配器(
            self.模型, rank=int(checkpoint["rank"]), alpha=int(checkpoint["alpha"])
        )
        适配器.载入快照(checkpoint["适配器A"], checkpoint["适配器B"], self.设备)
        with torch.no_grad():
            缓存 = 微调.计算有效权重缓存(self.模型, 适配器)
        self._适配权重缓存 = {
            名称: 张量.to(self.设备).to(self.权重dtype) for 名称, 张量 in 缓存.items()
        }
        self.适配器标识 = f"{checkpoint['基座标识']}_lora_s{checkpoint.get('step', 0)}"
        self._权重版本 += 1  # 快速步进：W_eff 变更 → 下次步进惰性重建 TS 模块

    def _权重(self, 名字: str) -> torch.Tensor:
        """
        前向取权重：挂载适配器时被适配投影取 W_eff 缓存（未适配的名字回退 z 原值），
        无适配器时恒等返回 z[名字]（数值路径与改动前完全一致）。
        """
        if self._适配权重缓存 is None:
            return self.模型.z[名字]
        return self._适配权重缓存.get(名字, self.模型.z[名字])

    # ──────────────────────────────────────────────
    # 快速步进（提速②）：TS 整环单步
    # ──────────────────────────────────────────────

    def _构建ts步进(self) -> None:
        """从 _权重（自动含适配器 W_eff）构建 jit.script 整环步进模块。"""
        z = self.模型.z
        参数: List[torch.Tensor] = []
        for i in range(self.n_layer):
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."
            bbb = f"blocks.{i}."
            参数.extend([
                self._权重(bbb + "ln1.weight"), self._权重(bbb + "ln1.bias"),            # 0,1
                self._权重(att + "x_r"), self._权重(att + "x_w"), self._权重(att + "x_k"),  # 2-4
                self._权重(att + "x_v"), self._权重(att + "x_a"), self._权重(att + "x_g"),  # 5-7
                self._权重(att + "w0"), self._权重(att + "w1"), self._权重(att + "w2"),    # 8-10
                self._权重(att + "a0"), self._权重(att + "a1"), self._权重(att + "a2"),    # 11-13
                self._权重(att + "v0"), self._权重(att + "v1"), self._权重(att + "v2"),    # 14-16
                self._权重(att + "g1"), self._权重(att + "g2"),                            # 17-18
                self._权重(att + "k_k"), self._权重(att + "k_a"), self._权重(att + "r_k"),  # 19-21
                self._权重(att + "receptance.weight"), self._权重(att + "key.weight"),      # 22-23
                self._权重(att + "value.weight"), self._权重(att + "output.weight"),        # 24-25
                self._权重(att + "ln_x.weight"), self._权重(att + "ln_x.bias"),             # 26-27
                self._权重(bbb + "ln2.weight"), self._权重(bbb + "ln2.bias"),               # 28-29
                self._权重(ffn + "x_k"), self._权重(ffn + "key.weight"), self._权重(ffn + "value.weight"),  # 30-32
            ])
        模块 = _RWKE_TS步进模块(
            参数,
            z["emb.weight"], z["ln_out.weight"], z["ln_out.bias"], z["head.weight"],
            self.n_layer, self.n_head, self.head_size, self.n_embd, self.末N层捕获,
        )
        self._ts模块 = torch.jit.script(模块)
        self._ts权重版本 = self._权重版本

    @torch.no_grad()
    def _适配前向序列(self, token列表: List[int], state: List[torch.Tensor]):
        """
        挂载适配器时的预填充：逐行镜像 rwkv.model 非DE forward_seq（model.py
        L458-487），仅 6 类被适配投影经 _权重 换 W_eff，其余权重与控制流同官方。

        不复用官方 forward_seq 的原因：其权重直取 z，无法叠加 ΔW；也不改走逐 token
        _前向单步 的原因：实测 forward_seq（TMix_seq）与逐 token forward_one（TMix_one）
        存在 gemv/gemm 舍入级差异（fp16 logits 最大 ~1.5e-2），镜像同路径才能与
        "合并权重模型直接构造引擎"的预填充逐位一致（合并等价性要求）。

        返回 (末位 logits [1, vocab], state)；state 原地演进。
        """
        z = self.模型.z
        x = z["emb.weight"][token列表]
        v_first = torch.empty_like(x)

        for i in range(self.n_layer):
            bbb = f"blocks.{i}."
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."

            xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"])

            xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_seq(
                i, self.n_head, self.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1],
                z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"], z[att + "x_a"], z[att + "x_g"],
                z[att + "w0"], z[att + "w1"], z[att + "w2"], z[att + "a0"], z[att + "a1"], z[att + "a2"],
                z[att + "v0"], z[att + "v1"], z[att + "v2"],
                z[att + "g1"], z[att + "g2"], z[att + "k_k"], z[att + "k_a"], z[att + "r_k"],
                self._权重(att + "receptance.weight"), self._权重(att + "key.weight"),
                self._权重(att + "value.weight"), self._权重(att + "output.weight"),
                z[att + "ln_x.weight"], z[att + "ln_x.bias"],
            )
            x = x + xx

            xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
            xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(
                xx, state[i * 3 + 2], z[ffn + "x_k"],
                self._权重(ffn + "key.weight"), self._权重(ffn + "value.weight"),
            )
            x = x + xx

        x = F.layer_norm(x[-1, :], (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
        logits = x @ z["head.weight"]
        return logits.unsqueeze(0), state

    def _预填充(self, token列表: List[int], state: List[torch.Tensor]):
        """
        prompt 预填充分派：无适配器走官方 forward_seq（改动前路径，逐位不变）；
        挂载适配器走 _适配前向序列（forward_seq 权重直取 z，无法叠加 ΔW）。
        """
        if self._适配权重缓存 is None:
            return self.模型.forward(token列表, state)
        return self._适配前向序列(token列表, state)

    # ──────────────────────────────────────────────
    # 自实现前向：单 token，逐层记录（Task 4.3 核心）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _前向单步(self, token_id: int, state: List[torch.Tensor]):
        """
        复刻 rwkv.model 非DE forward_one（L336-363），并记录每层残差输出。

        被适配的 6 类投影（att.receptance/key/value/output、ffn.key/value）经 _权重
        取值：挂载适配器时为 W_eff，否则与 z[名字] 逐位一致。

        返回 (logits [1, vocab], state, hidden_state [n_embd])；
        hidden_state = 末N层 CMix 残差后输出的平均（对应 V1 hook 捕获语义）。
        """
        if self.快速步进:
            return self._前向单步快(token_id, state)

        z = self.模型.z
        x = z["emb.weight"][int(token_id)]
        v_first = torch.empty_like(x)
        层输出: List[torch.Tensor] = []

        for i in range(self.n_layer):
            bbb = f"blocks.{i}."
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."

            xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"])

            xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_one(
                i, self.n_head, self.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1],
                z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"], z[att + "x_a"], z[att + "x_g"],
                z[att + "w0"], z[att + "w1"], z[att + "w2"], z[att + "a0"], z[att + "a1"], z[att + "a2"],
                z[att + "v0"], z[att + "v1"], z[att + "v2"],
                z[att + "g1"], z[att + "g2"], z[att + "k_k"], z[att + "k_a"], z[att + "r_k"],
                self._权重(att + "receptance.weight"), self._权重(att + "key.weight"),
                self._权重(att + "value.weight"), self._权重(att + "output.weight"),
                z[att + "ln_x.weight"], z[att + "ln_x.bias"],
            )
            x = x + xx

            xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"])
            xx, state[i * 3 + 2] = RWKV_x070_CMix_one(
                xx, state[i * 3 + 2], z[ffn + "x_k"],
                self._权重(ffn + "key.weight"), self._权重(ffn + "value.weight"),
            )
            x = x + xx

            层输出.append(x.detach())

        n = min(self.末N层捕获, len(层输出))
        hidden_state = torch.stack(层输出[-n:]).mean(dim=0)

        x = F.layer_norm(x, (self.n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
        logits = x @ z["head.weight"]
        return logits.unsqueeze(0), state, hidden_state

    def _前向单步快(self, token_id: int, state: List[torch.Tensor]):
        """
        快速步进（提速②）：TS 整环单步（_RWKE_TS步进模块）。

        - 数值与旧路径同构（逐算子镜像同一官方算子），跨核实现差异为 fp16 噪声级
          （30 步 logits 最大差 ~0.14，见 _研究_提速/profile_与原型A.py）；
        - TS List[Tensor] 按值传递 → 新状态列表函数式返回，调用方（生成循环 L822）
          本就逐帧重绑 state，兼容无侵入；
        - hidden = 末N层捕获 残差流均值，与旧路径捕获语义一致。
        """
        if self._ts模块 is None or self._ts权重版本 != self._权重版本:
            self._构建ts步进()
        # 提速⑤：CUDA 图模式（状态自续 + 零逐帧拷贝）；CPU 设备 / 图未就绪时走无图 TS
        if self._图 is None or self._图权重版本 != self._权重版本:
            if self.设备.type == "cuda":
                self._构建图步进()
            else:
                self._图 = False  # 标记：本引擎不适用图模式
        if self._图 is False:
            self._步进token缓冲.fill_(int(token_id))  # 提速④：复用 0 维缓冲
            logits, 新状态, hidden = self._ts模块(self._步进token缓冲, state)
            return logits.unsqueeze(0), 新状态, hidden
        # 图模式：状态不同源（新轮/软重锚/外部传入）→ copy-in 一次；同源（上一步图输出）
        # → 零拷贝直接 replay（图内 copy 回写自续）
        if state is not self._图状态列表:
            for i in range(len(state)):
                self._图状态列表[i].copy_(state[i])
        self._步进token缓冲.fill_(int(token_id))
        self._图.replay()
        # hidden/logits 必须 clone：图缓冲在下次 replay 时被覆写（池/回调可能持有引用）
        return self._图logits缓冲.unsqueeze(0), self._图状态列表, self._图hidden缓冲.clone()

    def _构建图步进(self) -> None:
        """捕获 CUDA Graph：TS 整环步进 + 状态回写（图内自续）。"""
        self._图状态列表 = [t.clone() for t in self.模型.generate_zero_state()]
        if self._图状态列表 and self._图状态列表[0].device != self.设备:
            self._图状态列表 = [t.to(self.设备) for t in self._图状态列表]
        self._图logits缓冲 = torch.zeros(self.vocab_size, dtype=self.权重dtype, device=self.设备)
        self._图hidden缓冲 = torch.zeros(self.n_embd, dtype=self.权重dtype, device=self.设备)
        self._步进token缓冲.fill_(0)
        ts模块 = self._ts模块

        def _图体():
            lg, ns, hd = ts模块(self._步进token缓冲, self._图状态列表)
            for i in range(len(self._图状态列表)):
                self._图状态列表[i].copy_(ns[i])
            self._图logits缓冲.copy_(lg.view(-1))
            self._图hidden缓冲.copy_(hd)

        # 侧流预热（图池分配就位）→ 捕获
        torch.cuda.synchronize()
        预热流 = torch.cuda.Stream()
        预热流.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(预热流):
            for _ in range(3):
                _图体()
        torch.cuda.current_stream().wait_stream(预热流)
        torch.cuda.synchronize()
        try:
            self._图 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._图):
                _图体()
        except Exception as e:
            warnings.warn(f"CUDA Graph 捕获失败（回退无图 TS 步进）：{type(e).__name__}: {e}")
            self._图 = False
            torch.cuda.synchronize()
            return
        self._图权重版本 = self._权重版本

    # ──────────────────────────────────────────────
    # 注入 + 捕获（V1架构.py L428-477 逐行等价）
    # ──────────────────────────────────────────────

    def _池追加(self, hidden: torch.Tensor, 权重: float) -> None:
        """提速⑥：捕获向量写入 GPU 池镜像（无 CPU 同步；容量满移除最旧）。"""
        cap = self._池V.shape[0]
        if self._池n >= cap:
            self._池V[:-1] = self._池V[1:]
            self._池W[:-1] = self._池W[1:]
            self._池TS[:-1] = self._池TS[1:]
            self._池n = cap - 1
        self._池V[self._池n] = hidden.detach().float()
        self._池W[self._池n] = 权重
        self._池TS[self._池n] = self.回响池.当前步数
        self._池n += 1

    def _池GPU质心(self) -> Optional[torch.Tensor]:
        """提速⑥：向量化衰减质心（全 GPU，无同步）。

        语义对齐 回响池.计算质心()：有效权重 w_i = W_i × exp(-γ·(t−ts_i))，
        低于 eviction_threshold 的项不参与（对应池的衰减淘汰），返回 None 表示空池。
        """
        n = self._池n
        if n == 0:
            return None
        dt = (self.回响池.当前步数 - self._池TS[:n]).float()
        w = self._池W[:n] * torch.exp(-self.γ衰减 * dt)
        w = w * (w > self.回响池.eviction_threshold).float()  # 对齐池衰减淘汰
        # 全 GPU 无同步：总权重≤0 的退化情形经 clamp 得全零质心（注入偏置为 0，语义等价）
        总权重 = w.sum().clamp_min(1e-12)
        return (w.unsqueeze(1) * self._池V[:n]).sum(dim=0) / 总权重

    def _同步回响池(self) -> None:
        """提速⑥：轮末把 GPU 镜像批量同步回 回响池 对象（一次 D2H；保持池对象权威完整，
        历史/信息()/情感展示读取不受影响）。存活判定与池衰减淘汰语义一致。"""
        池 = self.回响池
        n = self._池n
        if n == 0:
            池.向量列表 = []
            池.权重列表 = []
            池.时间戳列表 = []
            池.轮次列表 = []
            池._质心缓存 = None
            return
        t = 池.当前步数
        w_eff = (self._池W[:n] * torch.exp(-self.γ衰减 * (t - self._池TS[:n]).float())).cpu()
        V = self._池V[:n].cpu()
        TS = self._池TS[:n].cpu()
        存活 = [i for i in range(n) if float(w_eff[i]) > 池.eviction_threshold]
        池.向量列表 = [V[i] for i in 存活]
        池.权重列表 = [float(self._池W[i]) for i in 存活]
        池.时间戳列表 = [int(TS[i]) for i in 存活]
        池.轮次列表 = [池.轮次ID] * len(存活)
        池._质心缓存 = None

    @torch.no_grad()
    def _注入偏置(self, logits: torch.Tensor) -> torch.Tensor:
        """回响池质心经随机投影映射到 logits 空间作为偏置（阶段λ选择结构保留）。"""
        if not self.启用注入:
            return logits

        if self.当前阶段 == "正文" and self.正文阶段λ == 0.0:
            return logits

        质心 = self._池GPU质心()  # 提速⑥：GPU 镜像向量化质心（原 池.计算质心() CPU O(n) 且每步重算）
        if 质心 is None:
            return logits

        if self.当前阶段 == "思考" and self.思考阶段λ is not None:
            有效λ = self.思考阶段λ
        elif self.当前阶段 == "正文":
            有效λ = self.正文阶段λ
        else:
            有效λ = self.λ强度

        偏置 = 质心 @ self.投影矩阵GPU  # 提速①：GPU 常驻副本（原 self.投影矩阵.to(self.设备) 每步 537MB 搬运）
        偏置 = 偏置 * 有效λ

        if self.偏置归一:
            # RMS 归一：偏置强度有界（0.5×λ），与池质心范数漂移解耦；方向（情感梯度）保留
            偏置 = 偏置 / (偏置.std() + 1e-6) * (abs(有效λ) * 0.5)
        if self.注入衰减:
            偏置 = 偏置 * self._注入衰减系数

        if self.eos保护:
            # eos 维度去偏置：情感回响不直接改变终止行为
            # （RWKV7-World-1.5B 实测：任意 λ>0 会抬高 eos 排名导致"注入即停"）
            偏置[int(self.eos_token_id)] = 0.0

        return logits + 偏置.unsqueeze(0)

    @torch.no_grad()
    def _捕获回响(self, logits: torch.Tensor, hidden_state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        捕获准备（提速④）：计算 不确定性 = 1 - max_prob 的 GPU 张量并返回，【不同步】。

        τ 判定与入池由生成循环在采样同步点之后提交（float(不确定性t) 届时零等待），
        数值语义与旧实现（本方法内 .item() + 立即入池）等价，省一次流水线同步泡。
        启用捕获=False 或 hidden_state=None（预填充步）返回 None。
        """
        if not self.启用捕获:
            return None
        if hidden_state is None:
            return None

        probs = F.softmax(logits.float(), dim=-1)
        return probs.max()

    # ──────────────────────────────────────────────
    # 阶段切换检测（V1架构.py L405-422 结构保留）
    # ──────────────────────────────────────────────

    def _检测阶段切换(self, 新token_id: int) -> bool:
        """World 模型思考标记对为空 → 恒无切换；保留结构与 V1 对齐。

        提速④：无思考标记的模型（World 系）直接短路，省每步 2 次 TRIE decode
        （本方法 + 已解码文本各一次的冗余路径）。"""
        if not self.思考标记对 or self.思考标记对 == ("", ""):
            return False
        新文本 = self.分词器.decode([新token_id])
        self.当前阶段文本 = getattr(self, "当前阶段文本", "") + 新文本

        if (
            self.当前阶段 == "思考"
            and self.思考标记对[1] in self.当前阶段文本
        ):
            self.当前阶段 = "正文"
            return True
        return False

    # ──────────────────────────────────────────────
    # 采样链（V1架构.py L563-587 逐行等价）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _采样(self, logits: torch.Tensor, 温度: float, top_p: float, top_k: int, 快速: bool = False) -> torch.Tensor:
        """温度缩放 → top-p（stable sort + scatter 反排）→ top-k（阈值过滤）→ multinomial。

        快速=True（提速⑧，交互口径）：top-k 先行，只在 top-k 候选内做 top-p——
        省去 65536 全表 stable sort+cumsum+scatter（~3ms → ~1ms）。语义差异：
        top-p 截断阈值相对 top-k 质量归一（截断点略提前），认证复现请用快速=False。
        """
        logits = logits / 温度

        if 快速 and top_k > 0:
            k = min(top_k, logits.size(-1))
            topv, topi = torch.topk(logits.view(-1), k)
            sp = F.softmax(topv, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            保留 = (cum - sp) <= top_p  # 与原实现 shift 语义一致：保留使前缀累计 ≤ p
            保留[0] = True
            sp = sp * 保留.float()
            idx = torch.multinomial(sp, num_samples=1)
            return topi[idx].unsqueeze(0)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove,
            )
            logits[indices_to_remove] = float("-inf")

        if top_k > 0:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            threshold = top_k_values[:, -1].unsqueeze(-1)
            logits[logits < threshold] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ──────────────────────────────────────────────
    # State 挂载 / 导出（Task 4.5，经 状态接口）
    # ──────────────────────────────────────────────

    def _对齐状态(self, state: Optional[List[torch.Tensor]]) -> Optional[List[torch.Tensor]]:
        """张量组搬到模型设备并核对 dtype：att/ffn 前值 → 权重 dtype，att_kv → fp32。"""
        if state is None:
            return None
        对齐后: List[torch.Tensor] = []
        for i in range(self.n_layer):
            对齐后.append(state[i * 3 + 0].to(self.设备).to(self.权重dtype))
            对齐后.append(state[i * 3 + 1].to(self.设备).float())
            对齐后.append(state[i * 3 + 2].to(self.设备).to(self.权重dtype))
        return 对齐后

    def 挂载日记状态(self, 状态文件路径: str, 期望模型标识: Optional[str] = None) -> dict:
        """
        加载 .pth 日记 State 并挂载（每次生成前克隆，不污染源）。

        经 状态接口.加载状态 自动分派格式，并做模型标识校验（防 A 模型 State 挂到 B 模型）。
        返回 .pth 内的元数据 dict。
        """
        状态对象, 元数据 = 加载状态(
            状态文件路径,
            期望模型标识 if 期望模型标识 is not None else self.模型标识,
            设备=None,
        )
        if 状态对象.原始状态 is not None and len(状态对象.原始状态) != self.n_layer * 3:
            raise ValueError(
                f"State 层数不匹配：文件含 {len(状态对象.原始状态)} 张量，"
                f"模型需要 n_layer*3 = {self.n_layer * 3}"
            )
        self._挂载state = self._对齐状态(状态对象.原始状态)
        return 元数据

    def 清除挂载状态(self) -> None:
        """卸载日记 State，恢复零态起步。"""
        self._挂载state = None

    def 挂载状态(self, state) -> None:
        """
        直接挂载一份已对齐的 State（不落盘），语义等价于 挂载日记状态。

        供实验/预热复用：如"人设卡预热轮"跑完后把结果 State 挂上，
        此后每次新会话都从"已是角色"的状态起算。
        """
        self._挂载state = self._对齐状态(state)

    def 导出状态(self, 路径: str) -> dict:
        """把最近一次生成结束时的 State 存为 .pth（CPU 张量 + 元数据；挂载适配器时带适配器标识）。"""
        if self._末state is None:
            raise RuntimeError("尚未执行过生成，无状态可导出")
        cpu_state = [t.detach().to("cpu") for t in self._末state]
        附加元数据 = {"适配器标识": self.适配器标识} if self.适配器标识 else None
        return RWKV状态(cpu_state).保存(
            路径, self.模型标识, self.hidden_dim, self.n_layer, 附加元数据=附加元数据
        )

    # ──────────────────────────────────────────────
    # 架构级身份记忆挂载（状态注入.py：槽位锁定 + 每步回注）
    # ──────────────────────────────────────────────

    def 挂载身份记忆(self, 记忆, α: Optional[float] = None) -> dict:
        """
        挂载状态级联想记忆（由 状态注入.构造身份记忆 产出）。

        与"日记 State 整体灌入"的区别：只占用少量 (层, 头) 槽，
        槽里是 key→value 联想矩阵（内容寻址），且每步生成后被回注
        `S_slot ← α·S_slot + (1-α)·S_id`，人设不被对话擦除。
        """
        self._身份记忆 = 记忆
        if α is not None:
            self.身份回注系数 = float(α)
        return {"槽数": len(记忆), "层": 记忆.层集合, "头": 记忆.头集合,
                "回注系数": self.身份回注系数}

    def 解除身份记忆(self) -> None:
        """卸载身份记忆（后续生成不再回注，槽位回归普通对话语义）。"""
        self._身份记忆 = None

    def 身份衰减(self) -> dict:
        """探测当前末态里身份槽相对基准的漂移（0=完全锁定）。"""
        if self._身份记忆 is None or self._末state is None:
            return {"槽数": 0, "平均漂移": 0.0, "最大漂移": 0.0, "方向余弦": 1.0}
        import 状态注入 as _状态注入
        return _状态注入.探测(self, self._末state, self._身份记忆)

    # ──────────────────────────────────────────────
    # 输出链路挂载（导向链.py：情感/身份/复制 三路加性先验）
    # ──────────────────────────────────────────────

    def 挂载导向链(self, 配置: Optional[dict] = None) -> dict:
        """
        挂载输出导向链：每轮采样链前对 logits 叠加若干方向的加性先验
        `logits += Σ β_i·m_i`（三路语义见 导向链.py）。

        配置形如 `{"情感": {"强度": 1.5}, "身份": {"强度": 1.2}, "复制": {"强度": 1.2}}`；
        传 None 时默认挂"情感"一路（与旧的 情感导向 行为等价）。
        """
        import 导向链 as _导向链模块
        if 配置 is None:
            配置 = {"情感": {"强度": 1.5}}
        self._导向链 = _导向链模块.导向链(self, 配置)
        return {"构件": {c.名称: c.强度 for c in self._导向链.构件},
                "覆盖": {c.名称: c.覆盖统计() for c in self._导向链.构件}}

    def 解除导向链(self) -> None:
        """卸载输出导向链（后续生成不再做任何采样级加性偏置）。"""
        self._导向链 = None

    def 导向诊断(self) -> dict:
        """最近一轮导向链生效情况（情感标签、复制词、各路强度）。"""
        return dict(self._导向链.最近诊断) if self._导向链 is not None else {}

    def 本轮情感(self) -> Optional[str]:
        """最近一轮生效的情感标签（供实验记录）。"""
        return self._本轮情感

    # ──────────────────────────────────────────────
    # 对外主接口：单轮生成（Task 4.4）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 生成(
        self,
        消息: str,
        种子: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        温度: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        重复惩罚: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        停止标记: Optional[str] = None,
        logits_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        轮次回调: Optional[Callable[[int, object], None]] = None,
        token回调: Optional[Callable[[str], None]] = None,
        快速采样: Optional[bool] = None,
        保留池: bool = False,
        延续对话: bool = False,
        情感: Optional[str] = None,
    ) -> str:
        """
        单轮对话生成：聊天模板化 → 预填充 → 自回归（惩罚→注入→回调→捕获→采样→推进）。

        生成循环顺序与 V1架构.py L537-605 逐步一致；采样参数缺省读统一采样协议。

        Parameters
        ----------
        保留池 : bool
            False 时生成前清空回响池（独立单轮实验语义）；True 时跨轮累积（连续对话语义）。
        延续对话 : bool
            True 时从上一轮结束 State 继续（多轮对话）：prompt 仅追加本轮
            "\\n\\nUser: {消息}\\n\\nAssistant:"，历史上下文由 State 天然携带
            （O(1) 不随轮数增长——RWKV 超长记忆核心卖点）。需 保留池=True 配合。
        轮次回调 : Callable[当前步数, 回响池]
            每步推进后调用（与 V1 一致，供滑动窗口策略 / 实验记录）。
        logits_callback : Callable[步数, 注入后logits]
            每步注入后、捕获前调用（与 V1 一致，供语义熵计算 / 实验记录）。

        Returns
        -------
        str
            回复文本（截断停止标记之后内容，去除首尾空白）。
        """
        协议 = self.采样协议
        种子 = 协议.get("种子", 42) if 种子 is None else 种子
        max_new_tokens = 协议.get("最大新Token", 256) if max_new_tokens is None else max_new_tokens
        温度 = 协议.get("温度", 1.0) if 温度 is None else 温度
        top_p = 协议.get("top_p", 0.9) if top_p is None else top_p
        top_k = int(协议.get("top_k", 50)) if top_k is None else top_k
        重复惩罚 = 协议.get("重复惩罚", 1.05) if 重复惩罚 is None else 重复惩罚
        快速采样 = bool(协议.get("快速采样", False)) if 快速采样 is None else bool(快速采样)
        eos_token_id = self.eos_token_id if eos_token_id is None else eos_token_id
        停止标记 = self.聊天模板.get("停止标记", "\n\nUser:") if 停止标记 is None else 停止标记

        if 种子 is not None:
            torch.manual_seed(int(种子))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(种子))

        if not 保留池:
            self.清空回响()  # 提速⑥：池对象 + GPU 镜像同步清

        # ---- 输出链路：本轮目标情感 + （复制导向）用户话语编码 ----
        if self._导向链 is not None:
            self._本轮情感 = self._导向链.准备轮(消息, 情感)
        else:
            self._本轮情感 = 情感

        self.当前阶段 = "思考"
        self.当前阶段文本 = ""
        self.当前hidden_state = None

        # ---- 聊天模板化（Task 4.2；延续对话=多轮：仅追加本轮，State 携带历史）----
        重锚发生 = False
        if 延续对话 and self._末state is not None:
            追加模板 = self.聊天模板.get("多轮追加", "\n\nUser: {消息}\n\nAssistant:")
            prompt_ids = self.分词器.encode(追加模板.format(消息=消息))
            惩罚集合 = self._会话token集合  # 会话级：跨轮累积防复读

            if self._会话token数 + len(prompt_ids) > self.上下文预算:
                # ---- 软重锚：超训练窗口预算，回到挂载/零态并重填最近轮 ----
                # （挂载日记 State 时人设记忆随 State 恢复——挂日记的核心价值）
                if self._挂载state is not None:
                    state = [t.clone() for t in self._挂载state]
                else:
                    state = self.模型.generate_zero_state()
                if self._最近轮token:
                    _, state = self._预填充(list(self._最近轮token), state)
                self._会话token数 = len(self._最近轮token)
                重锚发生 = True
            else:
                state = [t.clone() for t in self._末state]
        else:
            self._会话token集合 = set()  # 新会话：清空
            惩罚集合 = self._会话token集合
            self._会话token数 = 0
            self._最近轮token = []
            重锚发生 = True  # 全新预填充
            模板 = self.聊天模板.get("单轮", "User: {消息}\n\nAssistant:")
            prompt_ids = self.分词器.encode(模板.format(消息=消息))
            # ---- 初始 State：挂载克隆 or 零态 ----
            if self._挂载state is not None:
                state = [t.clone() for t in self._挂载state]
            else:
                state = self.模型.generate_zero_state()

        # 设备对齐守卫：rwkv pip 的 generate_zero_state 依赖模块级全局 DEVICE，
        # 多引擎并存时该全局可能被其他实例的构造改写（如 CPU 回退），此处强制对齐
        if state and state[0].device != self.设备:
            state = [t.to(self.设备) for t in state]

        # ---- 身份记忆装订 + 回注钩子（无挂载时 _身份注入=None，路径逐位不变）----
        _身份注入 = None
        if self._身份记忆 is not None:
            import 状态注入 as _身份注入
            _身份注入.写入(self, state, self._身份记忆)  # 槽位硬装订基准（会话起点确定）

        # 提速③：惩罚集合 → GPU 缓冲（每轮一次 H2D；集合与缓冲同序无需一致，
        # 向量化除法对集合成员逐个生效，与原 Python 循环数值等价）
        if 重复惩罚 != 1.0 and 惩罚集合:
            集合列表 = list(惩罚集合)[: self._惩罚缓冲.numel()]
            self._惩罚缓冲[: len(集合列表)] = torch.tensor(集合列表, dtype=torch.int64)
            self._惩罚数 = len(集合列表)
        else:
            self._惩罚数 = 0

        # ---- 预填充（prompt 末位 logits；state 原地演进）：分派见 _预填充 ----
        logits, state = self._预填充(prompt_ids, state)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # 统一 [1, vocab]（与 V1 张量形状约定一致）
        if _身份注入 is not None:
            # prompt 预填充会往槽里写对话 token，立即回注（几何泄漏，人设不被冲淡）
            _身份注入.回注(self, state, self._身份记忆, self.身份回注系数)

        生成token序列: List[int] = []
        已解码文本 = ""
        连续计数: dict = {}      # token → 连续出现次数（复读压制用）
        上一token: Optional[int] = None

        for 步 in range(int(max_new_tokens)):
            # 注入衰减系数（进度调度：前强后弱，下限 0.4）
            if self.注入衰减:
                self._注入衰减系数 = max(0.4, 1.0 - 0.6 * 步 / max(int(max_new_tokens), 1))

            # (0) 重复惩罚：对已生成 token 集合 logits /= penalty（V1 L548-550；
            #     延续对话时集合为会话级累积，防止跨轮复读）
            #     提速③：GPU 缓冲向量化（原 Python 逐 token 循环 O(会话长度)）
            if self._惩罚数:
                logits.view(-1)[self._惩罚缓冲[: self._惩罚数]] /= 重复惩罚

            # (0.5) 连续重复压制：同一 token 连续出现 3 次起 logits 减半
            #      （保留"哈哈"等 2 连叠词，专破注入偏置/长会话下的复读机 attractor）
            if 上一token is not None and 连续计数.get(上一token, 0) >= 3:
                logits[0, 上一token] /= 2.0

            # (0.6) 开场 eos 禁步：前 eos禁步数 步不采样 eos（防挂载 State 下"开场即终止"）
            if 步 < self.eos禁步数:
                logits[0, int(self.eos_token_id)] = float("-inf")

            # (1) 注入：回响池质心偏置（V1 L553）
            logits = self._注入偏置(logits)

            # (1.5) 输出链路：情感/身份/复制 三路加性先验（导向链.py）
            if self._导向链 is not None:
                logits = self._导向链.应用(logits, {"情感": self._本轮情感})

            # 外部日志回调（语义熵 / 实验记录；V1 L556-557）
            if logits_callback is not None:
                logits_callback(步, logits)

            # (2) 捕获准备：hidden_state 不确定性计算（GPU 排队，不同步；V1 L560；
            #     预填充步 hidden=None 跳过）。提交（τ 判定 + 入池）延后到采样同步之后，
            #     与下一步注入的先后顺序不变——数值语义等价，省一次流水线同步泡
            不确定性t = self._捕获回响(logits, self.当前hidden_state)

            # (3) 采样链（V1 L563-587：温度 → top-p → top-k → multinomial）
            下一个token = self._采样(logits, float(温度), float(top_p), int(top_k), 快速=快速采样)
            token_id = int(下一个token.reshape(-1)[0].item())  # 唯一同步点
            if 不确定性t is not None:
                不确定性 = 1.0 - float(不确定性t)  # GPU 已排空，此次读取零等待
                if 不确定性 > self.τ阈值:
                    self._池追加(self.当前hidden_state, 不确定性)  # 提速⑥：GPU 镜像（池对象轮末批量同步）

            # (4) 阶段检测（V1 L590；World 模型恒"思考"）
            self._检测阶段切换(token_id)

            生成token序列.append(token_id)
            惩罚集合.add(token_id)
            # 提速③：新 token 追加进惩罚 GPU 缓冲（集合去重 → 长度增长即新 token）
            if self._惩罚数 < self._惩罚缓冲.numel() and len(惩罚集合) > self._惩罚数:
                self._惩罚缓冲[self._惩罚数] = token_id
                self._惩罚数 += 1
            增量文本 = self.分词器.decode([token_id])
            已解码文本 += 增量文本
            if token回调 is not None and 增量文本:
                token回调(增量文本)  # 真流式：增量文本回调（decode 已在本步完成，零额外开销）
            连续计数[token_id] = 连续计数.get(token_id, 0) + 1 if token_id == 上一token else 1
            上一token = token_id

            # (5) 推进回响池步数（V1 L596）
            self.回响池.推进()

            # (6) 轮次回调（V1 L599-600）
            if 轮次回调 is not None:
                轮次回调(self.回响池.当前步数, self.回响池)

            if token_id == eos_token_id:
                break
            if 停止标记 and 停止标记 in 已解码文本:
                break

            # (7) 前向下一个 token（自实现逐层前向，产出下一步 logits 与 hidden_state）
            logits, state, self.当前hidden_state = self._前向单步(token_id, state)
            if _身份注入 is not None:
                _身份注入.回注(self, state, self._身份记忆, self.身份回注系数)

        self._末state = state
        # 会话计数与最近轮缓冲（软重锚用；缓冲保留最近约 2 轮量）
        self._会话token数 += len(prompt_ids) + len(生成token序列)
        self._最近轮token = (self._最近轮token + list(prompt_ids) + 生成token序列)[-320:]
        self._同步回响池()  # 提速⑥：GPU 镜像 → 池对象批量同步（一次 D2H）

        # ---- 解码后处理：剔除 eos（无文本表示，混入会显示为乱码）、截断停止标记、去回复后缀 ----
        文本 = self.分词器.decode([t for t in 生成token序列 if t != int(self.eos_token_id)])
        for 停 in (停止标记, "\n\nAssistant:"):  # 后者：模型自续对话标记（自问自答污染）
            if 停 and 停 in 文本:
                文本 = 文本.split(停)[0]
        回复后缀 = self.聊天模板.get("回复后缀", "\n\n")
        if 回复后缀 and 文本.endswith(回复后缀):
            文本 = 文本[: -len(回复后缀)]
        return 文本.strip()

    def 清空回响(self) -> None:
        """清空回响池（对象 + GPU 镜像同步清）。"""
        self.回响池.清空()
        self._池n = 0

    # ──────────────────────────────────────────────
    # 观测辅助
    # ──────────────────────────────────────────────

    def 信息(self) -> str:
        """引擎概况一行摘要（调试用）。"""
        return (
            f"RWKE引擎(标识={self.模型标识}, 策略='{self.加载策略}', 设备={self.设备}, "
            f"dtype={self.权重dtype}, n_layer={self.n_layer}, n_embd={self.n_embd}, "
            f"vocab={self.vocab_size}, λ={self.λ强度}, γ={self.γ衰减}, τ={self.τ阈值}, "
            f"投影种子={self.采样协议.get('投影种子', 42)}, 末N层={self.末N层捕获}, "
            f"池大小={self.回响池.大小}, 挂载State={'有' if self._挂载state is not None else '无'}, "
            f"适配器={self.适配器标识 or '无'})"
        )
