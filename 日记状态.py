# -*- coding: utf-8 -*-
"""
RWKE 日记状态模块
=================

将角色日记"灌入" RWKV-7 State，产出可挂载到引擎的日记状态文件：

    日记文件(.json/.jsonl/.md) → World tokenizer 编码 → 分段流式预填充
    （每段一次 forward_seq，State O(1) 固定尺寸，显存与日记长度无关）
    → RoleB_状态_<模型标识>.pth（经 状态接口 序列化）

与 Transformer 路线的本质差异：KV 缓存预填充需 O(n) 显存且受上下文窗口限制；
RWKV State 预填充显存恒定，日记长度只影响预填充耗时。

用法
----
    # 对矩阵内模型生成日记状态（需已下载模型）
    python 日记状态.py --模型 RWKV7-World-0.4B
    python 日记状态.py --模型 RWKV7-World-1.5B --日记 <路径> --段大小 512

    # 无模型解析自检（只验证日记解析与分词，不加载模型）
    python 日记状态.py --自检解析

日记格式兼容
------------
- .json ：dict 含"日记列表"（完整导出格式，见 元数据/人设/日记列表），或直接是日记条目 list；
- .jsonl：每行一个 JSON 对象（日记条目或含"正文"字段的对象）；
- .md   ：以 '# ' / '## ' 标题分节，每节为一篇（首行=标题，其余=正文）。

条目取"标题 + 正文"拼接；其余字段（年龄/主题等）仅作统计输出。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 复用 rwke引擎 的环境设置（RWKV_V7_ON / CUDA 降级）、模型工厂与分词器来源
import rwke引擎 as 引擎模块
from rwke引擎 import RWKV, TRIE_TOKENIZER  # noqa: E402
from 状态接口 import RWKV状态  # noqa: E402

_RWKE_DIR = Path(__file__).resolve().parent

# 默认日记来源（tasks.md 指定；两处同级副本取其一，存在即用）
_默认日记候选 = [
    _RWKE_DIR.parent / "emocompanion-ai-emotion" / "最后版本" / "数据" / "日记" / "RoleB_日记.json",
    _RWKE_DIR.parent / "最后版本！" / "数据" / "日记" / "RoleB_日记.json",
]


# ======================================================================
# 日记解析（.json / .jsonl / .md 兼容）
# ======================================================================

def _条目转文本(条目: dict) -> str:
    """日记条目 → 预填充文本片段（标题 + 正文）。"""
    标题 = str(条目.get("标题", "")).strip()
    正文 = str(条目.get("正文", "")).strip()
    if 标题 and 正文:
        return f"{标题}\n{正文}"
    return 正文 or 标题


def _人设转文本(人设: dict) -> str:
    """人设 dict → 角色自我介绍片段（空 dict 返回空串）。"""
    if not 人设:
        return ""
    部件 = []
    if 人设.get("姓名"):
        部件.append(f"我叫{人设['姓名']}")
    if 人设.get("性别"):
        部件.append(f"是{人设['性别']}孩" if 人设["性别"] in ("女", "男") else str(人设["性别"]))
    if 人设.get("出生年份"):
        部件.append(f"{人设['出生年份']}年出生")
    if 人设.get("出生地"):
        部件.append(f"来自{人设['出生地']}")
    if 人设.get("性格"):
        部件.append(f"性格{人设['性格']}")
    if 人设.get("人设描述"):
        部件.append(str(人设["人设描述"]))
    if 人设.get("关键经历"):
        部件.append(str(人设["关键经历"]))
    if not 部件:
        return ""
    return "这是一段关于我的自我介绍：" + "，".join(部件) + "。"


def _从dict提取(data: dict) -> tuple:
    """完整导出 dict → (人设文本, [日记条目dict...], 角色名)。"""
    人设文本 = _人设转文本(data.get("人设", {}) or {})
    角色名 = str((data.get("元数据", {}) or {}).get("角色名")
                 or (data.get("人设", {}) or {}).get("姓名") or "RoleB")
    列表 = data.get("日记列表", [])
    if not isinstance(列表, list):
        列表 = []
    return 人设文本, [e for e in 列表 if isinstance(e, dict)], 角色名


def 解析日记文件(路径: str) -> dict:
    """
    解析日记文件 → {'角色名': str, '人设文本': str, '片段列表': [str], '篇数': int}

    兼容 .json / .jsonl / .md 三种格式（按扩展名分派，.json 解析失败自动尝试逐行模式）。
    """
    p = Path(路径)
    if not p.exists():
        raise FileNotFoundError(f"日记文件不存在：{p}")
    原始 = p.read_text(encoding="utf-8", errors="replace")

    if p.suffix.lower() == ".md":
        片段列表: List[str] = []
        当前: List[str] = []
        for 行 in 原始.splitlines():
            if 行.startswith("# ") or 行.startswith("## "):
                if 当前:
                    片段列表.append("\n".join(当前).strip())
                当前 = [行.lstrip("# ").strip()]
            elif 当前 is not None:
                当前.append(行)
        if 当前:
            片段列表.append("\n".join(当前).strip())
        片段列表 = [s for s in 片段列表 if s]
        return {"角色名": p.stem, "人设文本": "", "片段列表": 片段列表, "篇数": len(片段列表)}

    if p.suffix.lower() == ".jsonl":
        条目列表: List[dict] = []
        for 行号, 行 in enumerate(原始.splitlines(), 1):
            行 = 行.strip()
            if not 行:
                continue
            try:
                对象 = json.loads(行)
            except json.JSONDecodeError:
                continue
            if isinstance(对象, dict):
                条目列表.append(对象)
            elif isinstance(对象, list):
                条目列表.extend(e for e in 对象 if isinstance(e, dict))
        人设文本, 角色名 = "", "RoleB"
        片段 = []
        for e in 条目列表:
            if "日记列表" in e:  # 整个导出 dict 恰好一行
                人设文本, 子列表, 角色名 = _从dict提取(e)
                片段.extend(_条目转文本(x) for x in 子列表 if _条目转文本(x))
            else:
                文本 = _条目转文本(e)
                if 文本:
                    片段.append(文本)
        return {"角色名": 角色名, "人设文本": 人设文本, "片段列表": 片段, "篇数": len(片段)}

    # ---- .json：完整导出 dict 或条目 list ----
    try:
        data = json.loads(原始)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败（{p}）：{e}")

    if isinstance(data, dict):
        人设文本, 条目列表, 角色名 = _从dict提取(data)
        片段 = [t for t in (_条目转文本(x) for x in 条目列表) if t]
        return {"角色名": 角色名, "人设文本": 人设文本, "片段列表": 片段, "篇数": len(片段)}

    if isinstance(data, list):
        片段 = [t for t in (_条目转文本(x) for x in data if isinstance(x, dict)) if t]
        return {"角色名": "RoleB", "人设文本": "", "片段列表": 片段, "篇数": len(片段)}

    raise ValueError(f"不支持的日记 JSON 顶层类型：{type(data).__name__}（{p}）")


def 组装预填充文本(解析结果: dict) -> str:
    """人设 + 各篇日记按顺序拼接（段间空行分隔）。"""
    部分 = []
    if 解析结果.get("人设文本"):
        部分.append(解析结果["人设文本"])
    部分.extend(解析结果["片段列表"])
    return "\n\n".join(部分)


# ======================================================================
# 分段流式预填充
# ======================================================================

def 流式预填充(模型, token列表: List[int], 段大小: int = 512, 日志前缀: str = "") -> List[torch.Tensor]:
    """
    分段流式预填充：token 列表切块，逐块 forward（forward_seq），State 原地演进。

    返回最终 State（扁平 List[n_layer*3]，与 O(1) 显存承诺一致）。
    """
    state = 模型.generate_zero_state()
    总数 = len(token列表)
    if 总数 == 0:
        return state

    段数 = (总数 + 段大小 - 1) // 段大小
    for 段号 in range(段数):
        起 = 段号 * 段大小
        止 = min(起 + 段大小, 总数)
        块 = token列表[起:止]
        _, state = 模型.forward(块, state)  # list 且 len>1 → forward_seq
        if 段数 > 1:
            print(f"  {日志前缀}预填充段 {段号 + 1}/{段数}（token {起}–{止}）", flush=True)
    return state


# ======================================================================
# 主流程：日记 → State .pth
# ======================================================================

def 生成日记状态(
    模型标识: str,
    日记路径: Optional[str] = None,
    输出目录: Optional[str] = None,
    段大小: int = 512,
) -> Path:
    """
    对矩阵内指定模型执行：解析日记 → 编码 → 流式预填充 → 保存 .pth。

    返回生成的 .pth 路径。
    """
    配置 = 引擎模块._读系统配置()
    矩阵 = 配置.get("模型矩阵", {})
    if 模型标识 not in 矩阵:
        raise KeyError(f"模型标识 '{模型标识}' 不在模型矩阵中：{list(矩阵)}")
    条目 = 矩阵[模型标识]

    日记p = Path(日记路径) if 日记路径 else next((c for c in _默认日记候选 if c.exists()), None)
    if 日记p is None:
        raise FileNotFoundError("未找到默认日记文件，请用 --日记 指定路径")
    print(f"[1/4] 解析日记：{日记p}")
    解析结果 = 解析日记文件(str(日记p))
    if not 解析结果["片段列表"]:
        raise ValueError(f"日记中未提取到任何正文片段：{日记p}")
    print(f"      角色名={解析结果['角色名']}，篇数={解析结果['篇数']}")
    文本 = 组装预填充文本(解析结果)
    print(f"      预填充文本总长：{len(文本)} 字符")

    print(f"[2/4] 加载模型：{模型标识}")
    模型路径 = Path(条目["路径"])
    if not 模型路径.is_absolute():
        模型路径 = _RWKE_DIR / 模型路径
    rwkv路径 = str(模型路径.with_suffix("")) if 模型路径.suffix == ".pth" else str(模型路径)

    推理环境 = 配置.get("推理环境", {})
    策略 = 推理环境.get("加载策略", "cuda fp16")
    回退策略 = 推理环境.get("CPU回退策略", "cpu fp32")
    try:
        模型 = RWKV(rwkv路径, 策略)
        实际策略 = 策略
    except Exception as e:
        print(f"      策略 '{策略}' 失败（{e}），回退 '{回退策略}'")
        模型 = RWKV(rwkv路径, 回退策略)
        实际策略 = 回退策略
    print(f"      加载策略：{实际策略}")

    print("[3/4] 编码与流式预填充")
    词表 = __import__("os").path.join(__import__("os").path.dirname(引擎模块._rwkv包.__file__),
                                      "rwkv_vocab_v20230424.txt")
    分词器 = TRIE_TOKENIZER(词表)
    token列表 = 分词器.encode(文本)
    print(f"      token 总数：{len(token列表)}，段大小：{段大小}")
    state = 流式预填充(模型, token列表, 段大小=段大小, 日志前缀="")

    print("[4/4] 保存日记状态")
    输出目录p = Path(输出目录) if 输出目录 else _RWKE_DIR / "states"
    输出目录p.mkdir(parents=True, exist_ok=True)
    输出路径 = 输出目录p / f"{解析结果['角色名']}_状态_{模型标识}.pth"
    RWKV状态([t.detach().to("cpu") for t in state]).保存(
        str(输出路径), 模型标识, int(条目["hidden_dim"]), int(条目["n_layer"])
    )
    print(f"      已保存：{输出路径}")
    print(f"      元数据：模型标识={模型标识}, hidden_dim={条目['hidden_dim']}, n_layer={条目['n_layer']}")
    return 输出路径


# ======================================================================
# CLI
# ======================================================================

def _自检解析(日记路径: Optional[str]) -> int:
    """无模型自检：日记解析 + World tokenizer 编码往返。"""
    日记p = Path(日记路径) if 日记路径 else next((c for c in _默认日记候选 if c.exists()), None)
    if 日记p is None:
        print("[自检] 未找到日记文件")
        return 1
    print(f"[自检] 解析：{日记p}")
    解析结果 = 解析日记文件(str(日记p))
    print(f"  角色名：{解析结果['角色名']}")
    print(f"  篇数：{解析结果['篇数']}")
    print(f"  人设文本：{解析结果['人设文本'][:60]}...")
    文本 = 组装预填充文本(解析结果)
    print(f"  预填充文本：{len(文本)} 字符")
    首片段 = 解析结果["片段列表"][0]
    print(f"  首片段预览：{首片段[:60].replace(chr(10), ' ')}...")

    分词器 = TRIE_TOKENIZER(__import__("os").path.join(
        __import__("os").path.dirname(引擎模块._rwkv包.__file__), "rwkv_vocab_v20230424.txt"))
    token列表 = 分词器.encode(文本)
    还原 = 分词器.decode(token列表)
    往返一致 = (还原.strip() == 文本.strip())
    print(f"  编码 token 数：{len(token列表)}，解码往返一致：{往返一致}")
    if not 往返一致:
        print(f"  往返差异预览：原={文本[:50]!r} 还原={还原[:50]!r}")
        return 1
    print("[自检] 日记解析与分词通过（模型预填充验证需先下载模型，由测试_冒烟.py 承接）")
    return 0


def main() -> int:
    解析器 = argparse.ArgumentParser(description="RWKE 日记状态：日记 → RWKV-7 State .pth")
    解析器.add_argument("--模型", default=None, help="模型矩阵标识（如 RWKV7-World-0.4B）")
    解析器.add_argument("--日记", default=None, help="日记文件路径（.json/.jsonl/.md）")
    解析器.add_argument("--输出目录", default=None, help="State 输出目录（默认 RWKE/states）")
    解析器.add_argument("--段大小", type=int, default=512, help="流式预填充每段 token 数（默认 512）")
    解析器.add_argument("--自检解析", action="store_true", help="只做日记解析+分词自检，不加载模型")
    参数 = 解析器.parse_args()

    try:
        if 参数.自检解析:
            return _自检解析(参数.日记)
        if not 参数.模型:
            print("错误：未指定 --模型（或使用 --自检解析）")
            return 2
        生成日记状态(参数.模型, 参数.日记, 参数.输出目录, 参数.段大小)
        return 0
    except Exception as e:
        print(f"错误：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
