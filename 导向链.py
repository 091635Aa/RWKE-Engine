# -*- coding: utf-8 -*-
"""
导向链：输出链路的底层可控化（情感 / 身份 / 复制 三路加性先验）
==============================================================

定位
----
改造前，RWKE 的输出只有两条"可控"通路：
  1. LoRA 适配器——训练期把语气/人设写进权重，**静态、不可逐轮指定**；
  2. 回响池质心偏置——无方向的整体漂移，**不针对具体维度**。
本模块补上第三层：**采样前的加性先验链**，每一步都对 logits 施加若干
方向明确、可逐轮切换的偏置，且全部发生在 top-p/top-k 之后生效的可能点之前
（即只改"倾向"，分布形状仍由模型自己决定）。

三路导向（都可独立开关，按顺序叠加）
------------------------------------
- 情感导向：目标情感语汇的 token 指示向量加性重加权。概率形式即 ETD 的离散
  特化 `p' ∝ p·exp(β·m) = p^(1-α)·q^α`（把情感先验 q 取成该情感语汇上的
  最大熵分布）；目标情感优先取显式指定，否则从**用户消息**语汇判定。
- 身份导向：自称词（角色名）的加性偏置——修的是"模型把自己叫成小圆/小猪/小猫"
  这一已观测到的身份漂移；只做正偏置、不动其它词，避免破坏自然表达。
- 复制导向：用户本轮话语的内容词（词表趋同 / lexical entrainment）加性偏置。
  注意 World 分词对中文是**严格逐字**的（实测 "萌妹"→['萌','妹']、"ViewerA"→['桑','榆']），
  因此不存在"多字词=内容词"的 token 级代理；改用**高频功能字停用表**做字符级内容词
  抽取（"的了呢吗吧啊呀"等与 ASCII 片段整体丢弃）。它修的是"模型复读自己的口癖模板、
  完全不理用户说了什么"这一硬伤——评估集提问里本就含锚点（如"天鹅之梦是什么呀"），
  抬高用户内容字即有方向地把回答拉回问题本身。

设计边界
--------
- 三路均为**加性、有界、可关**；强度 β 的单位是 logit（温度 1.0 下 β=1.5
  约等于把目标 token 的相对概率抬到 e^1.5≈4.5 倍），β>3 会明显压过模型自身
  的语义选择、语言质量下降，故默认值都取在 1.2–1.5。
- 不触碰 eos；不修改任何模型权重；不依赖任何外部模型或分类器。
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

import 一致性评估 as 评估器  # 复用 情感词组映射（与角色日记同一套情感语汇）

# ----------------------------------------------------------------------
# 语汇表
# ----------------------------------------------------------------------

# 角色语气词变体（按情感分组）。刻意不含"啊/那个"——二者已由 LoRA 饱和，
# 再抬会把"每百字口癖数"推过 8 的惩罚拐点；同样剔除"什么/为什么/怎么/问"这类
# 高频功能字组合（World 分词逐字切分，"什么"→['什','么'] 抬的是功能字，会引来乱问）。
语气扩展: Dict[str, List[str]] = {
    "开心": ["开心", "高兴", "笑", "哈哈", "嘻嘻", "暖暖", "热闹", "美", "乐", "嘿"],
    "好奇": ["好奇", "猜", "新鲜", "咦", "难道", "想知道"],
    "娇羞": ["脸红", "害羞", "耳朵", "偷偷", "贴贴", "软软", "人家", "讨厌", "去去去", "扭扭"],
    "热情": ["热情", "欢迎", "大家", "朋友", "帮忙", "帮我", "喊", "一起", "各位", "来啦"],
    "紧张": ["紧张", "慌", "心跳", "手心", "汗", "攥", "咳", "小兔子", "嗓子", "腿"],
    "期待": ["期待", "盼", "希望", "机会", "名额", "报名", "最后一天", "相信", "一定", "快点"],
}

# 自称词：修"把自己叫成小圆/小猪/小猫"的身份漂移
自称词表: List[str] = ["EmoCompanion"]

# 高频功能字停用表（字符级）。World 分词对中文逐字切分，无法用 token 形态区分
# 内容词与功能词，故以人工停用表近似：命中即丢，其余中文单字视为内容字。
# 刻意不含 啊/那 —— 它们是角色口癖，属"应保留的角色特征"而非应丢弃的功能字。
功能字 = set(
    "的了呢吗吧哦嗯是不我有你他她它们这那个和与跟就都也在很要会想能说着过被把给对从到"
    "么什怎样谁哪太还只再又更没每让做去来说去到好可以因为所以如果但是而且然后"
)


def 情感词表() -> Dict[str, List[str]]:
    """评估语汇 ∪ 语气扩展（去重保序）。"""
    表: Dict[str, List[str]] = {}
    for 情感, 词 in 评估器.情感词组映射.items():
        合并 = list(词)
        for w in 语气扩展.get(情感, []):
            if w not in 合并:
                合并.append(w)
        表[情感] = 合并
    for 情感, 词 in 语气扩展.items():
        表.setdefault(情感, list(词))
    return 表


def _编码词表(引擎, 词列表: Sequence[str], 掩掉eos: bool = True) -> torch.Tensor:
    """词列表 → vocab 维 0/1 指示向量（GPU 常驻）。"""
    ids = set()
    for 词 in 词列表:
        for tid in 引擎.分词器.encode(词):
            if 掩掉eos and tid == int(引擎.eos_token_id):
                continue
            ids.add(int(tid))
    向量 = torch.zeros(引擎.vocab_size, dtype=torch.float32)
    if ids:
        向量[torch.tensor(sorted(ids), dtype=torch.int64)] = 1.0
    return 向量.to(引擎.设备)


def _内容词权重(引擎, 文本: str) -> Tuple[torch.Tensor, List[str]]:
    """
    用户话语 → 内容字指示向量。

    World 分词对中文逐字切分（"萌妹"→['萌','妹']），token 形态不携带词频信息，
    故走**字符级**抽取：汉字单字命中 功能字 停用表则丢，其余保留；
    纯 ASCII 片段（"User" / ":" / 空格）与 eos 一并丢弃。
    """
    权重 = torch.zeros(引擎.vocab_size, dtype=torch.float32)
    if not 文本:
        return 权重.to(引擎.设备), []
    ids, 词 = [], []
    for tid in 引擎.分词器.encode(文本):
        tid = int(tid)
        if tid == int(引擎.eos_token_id):
            continue
        面 = 引擎.分词器.decode([tid]).strip()
        if not 面 or not any("\u4e00" <= ch <= "\u9fff" for ch in 面):
            continue  # 空片 / 纯 ASCII（模板词）/ 纯标点（，？！）
        if all(ch in 功能字 for ch in 面):
            continue  # 整片都是功能字
        ids.append(tid)
        词.append(面)
    if ids:
        权重[torch.tensor(sorted(set(ids)), dtype=torch.int64)] = 1.0
    return 权重.to(引擎.设备), 词


# ----------------------------------------------------------------------
# 导向器
# ----------------------------------------------------------------------

class 导向器:
    名称 = "导向器"

    def __init__(self, 引擎, 强度: float):
        self.引擎 = 引擎
        self.强度 = float(强度)

    def 偏置(self, logits: torch.Tensor, 上下文: dict) -> torch.Tensor:
        return logits


class 情感导向器(导向器):
    名称 = "情感"

    def __init__(self, 引擎, 强度: float = 1.5):
        super().__init__(引擎, 强度)
        self.词表 = 情感词表()
        self._向量 = {情感: _编码词表(引擎, 词) for 情感, 词 in self.词表.items()}

    def 覆盖统计(self) -> Dict[str, int]:
        return {k: int(v.sum().item()) for k, v in self._向量.items()}

    def 预测(self, 文本: str) -> Tuple[Optional[str], List[str]]:
        """语汇命中判定：返回 (情感标签, 命中词)；无命中 (None, [])。"""
        if not 文本:
            return None, []
        最佳, 最佳命中 = None, []
        for 情感, 词列表 in self.词表.items():
            命中 = [w for w in 词列表 if w in 文本]
            if len(命中) > len(最佳命中):
                最佳, 最佳命中 = 情感, 命中
        return 最佳, 最佳命中

    def 偏置(self, logits, 上下文):
        情感 = 上下文.get("情感")
        向量 = self._向量.get(情感)
        if 向量 is None or self.强度 == 0.0:
            return logits
        logits[0] = logits[0] + 向量.to(dtype=logits.dtype) * self.强度
        return logits


class 身份导向器(导向器):
    名称 = "身份"

    def __init__(self, 引擎, 强度: float = 1.2):
        super().__init__(引擎, 强度)
        self._向量 = _编码词表(引擎, 自称词表)

    def 覆盖统计(self) -> Dict[str, int]:
        return {"自称": int(self._向量.sum().item())}

    def 偏置(self, logits, 上下文):
        if self.强度 == 0.0:
            return logits
        logits[0] = logits[0] + self._向量.to(dtype=logits.dtype) * self.强度
        return logits


class 复制导向器(导向器):
    名称 = "复制"

    def __init__(self, 引擎, 强度: float = 1.2):
        super().__init__(引擎, 强度)
        self._缓存文本: Optional[str] = None
        self._缓存向量: Optional[torch.Tensor] = None
        self._缓存词: List[str] = []

    def 准备(self, 用户消息: str) -> None:
        if 用户消息 == self._缓存文本:
            return
        self._缓存文本 = 用户消息
        向量, 词 = _内容词权重(self.引擎, 用户消息 or "")
        self._缓存向量, self._缓存词 = 向量, 词

    def 覆盖统计(self) -> Dict[str, int]:
        """最近一轮从用户话语抽出的内容字数（未 准备 时为 0）。"""
        return {"内容字": len(self._缓存词)}

    def 偏置(self, logits, 上下文):
        if self.强度 == 0.0 or self._缓存向量 is None:
            return logits
        logits[0] = logits[0] + self._缓存向量.to(dtype=logits.dtype) * self.强度
        return logits


# ----------------------------------------------------------------------
# 导向链
# ----------------------------------------------------------------------

class 导向链:
    """
    有序导向器集合：每步 `logits ← logits + Σ β_i · m_i`。

    诊断：`最近诊断` 记录本轮各路的生效情况（情感标签、复制词、覆盖 token 数），
    供实验记录与"哪一路在起作用"的归因。
    """

    构件表 = {"情感": 情感导向器, "身份": 身份导向器, "复制": 复制导向器}

    def __init__(self, 引擎, 配置: Optional[dict] = None):
        self.引擎 = 引擎
        self.构件: List[导向器] = []
        self.构件配置: Dict[str, dict] = dict(配置 or {})
        for 名, 参数 in self.构件配置.items():
            类 = self.构件表.get(名)
            if 类 is None:
                raise KeyError(f"未知导向器 '{名}'，可选：{list(self.构件表)}")
            self.构件.append(类(引擎, 强度=float((参数 or {}).get("强度", 1.2))))
        self._复制件 = next((c for c in self.构件 if isinstance(c, 复制导向器)), None)
        self._情感件 = next((c for c in self.构件 if isinstance(c, 情感导向器)), None)
        self.最近诊断: dict = {}

    # ---- 生命周期 ----

    def 准备轮(self, 用户消息: str, 情感: Optional[str] = None) -> Optional[str]:
        """轮开始：解析目标情感、（复制导向器）编码用户话语；返回本轮情感标签。"""
        标签 = 情感
        if 标签 is None and self._情感件 is not None:
            标签 = self._情感件.预测(用户消息 or "")[0]
        if self._复制件 is not None:
            self._复制件.准备(用户消息 or "")
        诊断 = {"情感": 标签, "复制词": list(self._复制件._缓存词) if self._复制件 else [],
                "强度": {c.名称: c.强度 for c in self.构件}}
        self.最近诊断 = 诊断
        return 标签

    def 应用(self, logits: torch.Tensor, 上下文: dict) -> torch.Tensor:
        for c in self.构件:
            logits = c.偏置(logits, 上下文)
        return logits

    def 摘要(self) -> str:
        return "导向链(" + "、".join(f"{c.名称}β={c.强度}" for c in self.构件) + ")"