# -*- coding: utf-8 -*-
"""
RWKE 状态级联想记忆注入（架构级身份记忆 / 超长期记忆）
=====================================================

问题（现有"日记状态"路径的两个结构缺陷）
----------------------------------------
1. **人设与对话抢同一批 head**：日记状态把角色信息整体灌进全部 State
   （n_layer × 全部 head）。RWKV-7 每步 forward 都对 att_kv 做 delta-rule
   `S ← S·w + S@ab + v⊗k`（遗忘 + 擦除 + 写入），人设与对话共用 head 时，
   对话每步都在擦同一批矩阵 → 长会话下"越想不起自己是谁"。
2. **只有质心、没有寻址**：State 被当成一个整体向量用，只能"整体像不像"，
   不能"按问题精确召回剧情"。而 att_kv 在数学上本来就是
   key→value 联想矩阵（`S@r` 即内容寻址读取），这层结构此前完全没被利用。

本模块（不改官方算子，纯状态空间改造）
--------------------------------------
1. **身份槽锁定 Slot Locking**：每层只划出少量 head 作为身份/剧情专用槽，
   其余 head 继续承载对话 → 人设与对话在状态空间**解耦**，互不擦除；
2. **结构化写入**：记忆文本仍走模型自身的 wkv7 写入路径（delta rule），
   槽里存的是**真实的外积联想矩阵**（Σ v_t⊗kk_t），不是平均向量；
   读取 `S_slot @ r` 即按当前 token 的 query 做内容寻址 → "明确的身份"；
3. **状态回注 Re-injection**：每步 forward 后 `S_slot ← α·S_slot + (1-α)·S_id`，
   α<1 使人设以几何级泄漏保持，**永不遗忘**（超长期记忆的结构保证）；
4. **衰减探测**：给出槽漂移（相对基准的偏移）与方向余弦，供在线监测 /
   低于阈值触发重写。

槽位分配
--------
默认取后 50% 层（高层语义更抽象）各 2 个 head → 1.5B 为 12 层 × 2 = 24 槽，
仅占全 State 的 3%（24 / 768 个 head 矩阵），对话容量几乎无损。
可用 排序槽位() 按"人设文本响应强度"数据化挑槽。

对外接口
--------
    记忆 = 构造身份记忆(引擎, 文本, 层列表=None, 头列表=None, 缩放=1.0)
    引擎.挂载身份记忆(记忆, α=0.92)      # 引擎侧最小钩子（rwke引擎.py）
    state → 写入/回注 由引擎生成循环自动调用
    引擎.身份衰减() → {"平均漂移":…, "最大漂移":…, "方向余弦":…}
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ----------------------------------------------------------------------
# 槽位分配
# ----------------------------------------------------------------------

# 默认：后 50% 层（起于 n_layer//2），每层取前 每层头数 个 head
_默认层起点比例 = 0.5
_默认每层头数 = 2


def 默认槽位(引擎, 每层头数: int = _默认每层头数) -> Tuple[List[int], List[int]]:
    """默认槽位：后 50% 层 × 前 N 个 head。返回 (层列表, 头列表)。"""
    起 = int(引擎.n_layer * _默认层起点比例)
    层列表 = list(range(起, 引擎.n_layer))
    头列表 = list(range(min(int(每层头数), 引擎.n_head)))
    return 层列表, 头列表


class 身份记忆:
    """一支被锁定的联想记忆：若干 (层, 头) 槽 + 各自的基准矩阵 S_id [N, N] fp32。"""

    def __init__(self, 槽: List[Tuple[int, int, torch.Tensor]], 文本: str,
                 元信息: Optional[dict] = None):
        self.槽 = 槽
        self.文本 = 文本
        self.元信息 = dict(元信息 or {})

    def __len__(self) -> int:
        return len(self.槽)

    @property
    def 层集合(self) -> List[int]:
        return sorted({层 for 层, _, _ in self.槽})

    @property
    def 头集合(self) -> List[int]:
        return sorted({头 for _, 头, _ in self.槽})

    def 摘要(self) -> str:
        return (f"身份记忆(槽数={len(self.槽)}, 层={self.层集合[0]}..{self.层集合[-1]}"
                f"({len(self.层集合)}层), 头={self.头集合}, "
                f"token数={self.元信息.get('token数', '?')}, "
                f"基准范数均值={self.元信息.get('基准范数均值', '?'):.3f}"
                if isinstance(self.元信息.get('基准范数均值'), float) else
                f"身份记忆(槽数={len(self.槽)}, 层={self.层集合}, 头={self.头集合})")


# ----------------------------------------------------------------------
# 构造（走模型自身的 wkv7 写入路径）
# ----------------------------------------------------------------------

def _对齐(引擎, state):
    """复用引擎的设备/dtype 对齐（att_kv → fp32，前值 → 权重 dtype）。"""
    if state is None:
        return None
    if hasattr(引擎, "_对齐状态"):
        return 引擎._对齐状态(state)
    if state and state[0].device != 引擎.设备:
        return [t.to(引擎.设备) for t in state]
    return state


def 构造身份记忆(
    引擎,
    文本: str,
    层列表: Optional[Sequence[int]] = None,
    头列表: Optional[Sequence[int]] = None,
    缩放: float = 1.0,
    段大小: int = 512,
    进度: bool = False,
    槽位对: Optional[Sequence[Tuple[int, int]]] = None,
) -> 身份记忆:
    """
    把 文本 写进指定 (层, 头) 槽：零态 → 分段预填充（走引擎 _预填充，
    挂适配器时权重为 W_eff）→ 抽取槽矩阵 × 缩放 作为基准 S_id。

    「结构化写入」的落点：预填充本身就是官方 wkv7 递推
    `S ← S·w + S@ab + v⊗k`，因此槽里是**逐 token 外积累加的真实联想矩阵**；
    本函数只负责把它**限定在指定 head 上**并冻结为基准。

    槽位指定两选一：
    - 槽位对：显式 (层, 头) 配对（排序槽位 的数据化结果走这条，**不做叉积**）；
    - 层列表 × 头列表：笛卡尔积（"后 N 层 × 前 M 头"这类规则分配走这条）。
    """
    槽位对 = [(int(l), int(h)) for l, h in 槽位对] if 槽位对 is not None else None
    if 槽位对 is None:
        层列表 = list(层列表) if 层列表 is not None else 默认槽位(引擎)[0]
        头列表 = list(头列表) if 头列表 is not None else 默认槽位(引擎)[1]
        层列表 = [int(层) for 层 in 层列表 if 0 <= int(层) < 引擎.n_layer]
        头列表 = [int(头) for 头 in 头列表 if 0 <= int(头) < 引擎.n_head]
        槽位对 = [(层, 头) for 层 in 层列表 for 头 in 头列表]
    else:
        槽位对 = [(l, h) for l, h in 槽位对 if 0 <= l < 引擎.n_layer and 0 <= h < 引擎.n_head]
        层列表 = sorted({l for l, _ in 槽位对})
        头列表 = sorted({h for _, h in 槽位对})
    if not 槽位对:
        raise ValueError(f"槽位为空：槽位对={槽位对}")

    token列表 = 引擎.分词器.encode(文本)
    state = _对齐(引擎, 引擎.模型.generate_zero_state())

    with torch.no_grad():
        段数 = max(1, (len(token列表) + 段大小 - 1) // 段大小)
        for 段号 in range(段数):
            块 = token列表[段号 * 段大小: (段号 + 1) * 段大小]
            if not 块:
                continue
            _, state = 引擎._预填充(块, state)
            if 进度 and 段数 > 1:
                print(f"    [身份写入] 段 {段号 + 1}/{段数}（token {段号 * 段大小}–"
                      f"{段号 * 段大小 + len(块)}）", flush=True)

        槽: List[Tuple[int, int, torch.Tensor]] = []
        范数 = []
        for 层, 头 in 槽位对:
            kv = state[层 * 3 + 1]
            基准 = (kv[头].detach().float() * float(缩放)).cpu().contiguous()
            槽.append((层, 头, 基准))
            范数.append(float(基准.norm()))

    元信息 = {
        "token数": len(token列表),
        "文本长度": len(文本),
        "层列表": 层列表,
        "头列表": 头列表,
        "缩放": float(缩放),
        "基准范数均值": sum(范数) / len(范数) if 范数 else 0.0,
        "基准范数最小": min(范数) if 范数 else 0.0,
        "基准范数最大": max(范数) if 范数 else 0.0,
    }
    return 身份记忆(槽, 文本, 元信息)


def 排序槽位(引擎, 文本: str, 槽位数: int = 24, 层起点比例: float = 0.5,
             每层上限: int = 2, 段大小: int = 512,
             排除位点: Optional[Sequence[Tuple[int, int]]] = None
             ) -> Tuple[List[int], List[int], List[float]]:
    """
    数据化挑槽：按"该 (层, 头) 对 文本 的响应强度"（基准矩阵 Frobenius 范数）
    在层起点比例之后的层里排序，取前 槽位数 个（每层至多 每层上限 个）。

    依据：响应越强的 head 对这段文本越"敏感"，优先征用为专用槽。
    排除位点：已被其它事实占用的 (层, 头)（多事实记忆库要求槽带互不相交，
    见 合并记忆 的重叠校验）。
    返回 (层列表, 头列表, 分数列表)，层/头一一对应。
    """
    起 = int(引擎.n_layer * float(层起点比例))
    排除 = {(int(l), int(h)) for l, h in (排除位点 or [])}
    token列表 = 引擎.分词器.encode(文本)
    state = _对齐(引擎, 引擎.模型.generate_zero_state())
    with torch.no_grad():
        for 起i in range(0, len(token列表), 段大小):
            块 = token列表[起i: 起i + 段大小]
            if 块:
                _, state = 引擎._预填充(块, state)
        scores: List[Tuple[float, int, int]] = []
        for 层 in range(起, 引擎.n_layer):
            kv = state[层 * 3 + 1]
            for 头 in range(引擎.n_head):
                if (层, 头) in 排除:
                    continue
                scores.append((float(kv[头].float().norm()), 层, 头))
    scores.sort(reverse=True)
    层列表, 头列表, 分数列表, 每层计数 = [], [], [], {}
    for 分, 层, 头 in scores:
        if len(层列表) >= 槽位数:
            break
        if 每层计数.get(层, 0) >= 每层上限:
            continue
        每层计数[层] = 每层计数.get(层, 0) + 1
        层列表.append(层)
        头列表.append(头)
        分数列表.append(分)
    return 层列表, 头列表, 分数列表


# ----------------------------------------------------------------------
# 状态操作：写入 / 回注 / 探测
# ----------------------------------------------------------------------

def 合并记忆(记忆列表: Sequence["身份记忆"], 名称: str = "合并记忆") -> 身份记忆:
    """
    把多支记忆合并为一支（要求槽位 (层, 头) 互不重叠），供引擎一次性挂载。

    用途：把"身份"与"剧情/日记"分到互不重叠的槽带上——两套记忆各自独立
    写入、独立回注，互不擦除，从而在状态空间里实现多路超长期记忆。
    """
    槽, 文本部分, 元 = [], [], {}
    for 记忆 in 记忆列表:
        if 记忆 is None:
            continue
        槽.extend(记忆.槽)
        文本部分.append(记忆.文本)
        元.update(记忆.元信息)
    位点 = [(层, 头) for 层, 头, _ in 槽]
    if len(set(位点)) != len(位点):
        raise ValueError("合并记忆时槽位 (层, 头) 出现重叠，请分配互不相交的槽带")
    return 身份记忆(槽, "\n\n".join(文本部分), 元)


def 写入(引擎, state, 记忆: 身份记忆, 强度: float = 1.0) -> None:
    """硬写入：槽 ← 基准 × 强度（覆盖槽内一切对话痕迹，用于会话开头装订）。"""
    if state is None or 记忆 is None:
        return
    with torch.no_grad():
        for 层, 头, 基准 in 记忆.槽:
            kv = state[层 * 3 + 1]
            目标 = 基准.to(device=kv.device, dtype=kv.dtype)
            if float(强度) == 1.0:
                kv[头].copy_(目标)
            else:
                kv[头].copy_(目标 * float(强度))


def 回注(引擎, state, 记忆: 身份记忆, α: float = 0.92) -> None:
    """
    软回注：槽 ← α·槽 + (1-α)·基准（每步 forward 后调用）。

    α<1 ⇒ 人设以 (1-α) 的几何泄漏被持续写回，**不会被对话擦除**；
    α 越大人设越"硬"（对话对槽的影响越小），α=0 即完全硬锁定。
    原地修改（int 索引取 view），CUDA 图模式下直接改常驻状态缓冲亦安全。
    """
    if state is None or 记忆 is None:
        return
    α = float(α)
    with torch.no_grad():
        for 层, 头, 基准 in 记忆.槽:
            kv = state[层 * 3 + 1]
            视 = kv[头]
            if α <= 0.0:
                视.copy_(基准.to(device=kv.device, dtype=kv.dtype))
                continue
            视.mul_(α).add_(基准.to(device=kv.device, dtype=kv.dtype), alpha=1.0 - α)


def 探测(引擎, state, 记忆: 身份记忆) -> Dict[str, float]:
    """
    衰减探测：槽相对基准的偏移与方向余弦。

    平均漂移 = mean(‖S - S_id‖) / mean(‖S_id‖)：0=完全锁定，1≈被冲淡到无关。
    方向余弦 1.0 = 槽方向与基准完全一致（未被人设外信息污染）。
    """
    if state is None or 记忆 is None:
        return {"槽数": 0, "平均漂移": 0.0, "最大漂移": 0.0, "方向余弦": 1.0}
    漂移表, 余弦表 = [], []
    with torch.no_grad():
        for 层, 头, 基准 in 记忆.槽:
            现 = state[层 * 3 + 1][头].float().cpu()
            基准n = 基准.norm()
            if float(基准n) < 1e-8:
                continue
            漂移表.append(float((现 - 基准).norm()) / float(基准n))
            余弦表.append(float(torch.nn.functional.cosine_similarity(
                现.reshape(-1), 基准.reshape(-1), dim=0)))
    if not 漂移表:
        return {"槽数": 0, "平均漂移": 0.0, "最大漂移": 0.0, "方向余弦": 0.0}
    return {
        "槽数": len(漂移表),
        "平均漂移": round(sum(漂移表) / len(漂移表), 4),
        "最大漂移": round(max(漂移表), 4),
        "方向余弦": round(sum(余弦表) / len(余弦表), 4),
    }