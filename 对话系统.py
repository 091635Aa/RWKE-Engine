# -*- coding: utf-8 -*-
"""
RWKE 简易对话系统：历史记录 + 角色情感
========================================

基于 rwke引擎.RWKE引擎 的可交互对话封装，补齐两层"产品壳"能力：

1. 历史记录
   - 上下文记忆：多轮对话经 RWKV State 天然携带（O(1) 常量空间，引擎"延续对话"模式）；
   - 留痕落盘：每轮自动保存到 历史记录/{角色}_{模型}_{会话ID}.json，
     含时间戳、双方消息、情感快照；/历史 可回看，/新会话 开新档。

2. 角色情感
   - 人设：挂载 states/ 下的角色日记 State（RoleB/EmoCompanion），模型标识不符自动跳过；
   - 情绪跟踪：本地情感词表对"用户消息 / 角色回复"打分（[-1, 1]），
     角色心情值经指数平滑维护（随时间向 0 回落），映射五档情绪标签；
   - 情绪 → 生成反馈调制：|心情| 越大情感回响注入越强（λ 放大 0.6~1.4 倍），
     心情好温度略升（更活泼）、低落温度略降（更收敛）。

用法
----
    python 对话系统.py                        # RoleB × RWKV7-World-0.4B（最快）
    python 对话系统.py --角色 EmoCompanion --模型 RWKV7-G1-1.5B
    python 对话系统.py --无人设               # 不挂日记 State，裸模型起步
    python 对话系统.py --字数 160             # 每轮回复 token 上限

会话内命令
----------
    /帮助  /历史 [N]  /情感  /新会话  /保存  /退出
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rwke引擎 import RWKE引擎  # noqa: E402

_RWKE_DIR = Path(__file__).resolve().parent
_历史目录 = _RWKE_DIR / "历史记录"


# ======================================================================
# 情感打分：本地简易词表（与 测试_对比.py 同源口径，零外部依赖）
# ======================================================================

_正面词 = {"开心", "高兴", "快乐", "幸福", "喜悦", "兴奋", "愉快", "甜蜜", "喜欢", "爱",
           "温暖", "温柔", "安慰", "美好", "感动", "棒", "爽", "舒服", "满意", "幸运",
           "期待", "放心", "甜", "乖", "可爱", "治愈", "亲爱", "想你", "笑", "元气",
           "谢谢", "感谢", "加油", "有趣", "好玩", "惊喜", "放松", "满足", "骄傲", "幸福"}
_负面词 = {"难过", "悲伤", "伤心", "痛苦", "失落", "心碎", "沮丧", "哭泣", "委屈", "心疼",
           "绝望", "崩溃", "心累", "扎心", "孤独", "焦虑", "烦", "累", "害怕", "担心",
           "生气", "讨厌", "对不起", "自责", "自卑", "迷茫", "没用", "差劲", "压力", "哭",
           "无聊", "尴尬", "紧张", "后悔", "失望", "抱歉", "辛苦", "生病", "难受", "想你"}

# 情绪标签五档（心情值 m ∈ [-1, 1]）
_情绪档位 = [
    (0.5, "兴高采烈"),
    (0.15, "愉快"),
    (-0.15, "平静"),
    (-0.5, "低落"),
    (-1.01, "难过"),
]


def 情感得分(文本: str) -> float:
    """文本情感得分 ∈ [-1, 1]：正负词命中差占优为正，反之为负，无命中为 0。

    命中数 <3 时按比例弱化（单个词不轻易带偏心情）。
    """
    if not 文本:
        return 0.0
    正 = sum(1 for w in _正面词 if w in 文本)
    负 = sum(1 for w in _负面词 if w in 文本)
    总 = 正 + 负
    if 总 == 0:
        return 0.0
    方向 = (正 - 负) / 总
    强度 = min(1.0, 总 / 3.0)
    return round(max(-1.0, min(1.0, 方向 * 强度)), 4)


def 情绪标签(心情值: float) -> str:
    """心情值 → 情绪标签。"""
    for 阈值, 标签 in _情绪档位:
        if 心情值 >= 阈值:
            return 标签
    return _情绪档位[-1][1]


def 情感条(心情值: float, 宽: int = 10) -> str:
    """心情值可视条：中点为平静，右侧正向左侧负向。"""
    填充 = int(round((心情值 + 1.0) / 2.0 * 宽))
    return "█" * 填充 + "░" * (宽 - 填充)


# ======================================================================
# 对话系统主体
# ======================================================================

class 对话系统:
    """RWKE 对话系统：历史记录落盘 + 角色情感跟踪/反馈。"""

    # 心情指数平滑系数：用户输入影响大（共情），角色自己的话影响小（自我沉淀）
    _用户情感权重 = 0.25
    _回复情感权重 = 0.15

    def __init__(
        self,
        模型标识: str = "RWKV7-World-0.4B",
        角色: str = "RoleB",
        挂人设: bool = True,
        每轮字数: int = 128,
    ) -> None:
        print(f"[加载模型] {模型标识}（首次加载需数十秒）…")
        self.引擎 = RWKE引擎(模型标识)
        self.模型标识 = 模型标识
        self.角色 = 角色
        self.每轮字数 = int(每轮字数)
        self.基础λ = self.引擎.λ强度  # 扫描表基准值，每轮按心情调制后写回引擎

        # 角色情感状态
        self.心情值 = 0.0

        # 历史记录
        self.历史: List[dict] = []
        self.会话ID = time.strftime("%Y%m%d_%H%M%S")
        self.状态文件路径: Optional[Path] = None
        self.人设已挂载 = False
        if 挂人设:
            self.挂载人设()

        print(f"[引擎] {self.引擎.信息()}")
        print(f"[人设] {self.角色}{'（日记 State 已挂载）' if self.人设已挂载 else '（未挂载，裸模型起步）'}")
        print(f"[历史] {_历史目录 / (self.当前文件名 + '.json')}")

    # ──────────────────────────────────────────────
    # 人设挂载
    # ──────────────────────────────────────────────

    @property
    def 当前文件名(self) -> str:
        return f"{self.角色}_{self.模型标识}_{self.会话ID}"

    def 挂载人设(self) -> None:
        """挂载角色日记 State（人设记忆随 State 恢复）；文件缺失则跳过不阻断。"""
        路径 = _RWKE_DIR / "states" / f"{self.角色}_状态_{self.模型标识}.pth"
        if not 路径.exists():
            print(f"[人设] 跳过：State 不存在（{路径.name}）")
            return
        try:
            self.引擎.挂载日记状态(str(路径), self.模型标识)
            self.状态文件路径 = 路径
            self.人设已挂载 = True
            print(f"[人设] 已挂载 {self.角色} 日记 State：{路径.name}")
        except Exception as e:
            print(f"[人设] 挂载失败（{e}），裸模型起步")
            self.人设已挂载 = False

    # ──────────────────────────────────────────────
    # 每轮对话：情感跟踪 → 参数调制 → 生成 → 落盘
    # ──────────────────────────────────────────────

    def 发消息(self, 消息: str, token回调: Optional[Callable[[str], None]] = None) -> dict:
        """处理一轮用户消息：返回本轮记录（已追加进历史并落盘）。

        token回调 : 真流式钩子——每生成一段增量文本即调用（引擎 decode 后原样透传）。
        """
        # (1) 用户情绪 → 角色共情（心情向用户情绪靠拢）
        用户情感 = 情感得分(消息)
        self.心情值 = self._平滑(self.心情值, 用户情感, self._用户情感权重)

        # (2) 心情调制生成参数：
        #     λ：|心情| 越大情感回响注入越强（0.6~1.4 倍基准）
        #     温度：心情好更活泼（+），低落更收敛（−），限幅 [0.85, 1.15]
        self.引擎.λ强度 = self.基础λ * (0.6 + 0.8 * abs(self.心情值))
        温度 = max(0.85, min(1.15, 1.0 + 0.15 * self.心情值))

        # (3) 生成（延续对话：State 携带历史，O(1)；回响池跨轮累积）
        回复 = self.引擎.生成(
            消息,
            max_new_tokens=self.每轮字数,
            温度=温度,
            保留池=True,
            延续对话=(len(self.历史) > 0),
            token回调=token回调,
        )

        # (4) 角色自我沉淀：自己说的话也会影响心情
        回复情感 = 情感得分(回复)
        self.心情值 = self._平滑(self.心情值, 回复情感, self._回复情感权重)

        # (5) 记录并落盘
        记录 = {
            "轮号": len(self.历史) + 1,
            "时间": time.strftime("%H:%M:%S"),
            "用户": 消息,
            "回复": 回复,
            "情感": {
                "用户得分": 用户情感,
                "回复得分": 回复情感,
                "心情值": round(self.心情值, 4),
                "标签": 情绪标签(self.心情值),
                "注入λ": round(self.引擎.λ强度, 4),
                "温度": round(温度, 3),
            },
        }
        self.历史.append(记录)
        self.保存历史()
        return 记录

    @staticmethod
    def _平滑(旧值: float, 新值: float, 权重: float) -> float:
        """指数平滑一步并向 0 轻微回落（情绪自然消退）。"""
        新值 = 旧值 * (1.0 - 权重) + 新值 * 权重
        新值 *= 0.98  # 时间衰减：没有新刺激就慢慢归于平静
        return round(max(-1.0, min(1.0, 新值)), 4)

    # ──────────────────────────────────────────────
    # 历史记录：保存 / 展示
    # ──────────────────────────────────────────────

    def 保存历史(self) -> Path:
        """整档保存当前会话历史（JSON，ensure_ascii=False）。"""
        _历史目录.mkdir(exist_ok=True)
        档案 = {
            "会话ID": self.会话ID,
            "角色": self.角色,
            "模型": self.模型标识,
            "人设已挂载": self.人设已挂载,
            "创建时间": self.会话ID,
            "更新时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "轮数": len(self.历史),
            "轮次": self.历史,
        }
        路径 = _历史目录 / f"{self.当前文件名}.json"
        路径.write_text(json.dumps(档案, ensure_ascii=False, indent=2), encoding="utf-8")
        return 路径

    def 展示历史(self, 最近N轮: int = 5) -> str:
        """回看最近 N 轮对话（含情感快照）。"""
        if not self.历史:
            return "（本会话还没有对话记录）"
        行 = [f"—— 最近 {min(最近N轮, len(self.历史))} 轮（共 {len(self.历史)} 轮）——"]
        for r in self.历史[-最近N轮:]:
            情 = r["情感"]
            行.append(f"第{r['轮号']}轮 [{r['时间']}]")
            行.append(f"  你  ：{r['用户'][:60]}{'…' if len(r['用户']) > 60 else ''}"
                      f"（情绪 {情['用户得分']:+.2f}）")
            回复 = r["回复"][:60] + ("…" if len(r["回复"]) > 60 else "")
            行.append(f"  {self.角色}：{回复 if 回复 else '（空白）'}")
        return "\n".join(行)

    def 情感状态文本(self) -> str:
        """当前角色情感状态详单。"""
        标 = 情绪标签(self.心情值)
        λ描述 = f"{self.引擎.λ强度:.3f}（基准 {self.基础λ:.3f} × {0.6 + 0.8 * abs(self.心情值):.2f}）"
        return (
            f"{self.角色} 当前情感状态\n"
            f"  情绪标签：{标}\n"
            f"  心情值　：{self.心情值:+.2f}  {情感条(self.心情值)}\n"
            f"  回响注入：λ = {λ描述}\n"
            f"  采样温度：{max(0.85, min(1.15, 1.0 + 0.15 * self.心情值)):.2f}\n"
            f"  情感场　：回响池 {self.引擎.回响池.大小} 条回响"
        )

    # ──────────────────────────────────────────────
    # 新会话
    # ──────────────────────────────────────────────

    def 新会话(self) -> None:
        """开新档：清空历史、重置情感、重挂人设 State（对话记忆归零）。"""
        self.历史 = []
        self.心情值 = 0.0
        self.会话ID = time.strftime("%Y%m%d_%H%M%S")
        self.引擎.清空回响()
        if self.人设已挂载 and self.状态文件路径:
            self.引擎.挂载日记状态(str(self.状态文件路径), self.模型标识)
        print(f"[新会话] {self.会话ID}（{self.角色} 情绪已复位为平静）")


# ======================================================================
# CLI 交互循环
# ======================================================================

_帮助文本 = """会话内命令：
  /帮助          显示本帮助
  /历史 [N]      回看最近 N 轮对话（默认 5）
  /情感          查看角色当前情感状态
  /新会话        清空历史、重置情绪、重挂人设
  /保存          立即保存历史（每轮已自动保存）
  /退出          保存并退出"""


def main() -> int:
    解析器 = argparse.ArgumentParser(description="RWKE 简易对话系统（历史记录 + 角色情感）")
    解析器.add_argument("--模型", default="RWKV7-World-0.4B")
    解析器.add_argument("--角色", default="RoleB", help="角色名（对应 states/ 下的日记 State）")
    解析器.add_argument("--无人设", action="store_true", help="不挂载日记 State")
    解析器.add_argument("--字数", type=int, default=128, help="每轮回复 token 上限")
    参数 = 解析器.parse_args()

    系统 = 对话系统(
        模型标识=参数.模型,
        角色=参数.角色,
        挂人设=not 参数.无人设,
        每轮字数=参数.字数,
    )

    print("\n" + _帮助文本)
    print(f"\n[RWKE 对话系统已启动] 和 {系统.角色} 说点什么吧（输入 /退出 结束）\n")

    while True:
        try:
            消息 = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not 消息:
            continue

        # ---- 会话内命令 ----
        低 = 消息.lower()
        if 低 in ("/退出", "/exit", "/quit", "exit", "quit", "q"):
            break
        if 低 in ("/帮助", "/help", "/?"):
            print(_帮助文本 + "\n")
            continue
        if 低 == "/情感":
            print(系统.情感状态文本() + "\n")
            continue
        if 低 == "/新会话":
            系统.新会话()
            print()
            continue
        if 低 == "/保存":
            路径 = 系统.保存历史()
            print(f"[已保存] {路径}\n")
            continue
        if 低.startswith("/历史"):
            parts = 消息.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            print(系统.展示历史(n) + "\n")
            continue
        if 消息.startswith("/"):
            print(f"未知命令：{消息}（/帮助 查看命令）\n")
            continue

        # ---- 正常对话 ----
        try:
            记录 = 系统.发消息(消息)
        except Exception as e:
            print(f"[生成异常] {type(e).__name__}: {e}\n")
            continue
        print(f"{系统.角色}：{记录['回复'] if 记录['回复'] else '（空白）'}")
        情 = 记录["情感"]
        print(f"   └ 第{记录['轮号']}轮 · {情['标签']} {情感条(情['心情值'])} "
              f"心情{情['心情值']:+.2f} · λ={情['注入λ']:.3f} · T={情['温度']:.2f}（已存档）\n")

    系统.保存历史()
    print(f"\n[会话已保存] 共 {len(系统.历史)} 轮 → {系统.当前文件名}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
