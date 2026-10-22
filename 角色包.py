# -*- coding: utf-8 -*-
"""
RWKE 角色资源包（外挂式角色插件）
==================================

把角色的全部"人格积木"打成一个**可按名下载/加载**的资源包，加载即带上该角色的
音色/情感/身份锚点，无需改代码：

    角色包/<角色名>/
        角色.json      清单（基座、导向链、状态注入、采样预设、文件 sha256）
        适配器.pt      角色 LoRA（音色 + 情感，由 微调.py 产出）
        人设卡.txt     人设/锚点/口癖文本（预热写入 State）
        状态.pth       可选：日记 State（免预热直接挂载）

设计要点
--------
- **清单驱动**：装载端只读 角色.json，不认识的角色也能装（零代码改动扩展角色）；
- **校验入包**：每个文件带 sha256，下载/拷贝后校验，损坏即拒装；
- **装配复用现役积木**：适配器走 ``引擎._挂载适配器``、身份走 ``状态注入``、
  输出偏置走 ``引擎.挂载导向链``、日记态走 ``引擎.挂载日记状态``——不新增前向路径；
- **可分发**：目录或 zip 均可；支持 http(s) 直接下载。

用法
----
    python 角色包.py 列表
    python 角色包.py 打包 --角色 EmoCompanion --输出 dist/EmoCompanion.zip
    python 角色包.py 安装 dist/EmoCompanion.zip            # 目录/zip/URL 均可
    python 角色包.py 校验 EmoCompanion
    python 角色包.py 装配 --角色包 EmoCompanion --模型 RWKV7-G1-1.5B   # 需 GPU

供 服务器.py 调用的 API
-----------------------
    列出角色包(根=None) -> List[dict]
    读清单(名称, 根=None) -> dict
    校验角色包(名称, 根=None) -> dict
    安装角色包(源, 根=None, 覆盖=False) -> dict
    打包角色包(名称, 根=None, 输出=None) -> Path
    装配引擎(引擎, 名称, 根=None, 挂身份=True, 挂导向=True, 用状态=True) -> dict
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_RWKE_DIR = Path(__file__).resolve().parent
默认根 = _RWKE_DIR / "角色包"
清单名 = "角色.json"
格式版本 = 1

# 文件槽位 → 允许的扩展名（防把任意文件塞进包）
_槽位后缀 = {"适配器": (".pt",), "人设卡": (".txt", ".md"), "状态": (".pth",)}


class 角色包异常(RuntimeError):
    """角色包结构/校验失败（绝不静默装载损坏资源）。"""


# ======================================================================
# 清单读写 / 校验
# ======================================================================

def _sha256(路径: Path) -> str:
    哈希 = hashlib.sha256()
    with 路径.open("rb") as 文件:
        for 块 in iter(lambda: 文件.read(1 << 20), b""):
            哈希.update(块)
    return 哈希.hexdigest()


def _包目录(名称: str, 根: Path) -> Path:
    return Path(根) / 名称


def 读清单(名称: str, 根: Optional[Path] = None) -> dict:
    """读取角色包清单；结构非法即抛 角色包异常。"""
    根 = Path(根) if 根 else 默认根
    目录 = _包目录(名称, 根)
    清单路径 = 目录 / 清单名
    if not 清单路径.exists():
        raise 角色包异常(f"角色包不存在：{目录}（缺 {清单名}）")
    清单 = json.loads(清单路径.read_text(encoding="utf-8"))
    if int(清单.get("格式版本", 0)) > 格式版本:
        raise 角色包异常(f"角色包格式版本 {清单.get('格式版本')} 高于本程序支持的 {格式版本}，请更新程序")
    for 必需 in ("角色", "基座", "文件"):
        if 必需 not in 清单:
            raise 角色包异常(f"角色包清单缺少必需字段 '{必需}'：{清单路径}")
    if "适配器" not in 清单["文件"]:
        raise 角色包异常(f"角色包缺少适配器文件条目：{清单路径}")
    清单["_目录"] = str(目录)
    return 清单


def 校验角色包(名称: str, 根: Optional[Path] = None) -> dict:
    """逐文件核对存在性与 sha256（缺失/损坏/多余文件都报告）。"""
    清单 = 读清单(名称, 根)
    目录 = Path(清单["_目录"])
    结果: Dict[str, dict] = {}
    for 槽位, 条目 in 清单["文件"].items():
        路径 = 目录 / 条目["路径"]
        if not 路径.exists():
            结果[槽位] = {"状态": "缺失", "路径": str(路径)}
            continue
        实际字节 = 路径.stat().st_size
        实际哈希 = _sha256(路径)
        期望哈希 = 条目.get("sha256")
        状态 = "通过"
        if 期望哈希 and 实际哈希 != 期望哈希:
            状态 = "哈希不符"
        elif 条目.get("字节") is not None and int(条目["字节"]) != 实际字节:
            状态 = "字节数不符"
        结果[槽位] = {"状态": 状态, "路径": 条目["路径"], "字节": 实际字节,
                      "sha256": 实际哈希 if 状态 != "通过" else 期望哈希}
    通过 = all(v["状态"] == "通过" for v in 结果.values())
    return {"角色": 清单["角色"], "通过": 通过, "明细": 结果}


def 列出角色包(根: Optional[Path] = None) -> List[dict]:
    """扫描根目录下全部合法角色包（清单损坏的条目也列出并标注）。"""
    根 = Path(根) if 根 else 默认根
    if not 根.exists():
        return []
    结果 = []
    for 子 in sorted(根.iterdir()):
        if not 子.is_dir() or not (子 / 清单名).exists():
            continue
        try:
            清单 = 读清单(子.name, 根)
            结果.append({"角色": 清单["角色"], "显示名": 清单.get("显示名", 清单["角色"]),
                         "描述": 清单.get("描述", ""), "版本": 清单.get("版本", ""),
                         "基座": 清单["基座"], "目录": str(子),
                         "文件": {k: v["路径"] for k, v in 清单["文件"].items()}})
        except Exception as 异常:
            结果.append({"角色": 子.name, "目录": str(子), "错误": str(异常)})
    return 结果


# ======================================================================
# 打包 / 安装
# ======================================================================

def _写条目(路径: Path) -> dict:
    return {"路径": 路径.name, "字节": 路径.stat().st_size, "sha256": _sha256(路径)}


def 建包(名称: str, 基座: str, 适配器: Path, 人设卡: Optional[Path] = None,
        状态: Optional[Path] = None, 显示名: Optional[str] = None, 描述: str = "",
        版本: str = "1.0", 导向链: Optional[dict] = None, 状态注入: Optional[dict] = None,
        采样: Optional[dict] = None, 根: Optional[Path] = None, 覆盖: bool = False) -> Path:
    """
    组装一个角色包目录：拷贝积木 + 生成 角色.json（含 sha256）。

    默认档位与现役部署口径一致（导向链三路、状态注入 选槽24、统一采样协议），
    保证"装上即与 服务器.py 现役EmoCompanion表现一致"。
    """
    根 = Path(根) if 根 else 默认根
    目录 = _包目录(名称, 根)
    if 目录.exists():
        if not 覆盖:
            raise 角色包异常(f"角色包已存在：{目录}（--覆盖 可重建）")
        shutil.rmtree(目录)
    目录.mkdir(parents=True)

    文件条目: Dict[str, dict] = {}
    适配器目标 = 目录 / f"适配器{适配器.suffix.lower()}"
    shutil.copy2(适配器, 适配器目标)
    文件条目["适配器"] = _写条目(适配器目标)

    if 人设卡 is not None:
        人设目标 = 目录 / f"人设卡{人设卡.suffix.lower()}"
        shutil.copy2(人设卡, 人设目标)
        文件条目["人设卡"] = _写条目(人设目标)
    if 状态 is not None:
        状态目标 = 目录 / f"状态{状态.suffix.lower()}"
        shutil.copy2(状态, 状态目标)
        文件条目["状态"] = _写条目(状态目标)

    清单 = {
        "格式版本": 格式版本,
        "角色": 名称,
        "显示名": 显示名 or 名称,
        "描述": 描述,
        "版本": str(版本),
        "基座": 基座,
        "文件": 文件条目,
        "导向链": 导向链 if 导向链 is not None else
                  {"情感": {"强度": 1.5}, "身份": {"强度": 1.2}, "复制": {"强度": 1.2}},
        "状态注入": 状态注入 if 状态注入 is not None else
                    {"自动选槽": 24, "每层上限": 2, "α": 0.92, "缩放": 1.0},
        "采样": 采样 if 采样 is not None else
                {"温度": 1.0, "top_p": 0.9, "top_k": 50, "种子": 42, "每轮字数": 192,
                 "预热预算": 384},
    }
    (目录 / 清单名).write_text(json.dumps(清单, ensure_ascii=False, indent=2), encoding="utf-8")
    return 目录


def 打包角色包(名称: str, 根: Optional[Path] = None, 输出: Optional[Path] = None) -> Path:
    """把角色包目录打成 zip（分发用）；打包前先校验。"""
    根 = Path(根) if 根 else 默认根
    校验 = 校验角色包(名称, 根)
    if not 校验["通过"]:
        raise 角色包异常(f"角色包校验未通过，拒绝打包：{校验['明细']}")
    目录 = _包目录(名称, 根)
    输出 = Path(输出) if 输出 else 根 / f"{名称}.角色包.zip"
    输出.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(输出, "w", zipfile.ZIP_DEFLATED) as 压缩:
        for 文件 in sorted(目录.rglob("*")):
            if 文件.is_file():
                压缩.write(文件, 文件.relative_to(目录.parent))
    return 输出


def 安装角色包(源: str, 根: Optional[Path] = None, 覆盖: bool = False) -> dict:
    """
    安装角色包：源可为本地目录 / zip / http(s) URL。

    安装后逐文件校验 sha256，任何不符即回滚清目录（绝不留下半装状态）。
    """
    根 = Path(根) if 根 else 默认根
    根.mkdir(parents=True, exist_ok=True)
    临时 = Path(tempfile.mkdtemp(prefix="角色包_安装_"))
    try:
        源文 = str(源)
        if 源文.startswith(("http://", "https://")):
            压缩包 = 临时 / "下载.zip"
            print(f"        下载 {源文} …", flush=True)
            urllib.request.urlretrieve(源文, 压缩包)
            解包目录 = _解压到临时(压缩包, 临时)
        else:
            本地 = Path(源文)
            if not 本地.exists():
                raise 角色包异常(f"安装源不存在：{本地}")
            解包目录 = _解压到临时(本地, 临时) if 本地.is_file() else 本地
        # 兼容包内再套一层目录
        if not (解包目录 / 清单名).exists():
            候选 = [d for d in 解包目录.iterdir() if d.is_dir() and (d / 清单名).exists()]
            if len(候选) == 1:
                解包目录 = 候选[0]
            else:
                raise 角色包异常(f"安装源内未找到 {清单名}：{解包目录}")
        清单 = json.loads((解包目录 / 清单名).read_text(encoding="utf-8"))
        名称 = 清单["角色"]
        目标 = _包目录(名称, 根)
        if 目标.exists():
            if not 覆盖:
                raise 角色包异常(f"角色 '{名称}' 已安装（{目标}）；--覆盖 可替换")
            shutil.rmtree(目标)
        shutil.copytree(解包目录, 目标)
        try:
            校验 = 校验角色包(名称, 根)
            if not 校验["通过"]:
                raise 角色包异常(f"安装后校验未通过：{校验['明细']}")
        except Exception:
            shutil.rmtree(目标, ignore_errors=True)   # 回滚，不留半装状态
            raise
        return {"角色": 名称, "路径": str(目标), "校验": 校验["明细"]}
    finally:
        shutil.rmtree(临时, ignore_errors=True)


def _解压到临时(压缩包: Path, 临时: Path) -> Path:
    目标 = 临时 / "解包"
    目标.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(压缩包) as 压缩:
        for 成员 in 压缩.namelist():     # 防 zip 路径穿越
            if 成员.startswith(("/", "\\")) or ".." in Path(成员).parts:
                raise 角色包异常(f"压缩包内含非法路径：{成员}")
        压缩.extractall(目标)
    return 目标


# ======================================================================
# 装配（把角色包挂到引擎上；需要已加载的引擎，故仅在有 GPU 时可用）
# ======================================================================

def 装配引擎(引擎, 名称: str, 根: Optional[Path] = None, 挂身份: bool = True,
             挂导向: bool = True, 用状态: bool = True, 采样: bool = True) -> dict:
    """
    按角色包清单装配引擎：适配器 → 日记态（或人设卡预热）→ 状态级身份记忆 → 导向链 → 采样。

    装配顺序与 服务器.py 现役口径一致（预热在导向链之前，避免偏置污染人设预填充）。
    返回各步结果明细，便于日志与排障。
    """
    清单 = 读清单(名称, 根)
    目录 = Path(清单["_目录"])
    明细: Dict[str, object] = {"角色": 清单["角色"], "基座": 清单["基座"]}

    基座不符 = 清单["基座"] != getattr(引擎, "模型标识", 清单["基座"])
    if 基座不符:
        raise 角色包异常(
            f"角色包基座 '{清单['基座']}' 与当前引擎 '{getattr(引擎, '模型标识', '?')}' 不符（防错挂）")

    # (1) 角色 LoRA（音色 + 情感）
    适配器路径 = 目录 / 清单["文件"]["适配器"]["路径"]
    引擎._挂载适配器(str(适配器路径))
    明细["适配器"] = 适配器路径.name

    # (2) 人设入场：优先挂日记 State（免预热）；否则走人设卡预热写入 State
    人设路径 = 目录 / 清单["文件"]["人设卡"]["路径"] if "人设卡" in 清单["文件"] else None
    状态路径 = 目录 / 清单["文件"]["状态"]["路径"] if "状态" in 清单["文件"] else None
    if 用状态 and 状态路径 is not None and 状态路径.exists():
        引擎.挂载日记状态(str(状态路径), 清单["基座"])
        明细["人设"] = f"日记State：{状态路径.name}"
    elif 人设路径 is not None and 人设路径.exists():
        人设文本 = 人设路径.read_text(encoding="utf-8").strip()
        预热预算 = int(清单.get("采样", {}).get("预热预算", 384))
        引擎.生成(人设文本, max_new_tokens=预热预算,
                  温度=float(清单.get("采样", {}).get("温度", 1.0)),
                  top_p=float(清单.get("采样", {}).get("top_p", 0.9)),
                  top_k=int(清单.get("采样", {}).get("top_k", 50)))
        引擎.挂载状态([t.clone() for t in 引擎._末state])
        明细["人设"] = f"人设卡预热：{人设路径.name}"
    else:
        明细["人设"] = "无（裸基座）"

    # (3) 状态级身份记忆（架构级身份锚点：槽位锁定 + 每步回注）
    if 挂身份 and 清单.get("状态注入") and 人设路径 is not None and 人设路径.exists():
        import 状态注入 as _状态注入
        参数 = dict(清单["状态注入"])
        人设文本 = 人设路径.read_text(encoding="utf-8").strip()
        if 参数.get("自动选槽"):
            层列表, 头列表, _分数 = _状态注入.排序槽位(
                引擎, 人设文本, 槽位数=int(参数["自动选槽"]),
                每层上限=int(参数.get("每层上限", 2)),
                层起点比例=float(参数.get("层起点比例", 0.5)))
            槽位对 = list(zip(层列表, 头列表))
        else:
            槽位对 = None
        记忆 = _状态注入.构造身份记忆(引擎, 人设文本, 槽位对=槽位对,
                                     缩放=float(参数.get("缩放", 1.0)))
        明细["身份记忆"] = 引擎.挂载身份记忆(记忆, α=float(参数.get("α", 0.92)))

    # (4) 输出导向链（情感 / 身份 / 复制三路加性先验）
    if 挂导向 and 清单.get("导向链"):
        明细["导向链"] = 引擎.挂载导向链(清单["导向链"])

    # (5) 采样预设（温/top_p/top_k/种子 覆盖统一协议的对应键）
    if 采样 and 清单.get("采样"):
        for 键 in ("温度", "top_p", "top_k", "种子"):
            if 键 in 清单["采样"]:
                引擎.采样协议[键] = 清单["采样"][键]
        明细["采样"] = {k: 清单["采样"][k] for k in ("温度", "top_p", "top_k", "种子") if k in 清单["采样"]}
    return 明细


# ======================================================================
# CLI
# ======================================================================

def _默认积木(名称: str) -> dict:
    """EmoCompanion的现役积木默认位置（打包默认源）。"""
    基座 = "RWKV7-G1-1.5B"
    return {
        "基座": 基座,
        "适配器": _RWKE_DIR / "适配器" / f"{基座}_lora_best.pt",
        "人设卡": _RWKE_DIR / "人设卡_EmoCompanion.txt",
        "状态": _RWKE_DIR / "states" / f"{名称}_状态_{基座}.pth",
        "显示名": 名称,
        "描述": "EmoCompanion的直播间新人主播｜元气萌妹，温暖活泼略带娇羞，开播第十五天",
    }


def main() -> int:
    解析器 = argparse.ArgumentParser(description="RWKE 角色资源包（外挂式角色插件）")
    子 = 解析器.add_subparsers(dest="命令")

    列 = 子.add_parser("列表", help="列出已安装角色包")
    列.add_argument("--根", default=None)

    打 = 子.add_parser("打包", help="组装/打包角色包")
    打.add_argument("--角色", default="EmoCompanion")
    打.add_argument("--模型", default="RWKV7-G1-1.5B")
    打.add_argument("--适配器", default=None, help="角色 LoRA .pt（默认 适配器/<模型>_lora_best.pt）")
    打.add_argument("--人设卡", default=None)
    打.add_argument("--状态", default=None, help="日记 State（默认 states/<角色>_状态_<模型>.pth；'无' 表示不含）")
    打.add_argument("--输出", default=None, help="zip 输出路径；给 '目录' 则只组装不压缩")
    打.add_argument("--根", default=None)
    打.add_argument("--覆盖", action="store_true")

    装 = 子.add_parser("安装", help="安装角色包（目录/zip/URL）")
    装.add_argument("源")
    装.add_argument("--根", default=None)
    装.add_argument("--覆盖", action="store_true")

    校 = 子.add_parser("校验", help="校验角色包 sha256")
    校.add_argument("名称")
    校.add_argument("--根", default=None)

    装2 = 子.add_parser("装配", help="把角色包装配到引擎（需 GPU）")
    装2.add_argument("--角色包", required=True)
    装2.add_argument("--模型", default=None, help="引擎基座标识；默认取清单")
    装2.add_argument("--根", default=None)
    装2.add_argument("--无人设", action="store_true", help="不挂身份记忆")
    装2.add_argument("--无状态", action="store_true", help="不用日记 State，改走人设卡预热")

    参数 = 解析器.parse_args()
    根 = Path(参数.根) if getattr(参数, "根", None) else None

    if 参数.命令 == "列表":
        条目 = 列出角色包(根)
        if not 条目:
            print(f"（{根 or 默认根} 下无角色包）")
        for 项 in 条目:
            if "错误" in 项:
                print(f"  [损坏] {项['角色']}：{项['错误']}")
            else:
                print(f"  {项['角色']}（v{项['版本']}，基座 {项['基座']}）：{项['描述']}")
                print(f"        文件：{项['文件']}")
        return 0

    if 参数.命令 == "校验":
        结果 = 校验角色包(参数.名称, 根)
        for 槽位, 条 in 结果["明细"].items():
            print(f"  {槽位}: {条['状态']}  {条.get('路径', '')}")
        print(("校验通过" if 结果["通过"] else "校验未通过") + f"：{结果['角色']}")
        return 0 if 结果["通过"] else 1

    if 参数.命令 == "打包":
        默认 = _默认积木(参数.角色)
        适配器 = Path(参数.适配器) if 参数.适配器 else 默认["适配器"]
        if not 适配器.exists():
            print(f"适配器不存在：{适配器}")
            return 1
        人设卡 = Path(参数.人设卡) if 参数.人设卡 else (默认["人设卡"] if 默认["人设卡"].exists() else None)
        状态 = None
        if 参数.状态 == "无":
            状态 = None
        elif 参数.状态:
            状态 = Path(参数.状态)
        elif 默认["状态"].exists():
            状态 = 默认["状态"]
        目录 = 建包(参数.角色, 参数.模型 or 默认["基座"], 适配器, 人设卡=人设卡, 状态=状态,
                    显示名=默认["显示名"], 描述=默认["描述"], 根=根, 覆盖=参数.覆盖)
        print(f"已组装角色包：{目录}")
        for 槽位, 条 in 校验角色包(参数.角色, 根)["明细"].items():
            print(f"  {槽位}: {条['状态']}（{条.get('字节', '?')} 字节）")
        if 参数.输出 and 参数.输出 != "目录":
            zip路径 = 打包角色包(参数.角色, 根, Path(参数.输出))
            print(f"已压缩分发包：{zip路径}")
        return 0

    if 参数.命令 == "安装":
        结果 = 安装角色包(参数.源, 根, 覆盖=参数.覆盖)
        print(f"已安装角色 '{结果['角色']}' → {结果['路径']}")
        return 0

    if 参数.命令 == "装配":
        清单 = 读清单(参数.角色包, 根)
        from rwke引擎 import RWKE引擎
        引擎 = RWKE引擎(模型标识=参数.模型 or 清单["基座"])
        明细 = 装配引擎(引擎, 参数.角色包, 根, 挂身份=not 参数.无人设, 用状态=not 参数.无状态)
        print("装配完成：")
        for 键, 值 in 明细.items():
            print(f"  {键}: {值}")
        return 0

    解析器.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())