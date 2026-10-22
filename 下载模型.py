# -*- coding: utf-8 -*-
"""
============================================================
 RWKE（RWKV-Kernel-Engine）模型矩阵下载脚本
============================================================
功能：
  1. 检查 / 下载 RWKE 所需模型矩阵到 RWKE/models/ 目录：
     - RWKV7-World-0.4B      （必需，单文件 .pth，约 0.86 GB）
     - RWKV7-World-1.5B      （必需，单文件 .pth，约 2.85 GB）
     - Qwen2.5-1.5B-Instruct （必需，Transformer 基线，HF 目录格式多文件，约 2.88 GB）
     - RWKV7-G1-1.5B         （可选，默认不下载，--含G1 启用，约 2.85 GB）
  2. 支持下载源：hf-mirror.com（HF 国内镜像，默认）/ ModelScope（国内官方源）。
  3. 流式下载 + .part 临时文件 + 完成后原子重命名；已存在文件跳过（--强制 重下）。
  4. 完整性校验：文件大小 > 0 且与预期大小偏差 <= 5%（另与 HTTP Content-Length 对照）。

用法示例：
  python 下载模型.py --check                    # 只检查各模型文件状态，退出码恒为 0
  python 下载模型.py                            # 下载全部必需模型（hf-mirror 源）
  python 下载模型.py --源 modelscope            # 改用 ModelScope 源
  python 下载模型.py --仅模型 RWKV7-World-0.4B  # 只下载指定模型（支持子串匹配）
  python 下载模型.py --含G1                     # 额外下载可选的 RWKV7-G1-1.5B
  python 下载模型.py --强制                     # 已存在的文件也重新下载

退出码：--check 模式恒为 0；下载模式下任一文件下载/校验失败则为 1。

依赖：下载模式需 requests（pip install requests）；进度条用 tqdm（可选，
      未安装则降级为简单百分比打印）。--check 模式零第三方依赖。

URL 核实记录（2026-09-03 通过 HF / ModelScope 官方 API 核实仓库与文件清单，
文件大小与 SHA256 两源一致；若失效请到 HF 页面核对）：
  - https://huggingface.co/BlinkDL/rwkv-7-world    （镜像：https://hf-mirror.com/BlinkDL/rwkv-7-world）
  - https://huggingface.co/BlinkDL/rwkv7-g1        （镜像：https://hf-mirror.com/BlinkDL/rwkv7-g1）
  - https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
  - https://modelscope.cn/models/Blink_DL/rwkv-7-world
  - https://modelscope.cn/models/Blink_DL/rwkv7-g1
  - https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct
============================================================
"""

import argparse
import os
import sys
import time

# ---------- Windows 中文控制台输出保护（避免 GBK 控制台下 emoji 崩溃） ----------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------- tqdm 可选导入（未安装则降级为简单打印） ----------
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================
# 常量表：下载源前缀
# ============================================================
脚本目录 = os.path.dirname(os.path.abspath(__file__))
模型根目录 = os.path.join(脚本目录, "models")

# 大小校验容差（不同源的同一文件可能相差几十字节，如 tokenizer_config.json）
大小容差 = 0.05  # 5%

# ============================================================
# 常量表：模型矩阵（URL 已核实，若失效请到 HF 页面核对）
#   hfmirror   = https://hf-mirror.com/{仓库}/resolve/main/{文件}   （HF 主站把前缀换成 huggingface.co 即可）
#   modelscope = https://modelscope.cn/models/{仓库}/resolve/master/{文件}
# ============================================================
模型矩阵 = [
    {
        "名称": "RWKV7-World-0.4B",
        "必需": True,
        "类型": "单文件",
        "说明": "RWKV-7 World 0.4B 基底模型（单文件 .pth）",
        "文件名": "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
        "预期大小": 901_794_466,  # 约 0.86 GB
        "SHA256": "c08cf602b33e59a5717b8f0cafaf7f04c50c1c67166e477aa2f47c5ca180da4a",  # 备查（HF LFS oid 与 ModelScope Sha256 一致）
        "URL": {
            "hfmirror": "https://hf-mirror.com/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
            "modelscope": "https://modelscope.cn/models/Blink_DL/rwkv-7-world/resolve/master/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
        },
    },
    {
        "名称": "RWKV7-World-1.5B",
        "必需": True,
        "类型": "单文件",
        "说明": "RWKV-7 World 1.5B 基底模型（单文件 .pth）",
        "文件名": "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
        "预期大小": 3_055_062_345,  # 约 2.85 GB
        "SHA256": "23e13ff62ac6f80b28910ffc230324939e80ddb00b98f892a5951c0da071e700",  # 备查
        "URL": {
            "hfmirror": "https://hf-mirror.com/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
            "modelscope": "https://modelscope.cn/models/Blink_DL/rwkv-7-world/resolve/master/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
        },
    },
    {
        "名称": "Qwen2.5-1.5B-Instruct",
        "必需": True,
        "类型": "目录",
        "说明": "Qwen2.5-1.5B 指令模型（Transformer 基线，HF 目录格式多文件）",
        "子目录": "Qwen2.5-1.5B-Instruct",
        # 仓库实际文件清单（2026-09-03 核实），含各文件预期字节数
        "文件": [
            ("config.json", 660),
            ("generation_config.json", 242),
            ("model.safetensors", 3_087_467_144),  # 约 2.88 GB
            ("tokenizer.json", 7_031_645),
            ("tokenizer_config.json", 7_305),
            ("vocab.json", 2_776_833),
            ("merges.txt", 1_671_839),
            ("LICENSE", 11_343),
            ("README.md", 4_917),
        ],
        "URL模板": {
            "hfmirror": "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/{文件}",
            "modelscope": "https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/resolve/master/{文件}",
        },
    },
    {
        "名称": "RWKV7-G1-1.5B",
        "必需": False,  # 可选模型，默认不下载，--含G1 启用
        "类型": "单文件",
        "说明": "RWKV7-G1 推理模型 1.5B（单文件 .pth，选 g1j 最新版）",
        "文件名": "rwkv7-g1j-1.5b-20260831-ctx16384.pth",
        "预期大小": 3_055_444_605,  # 约 2.85 GB
        "SHA256": "c43176881caf85fe22ad654ab02e7519260d560f3d20420ab590adb0c823860f",  # 备查
        # 备注：仓库内另有备选版本 rwkv7-g1i-1.5b-20260805-ctx16384.pth（大小相同 3,055,444,605 字节），
        #       如需换版本直接改上面的"文件名"与"预期大小"即可
        "URL": {
            "hfmirror": "https://hf-mirror.com/BlinkDL/rwkv7-g1/resolve/main/rwkv7-g1j-1.5b-20260831-ctx16384.pth",
            "modelscope": "https://modelscope.cn/models/Blink_DL/rwkv7-g1/resolve/master/rwkv7-g1j-1.5b-20260831-ctx16384.pth",
        },
    },
]


# ============================================================
# 工具函数
# ============================================================
def 格式大小(字节数):
    """字节数转人类可读字符串。"""
    if 字节数 is None:
        return "?"
    值 = float(字节数)
    for 单位 in ("B", "KB", "MB", "GB", "TB"):
        if 值 < 1024 or 单位 == "TB":
            return f"{值:.2f} {单位}" if 单位 != "B" else f"{int(值)} B"
        值 /= 1024
    return f"{值:.2f} TB"


def 大小匹配(实际, 预期):
    """判断实际大小是否与预期相符：>0 且（无预期 或 偏差在容差内）。"""
    if 实际 is None or 实际 <= 0:
        return False
    if not 预期:
        return True
    return abs(实际 - 预期) <= 预期 * 大小容差


def 模型文件列表(模型):
    """
    展开模型的 (本地路径, URL) 列表。
    单文件模型 -> [(models/xxx.pth, 完整URL)]
    目录模型   -> [(models/Qwen2.5-1.5B-Instruct/xxx, 完整URL) * N]
    """
    结果 = []
    if 模型["类型"] == "单文件":
        本地 = os.path.join(模型根目录, 模型["文件名"])
        结果.append((本地, 模型["URL"]))
    else:
        目录 = os.path.join(模型根目录, 模型["子目录"])
        for 文件名, _大小 in 模型["文件"]:
            本地 = os.path.join(目录, 文件名)
            URL = {源: 模板.format(文件=文件名) for 源, 模板 in 模型["URL模板"].items()}
            结果.append((本地, URL))
    return 结果


def 选取模型(参数):
    """根据 --仅模型 / --含G1 决定要处理的模型集合。"""
    if 参数.only:
        # 显式点名：在全部模型（含可选）中做不区分大小写的子串匹配
        关键词 = [k.lower() for k in 参数.only]
        return [m for m in 模型矩阵 if any(k in m["名称"].lower() for k in 关键词)]
    return [m for m in 模型矩阵 if m["必需"] or 参数.with_g1]


def 检查文件状态(本地路径, 预期大小):
    """返回本地文件状态：'已就位' / '需下载' / '异常'（存在但大小不符）。"""
    if not os.path.exists(本地路径):
        return "需下载"
    实际 = os.path.getsize(本地路径)
    if 大小匹配(实际, 预期大小):
        return "已就位"
    return "异常"


# ============================================================
# 下载（流式 + .part 临时文件 + 完成后重命名）
# ============================================================
def 下载文件(URL, 本地路径, 预期大小):
    """
    流式下载单个文件。成功返回 True；任何失败抛异常（由调用方捕获打印 ❌）。
    """
    try:
        import requests  # 延迟导入：--check 模式不依赖第三方库
    except ImportError:
        raise RuntimeError("缺少 requests 库，请先执行：pip install requests")

    确定URL = URL if isinstance(URL, str) else URL.get(确定源, URL.get("hfmirror", list(URL.values())[0]))
    临时路径 = 本地路径 + ".part"
    os.makedirs(os.path.dirname(本地路径), exist_ok=True)

    头 = {"User-Agent": "RWKE-model-downloader/1.0"}  # HTTP 头仅支持 latin-1，不能含非 ASCII
    开始 = time.time()
    with requests.get(确定URL, stream=True, timeout=(10, 120), headers=头, allow_redirects=True) as 响应:
        if 响应.status_code != 200:
            raise RuntimeError(f"HTTP {响应.status_code}（URL: {确定URL}）")
        总长 = int(响应.headers.get("Content-Length") or 0)

        进度条 = None
        if tqdm is not None and 总长:
            进度条 = tqdm(total=总长, unit="B", unit_scale=True, unit_divisor=1024,
                          desc=f"      ↓ {os.path.basename(本地路径)}", leave=False)
        行内进度 = False
        已写 = 0
        try:
            with open(临时路径, "wb") as f:
                for 块 in 响应.iter_content(chunk_size=1024 * 1024):  # 1MB 块
                    if not 块:
                        continue
                    f.write(块)
                    已写 += len(块)
                    if 进度条 is not None:
                        进度条.update(len(块))
                    elif 总长:
                        print(f"\r      ↓ {os.path.basename(本地路径)}  {已写 * 100 // 总长:3d}%"
                              f"（{格式大小(已写)} / {格式大小(总长)}）", end="", flush=True)
                        行内进度 = True
        except Exception:
            # 下载中断：清理 .part 残留
            if 进度条 is not None:
                进度条.close()
            if 行内进度:
                print()
            if os.path.exists(临时路径):
                os.remove(临时路径)
            raise

        if 进度条 is not None:
            进度条.close()
        elif 行内进度:
            print()  # \r 行收尾换行

    # ---------- 完整性校验 ----------
    实际 = os.path.getsize(临时路径)
    if 总长 and 实际 != 总长:
        os.remove(临时路径)
        raise RuntimeError(f"与 Content-Length 不符（实际 {实际} / 声明 {总长} 字节）")
    if not 大小匹配(实际, 预期大小):
        os.remove(临时路径)
        raise RuntimeError(f"与预期大小不符（实际 {格式大小(实际)} / 预期 {格式大小(预期大小)}）")

    # 下载完成后原子重命名
    os.replace(临时路径, 本地路径)
    耗时 = time.time() - 开始
    print(f"      ✔ 完成 {格式大小(实际)}，耗时 {耗时:.1f} 秒")


# ============================================================
# 单模型处理
# ============================================================
def 处理模型(模型, 参数):
    """
    处理一个模型的所有文件。
    返回 (就位数, 需下载数, 失败数, 需下载总字节)。
    """
    就位 = 需下载 = 失败 = 0
    待下载字节 = 0
    文件清单 = 模型文件列表(模型)

    for 本地路径, URL in 文件清单:
        文件名 = os.path.basename(本地路径)
        预期大小 = None
        if 模型["类型"] == "单文件":
            预期大小 = 模型["预期大小"]
        else:
            for 名, 大小 in 模型["文件"]:
                if 名 == 文件名:
                    预期大小 = 大小
                    break

        状态 = 检查文件状态(本地路径, 预期大小)
        相对路径 = os.path.relpath(本地路径, 脚本目录)

        if 状态 == "已就位" and not 参数.force:
            print(f"   ✅ 已就位  {相对路径}（{格式大小(os.path.getsize(本地路径))}）")
            就位 += 1
            continue

        if 参数.check_only:
            if 状态 == "需下载":
                print(f"   ⬇️ 需下载  {相对路径}（预期 {格式大小(预期大小)}）")
                需下载 += 1
                待下载字节 += 预期大小 or 0
            elif 状态 == "异常":
                print(f"   ❌ 异常    {相对路径}（实际 {格式大小(os.path.getsize(本地路径))}"
                      f" ≠ 预期 {格式大小(预期大小)}，需重新下载）")
                失败 += 1
                待下载字节 += 预期大小 or 0
            continue

        # ---------- 下载模式 ----------
        if 状态 == "异常":
            print(f"   ⚠️ 本地文件大小异常，将重新下载：{相对路径}")
        elif 参数.force and 状态 == "已就位":
            print(f"   ♻ --强制 重新下载：{相对路径}")

        print(f"   ⬇️ 下载 {文件名}（预期 {格式大小(预期大小)}，源：{确定源}）")
        try:
            下载文件(URL, 本地路径, 预期大小)
            就位 += 1
        except Exception as 异常:
            print(f"   ❌ 失败    {文件名}：{异常}")
            失败 += 1

    return 就位, 需下载, 失败, 待下载字节


# ============================================================
# 主入口
# ============================================================
def 主():
    解析 = argparse.ArgumentParser(
        prog="下载模型.py",
        description="RWKE 模型矩阵下载脚本（RWKV7-World-0.4B / 1.5B、Qwen2.5-1.5B-Instruct、可选 RWKV7-G1-1.5B）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    解析.add_argument("--源", "--source", dest="source", choices=["hfmirror", "modelscope"],
                      default="hfmirror", help="下载源：hfmirror=hf-mirror.com 镜像（默认），modelscope=ModelScope")
    解析.add_argument("--仅模型", "--only", dest="only", nargs="+", metavar="名称",
                      help="只处理名称含关键字的模型（子串匹配，如：RWKV7-World-0.4B）")
    解析.add_argument("--check", action="store_true", dest="check_only",
                      help="只检查不下载（退出码恒为 0）")
    解析.add_argument("--含G1", "--with-g1", action="store_true", dest="with_g1",
                      help="包含可选模型 RWKV7-G1-1.5B（默认不下载）")
    解析.add_argument("--强制", "--force", action="store_true", dest="force",
                      help="已存在的文件也重新下载")
    参数 = 解析.parse_args()

    global 确定源
    确定源 = 参数.source

    选中 = 选取模型(参数)
    if not 选中:
        print("❌ 没有匹配到任何模型，可用模型名：")
        for m in 模型矩阵:
            print(f"   - {m['名称']}{'（可选，--含G1 启用）' if not m['必需'] else ''}")
        sys.exit(0 if 参数.check_only else 1)

    模式 = "检查模式（--check，只检查不下载）" if 参数.check_only else \
           f"下载模式（源：{确定源}{'，含G1' if 参数.with_g1 else ''}{'，--强制重下' if 参数.force else ''}）"
    print("=" * 60)
    print(f"RWKE 模型矩阵 —— {模式}")
    print(f"目标目录：{模型根目录}")
    print("=" * 60)

    总就位 = 总需下载 = 总失败 = 0
    总待下载字节 = 0
    for 序号, 模型 in enumerate(选中, 1):
        可选标记 = "（可选）" if not 模型["必需"] else ""
        print(f"\n[{序号}/{len(选中)}] {模型['名称']}{可选标记} —— {模型['说明']}")
        就位, 需下载, 失败, 待下载字节 = 处理模型(模型, 参数)
        总就位 += 就位
        总需下载 += 需下载
        总失败 += 失败
        总待下载字节 += 待下载字节

    print("\n" + "=" * 60)
    print(f"汇总：✅已就位 {总就位} 个文件 | ⬇️需下载 {总需下载} 个文件 | ❌失败/异常 {总失败} 个文件")
    if 总待下载字节:
        print(f"预计还需下载约 {格式大小(总待下载字节)}")
    print("=" * 60)

    # 退出码：--check 恒为 0；下载模式任一失败则为 1
    if 参数.check_only:
        sys.exit(0)
    sys.exit(1 if 总失败 > 0 else 0)


if __name__ == "__main__":
    主()
