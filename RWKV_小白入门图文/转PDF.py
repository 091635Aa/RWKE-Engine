# -*- coding: utf-8 -*-
"""
RWKV 介绍文档 · Markdown → PDF（Edge 无头打印）
特点：HTML 生成到 MD 同目录，保证 assets/*.svg 相对路径可用；
     紧凑排版（10.5pt / 1.65 行距 / 1.5cm 页边距）保证页面利用率 ≥80%。

用法: python 转PDF.py
"""
import sys, time, subprocess
from pathlib import Path

import markdown

EDGE = r"<path>\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
目录 = Path(__file__).parent
MD = 目录 / "RWKV_小白超入门_图文介绍.md"
PDF = 目录.parent / "PDF转化" / "RWKV_小白超入门_图文介绍.pdf"
HTML = 目录 / "_tmp_doc.html"

CSS = """
@page { size: A4; margin: 1.5cm 1.7cm 1.6cm 1.7cm;
  @bottom-center { content: "RWKV 小白超入门 · 第 " counter(page) " / " counter(pages) " 页";
                    font-size: 8pt; color: #777; } }
html, body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; font-size: 10.5pt;
             line-height: 1.65; color: #222; margin: 0; padding: 0; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 10pt 0; color: #1a3c6e; }
h2 { font-size: 13pt; color: #1a3c6e; border-bottom: 1pt solid #c8d4e4;
     padding-bottom: 2pt; margin: 14pt 0 6pt 0; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 10pt 0 4pt 0; page-break-after: avoid; }
p { margin: 5pt 0; }
ul, ol { margin: 5pt 0; padding-left: 20pt; }
li { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0; font-size: 9pt; }
th { background: #eef3f9; border: 0.5pt solid #9db2c8; padding: 3pt 5pt; text-align: center; }
td { border: 0.5pt solid #b7c6d6; padding: 3pt 5pt; }
tr { page-break-inside: avoid; }
blockquote { color: #444; border-left: 3pt solid #9db2c8; background: #f7fafd;
             padding: 4pt 8pt; margin: 6pt 0; page-break-inside: avoid; }
blockquote p { margin: 2pt 0; }
code { font-family: Consolas, monospace; background: #f4f4f4; font-size: 8.5pt; padding: 0 2pt; }
pre { background: #f6f8fa; border: 0.5pt solid #ddd; padding: 6pt 8pt; font-size: 8.5pt;
      white-space: pre-wrap; page-break-inside: avoid; margin: 6pt 0; }
pre code { background: none; padding: 0; }
img { display: block; width: 93%; height: auto; margin: 8pt auto; page-break-inside: avoid; }
hr { border: 0; border-top: 1pt solid #ccc; margin: 10pt 0; }
strong { color: #1a3c6e; }
a { color: #1a3c6e; text-decoration: none; }
"""


def 主():
    md文本 = MD.read_text(encoding="utf-8")
    body = markdown.markdown(md文本, extensions=["tables", "fenced_code", "sane_lists"])
    标题 = md文本.splitlines()[0].lstrip("# ").strip()

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{标题}</title>
<style>{CSS}</style></head><body>{body}</body></html>"""
    HTML.write_text(html, encoding="utf-8")

    PDF.parent.mkdir(parents=True, exist_ok=True)
    命令 = [
        EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--user-data-dir={目录 / '_tmp_edge_profile'}",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ]
    proc = subprocess.Popen(命令, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        if PDF.exists() and PDF.stat().st_size > 0:
            break
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    else:
        proc.kill()
        print("超时：Edge 未在 60s 内完成")
        sys.exit(1)
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    time.sleep(1)

    if not PDF.exists() or PDF.stat().st_size == 0:
        print("错误：未生成 PDF")
        sys.exit(1)
    print(f"PDF 已生成: {PDF} ({PDF.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    主()
