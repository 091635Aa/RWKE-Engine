# -*- coding: utf-8 -*-
"""
RWKE OpenAI 兼容服务器：AIRI 应用落地入口
==========================================

把 rwke引擎（RWKV-7 + 情感回响注入 + 日记 State 人设）包成 OpenAI 兼容 API，
供 AIRI（或任何 OpenAI 兼容客户端）以自定义提供者方式接入：

    POST /v1/chat/completions   对话（支持 stream SSE / 非流式）
    GET  /v1/models             模型列表（AIRI 拉模型用）
    GET  /api/health            就绪状态（模型加载中返回 loading）
    GET  /api/emotion           角色当前情感状态（心情值/标签/λ/回响池）
    POST /api/session/reset     重开会话（清历史、复位情绪、重挂人设 State）
    GET  /api/history?n=5       最近 N 轮对话

会话语义
--------
- 人设/记忆由日记 State 携带（O(1) 常量空间），system 消息不进模型；
- 每请求只取最后一条 user 消息生成，历史由 RWKV State 延续（延续对话模式）；
- 会话对齐：请求里的 assistant 轮数少于服务端已存轮数 → 判定客户端重开聊天，
  自动 重置会话 再生成（防 State 残留旧记忆）；
- 角色心情由 对话系统 维护：用户/回复情感 → 心情值 → λ 与温度调制。

用法
----
    python 服务器.py                          # EmoCompanion（正式角色：日记 State + 角色 LoRA）× G1-1.5B
    python 服务器.py --角色 RoleB --无适配器    # RoleB（测试代号，仅日记 State）
    python 服务器.py --无适配器               # EmoCompanion裸基座对比
    python 服务器.py --快速采样               # top-k 先行快速采样（认证复现勿开）

AIRI 配置（桌面版 设置 → Providers → Add Provider → OpenAI Compatible）
------------------------------------------------------------------------
    Base URL : http://127.0.0.1:8102/v1
    API Key  : rwke（任意非空字符串）
    Model    : 从列表拉取（rwkv7-g1-1.5b）
"""

import argparse
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from 对话系统 import 对话系统, 情感条

_RWKE_DIR = Path(__file__).resolve().parent

# ======================================================================
# 全局会话持有者：模型在后台线程懒加载，加载期间服务已可应答 /api/health
# ======================================================================

_锁 = threading.Lock()          # 引擎生成串行化（非线程安全）
_就绪 = False
_预热成功 = False
_加载错误: Optional[str] = None
_参数: argparse.Namespace = argparse.Namespace(
        模型="RWKV7-G1-1.5B", 角色="EmoCompanion", 无人设=False, 字数=192,
        主机="127.0.0.1", 端口=8102, 适配器=None, 无适配器=False, 导向="关", 角色包=None,
    )
_系统: Optional[对话系统] = None

# 输出导向链档位（--导向）：三路默认强度见 导向链.py；"关" 时不挂载（逐位保持旧行为）
_导向档位 = {
    "关": None,
    "情感": {"情感": {"强度": 1.5}},
    "身份": {"身份": {"强度": 1.2}},
    "复制": {"复制": {"强度": 1.2}},
    "三路": {"情感": {"强度": 1.5}, "身份": {"强度": 1.2}, "复制": {"强度": 1.2}},
}


def _导向配置(档位: str) -> Optional[dict]:
    if 档位 not in _导向档位:
        raise KeyError(f"未知导向档位 '{档位}'，可选：{list(_导向档位)}")
    return _导向档位[档位]


def _启动引擎() -> None:
    global _系统, _就绪, _加载错误, _预热成功
    try:
        _系统 = 对话系统(
            模型标识=_参数.模型,
            角色=_参数.角色,
            挂人设=(not _参数.无人设 and not _参数.角色包),
            每轮字数=_参数.字数,
        )
        # 角色资源包（角色包.py）：一键装配 适配器 + 人设 + 身份记忆 + 导向链 + 采样预设
        角色包已装配 = False
        if _参数.角色包:
            import 角色包 as _角色包模块
            明细 = _角色包模块.装配引擎(_系统.引擎, _参数.角色包, 挂身份=not _参数.无人设)
            print(f"[服务器] 角色包 '{_参数.角色包}' 已装配：{明细}")
            角色包已装配 = True
        if not 角色包已装配 and _参数.适配器:
            _系统.引擎._挂载适配器(_参数.适配器)
            print(f"[服务器] 已挂载角色 LoRA 适配器：{_参数.适配器}")
        if 角色包已装配:
            pass
        elif _参数.快速采样:
            _系统.引擎.采样协议["快速采样"] = True  # 交互提速；认证脚本不设此键口径不变
            print("[服务器] 快速采样已开启（top-k 先行）")
        # 预热轮：G1 挂日记 State 的首轮易进长思考（吃满预算），把这一步移到启动期——
        # 预热成功则丢弃历史但保留 State（真实请求从"已开口"的会话态继续，直接出正文）
        # 人设卡（提速⑧人设侧）：预热消息用角色卡——人设/锚点/口癖随 State 一次预填充，
        # 此后每轮对话从"已是EmoCompanion"的状态继续（O(1) 随身携带）
        预热消息 = "你好呀"
        卡路径 = Path(_参数.人设卡) if _参数.人设卡 else None
        if 卡路径 is None and _参数.角色 == "EmoCompanion" and not _参数.无人设:
            默认卡 = _RWKE_DIR / "人设卡_EmoCompanion.txt"
            卡路径 = 默认卡 if 默认卡.exists() else None
        if 卡路径 and 卡路径.exists() and not 角色包已装配:
            预热消息 = 卡路径.read_text(encoding="utf-8").strip()
            print(f"[服务器] 人设卡已加载：{卡路径.name}（随预热写入 State）")
        if not _参数.无人设 and not 角色包已装配:
            try:
                _系统.每轮字数 = max(_参数.字数, 384)  # 人设卡轮：预算给足让思考完成并打招呼
                预热 = _剥思考(_系统.发消息(预热消息)["回复"] or "")
                _系统.每轮字数 = _参数.字数
                if len(预热) >= 4:
                    _系统.历史 = []
                    _预热成功 = True
                    print(f"[服务器] 预热完成（人设已入State）：{预热[:40]}")
                else:
                    _系统.新会话()
                    print("[服务器] 预热回复过短，已重置会话（首轮将走完整思考）")
            except Exception as e:
                print(f"[服务器] 预热失败（忽略）：{e}")
        # 输出导向链（导向链.py）：预热之后再挂，避免偏置污染人设卡预填充本身
        导向配置 = None if 角色包已装配 else _导向配置(_参数.导向)
        if 导向配置 is not None:
            信息 = _系统.引擎.挂载导向链(导向配置)
            print(f"[服务器] 导向链已挂载：{信息['构件']}")
        _就绪 = True
        print(f"[服务器] 就绪：http://{_参数.主机}:{_参数.端口}/v1")
    except Exception as e:
        _加载错误 = f"{type(e).__name__}: {e}"
        print(f"[服务器] 引擎启动失败：{_加载错误}")


def _模型ID() -> str:
    return _参数.模型.lower()


app = FastAPI(title="RWKE OpenAI 兼容服务器", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_启动引擎, daemon=True).start()


# ======================================================================
# 请求/响应模型（只取 OpenAI 兼容所需字段，其余忽略）
# ======================================================================

class _消息(BaseModel):
    role: str
    content: str


class _对话请求(BaseModel):
    model: Optional[str] = None
    messages: List[_消息]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None  # 忽略：温度由角色心情调制
    user: Optional[str] = None


def _确保就绪() -> 对话系统:
    if _加载错误:
        raise HTTPException(500, f"引擎启动失败：{_加载错误}")
    if not _就绪 or _系统 is None:
        raise HTTPException(503, "模型加载中，请稍后重试（G1-1.5B 首次加载数十秒）")
    return _系统


def _最后用户消息(messages: List[_消息]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    raise HTTPException(400, "messages 中没有 user 消息")


def _会话对齐(系统: 对话系统, messages: List[_消息]) -> None:
    """客户端历史比服务端短 → 判定重开聊天，重置会话（清 State 残留记忆）。"""
    已完成轮数 = sum(1 for m in messages if m.role == "assistant")
    if 已完成轮数 < len(系统.历史):
        系统.新会话()


def _剥思考(文本: str) -> str:
    """清洗模型输出：剥掉 G1 思考模型的 <think>...</think> 块 + 截断尾部复读循环。"""
    if "</think>" in 文本:
        文本 = 文本.split("</think>", 1)[1]
    elif "<think>" in 文本:
        return ""  # 思考未闭合（预算被吃满）
    # 尾部复读截断：① 同一 1~4 字片段连续 ≥6 次；② 同一 4~16 字短语连续 ≥4 次
    import re
    文本 = re.sub(r"(.{1,4})\1{5,}\s*$", lambda m: m.group(1) * 2, 文本)
    文本 = re.sub(r"(.{4,16})\1{3,}\s*$", r"\1", 文本)
    return 文本.strip()


# ======================================================================
# OpenAI 兼容端点
# ======================================================================

@app.get("/v1/models")
def 列模型():
    return {
        "object": "list",
        "data": [{
            "id": _模型ID(),
            "object": "model",
            "created": 0,
            "owned_by": "rwke",
        }],
    }


@app.post("/v1/chat/completions")
def 对话补全(req: _对话请求):
    系统 = _确保就绪()
    用户消息 = _最后用户消息(req.messages)
    完成ID = f"chatcmpl-rwke-{uuid.uuid4().hex[:12]}"
    时间戳 = int(time.time())
    模型名 = req.model if req.model else _模型ID()

    # 每请求随机种子：采样协议默认固定 42（认证复现口径），交互对话需逐轮变化防复读
    import random
    种子 = random.randint(1, 2**31 - 1)

    def _应用请求参数():
        """锁内执行：会话对齐 + 预算/种子设置（流式在 worker 内、非流式在端点内调用）。"""
        _会话对齐(系统, req.messages)
        if req.max_tokens is not None:
            基础预算 = max(16, min(512, int(req.max_tokens)))
        else:
            基础预算 = 256
        # 会话首轮：G1 思考块吃满小预算会导致正文为空，首轮放宽到 512 让思考完成并出正文
        # （预热成功时首轮已是"已开口"状态，走常规预算）
        系统.每轮字数 = max(基础预算, 512) if (len(系统.历史) == 0 and not _预热成功) else 基础预算
        系统.引擎.采样协议["种子"] = 种子

    if not req.stream:
        with _锁:
            _应用请求参数()
            记录 = 系统.发消息(用户消息)

        回复 = _剥思考(记录["回复"] or "") or "……"
        return {
            "id": 完成ID,
            "object": "chat.completion",
            "created": 时间戳,
            "model": 模型名,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": 回复},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(用户消息),
                "completion_tokens": len(回复),
                "total_tokens": len(用户消息) + len(回复),
            },
        }

    # ---- 真流式 SSE（提速⑦）：worker 线程持锁生成，token回调逐段推入队列；
    # SSE 生成器从队列消费——首字延迟 ≈ 预填充+1 步（此前为整段生成完再分片）----
    import queue as _queue
    from json import dumps as _dumps
    from fastapi.responses import StreamingResponse

    def 对块(块: dict) -> str:
        return _dumps(块, ensure_ascii=False)

    队列: _queue.Queue = _queue.Queue()

    def _worker():
        try:
            with _锁:
                _应用请求参数()
                记录 = 系统.发消息(用户消息, token回调=lambda s: 队列.put(s))
            队列.put(None)  # 正常结束哨兵
        except Exception as e:  # noqa: BLE001 —— 错误转 SSE error 事件
            队列.put(e)

    threading.Thread(target=_worker, daemon=True).start()

    def 事件流():
        首块 = {
            "id": 完成ID, "object": "chat.completion.chunk",
            "created": 时间戳, "model": 模型名,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {对块(首块)}\n\n"
        结束原因 = "stop"
        已输出 = False
        # 增量剥思考：三态（判定中→思考中→正文）。判定中暂缓首段直到确认是否 <think> 开头；
        # 思考中吞掉直到 </think>，其后正文照常透传（流式版 _剥思考）
        状态 = "判定中"
        缓冲 = ""

        def _透传(文本):
            nonlocal 已输出
            if 文本:
                已输出 = True
                块 = {
                    "id": 完成ID, "object": "chat.completion.chunk",
                    "created": 时间戳, "model": 模型名,
                    "choices": [{"index": 0, "delta": {"content": 文本}, "finish_reason": None}],
                }
                return f"data: {对块(块)}\n\n"
            return ""

        try:
            while True:
                段 = 队列.get()
                if 段 is None:
                    break
                if isinstance(段, Exception):
                    结束原因 = "error"
                    yield f"data: {对块({'error': {'message': f'{type(段).__name__}: {段}', 'type': 'rwke_internal_error'}})}\n\n"
                    break
                if not 段:
                    continue
                if 状态 == "判定中":
                    缓冲 += 段
                    if 缓冲.startswith("<think>"):
                        状态 = "思考中"
                        缓冲 = ""
                    elif not "<think>".startswith(缓冲):
                        状态 = "正文"
                        yield _透传(缓冲)
                elif 状态 == "思考中":
                    缓冲 += 段
                    if "</think>" in 缓冲:
                        状态 = "正文"
                        正文 = 缓冲.split("</think>", 1)[1]
                        缓冲 = ""
                        yield _透传(正文)
                else:
                    yield _透传(段)
        finally:
            if 状态 != "正文" and 缓冲 and not 缓冲.startswith("<think>"):
                yield _透传(缓冲)  # 判定中断流（异常/极短回复）：缓冲非思考则补发
            if not 已输出 and 结束原因 == "stop":
                yield _透传("……")  # 思考吃满预算兜底
            尾块 = {
                "id": 完成ID, "object": "chat.completion.chunk",
                "created": 时间戳, "model": 模型名,
                "choices": [{"index": 0, "delta": {}, "finish_reason": 结束原因}],
            }
            yield f"data: {对块(尾块)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(事件流(), media_type="text/event-stream")


# ======================================================================
# RWKE 扩展端点（情感可视化 / 会话管理）
# ======================================================================

@app.get("/api/health")
def 健康():
    return {
        "status": "ready" if _就绪 else ("error" if _加载错误 else "loading"),
        "模型": _参数.模型,
        "角色": _参数.角色,
        "人设": not _参数.无人设,
        "错误": _加载错误,
    }


@app.get("/api/emotion")
def 情感状态():
    系统 = _确保就绪()
    from 对话系统 import 情绪标签
    return {
        "角色": 系统.角色,
        "心情值": 系统.心情值,
        "标签": 情绪标签(系统.心情值),
        "心情条": 情感条(系统.心情值),
        "注入λ": round(系统.引擎.λ强度, 4),
        "基准λ": round(系统.基础λ, 4),
        "回响池条数": 系统.引擎.回响池.大小,
        "会话轮数": len(系统.历史),
    }


@app.post("/api/session/reset")
def 重置会话():
    系统 = _确保就绪()
    with _锁:
        系统.新会话()
    return {"ok": True, "会话ID": 系统.会话ID}


@app.get("/api/history")
def 历史(n: int = 5):
    系统 = _确保就绪()
    return {"轮数": len(系统.历史), "最近": 系统.历史[-max(1, n):]}


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    global _参数
    解析器 = argparse.ArgumentParser(description="RWKE OpenAI 兼容服务器（AIRI 落地入口）")
    解析器.add_argument("--模型", default="RWKV7-G1-1.5B")
    解析器.add_argument("--角色", default="EmoCompanion", help="正式角色=EmoCompanion；RoleB为测试代号")
    解析器.add_argument("--无人设", action="store_true", help="不挂载日记 State")
    解析器.add_argument("--适配器", default=None,
                        help="角色 LoRA .pt 路径；默认（角色=EmoCompanion时）自动挂 EmoCompanion best 检查点")
    解析器.add_argument("--无适配器", action="store_true", help="强制不挂角色 LoRA（对比用）")
    解析器.add_argument("--人设卡", default=None,
                        help="预热人设卡文本路径（角色=EmoCompanion时默认 人设卡_EmoCompanion.txt，随预热写入 State）")
    解析器.add_argument("--字数", type=int, default=192, help="每轮回复 token 上限")
    解析器.add_argument("--主机", default="127.0.0.1")
    解析器.add_argument("--端口", type=int, default=8102)
    解析器.add_argument("--快速采样", action="store_true",
                        help="top-k 先行快速采样（~3ms/步→~1ms；top-p 阈值相对 top-k 质量归一，认证复现勿开）")
    解析器.add_argument("--访问日志", action="store_true", help="开启 uvicorn 每请求访问日志（默认关）")
    解析器.add_argument("--导向", default="关", choices=list(_导向档位),
                        help="输出导向链档位（情感/身份/复制/三路；默认关=旧行为）")
    解析器.add_argument("--角色包", default=None,
                        help="角色资源包名（角色包/ 下，如 'EmoCompanion'）：一键装配适配器+人设+身份记忆+导向链；"
                             "给此参数时忽略 --适配器/--人设卡/--导向")
    _参数 = 解析器.parse_args()

    # EmoCompanion为正式角色：默认自动挂其角色 LoRA（--无适配器 可关；显式传 --适配器 时以显式为准）
    if _参数.适配器 is None and not _参数.无适配器 and _参数.角色 == "EmoCompanion" and not _参数.角色包:
        _参数.适配器 = str(_RWKE_DIR / "适配器" / "RWKV7-G1-1.5B_lora_best.pt")

    import uvicorn
    uvicorn.run(app, host=_参数.主机, port=_参数.端口, log_level="info", access_log=_参数.访问日志)


if __name__ == "__main__":
    main()
