# -*- coding: utf-8 -*-
"""
RWKE 统一状态接口
=================
将两种推理后端的"生成历史承载"抽象为统一的五操作接口：

    初始化 / 前进 / 克隆 / 保存 / 加载

后端对照（KV Cache vs RWKV State）：
- Transformer 后端：past_key_values（KV 缓存），随序列长度线性增长，O(n) 空间；
- RWKV 后端：固定尺寸 State，与序列长度无关，O(1) 常量空间。
  rwkv pip 包（RWKV-7 / x070）实测结构：state 为**扁平 List[torch.Tensor]**，
  长 n_layer*3，每层 3 张量：att_x_prev [n_embd]、
  att_kv [n_head, head_size, head_size] (fp32)、ffn_x_prev [n_embd]；
  每步推理 `logits, state = model.forward(token_id, state)`。
  为不依赖具体嵌套层级（未来版本可能为每层嵌套元组），
  克隆/序列化均按"张量树"递归实现（Tensor 为叶子，list/tuple 为分支）。

序列化约定：
- 一律在 CPU 上保存（避免 .pth 绑定 GPU 设备），加载时显式 map_location='cpu'；
- .pth 内容为 dict，含 'state' 张量组与元数据：
  '格式' / '模型标识' / 'hidden_dim' / 'n_layer' / '时间戳'；
- RWKV状态.保存 可选 附加元数据（如 {"适配器标识": ...}）并入元数据顶层，
  缺省不传时与既有格式完全一致；
- 加载时校验 '格式'，可选校验 '模型标识'（供引擎挂载日记 State 时防错挂）。
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import torch


# ======================================================================
# 张量树递归工具（Tensor 为叶子，list/tuple 为分支，其他类型原样返回）
# ======================================================================

def _克隆张量树(树):
    """深拷贝张量树：Tensor 叶子 .clone()，容器节点递归重建并保持原类型。"""
    if isinstance(树, torch.Tensor):
        return 树.clone()
    if isinstance(树, list):
        return [_克隆张量树(子) for 子 in 树]
    if isinstance(树, tuple):
        return tuple(_克隆张量树(子) for 子 in 树)
    return 树


def _张量树转CPU(树):
    """张量树全部叶子搬回 CPU（detach），容器结构保持；非容器非张量原样返回。"""
    if isinstance(树, torch.Tensor):
        return 树.detach().to('cpu')
    if isinstance(树, list):
        return [_张量树转CPU(子) for 子 in 树]
    if isinstance(树, tuple):
        return tuple(_张量树转CPU(子) for 子 in 树)
    return 树


def _张量树移至(树, 设备):
    """张量树全部叶子搬到指定设备，容器结构保持。"""
    if isinstance(树, torch.Tensor):
        return 树.to(设备)
    if isinstance(树, list):
        return [_张量树移至(子, 设备) for 子 in 树]
    if isinstance(树, tuple):
        return tuple(_张量树移至(子, 设备) for 子 in 树)
    return 树


# ======================================================================
# 抽象基类
# ======================================================================

class 推理状态(ABC):
    """统一状态接口：Transformer KV 缓存与 RWKV State 的公共操作抽象。"""

    @abstractmethod
    def 初始化(self):
        """重置为初始状态（等价于 state=None / 空 KV 缓存）。"""

    @abstractmethod
    def 前进(self, 新增状态增量):
        """将模型单步 forward 返回的状态增量并入当前状态。"""

    @abstractmethod
    def 克隆(self):
        """深拷贝语义：克隆体上'前进'不得影响原状态。"""

    @abstractmethod
    def 保存(self, 路径, 模型标识, hidden_dim, n_layer):
        """序列化为 .pth（CPU 张量 + 元数据），返回元数据 dict。"""

    @classmethod
    @abstractmethod
    def 加载(cls, 路径, 期望模型标识=None, 设备=None):
        """从 .pth 反序列化；返回 (状态对象, 元数据 dict)。"""


# ======================================================================
# RWKV State 适配器（O(1) 固定尺寸）
# ======================================================================

class RWKV状态(推理状态):
    """
    适配 rwkv pip 包（RWKV-7/x070）的 state：扁平 List[torch.Tensor]，长 n_layer*3。

    - state 为 None 表示初始状态；
    - 克隆/序列化按"张量树"递归实现，不依赖具体嵌套层级
      （Tensor 为叶子，list/tuple 为分支），兼容未来版本的嵌套结构变化；
    - 保存为 .pth：CPU 张量 + 元数据（'格式'='rwkv_state'）。
    """

    格式标识 = 'rwkv_state'

    def __init__(self, state=None):
        self._state = state

    # ---------------- 五操作 ----------------

    def 初始化(self):
        self._state = None

    def 前进(self, 新增状态增量):
        # RWKV State 是 O(1) 固定尺寸：每步直接整体替换
        self._state = 新增状态增量

    def 克隆(self):
        if self._state is None:
            return RWKV状态(None)
        return RWKV状态(_克隆张量树(self._state))

    def 保存(self, 路径, 模型标识, hidden_dim, n_layer, 附加元数据: Optional[dict] = None):
        """序列化为 .pth（CPU 张量 + 元数据），返回元数据 dict。

        附加元数据：可选键值对（如 {"适配器标识": ...}），并入元数据顶层；
        None/空 dict 时落盘内容与既有格式完全一致。
        """
        元数据 = {
            'state': self._张量组转CPU(self._state),
            '格式': self.格式标识,
            '模型标识': str(模型标识),
            'hidden_dim': int(hidden_dim),
            'n_layer': int(n_layer),
            '时间戳': datetime.now().isoformat(timespec='seconds'),
        }
        if 附加元数据:
            元数据.update(附加元数据)
        目录 = os.path.dirname(os.path.abspath(路径))
        if 目录:
            os.makedirs(目录, exist_ok=True)
        torch.save(元数据, 路径)
        return 元数据

    @classmethod
    def 加载(cls, 路径, 期望模型标识=None, 设备=None):
        数据 = torch.load(路径, map_location='cpu')
        if not isinstance(数据, dict) or 数据.get('格式') != cls.格式标识:
            raise ValueError(f"文件 {路径} 不是 '{cls.格式标识}' 格式的状态文件")
        cls._校验模型标识(数据, 期望模型标识)
        state = 数据['state']
        if 设备 is not None and state is not None:
            state = cls._张量组移至(state, 设备)
        return cls(state), 数据

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _张量组转CPU(state):
        # _张量树转CPU 对 None 原样返回
        return _张量树转CPU(state)

    @staticmethod
    def _张量组移至(state, 设备):
        return _张量树移至(state, 设备)

    # ---------------- 外部只读访问 ----------------

    @property
    def 原始状态(self):
        """供引擎每步 forward 直接使用的底层 state。"""
        return self._state

    def __len__(self):
        return 0 if self._state is None else len(self._state)


# ======================================================================
# Transformer KV 缓存适配器（O(n) 随序列增长）
# ======================================================================

class KV缓存状态(推理状态):
    """
    包装 HF transformers 的 past_key_values：
    tuple of tuples，每层为 (key, value) 二元组，各 [batch, n_head_kv, seq_len, head_dim]。

    注意：仅支持旧式 tuple 格式；若使用新版 Cache 对象（如 DynamicCache），
    可先经 .to_legacy_cache() 转换后再包装。
    """

    格式标识 = 'kv_cache'

    def __init__(self, past_key_values=None):
        self._pkv = past_key_values

    # ---------------- 五操作 ----------------

    def 初始化(self):
        self._pkv = None

    def 前进(self, 新增状态增量):
        # KV 缓存由模型 forward 内部追加，每步整体替换为最新引用
        self._pkv = 新增状态增量

    def 克隆(self):
        if self._pkv is None:
            return KV缓存状态(None)
        return KV缓存状态(_克隆张量树(self._pkv))

    def 保存(self, 路径, 模型标识, hidden_dim, n_layer):
        元数据 = {
            'state': self._张量组转CPU(self._pkv),
            '格式': self.格式标识,
            '模型标识': str(模型标识),
            'hidden_dim': int(hidden_dim),
            'n_layer': int(n_layer),
            '时间戳': datetime.now().isoformat(timespec='seconds'),
        }
        目录 = os.path.dirname(os.path.abspath(路径))
        if 目录:
            os.makedirs(目录, exist_ok=True)
        torch.save(元数据, 路径)
        return 元数据

    @classmethod
    def 加载(cls, 路径, 期望模型标识=None, 设备=None):
        数据 = torch.load(路径, map_location='cpu')
        if not isinstance(数据, dict) or 数据.get('格式') != cls.格式标识:
            raise ValueError(f"文件 {路径} 不是 '{cls.格式标识}' 格式的状态文件")
        cls._校验模型标识(数据, 期望模型标识)
        pkv = 数据['state']
        if 设备 is not None and pkv is not None:
            pkv = _张量树移至(pkv, 设备)
        return cls(pkv), 数据

    # ---------------- 内部工具 ----------------

    @staticmethod
    def _张量组转CPU(pkv):
        return _张量树转CPU(pkv)

    # ---------------- 外部只读访问 ----------------

    @property
    def 原始状态(self):
        """供引擎 forward(past_key_values=...) 直接使用的底层缓存。"""
        return self._pkv

    def __len__(self):
        return 0 if self._pkv is None else len(self._pkv)


# ======================================================================
# 顶层便捷函数
# ======================================================================

def 保存状态(状态对象, 路径, 模型标识, hidden_dim, n_layer):
    """统一保存入口：状态对象.保存()。"""
    return 状态对象.保存(路径, 模型标识, hidden_dim, n_layer)


def 加载状态(路径, 期望模型标识=None, 设备=None):
    """
    统一加载入口：按 .pth 元数据中的 '格式' 自动分派到对应适配器。
    返回 (状态对象, 元数据 dict)；模型标识不一致时抛出 ValueError。
    """
    数据 = torch.load(路径, map_location='cpu')
    if not isinstance(数据, dict) or '格式' not in 数据:
        raise ValueError(f"文件 {路径} 不是 RWKE 状态文件（缺少'格式'元数据）")
    格式 = 数据['格式']
    if 格式 == RWKV状态.格式标识:
        return RWKV状态.加载(路径, 期望模型标识, 设备)
    if 格式 == KV缓存状态.格式标识:
        return KV缓存状态.加载(路径, 期望模型标识, 设备)
    raise ValueError(f"未知的状态格式：'{格式}'（文件：{路径}）")


# 供各适配器复用的模型标识校验
def _校验模型标识(数据, 期望模型标识):
    if 期望模型标识 is not None and 数据.get('模型标识') != str(期望模型标识):
        raise ValueError(
            f"模型标识不匹配：状态文件为 '{数据.get('模型标识')}'，期望 '{期望模型标识}'。"
            f"（请勿将 A 模型的 State 挂载到 B 模型上）"
        )


# 给两个适配器类挂上共享校验方法
RWKV状态._校验模型标识 = staticmethod(_校验模型标识)
KV缓存状态._校验模型标识 = staticmethod(_校验模型标识)
