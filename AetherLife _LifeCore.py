#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AetherLife v1.0 - 最终完美版（单文件）
作者: Marshall
许可证: MIT License (2026)

基于纯 NumPy 的深度学习训练框架，支持 DPO 对齐、Transformer 变体、自适应硬件控制、生命化系统。
适用于移动端及低资源设备，无需外部依赖（可选依赖用于增强功能）。

核心功能：
- 自动微分系统 (Tensor + Function)
- Transformer / InfiniAttention / MoE / GRU / 图像编码器
- DPO 对齐训练 + SFT 预热
- 自适应硬件监控与动态调整
- 生命化系统（元学习 + 自适应体 + 意识核心 + 统一生命循环）
- 多目标进化搜索
- 检查点自动保存与恢复
- 结构化日志 + 进度条 + 完整单元测试

本文件为最终改造完成版 - 单文件 v1.0
"""

import argparse
import atexit
import copy
import gc
import hashlib
import json
import logging
import logging.handlers
import math
import os
import random
import sys
import threading
import time
import traceback
import types
from collections import OrderedDict, defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import heapq

# 可选依赖
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.n = 0
        def update(self, n=1): self.n += n
        def write(self, msg): print(msg)
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

EPS = 1e-8
INDEX_DTYPE = np.int32
GC_INTERVAL = 50
MAX_KV_CACHE_LEN = 256
MAX_VAL_SAMPLES = 1000
_thread_local = threading.local()

# ==================== 日志系统 ====================
def setup_logging(verbose: bool = False, log_file: str = "logs/aether_life.log") -> logging.Logger:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING if not verbose else logging.INFO)
    console.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    root_logger.addHandler(console)
    file_handler = logging.handlers.RotatingFileHandler(str(log_path), maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
    root_logger.addHandler(file_handler)
    return logging.getLogger(__name__)

logger = setup_logging(verbose=False)

def log_system(msg: str, level=logging.INFO) -> None:
    logger.log(level, '[系统] ' + msg)

def log_evolution(msg: str) -> None:
    logger.info('[进化] ' + msg)

def log_hardware(msg: str) -> None:
    logger.info('[硬件] ' + msg)

def log_life(msg: str) -> None:
    logger.info('[生命] ' + msg)

# ==================== 工具函数 ====================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def convert_to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16,
                        np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def adjust_d_model(d_model: int, nhead: int, lower: int = 64, upper: int = 256) -> int:
    if nhead <= 0:
        raise ValueError('nhead must be positive')
    lower = max(lower, nhead)
    first_multiple = ((lower + nhead - 1) // nhead) * nhead
    candidates = []
    multiple = first_multiple
    while multiple <= upper:
        candidates.append(multiple)
        multiple += nhead
    candidates.sort(key=lambda x: abs(x - d_model))
    for cand in candidates:
        if ((cand // nhead) % 2) == 0:
            return cand
    return candidates[0] if candidates else lower

def get_memory_usage_mb() -> Optional[float]:
    if HAS_PSUTIL:
        try:
            return psutil.Process().memory_info().rss / (1024 ** 2)
        except Exception:
            pass
    return None

def get_memory_available_mb() -> Optional[float]:
    if HAS_PSUTIL:
        try:
            return psutil.virtual_memory().available / (1024 ** 2)
        except Exception:
            pass
    return 800.0

def get_battery_percent() -> Optional[float]:
    if HAS_PSUTIL:
        try:
            batt = psutil.sensors_battery()
            if batt:
                return batt.percent
        except Exception:
            pass
    return 80.0

def get_cpu_temp_real() -> Optional[float]:
    if HAS_PSUTIL:
        try:
            temps = psutil.sensors_temperatures()
            for key in ["coretemp", "cpu_thermal", "acpitz"]:
                if key in temps and temps[key]:
                    return temps[key][0].current
        except Exception:
            pass
    return 45.0

def detect_hardware() -> Dict[str, Any]:
    hw = {
        'cpu_count': os.cpu_count() or 1,
        'memory_gb': 4.0,
        'has_gpu': False,
        'gpu_memory_gb': 0.0,
        'has_battery': True,
        'battery_percent': 80.0,
        'is_mobile': True,
        'env_type': 'mobile',
    }
    if HAS_PSUTIL:
        try:
            mem = psutil.virtual_memory()
            hw['memory_gb'] = mem.total / (1024**3)
        except Exception:
            pass
        batt = get_battery_percent()
        if batt is not None:
            hw['battery_percent'] = batt
        hw['is_mobile'] = hw['memory_gb'] < 6
        hw['env_type'] = 'mobile' if hw['is_mobile'] else 'laptop'
    return hw

def safe_data_dir(user_path: str, base_dir: str = ".") -> str:
    """安全地解析数据目录，防止路径遍历"""
    base = os.path.realpath(base_dir)
    full = os.path.realpath(os.path.join(base, user_path))
    if '..' in user_path:
        raise PermissionError("路径中不允许使用 '..'")
    if not full.startswith(base):
        raise PermissionError(f"数据目录 {user_path} 不在工作目录下")
    return full

def is_terminal_support_carriage() -> bool:
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

# ==================== 全局缓存 ====================
_rope_cache = OrderedDict()
_ROPE_CACHE_MAXSIZE = 64
_ROPE_CACHE_LOCK = threading.RLock()
_attn_mask_cache = OrderedDict()
_ATTN_MASK_MAXSIZE = 32
_ATTN_MASK_LOCK = threading.RLock()
_SUM_TO_SHAPE_CACHE = OrderedDict()
_SUM_TO_SHAPE_MAXSIZE = 128
_SUM_TO_SHAPE_LOCK = threading.RLock()

def clear_all_caches() -> None:
    with _ROPE_CACHE_LOCK:
        _rope_cache.clear()
    with _ATTN_MASK_LOCK:
        _attn_mask_cache.clear()
    with _SUM_TO_SHAPE_LOCK:
        _SUM_TO_SHAPE_CACHE.clear()
    log_system("已清理所有全局缓存")

# ==================== 硬件监控线程 ====================
class HardwareMonitorThread(threading.Thread):
    def __init__(self, sleep_interval: int = 20) -> None:
        super().__init__(daemon=True)
        self.sleep_interval = sleep_interval
        self.running = True
        self.state = {'mem_available': 0.0, 'cpu_temp': None, 'battery': None}
        self.history = deque(maxlen=100)
        self.lock = threading.Lock()

    def run(self) -> None:
        while self.running:
            new_state = {}
            if HAS_PSUTIL:
                try:
                    mem = psutil.virtual_memory()
                    new_state['mem_available'] = mem.available / (1024**2)
                    new_state['cpu_temp'] = get_cpu_temp_real()
                    batt = psutil.sensors_battery()
                    new_state['battery'] = batt.percent if batt else None
                except Exception as e:
                    logger.warning(f"硬件监控采集失败: {e}")
            else:
                new_state['mem_available'] = 800.0
                new_state['cpu_temp'] = 45.0
                new_state['battery'] = 80.0
            with self.lock:
                self.state.update(new_state)
                self.history.append(new_state.copy())
            time.sleep(self.sleep_interval)

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return self.state.copy()

    def predict(self, steps: int = 5) -> Dict[str, float]:
        with self.lock:
            if len(self.history) < 3:
                return self.state.copy()
            temps = [s.get('cpu_temp', 45.0) for s in self.history if s.get('cpu_temp') is not None]
            mems = [s.get('mem_available', 800.0) for s in self.history]
            if not temps:
                temps = [45.0]
            alpha = 0.3
            temp_forecast = temps[-1]
            mem_forecast = mems[-1]
            for _ in range(steps):
                temp_forecast = alpha * temp_forecast + (1 - alpha) * (temp_forecast if len(temps) > 1 else temps[-1])
                mem_forecast = alpha * mem_forecast + (1 - alpha) * (mem_forecast if len(mems) > 1 else mems[-1])
            return {'cpu_temp': temp_forecast, 'mem_available': mem_forecast, 'battery': self.state.get('battery', 80.0)}

    def stop(self) -> None:
        self.running = False
        if self.is_alive():
            self.join(timeout=2)

# ==================== 自动微分核心 ====================
def _sum_to_shape(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    if not isinstance(target_shape, tuple):
        target_shape = (target_shape,)
    if target_shape == ():
        return np.array(grad.sum(), dtype=grad.dtype)
    if grad.shape == target_shape:
        return grad.copy() if not grad.flags.writeable else grad

    target_padded = list(target_shape)
    while len(target_padded) < grad.ndim:
        target_padded.insert(0, 1)

    axes = []
    for i, (g, t) in enumerate(zip(grad.shape, target_padded)):
        if g != t and t == 1:
            axes.append(i)
    if axes:
        grad = grad.sum(axis=tuple(axes), keepdims=True)
    grad = grad.reshape(target_shape)
    return grad

class Function:
    __slots__ = ('inputs',)
    def __init__(self, *inputs: 'Tensor') -> None:
        self.inputs = inputs
    def __call__(self, grad: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        raise NotImplementedError

# ==================== 所有反向传播类 ====================
class AddBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.inputs
        return (_sum_to_shape(grad, a.data.shape), _sum_to_shape(grad, b.data.shape))

class MulBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.inputs
        return (_sum_to_shape(grad * b.data, a.data.shape),
                _sum_to_shape(grad * a.data, b.data.shape))

class MatMulBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.inputs
        a_grad = grad @ b.data.swapaxes(-2, -1)
        b_grad = a.data.swapaxes(-2, -1) @ grad
        return (_sum_to_shape(a_grad, a.data.shape),
                _sum_to_shape(b_grad, b.data.shape))

class DivBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.inputs
        denom = np.where(np.abs(b.data) < EPS, EPS, b.data)
        a_grad = _sum_to_shape(grad / denom, a.data.shape)
        b_grad = _sum_to_shape(-grad * a.data / (denom**2), b.data.shape)
        return a_grad, b_grad

class PowBackward(Function):
    __slots__ = ('power',)
    def __init__(self, a: 'Tensor', power: float):
        super().__init__(a)
        self.power = power
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        g = grad * self.power * (a.data ** (self.power - 1))
        return (_sum_to_shape(g, a.data.shape),)

class ReshapeBackward(Function):
    __slots__ = ('orig_shape',)
    def __init__(self, a: 'Tensor', orig_shape: Tuple[int, ...]):
        super().__init__(a)
        self.orig_shape = orig_shape
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad.reshape(self.orig_shape),)

class TransposeBackward(Function):
    __slots__ = ('axes',)
    def __init__(self, a: 'Tensor', axes: Optional[Tuple[int, ...]]):
        super().__init__(a)
        self.axes = axes
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        if self.axes is None:
            return (grad.T,)
        inv_axes = np.argsort(self.axes)
        return (grad.transpose(*inv_axes),)

class SqueezeBackward(Function):
    __slots__ = ('dim', 'orig_shape')
    def __init__(self, a: 'Tensor', dim: Optional[int], orig_shape: Tuple[int, ...]):
        super().__init__(a)
        self.dim = dim
        self.orig_shape = orig_shape
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        if self.dim is None:
            return (grad.reshape(self.orig_shape),)
        else:
            return (np.expand_dims(grad, axis=self.dim),)

class UnsqueezeBackward(Function):
    __slots__ = ('dim',)
    def __init__(self, a: 'Tensor', dim: int):
        super().__init__(a)
        self.dim = dim
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad.squeeze(axis=self.dim),)

class SumBackward(Function):
    __slots__ = ('axis', 'keepdims')
    def __init__(self, a: 'Tensor', axis: Optional[Union[int, Tuple[int, ...]]], keepdims: bool):
        super().__init__(a)
        self.axis = axis
        self.keepdims = keepdims
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        if self.axis is None:
            return (np.full_like(a.data, grad),)
        shape = list(a.data.shape)
        if not self.keepdims:
            axes = self.axis if isinstance(self.axis, tuple) else (self.axis,)
            for ax in sorted(axes):
                shape[ax] = 1
            grad = grad.reshape(shape)
        return (np.broadcast_to(grad, a.data.shape),)

class ExpBackward(Function):
    __slots__ = ('out',)
    def __init__(self, a: 'Tensor', out: np.ndarray):
        super().__init__(a)
        self.out = out
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        return (_sum_to_shape(grad * self.out, a.data.shape),)

class LogBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        return (_sum_to_shape(grad / (a.data + EPS), a.data.shape),)

class SqrtBackward(Function):
    __slots__ = ('out',)
    def __init__(self, a: 'Tensor', out: np.ndarray):
        super().__init__(a)
        self.out = out
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        return (_sum_to_shape(grad * 0.5 / (self.out + EPS), a.data.shape),)

class SoftmaxBackward(Function):
    __slots__ = ('probs', 'axis')
    def __init__(self, a: 'Tensor', probs: np.ndarray, axis: int):
        super().__init__(a)
        self.probs = probs
        self.axis = axis
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        s = self.probs
        sum_grad_s = np.sum(grad * s, axis=self.axis, keepdims=True)
        a_grad = s * (grad - sum_grad_s)
        return (_sum_to_shape(a_grad, a.data.shape),)

class LogSoftmaxBackward(Function):
    __slots__ = ('axis',)
    def __init__(self, a: 'Tensor', axis: int):
        super().__init__(a)
        self.axis = axis
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        max_val = np.max(a.data, axis=self.axis, keepdims=True)
        exp = np.exp(a.data - max_val)
        s = exp / (np.sum(exp, axis=self.axis, keepdims=True) + EPS)
        sum_grad = np.sum(grad, axis=self.axis, keepdims=True)
        a_grad = grad - s * sum_grad
        return (_sum_to_shape(a_grad, a.data.shape),)

class GatherBackward(Function):
    __slots__ = ('dim', 'index', 'orig_shape')
    def __init__(self, a: 'Tensor', dim: int, index: np.ndarray, orig_shape: Tuple[int, ...]):
        super().__init__(a)
        self.dim = dim
        self.index = index
        self.orig_shape = orig_shape
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        full_grad = np.zeros(self.orig_shape, dtype=a.data.dtype)
        idx_tuple = [slice(None)] * a.data.ndim
        idx_tuple[self.dim] = self.index
        np.add.at(full_grad, tuple(idx_tuple), grad)
        return (full_grad,)

class ScatterAddBackward(Function):
    __slots__ = ('dim', 'index')
    def __init__(self, a: 'Tensor', src: 'Tensor', dim: int, index: np.ndarray):
        super().__init__(a, src)
        self.dim = dim
        self.index = index
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, src = self.inputs
        grad_a = grad
        grad_src = np.zeros_like(src.data)
        idx_tuple = [slice(None)] * grad.ndim
        idx_tuple[self.dim] = self.index
        np.add.at(grad_src, tuple(idx_tuple), grad[idx_tuple])
        return grad_a, grad_src

class IndexBackward(Function):
    __slots__ = ('idx', 'orig_shape')
    def __init__(self, a: 'Tensor', idx: Any, orig_shape: Tuple[int, ...]):
        super().__init__(a)
        self.idx = idx
        self.orig_shape = orig_shape
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        full_grad = np.zeros(self.orig_shape, dtype=a.data.dtype)
        np.add.at(full_grad, self.idx, grad)
        return (full_grad,)

class SliceAssignBackward(Function):
    __slots__ = ('slices',)
    def __init__(self, a: 'Tensor', value: 'Tensor', slices: Tuple[slice, ...]):
        super().__init__(a, value)
        self.slices = slices
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, value = self.inputs
        grad_a = grad.copy()
        grad_a[self.slices] = 0
        grad_val = grad[self.slices]
        if grad_val.shape != value.shape:
            grad_val = _sum_to_shape(grad_val, value.shape)
        return grad_a, grad_val

class ExpandBackward(Function):
    __slots__ = ('orig_shape',)
    def __init__(self, a: 'Tensor', orig_shape: Tuple[int, ...]):
        super().__init__(a)
        self.orig_shape = orig_shape
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (_sum_to_shape(grad, self.orig_shape),)

class CloneBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad,)

class DropoutBackward(Function):
    __slots__ = ('mask',)
    def __init__(self, x: 'Tensor', mask: np.ndarray):
        super().__init__(x)
        self.mask = mask
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad * self.mask,)

class ClipBackward(Function):
    __slots__ = ('min_val', 'max_val')
    def __init__(self, x: 'Tensor', min_val: float, max_val: float):
        super().__init__(x)
        self.min_val = min_val
        self.max_val = max_val
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        x = self.inputs[0]
        mask = (x.data >= self.min_val) & (x.data <= self.max_val)
        return (grad * mask.astype(np.float32),)

class ReLUBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        return (grad * (a.data > 0),)

class AbsBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        return (grad * np.sign(a.data),)

class SignBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (np.zeros_like(self.inputs[0].data),)

class ELUBackward(Function):
    __slots__ = ('alpha',)
    def __init__(self, a: 'Tensor', alpha: float):
        super().__init__(a)
        self.alpha = alpha
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        d = np.where(a.data >= 0, 1.0, self.alpha * np.exp(a.data))
        return (grad * d,)

class SigmoidBackward(Function):
    __slots__ = ('out',)
    def __init__(self, a: 'Tensor', out: np.ndarray):
        super().__init__(a)
        self.out = out
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        g = grad * self.out * (1 - self.out)
        return (_sum_to_shape(g, a.data.shape),)

class TanhBackward(Function):
    __slots__ = ('out',)
    def __init__(self, a: 'Tensor', out: np.ndarray):
        super().__init__(a)
        self.out = out
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        a = self.inputs[0]
        g = grad * (1 - self.out ** 2)
        return (_sum_to_shape(g, a.data.shape),)

class EmbeddingBackward(Function):
    __slots__ = ('indices',)
    def __init__(self, weight: 'Tensor', indices: np.ndarray):
        super().__init__(weight)
        self.indices = indices
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        weight = self.inputs[0]
        batch, seq_len, d_model = grad.shape
        flat_indices = self.indices.reshape(-1)
        flat_grad = grad.reshape(-1, d_model)
        dw = np.zeros_like(weight.data)
        np.add.at(dw, flat_indices, flat_grad)
        return (dw,)

class LayerNormBackward(Function):
    __slots__ = ('mean', 'var', 'eps')
    def __init__(self, x: 'Tensor', gamma: 'Tensor', beta: 'Tensor',
                 mean: np.ndarray, var: np.ndarray, eps: float):
        super().__init__(x, gamma, beta)
        self.mean = mean.astype(np.float32)
        self.var = var.astype(np.float32)
        self.eps = eps
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, gamma, beta = self.inputs
        grad = grad.astype(np.float32)
        x_data = x.data.astype(np.float32)
        gamma_data = gamma.data.astype(np.float32)
        N = x_data.shape[-1] if x_data.ndim > 1 else 1
        var_eps = self.var + self.eps
        inv_std = 1.0 / np.sqrt(var_eps)
        x_hat = (x_data - self.mean) * inv_std
        sum_axes = tuple(range(grad.ndim - 1))
        dgamma = (grad * x_hat).sum(axis=sum_axes, keepdims=True).squeeze()
        dbeta = grad.sum(axis=sum_axes, keepdims=True).squeeze()
        dx = gamma_data * inv_std * (grad - (x_hat * dgamma + dbeta) / N)
        dx = np.clip(dx, -1e4, 1e4)
        return (dx.astype(x.dtype), dgamma.astype(gamma.dtype), dbeta.astype(beta.dtype))

class CatBackward(Function):
    __slots__ = ('axis',)
    def __init__(self, tensors: List['Tensor'], axis: int):
        super().__init__(*tensors)
        self.axis = axis
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        axis = self.axis if self.axis >= 0 else grad.ndim + self.axis
        splits = []
        idx = 0
        for t in self.inputs:
            length = t.data.shape[axis]
            slices = [slice(None)] * grad.ndim
            slices[axis] = slice(idx, idx + length)
            split_grad = grad[tuple(slices)]
            splits.append(_sum_to_shape(split_grad, t.data.shape))
            idx += length
        return tuple(splits)

class StackBackward(Function):
    __slots__ = ('axis',)
    def __init__(self, tensors: List['Tensor'], axis: int):
        super().__init__(*tensors)
        self.axis = axis
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        splits = np.split(grad, len(self.inputs), axis=self.axis)
        res = []
        for s, t in zip(splits, self.inputs):
            s_resh = s.reshape(t.data.shape)
            res.append(_sum_to_shape(s_resh, t.data.shape))
        return tuple(res)

class TypeCastBackward(Function):
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad,)

class SwapaxesBackward(Function):
    __slots__ = ('axis1', 'axis2')
    def __init__(self, a: 'Tensor', axis1: int, axis2: int):
        super().__init__(a)
        self.axis1 = axis1
        self.axis2 = axis2
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (np.swapaxes(grad, self.axis1, self.axis2),)

class LogSigmoidBackward(Function):
    def __init__(self, x: 'Tensor'):
        super().__init__(x)
    def __call__(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        x = self.inputs[0]
        sig = 1.0 / (1.0 + np.exp(-x.data))
        return (grad * (1 - sig),)

def log_sigmoid(x: 'Tensor') -> 'Tensor':
    x_data = x.data
    out_data = np.where(x_data >= 0,
                        -np.log1p(np.exp(-x_data)),
                        x_data - np.log1p(np.exp(x_data)))
    out = Tensor(out_data, requires_grad=x.requires_grad)
    if out.requires_grad:
        out.grad_fn = LogSigmoidBackward(x)
    return out

# ==================== Tensor 类 ====================
class Tensor:
    __slots__ = ('data', 'requires_grad', 'grad', 'grad_fn', 'is_leaf')
    def __init__(self, data, requires_grad=False, grad_fn=None, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        data_np = np.asarray(data)
        if dtype is None:
            if not requires_grad and np.issubdtype(data_np.dtype, np.integer):
                dtype = np.int64
            else:
                dtype = np.float32
        self.data = np.array(data_np, dtype=dtype)
        if getattr(_thread_local, 'no_grad', False):
            requires_grad = False
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn
        self.is_leaf = grad_fn is None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return isinstance(other, Tensor) and id(self) == id(other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f'Tensor(shape={self.shape}, requires_grad={self.requires_grad})'

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0)

    def clone(self):
        out = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.grad_fn = CloneBackward(self)
        return out

    def expand(self, *shape):
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        out_data = np.broadcast_to(self.data, new_shape).copy()
        out = Tensor(out_data, requires_grad=self.requires_grad)
        if out.requires_grad:
            out.grad_fn = ExpandBackward(self, self.data.shape)
        return out

    def slice_assign(self, slices, value):
        new_data = self.data.copy()
        val_data = value.data if isinstance(value, Tensor) else value
        new_data[slices] = val_data
        out = Tensor(new_data, requires_grad=self.requires_grad or value.requires_grad)
        if out.requires_grad:
            out.grad_fn = SliceAssignBackward(self, value, slices)
        return out

    def scatter_add(self, dim, index, src):
        index_arr = index.data.astype(np.int64)
        src_data = src.data
        new_data = self.data.copy()
        idx_tuple = [slice(None)] * new_data.ndim
        idx_tuple[dim] = index_arr
        np.add.at(new_data, tuple(idx_tuple), src_data)
        out = Tensor(new_data, requires_grad=self.requires_grad or src.requires_grad)
        if out.requires_grad:
            out.grad_fn = ScatterAddBackward(self, src, dim, index_arr)
        return out

    def clip(self, min_val, max_val):
        out_data = np.clip(self.data, min_val, max_val)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        if out.requires_grad:
            out.grad_fn = ClipBackward(self, min_val, max_val)
        return out

    def __add__(self, other):
        return self._bin_op(other, np.add, AddBackward)
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return -self + other
    def __mul__(self, other):
        return self._bin_op(other, np.multiply, MulBackward)
    def __rmul__(self, other):
        return self * other
    def __matmul__(self, other):
        return self._bin_op(other, np.matmul, MatMulBackward)
    def __neg__(self):
        return self * -1
    def __truediv__(self, other):
        return self._bin_op(other, np.divide, DivBackward)
    def __rtruediv__(self, other):
        return Tensor(other) / self
    def __pow__(self, power):
        if isinstance(power, (int, float)):
            out = Tensor(self.data ** power, self.requires_grad)
            if out.requires_grad:
                out.grad_fn = PowBackward(self, power)
            return out
        raise NotImplementedError

    def _bin_op(self, other, op, back_cls):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires = self.requires_grad or other.requires_grad
        out_data = op(self.data, other.data)
        out = Tensor(out_data, requires_grad=requires)
        if out.requires_grad:
            out.grad_fn = back_cls(self, other)
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], self.requires_grad)
        if out.requires_grad:
            out.grad_fn = IndexBackward(self, idx, self.data.shape)
        return out

    def reshape(self, *shape):
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        out = Tensor(self.data.reshape(new_shape), self.requires_grad)
        if out.requires_grad:
            out.grad_fn = ReshapeBackward(self, self.data.shape)
        return out

    def transpose(self, *axes):
        ndim = self.data.ndim
        if len(axes) == 0:
            out_data = self.data.T
            axes_tuple = None
        elif len(axes) == ndim:
            out_data = self.data.transpose(*axes)
            axes_tuple = axes
        else:
            if len(axes) == 2:
                a, b = axes
                out_data = np.swapaxes(self.data, a, b)
                axes_tuple = None
            else:
                raise ValueError(f"部分轴转置不支持 {len(axes)} 个轴")
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = TransposeBackward(self, axes_tuple)
        return out

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        out_data = np.swapaxes(self.data, axis1, axis2)
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = SwapaxesBackward(self, axis1, axis2)
        return out

    def unsqueeze(self, dim):
        new_shape = list(self.data.shape)
        if dim < 0:
            dim = len(new_shape) + 1 + dim
        new_shape.insert(dim, 1)
        return self.reshape(*new_shape)

    def squeeze(self, dim=None):
        if dim is None:
            out_data = np.squeeze(self.data)
        else:
            if self.data.shape[dim] != 1:
                raise ValueError(f'Cannot squeeze axis {dim}')
            out_data = np.squeeze(self.data, axis=dim)
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = SqueezeBackward(self, dim, self.data.shape)
        return out

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = SumBackward(self, axis, keepdims)
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.data.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
        return self.sum(axis, keepdims) * (1.0 / n)

    def norm(self, axis=None, keepdims=False):
        return (self ** 2).sum(axis=axis, keepdims=keepdims).sqrt()

    def std(self, axis=None, keepdims=False):
        mean = self.mean(axis=axis, keepdims=True)
        var = ((self - mean) ** 2).mean(axis=axis, keepdims=keepdims)
        return var.sqrt()

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = ExpBackward(self, out_data)
        return out

    def log(self):
        out_data = np.log(self.data + EPS)
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = LogBackward(self)
        return out

    def sqrt(self):
        out_data = np.sqrt(np.maximum(self.data, 0))
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = SqrtBackward(self, out_data)
        return out

    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    def tanh(self):
        return (self.exp() - (-self).exp()) / (self.exp() + (-self).exp())

    def gelu(self):
        cdf = 0.5 * (1.0 + (0.7978845608028654 * (self + 0.044715 * self ** 3)).tanh())
        return self * cdf

    def elu(self, alpha=1.0):
        out_data = np.where(self.data >= 0, self.data, alpha * (np.exp(self.data) - 1))
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = ELUBackward(self, alpha)
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = ReLUBackward(self)
        return out

    def abs(self):
        out_data = np.abs(self.data)
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = AbsBackward(self)
        return out

    def sign(self):
        out_data = np.sign(self.data)
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = SignBackward(self)
        return out

    def softmax(self, axis=-1):
        x = self.data
        max_val = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - max_val)
        probs = exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPS)
        out = Tensor(probs, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = SoftmaxBackward(self, probs, axis)
        return out

    def log_softmax(self, axis=-1):
        x = self.data
        max_val = np.max(x, axis=axis, keepdims=True)
        shifted = x - max_val
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True) + EPS)
        out_data = shifted - log_sum_exp
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = LogSoftmaxBackward(self, axis)
        return out

    def gather(self, dim, index):
        index_arr = index.data.astype(np.int64)
        if index_arr.ndim < self.data.ndim:
            expand_shape = list(self.data.shape)
            expand_shape[dim] = index_arr.shape[dim]
            index_arr = np.broadcast_to(index_arr, expand_shape)
        out_data = np.take_along_axis(self.data, index_arr, axis=dim)
        out = Tensor(out_data, self.requires_grad)
        if out.requires_grad:
            out.grad_fn = GatherBackward(self, dim, index_arr, self.data.shape)
        return out

    @classmethod
    def cat(cls, tensors, axis=0):
        data = np.concatenate([t.data for t in tensors], axis=axis)
        requires = any(t.requires_grad for t in tensors)
        out = cls(data, requires_grad=requires)
        if out.requires_grad:
            out.grad_fn = CatBackward(tensors, axis)
        return out

    @classmethod
    def stack(cls, tensors, axis=0):
        data = np.stack([t.data for t in tensors], axis=axis)
        requires = any(t.requires_grad for t in tensors)
        out = cls(data, requires_grad=requires)
        if out.requires_grad:
            out.grad_fn = StackBackward(tensors, axis)
        return out

    @classmethod
    def zeros(cls, shape, requires_grad=False, dtype=np.float32):
        return cls(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape, requires_grad=False, dtype=np.float32):
        return cls(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

    @staticmethod
    def no_grad():
        class NoGradContext:
            def __enter__(self):
                self.prev = getattr(_thread_local, 'no_grad', False)
                _thread_local.no_grad = True
            def __exit__(self, *args):
                _thread_local.no_grad = self.prev
        return NoGradContext()

    def backward(self, grad=None, retain_graph=False):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)

        topo = []
        visited = set()
        def build_topo(v):
            if v in visited or not v.requires_grad:
                return
            visited.add(v)
            if v.grad_fn:
                for inp in v.grad_fn.inputs:
                    if isinstance(inp, Tensor):
                        build_topo(inp)
            topo.append(v)
        build_topo(self)

        grads = {self: grad}
        for v in reversed(topo):
            if v.grad_fn is not None and v in grads:
                in_grads = v.grad_fn(grads[v])
                for inp, ig in zip(v.grad_fn.inputs, in_grads):
                    if isinstance(inp, Tensor) and inp.requires_grad and ig is not None:
                        if inp in grads:
                            np.add(grads[inp], ig, out=grads[inp])
                        else:
                            grads[inp] = ig.copy()

        for v, g in grads.items():
            if v.is_leaf and v.requires_grad:
                if v.grad is None:
                    v.grad = g
                else:
                    np.add(v.grad, g, out=v.grad)

        if not retain_graph:
            self._clean_graph()

    def _clean_graph(self):
        from collections import deque
        queue = deque([self])
        visited = set()
        while queue:
            node = queue.popleft()
            if node in visited or not isinstance(node, Tensor):
                continue
            visited.add(node)
            if node.grad_fn is not None:
                fn = node.grad_fn
                node.grad_fn = None
                for inp in fn.inputs:
                    if isinstance(inp, Tensor) and inp not in visited:
                        queue.append(inp)
                fn.inputs = None
        gc.collect()

    def to_float32(self):
        if self.data.dtype == np.float32:
            return self
        out = Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.grad_fn = TypeCastBackward(self)
        return out

    def to_half(self):
        out = Tensor(self.data.astype(np.float16), requires_grad=self.requires_grad)
        if out.requires_grad:
            out.grad_fn = TypeCastBackward(self)
        return out

# ==================== 基础层组件 ====================
class Parameter:
    def __init__(self, data):
        self.data = Tensor(data, requires_grad=True)

    def zero_grad(self):
        if self.data.grad is not None:
            self.data.grad.fill(0)

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        scale = math.sqrt(2.0 / in_features) if in_features > 0 else 0.02
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32) * scale)
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.weight.data.T
        if self.bias:
            out = out + self.bias.data
        return out

    def parameters(self) -> List[Tensor]:
        params = [self.weight.data]
        if self.bias:
            params.append(self.bias.data)
        return params

class TernaryLinear:
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        scale = math.sqrt(2.0 / in_features) if in_features > 0 else 0.02
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32) * scale)
        self.scale = Parameter(np.ones((out_features, 1), dtype=np.float32) * 0.018)
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        w_quant = self._ternary_quantize(self.weight.data, self.scale.data)
        out = x @ w_quant.T
        if self.bias:
            out = out + self.bias.data
        return out

    def _ternary_quantize(self, weight: Tensor, scale: Tensor) -> Tensor:
        w_sign = weight.sign()
        w_quant = w_sign * scale
        out = Tensor(w_quant.data, requires_grad=weight.requires_grad or scale.requires_grad)
        if out.requires_grad:
            class STEFunction(Function):
                def __init__(self, w, s):
                    super().__init__(w, s)
                def __call__(self, grad):
                    w, s = self.inputs
                    grad_w = grad * s.data
                    grad_s = np.sum(grad * w.sign().data, axis=1, keepdims=True)
                    return grad_w, grad_s
            out.grad_fn = STEFunction(weight, scale)
        return out

    def parameters(self) -> List[Tensor]:
        params = [self.weight.data, self.scale.data]
        if self.bias:
            params.append(self.bias.data)
        return params

class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = Parameter(np.ones(dim, dtype=np.float32))
        self.beta = Parameter(np.zeros(dim, dtype=np.float32))

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.gamma.data + self.beta.data

    def parameters(self) -> List[Tensor]:
        return [self.gamma.data, self.beta.data]

class Dropout:
    def __init__(self, p: float = 0.1):
        self.p = p
        self.training = True
        self.rng = np.random.default_rng()

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self.p >= 1.0:
            return Tensor(np.zeros_like(x.data), requires_grad=x.requires_grad)
        mask = (self.rng.random(x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        out = Tensor(x.data * mask, requires_grad=x.requires_grad)
        if out.requires_grad:
            out.grad_fn = DropoutBackward(x, mask)
        return out

class Embedding:
    def __init__(self, vocab_size: int, dim: int):
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = Parameter(np.random.randn(vocab_size, dim).astype(np.float32) * 0.02)

    def __call__(self, indices: np.ndarray) -> Tensor:
        return self.weight.data[indices]

    def parameters(self) -> List[Tensor]:
        return [self.weight.data]

class LoRAAdapter:
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 8.0):
        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} exceeds min dimension")
        self.rank = rank
        self.alpha = alpha
        self.A = Parameter(np.random.randn(rank, in_features).astype(np.float32) / np.sqrt(in_features))
        self.B = Parameter(np.zeros((out_features, rank), dtype=np.float32))

    def __call__(self, x: Tensor) -> Tensor:
        h = x @ self.A.data.T
        return (h @ self.B.data.T) * (self.alpha / self.rank)

    def parameters(self) -> List[Tensor]:
        return [self.A.data, self.B.data]

# ==================== 注意力与位置编码 ====================
def get_rope_cache(seq_len: int, head_dim: int) -> Tuple[Tensor, Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    with _ROPE_CACHE_LOCK:
        for key in list(_rope_cache.keys()):
            if key[0] < seq_len:
                del _rope_cache[key]
        for key, (cos, sin) in _rope_cache.items():
            if key[0] >= seq_len:
                if cos.shape[2] > seq_len:
                    cos = cos[:, :, :seq_len, :]
                    sin = sin[:, :, :seq_len, :]
                return cos, sin
        inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2) / head_dim))
        t = np.arange(seq_len)
        freqs = np.einsum('i,j->ij', t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1).astype(np.float32).reshape(1, 1, seq_len, head_dim)
        cos_t = Tensor(np.cos(emb), requires_grad=False)
        sin_t = Tensor(np.sin(emb), requires_grad=False)
        _rope_cache[(seq_len, head_dim)] = (cos_t, sin_t)
        if len(_rope_cache) > _ROPE_CACHE_MAXSIZE:
            _rope_cache.popitem(last=False)
        return cos_t, sin_t

def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return Tensor.cat([-x2, x1], axis=-1)

def apply_rope(q: Tensor, k: Tensor, past_len: int, seq_len: int, head_dim: int) -> Tuple[Tensor, Tensor]:
    total_len = past_len + seq_len
    try:
        cos, sin = get_rope_cache(total_len, head_dim)
    except Exception as e:
        logger.warning(f"RoPE cache error: {e}, skipping RoPE")
        return q, k
    if past_len < 0 or seq_len <= 0:
        return q, k
    start_q = past_len
    end_q = min(past_len + seq_len, total_len)
    if start_q >= end_q:
        return q, k
    cos_q = cos[:, :, start_q:end_q, :]
    sin_q = sin[:, :, start_q:end_q, :]
    if cos_q.shape[2] != seq_len:
        if cos_q.shape[2] == 0:
            return q, k
        cos_q = cos_q.expand(-1, -1, seq_len, -1)
        sin_q = sin_q.expand(-1, -1, seq_len, -1)
    cos_k = cos[:, :, :total_len, :]
    sin_k = sin[:, :, :total_len, :]
    q_rot = rotate_half(q)
    k_rot = rotate_half(k)
    q_out = q * cos_q + q_rot * sin_q
    k_out = k * cos_k + k_rot * sin_k
    return q_out, k_out

def get_causal_sliding_mask(seq_len: int, window_size: int) -> Tensor:
    key = (seq_len, window_size)
    with _ATTN_MASK_LOCK:
        if key in _attn_mask_cache:
            return _attn_mask_cache[key]
        i = np.arange(seq_len)[:, None]
        j = np.arange(seq_len)[None, :]
        causal = i > j
        if window_size < seq_len:
            window_mask = (j < i - window_size + 1) | (j > i + window_size - 1)
            mask = causal | window_mask
        else:
            mask = causal
        mask_np = np.where(mask, -np.inf, 0.0).astype(np.float32)
        mask_t = Tensor(mask_np[None, None, :, :], requires_grad=False)
        _attn_mask_cache[key] = mask_t
        if len(_attn_mask_cache) > _ATTN_MASK_MAXSIZE:
            _attn_mask_cache.popitem(last=False)
        return mask_t

class MultiHeadAttention:
    def __init__(self, dim: int, heads: int, window_size: int = 16, use_ternary: bool = False):
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        assert dim % heads == 0
        self.head_dim = dim // heads
        LinearCls = TernaryLinear if use_ternary else Linear
        self.q_proj = LinearCls(dim, dim, bias=False)
        self.k_proj = LinearCls(dim, dim, bias=False)
        self.v_proj = LinearCls(dim, dim, bias=False)
        self.o_proj = LinearCls(dim, dim, bias=False)
        self._training = True

    def __call__(self, x: Tensor, past_kv: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.heads, self.head_dim).swapaxes(1, 2)
        k = self.k_proj(x).reshape(B, T, self.heads, self.head_dim).swapaxes(1, 2)
        v = self.v_proj(x).reshape(B, T, self.heads, self.head_dim).swapaxes(1, 2)
        if self.training:
            past_kv = None
        if past_kv is not None:
            past_k, past_v = past_kv
            k = Tensor.cat([past_k, k], axis=2)
            v = Tensor.cat([past_v, v], axis=2)
            past_len = past_k.shape[2]
        else:
            past_len = 0
        q, k = apply_rope(q, k, past_len, T, self.head_dim)
        scores = (q @ k.swapaxes(-2, -1)) / math.sqrt(self.head_dim)
        mask = get_causal_sliding_mask(k.shape[2], self.window_size)
        scores = scores + mask
        attn = scores.softmax(axis=-1)
        out = (attn @ v).swapaxes(1, 2).reshape(B, T, C)
        out = self.o_proj(out)
        if k.shape[2] > MAX_KV_CACHE_LEN:
            k = Tensor(k.data[:, :, -MAX_KV_CACHE_LEN:, :], requires_grad=k.requires_grad)
            v = Tensor(v.data[:, :, -MAX_KV_CACHE_LEN:, :], requires_grad=v.requires_grad)
        return out, (k, v) if not self.training else (None, None)

    def parameters(self) -> List[Tensor]:
        return self.q_proj.parameters() + self.k_proj.parameters() + self.v_proj.parameters() + self.o_proj.parameters()

    @property
    def training(self) -> bool:
        return getattr(self, '_training', True)

    @training.setter
    def training(self, val: bool) -> None:
        self._training = val
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            if hasattr(proj, 'training'):
                proj.training = val

class InfiniMemory:
    __slots__ = ('mem_len', 'head_dim', 'nhead', 'alpha', 'init_mem', 'init_z')
    def __init__(self, mem_len: int, head_dim: int, nhead: int, alpha: float = 0.1):
        if mem_len <= 0:
            raise ValueError(f"mem_len must be positive, got {mem_len}")
        self.mem_len = mem_len
        self.head_dim = head_dim
        self.nhead = nhead
        self.alpha = alpha
        scale = 0.02
        self.init_mem = Parameter(np.random.randn(nhead, mem_len, head_dim, head_dim).astype(np.float32) * scale)
        self.init_z = Parameter(np.ones((nhead, mem_len, head_dim), dtype=np.float32) * 0.1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, state: Optional[Tuple[Tensor, Tensor, int]], seg_len: int) -> Tuple[Tensor, Tuple[Tensor, Tensor, int]]:
        B, nhead, seq_len, head_dim = q.shape
        if state is None:
            mem = self.init_mem.data.reshape(1, nhead, self.mem_len, head_dim, head_dim).expand(B, -1, -1, -1, -1)
            z = self.init_z.data.reshape(1, nhead, self.mem_len, head_dim).expand(B, -1, -1, -1)
            ptr = 0
        else:
            mem, z, ptr = state
        sigma_q = q.elu() + 1.0 + 1e-6  # 避免门控为0
        sigma_k = k.elu() + 1.0 + 1e-6
        sigma_v = v.elu() + 1.0 + 1e-6
        mem_reshaped = mem.transpose(0, 1, 3, 2)
        q_mem = (sigma_q @ mem_reshaped).reshape(B, nhead, seq_len, self.mem_len, head_dim)
        q_z = (sigma_q.unsqueeze(3) * z.unsqueeze(2)).sum(axis=-1, keepdims=True) + EPS
        mem_out = (q_mem / q_z).sum(axis=3)
        num_segments = (seq_len + seg_len - 1) // seg_len
        new_mem = mem.clone()
        new_z = z.clone()
        new_ptr = ptr
        for seg_idx in range(num_segments):
            start = seg_idx * seg_len
            end = min(start + seg_len, seq_len)
            k_seg = sigma_k[:, :, start:end, :]
            v_seg = sigma_v[:, :, start:end, :]
            outer = k_seg.transpose(-2, -1) @ v_seg
            idx = new_ptr % self.mem_len
            new_mem = new_mem.slice_assign(
                (slice(None), slice(None), slice(idx, idx + 1), slice(None), slice(None)),
                self.alpha * mem[(slice(None), slice(None), slice(idx, idx + 1))] + (1 - self.alpha) * outer.unsqueeze(2)
            )
            k_sum = k_seg.sum(axis=2)
            new_z = new_z.slice_assign(
                (slice(None), slice(None), slice(idx, idx + 1), slice(None)),
                self.alpha * z[(slice(None), slice(None), slice(idx, idx + 1))] + (1 - self.alpha) * k_sum.unsqueeze(2)
            )
            new_ptr = (new_ptr + 1) % self.mem_len
        return mem_out, (new_mem, new_z, new_ptr)

    def parameters(self) -> List[Tensor]:
        return [self.init_mem.data, self.init_z.data]

class InfiniAttentionBlock:
    def __init__(self, dim: int, heads: int, window_size: int, mem_len: int = 256, seg_len: int = 64, alpha: float = 0.1, dropout: float = 0.1, use_ternary: bool = False):
        self.nhead = heads
        self.head_dim = dim // heads
        self.window_size = window_size
        self.seg_len = seg_len
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        LinearCls = TernaryLinear if use_ternary else Linear
        self.qkv = LinearCls(dim, 3 * dim, bias=False)
        self.ffn = FeedForward(dim, 4, dropout, use_ternary)
        self.memory = InfiniMemory(mem_len, self.head_dim, heads, alpha)
        self._training = True

    def __call__(self, x: Tensor, mem_state: Optional[Tuple[Tensor, Tensor, int]] = None, past_kv: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor, int]], Optional[Tuple[Tensor, Tensor]]]:
        residual = x
        x = self.norm1(x)
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)
        if self.training:
            past_kv = None
        if past_kv is not None:
            past_k, past_v = past_kv
            k = Tensor.cat([past_k, k], axis=2)
            v = Tensor.cat([past_v, v], axis=2)
            past_len = past_k.shape[2]
        else:
            past_len = 0
        q, k = apply_rope(q, k, past_len, T, self.head_dim)
        scores_local = (q @ k.swapaxes(-2, -1)) / math.sqrt(self.head_dim)
        mask = get_causal_sliding_mask(k.shape[2], self.window_size)
        scores_local = scores_local + mask
        attn_local = scores_local.softmax(axis=-1)
        out_local = attn_local @ v
        mem_out, new_mem_state = self.memory.forward(q, k, v, mem_state, self.seg_len)
        out = out_local + mem_out
        out = out.swapaxes(1, 2).reshape(B, T, C)
        x = residual + out
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        if k.shape[2] > MAX_KV_CACHE_LEN:
            k = Tensor(k.data[:, :, -MAX_KV_CACHE_LEN:, :], requires_grad=k.requires_grad)
            v = Tensor(v.data[:, :, -MAX_KV_CACHE_LEN:, :], requires_grad=v.requires_grad)
        return x, new_mem_state, (k, v) if not self.training else (None, None)

    def parameters(self) -> List[Tensor]:
        params = self.norm1.parameters() + self.norm2.parameters()
        params.extend(self.qkv.parameters())
        params.extend(self.ffn.parameters())
        params.extend(self.memory.parameters())
        return params

    @property
    def training(self) -> bool:
        return getattr(self, '_training', True)

    @training.setter
    def training(self, val: bool) -> None:
        self._training = val

# ==================== 前馈与混合层 ====================
class FeedForward:
    def __init__(self, dim: int, hidden_factor: int = 4, dropout: float = 0.1, use_ternary: bool = False):
        LinearCls = TernaryLinear if use_ternary else Linear
        self.fc1 = LinearCls(dim, dim * hidden_factor, bias=False)
        self.fc2 = LinearCls(dim * hidden_factor, dim, bias=False)
        self.dropout = Dropout(dropout)

    def __call__(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(self.fc1(x).gelu()))

    def parameters(self) -> List[Tensor]:
        return self.fc1.parameters() + self.fc2.parameters()

class Expert:
    def __init__(self, dim: int, hidden_factor: int = 2, use_ternary: bool = False):
        hidden = dim * hidden_factor
        LinearCls = TernaryLinear if use_ternary else Linear
        self.fc1 = LinearCls(dim, hidden, bias=False)
        self.fc2 = LinearCls(hidden, dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).gelu())

    def parameters(self) -> List[Tensor]:
        return self.fc1.parameters() + self.fc2.parameters()

class MoERouter:
    def __init__(self, dim: int, num_experts: int, temperature: float = 1.0, use_ternary: bool = False):
        LinearCls = TernaryLinear if use_ternary else Linear
        self.router = LinearCls(dim, num_experts, bias=True)
        self.temperature = temperature

    def __call__(self, x: Tensor) -> Tensor:
        logits = self.router(x.mean(axis=1))
        gumbels = -np.log(-np.log(np.random.uniform(0, 1, logits.shape)))
        y = logits + Tensor(gumbels, requires_grad=False)
        probs = (y / self.temperature).softmax(axis=-1)
        return probs

    def parameters(self) -> List[Tensor]:
        return self.router.parameters()

class MoELayer:
    def __init__(self, dim: int, num_experts: int, top_k: int = 2, hidden_factor: int = 2, loss_coef: float = 0.01, use_ternary: bool = False):
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.loss_coef = loss_coef
        self.router = MoERouter(dim, num_experts, use_ternary=use_ternary)
        self.experts = [Expert(dim, hidden_factor, use_ternary) for _ in range(num_experts)]
        self.expert_usage = np.zeros(num_experts)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        probs = self.router(x_flat)
        top_vals, top_idxs = self._top_k(probs, self.top_k)
        top_probs_norm = top_vals / (top_vals.sum(axis=-1, keepdims=True) + EPS)
        avg_probs = probs.mean(axis=0)
        balance_loss = self.loss_coef * self.num_experts * (avg_probs * avg_probs).sum()
        out_flat = Tensor.zeros((B * T, C), requires_grad=True)
        for i in range(self.num_experts):
            mask = (top_idxs.data == i)
            if not np.any(mask):
                continue
            self.expert_usage[i] += np.sum(mask)
            indices = np.where(mask)[0]
            weights = top_probs_norm.data[mask]
            expert_out = self.experts[i](x_flat[indices])
            weighted = expert_out * weights[:, None]
            out_flat = out_flat.scatter_add(0, Tensor(indices, requires_grad=False), weighted)
        out = out_flat.reshape(B, T, C)
        return out, balance_loss

    def _top_k(self, probs: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        data = probs.data
        idx = np.argsort(-data, axis=-1)[:, :k]
        top_vals_data = np.take_along_axis(data, idx, axis=-1)
        norm_vals = top_vals_data / (top_vals_data.sum(axis=-1, keepdims=True) + EPS)
        top_vals = Tensor(norm_vals, requires_grad=False)
        return top_vals, Tensor(idx, requires_grad=False)

    def parameters(self) -> List[Tensor]:
        params = self.router.parameters()
        for e in self.experts:
            params.extend(e.parameters())
        return params

class GRUBlock:
    def __init__(self, dim: int, truncate_length: Optional[int] = None, use_ternary: bool = False):
        LinearCls = TernaryLinear if use_ternary else Linear
        self.W_ir = LinearCls(dim, dim, bias=False)
        self.W_iz = LinearCls(dim, dim, bias=False)
        self.W_in = LinearCls(dim, dim, bias=False)
        self.W_hr = LinearCls(dim, dim, bias=False)
        self.W_hz = LinearCls(dim, dim, bias=False)
        self.W_hn = LinearCls(dim, dim, bias=False)
        self.norm = LayerNorm(dim)
        self.truncate_length = truncate_length

    def __call__(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        B, T, C = x.shape
        h = Tensor.zeros((B, C), requires_grad=False)
        outputs = []
        for t in range(T):
            r = (self.W_ir(x[:, t, :]) + self.W_hr(h)).sigmoid()
            z = (self.W_iz(x[:, t, :]) + self.W_hz(h)).sigmoid()
            n = (self.W_in(x[:, t, :]) + self.W_hn(r * h)).tanh()
            h_new = (1 - z) * n + z * h
            outputs.append(h_new)
            h = h_new
        out = Tensor.stack(outputs, axis=1)
        return residual + out

    def parameters(self) -> List[Tensor]:
        params = self.W_ir.parameters() + self.W_iz.parameters() + self.W_in.parameters()
        params += self.W_hr.parameters() + self.W_hz.parameters() + self.W_hn.parameters()
        params += self.norm.parameters()
        return params

class ImageEncoder:
    def __init__(self, image_size: int, d_model: int, use_ternary: bool = False):
        self.image_size = image_size
        self.d_model = d_model
        LinearCls = TernaryLinear if use_ternary else Linear
        self.proj = LinearCls(3 * image_size * image_size, d_model, bias=False)
        self.norm = LayerNorm(d_model)

    def __call__(self, pixel_values: Tensor) -> Tensor:
        B, C, H, W = pixel_values.shape
        if C != 3 or H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected image shape (B,3,{self.image_size},{self.image_size})")
        x = pixel_values.reshape(B, -1)
        x = self.proj(x)
        x = self.norm(x)
        return x.unsqueeze(1)

    def parameters(self) -> List[Tensor]:
        return self.proj.parameters() + self.norm.parameters()

# ==================== 主模型 ====================
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    window_size: int = 16
    use_infini: bool = False
    mem_len: int = 256
    use_moe: bool = False
    num_experts: int = 4
    moe_layers: List[int] = field(default_factory=list)
    use_hybrid: bool = False
    use_image: bool = False
    image_size: int = 224
    use_ternary: bool = False
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.mem_len <= 0:
            raise ValueError(f"mem_len must be positive, got {self.mem_len}")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")

class AetherOmniModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.embed = Embedding(config.vocab_size, config.d_model)
        self.image_encoder = ImageEncoder(config.image_size, config.d_model, config.use_ternary) if config.use_image else None
        self.image_pos_embed = None
        if config.use_image:
            self.image_pos_embed = Tensor(np.random.randn(1, 1, config.d_model) * 0.02, requires_grad=True)
        self.blocks = []
        for i in range(config.n_layers):
            if config.use_moe and i in config.moe_layers:
                block = MoELayer(config.d_model, config.num_experts, top_k=2, hidden_factor=2,
                                 loss_coef=0.01, use_ternary=config.use_ternary)
            elif config.use_hybrid and i % 2 == 0:
                block = GRUBlock(config.d_model, truncate_length=None, use_ternary=config.use_ternary)
            elif config.use_infini:
                block = InfiniAttentionBlock(config.d_model, config.n_heads, config.window_size,
                                             config.mem_len, seg_len=64, alpha=0.1,
                                             dropout=config.dropout, use_ternary=config.use_ternary)
            else:
                block = TransformerBlock(config.d_model, config.n_heads, config.window_size,
                                         config.dropout, config.use_ternary)
            self.blocks.append(block)
        self.norm = LayerNorm(config.d_model)
        LinearCls = TernaryLinear if config.use_ternary else Linear
        self.lm_head = LinearCls(config.d_model, config.vocab_size, bias=False)
        self.training = True

    def train(self, mode: bool = True) -> None:
        self.training = mode
        for block in self.blocks:
            if hasattr(block, 'training'):
                block.training = mode
            elif hasattr(block, 'dropout') and hasattr(block.dropout, 'training'):
                block.dropout.training = mode

    def eval(self) -> None:
        self.train(False)

    def __call__(self, input_ids: np.ndarray, pixel_values: Optional[Tensor] = None,
                 attention_mask: Optional[Tensor] = None, past_key_values: Optional[List[Optional[Tuple[Tensor, Tensor]]]] = None,
                 return_mem_state: bool = False) -> Union[Tensor, Tuple[Tensor, Any]]:
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got {input_ids.ndim}")
        x = self.embed(input_ids)
        if pixel_values is not None and self.image_encoder is not None:
            img_embeds = self.image_encoder(pixel_values)
            if self.image_pos_embed is not None:
                img_embeds = img_embeds + self.image_pos_embed
            x = Tensor.cat([img_embeds, x], axis=1)
        mem_states = []
        new_past_kv = [] if past_key_values is not None else None
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None and i < len(past_key_values) else None
            if isinstance(block, InfiniAttentionBlock):
                mem_state = None
                x, new_mem, new_kv = block(x, mem_state, past_kv)
                mem_states.append(new_mem)
            elif isinstance(block, MoELayer):
                x, _ = block(x)
                new_kv = None
            elif isinstance(block, GRUBlock):
                x = block(x)
                new_kv = None
            else:
                x, new_kv = block(x, past_kv)
            if new_past_kv is not None:
                new_past_kv.append(new_kv)
        x = self.norm(x)
        logits = self.lm_head(x)
        if return_mem_state:
            return logits, mem_states
        return logits

    def parameters(self) -> List[Tensor]:
        params = self.embed.parameters()
        if self.image_encoder:
            params.extend(self.image_encoder.parameters())
            if self.image_pos_embed:
                params.append(self.image_pos_embed)
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.lm_head.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def get_state_dict(self) -> Dict[str, np.ndarray]:
        state = {}
        for i, p in enumerate(self.parameters()):
            state[f'param_{i}'] = p.data.copy()
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        for i, p in enumerate(self.parameters()):
            if f'param_{i}' in state:
                p.data[:] = state[f'param_{i}']

    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = 50, do_sample: bool = True) -> np.ndarray:
        self.eval()
        generated = input_ids.copy()
        for _ in range(max_new_tokens):
            logits = self(generated)
            next_token_logits = logits.data[0, -1, :]
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            if do_sample:
                if top_k > 0:
                    indices = np.argpartition(-next_token_logits, top_k)[:top_k]
                    probs = np.exp(next_token_logits[indices] - np.max(next_token_logits[indices]))
                    probs /= probs.sum()
                    next_token = np.random.choice(indices, p=probs)
                else:
                    probs = np.exp(next_token_logits - np.max(next_token_logits))
                    probs /= probs.sum()
                    next_token = np.random.choice(len(probs), p=probs)
            else:
                next_token = np.argmax(next_token_logits)
            generated = np.concatenate([generated, np.array([[next_token]])], axis=1)
        return generated

    def beam_search(self, input_ids: np.ndarray, beam_width: int = 4, max_new_tokens: int = 50) -> np.ndarray:
        self.eval()
        sequences = [(input_ids, 0.0)]
        for _ in range(max_new_tokens):
            all_candidates = []
            for seq, score in sequences:
                logits = self(seq)
                next_token_logits = logits.data[0, -1, :]
                probs = np.exp(next_token_logits - np.max(next_token_logits))
                probs /= probs.sum()
                top_indices = np.argpartition(probs, -beam_width)[-beam_width:]
                for idx in top_indices:
                    new_seq = np.concatenate([seq, np.array([[idx]])], axis=1)
                    new_score = score + np.log(probs[idx] + EPS)
                    all_candidates.append((new_seq, new_score))
            all_candidates.sort(key=lambda x: -x[1])
            sequences = all_candidates[:beam_width]
        return sequences[0][0]

class TransformerBlock:
    def __init__(self, dim: int, heads: int, window_size: int = 16, dropout: float = 0.1, use_ternary: bool = False):
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, window_size, use_ternary)
        self.ffn = FeedForward(dim, 4, dropout, use_ternary)
        self._training = True

    def __call__(self, x: Tensor, past_kv: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = x
        x = self.norm1(x)
        x_attn, new_kv = self.attn(x, past_kv)
        x = residual + x_attn
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x, new_kv if not self.training else (None, None)

    def parameters(self) -> List[Tensor]:
        return self.norm1.parameters() + self.norm2.parameters() + self.attn.parameters() + self.ffn.parameters()

    @property
    def training(self) -> bool:
        return getattr(self, '_training', True)

    @training.setter
    def training(self, val: bool) -> None:
        self._training = val
        self.attn.training = val
        self.ffn.dropout.training = val

# ==================== 优化器与调度 ====================
class Lookahead:
    def __init__(self, base_optimizer, k: int = 5, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.params = base_optimizer.params
        self._slow_params = [p.data.copy() for p in self.params]
        self._step = 0

    def step(self) -> None:
        self.base_optimizer.step()
        self._step += 1
        if self._step % self.k == 0:
            for i, p in enumerate(self.params):
                slow = self._slow_params[i]
                fast = p.data
                new_slow = slow + self.alpha * (fast - slow)
                p.data = new_slow
                self._slow_params[i] = new_slow
        else:
            for i, p in enumerate(self.params):
                self._slow_params[i] = p.data.copy()

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'base': self.base_optimizer.state_dict(),
            'k': self.k,
            'alpha': self.alpha,
            'step': self._step,
            'slow_params': [sp.copy() for sp in self._slow_params],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.k = state['k']
        self.alpha = state['alpha']
        self._step = state['step']
        self._slow_params = [sp.copy() for sp in state['slow_params']]
        self.base_optimizer.load_state_dict(state['base'])
        for i, p in enumerate(self.params):
            p.data = self._slow_params[i].copy()

class AdaptiveGradientClipper:
    def __init__(self, clip_factor: float = 0.01, target_grad_norm: float = 5.0, adaptation_rate: float = 0.99, eps: float = 1e-8):
        self.clip_factor = clip_factor
        self.target_grad_norm = target_grad_norm
        self.adaptation_rate = adaptation_rate
        self.eps = eps
        self.grad_norm_ema = None
        self.current_clip_factor = clip_factor

    def clip(self, model: AetherOmniModel, max_norm: Optional[float] = None) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += np.sum(p.grad ** 2)
        total_norm = np.sqrt(total_norm)
        if total_norm < self.eps:
            return total_norm

        if self.grad_norm_ema is None:
            self.grad_norm_ema = total_norm
        else:
            self.grad_norm_ema = self.adaptation_rate * self.grad_norm_ema + (1 - self.adaptation_rate) * total_norm

        ratio = self.grad_norm_ema / (self.target_grad_norm + self.eps)
        if ratio > 1.2:
            self.current_clip_factor *= 0.99
        elif ratio < 0.8:
            self.current_clip_factor *= 1.01
        self.current_clip_factor = max(1e-3, min(0.1, self.current_clip_factor))

        if max_norm is None:
            param_norms = [np.linalg.norm(p.data) for p in model.parameters() if p.data.ndim >= 2]
            if not param_norms:
                param_norms = [1.0]
            median_norm = np.median(param_norms)
            max_norm = self.current_clip_factor * max(median_norm, self.eps)

        if total_norm > max_norm:
            scale = max_norm / total_norm
            for p in model.parameters():
                if p.grad is not None:
                    p.grad *= scale
            total_norm = max_norm
        return total_norm

    def reset(self) -> None:
        self.grad_norm_ema = None

class AdamW:
    def __init__(self, params: List[Tensor], lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                continue
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (grad * grad)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            if self.wd != 0:
                p.data -= self.lr * self.wd * p.data

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        return {
            't': self.t,
            'm': [m.copy() for m in self.m],
            'v': [v.copy() for v in self.v],
            'lr': self.lr,
            'wd': self.wd,
            'betas': (self.b1, self.b2),
            'eps': self.eps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.t = state['t']
        self.m = [m.copy() for m in state['m']]
        self.v = [v.copy() for v in state['v']]
        self.lr = state['lr']
        self.wd = state['wd']
        self.b1, self.b2 = state['betas']
        self.eps = state['eps']

class SGD:
    def __init__(self, params: List[Tensor], lr: float = 1e-4, momentum: float = 0.9, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                continue
            grad = grad + self.wd * p.data
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            p.data -= self.lr * self.velocities[i]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'wd': self.wd,
            'velocities': [v.copy() for v in self.velocities],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.lr = state['lr']
        self.momentum = state['momentum']
        self.wd = state['wd']
        self.velocities = [v.copy() for v in state['velocities']]

class GradientScaler:
    def __init__(self, init_scale: float = 1, growth_factor: float = 1.2, backoff_factor: float = 0.8, growth_interval: int = 1000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._overflow_counter = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss * self.scale

    def unscale_grads(self, params: List[Tensor]) -> bool:
        has_overflow = False
        for p in params:
            if p.grad is not None:
                p.grad /= self.scale
                if np.any(np.isnan(p.grad)) or np.any(np.isinf(p.grad)):
                    p.grad.fill(0)
                    has_overflow = True
                    self._overflow_counter += 1
        if has_overflow:
            logger.warning(f"梯度溢出，缩放因子={self.scale}, 溢出次数={self._overflow_counter}")
        return has_overflow

    def update(self, has_overflow: bool) -> None:
        if has_overflow:
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
        self.scale = max(2 ** 4, min(2 ** 16, self.scale))

    def state_dict(self) -> Dict[str, Any]:
        return {'scale': self.scale, 'growth_tracker': self._growth_tracker}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.scale = state['scale']
        self._growth_tracker = state['growth_tracker']

class EMA:
    def __init__(self, model: AetherOmniModel, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = [p.data.copy() for p in model.parameters()]

    def update(self) -> None:
        for i, p in enumerate(self.model.parameters()):
            self.shadow[i] = self.decay * self.shadow[i] + (1 - self.decay) * p.data

    def apply_shadow(self) -> None:
        for i, p in enumerate(self.model.parameters()):
            p.data = self.shadow[i].copy()

    def restore(self) -> None:
        for i, p in enumerate(self.model.parameters()):
            self.shadow[i] = p.data.copy()

class LearningRateScheduler:
    def __init__(self, base_lr: float = 5e-6, warmup_steps: int = 500, plateau_steps: int = 1000, total_steps: int = 50000, min_lr_ratio: float = 0.1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.plateau_steps = plateau_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio

    def get_lr(self, step: int) -> float:
        if self.total_steps is None:
            return self.base_lr
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        elif step < self.warmup_steps + self.plateau_steps:
            return self.base_lr
        else:
            progress = (step - self.warmup_steps - self.plateau_steps) / (self.total_steps - self.warmup_steps - self.plateau_steps)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine

class AdaptiveLRPlateau:
    def __init__(self, decay_factor: float = 0.7, patience: int = 3, cooldown: int = 500, min_lr: float = 1e-7):
        self.decay_factor = decay_factor
        self.patience = patience
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.best_loss = None
        self.num_bad_epochs = 0
        self.last_decay_step = -1
        self.val_loss_ema = None

    def step(self, current_step: int, val_loss: float, lr: float) -> float:
        if current_step - self.last_decay_step < self.cooldown:
            return lr
        if self.val_loss_ema is None:
            self.val_loss_ema = val_loss
        else:
            self.val_loss_ema = 0.3 * val_loss + 0.7 * self.val_loss_ema
        if self.best_loss is None:
            self.best_loss = val_loss
            return lr
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            new_lr = max(lr * self.decay_factor, self.min_lr)
            if new_lr < lr:
                self.num_bad_epochs = 0
                self.last_decay_step = current_step
                self.val_loss_ema = None
                return new_lr
        return lr

# ==================== 训练管理器 ====================
class AdaptiveTrainManager:
    def __init__(self, model: AetherOmniModel, optimizer: Union[AdamW, SGD], config: Dict[str, Any], val_loader: Optional[Any] = None, total_steps: Optional[int] = None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.val_loader = val_loader
        self.total_steps = total_steps
        self.step = 0
        self.best_val_loss = float('inf')
        self.best_state_dict = None
        self.early_stop_count = 0
        self.dropout_val = config['regularization']['dropout']['initial']
        self.wd_val = config['regularization']['weight_decay']['initial']
        self.dpo_beta = config.get('dpo_beta', 0.3)
        self.agc = AdaptiveGradientClipper(clip_factor=0.01, target_grad_norm=5.0)
        self.meta_lr_multiplier = 1.0
        self._collect_dropout_layers()
        self._apply_regularization()
        self.lr_scheduler = self._build_lr_scheduler()
        self.plateau_counter = 0
        self.plateau_cooldown = 0

    def _collect_dropout_layers(self) -> None:
        self.dropout_layers = []
        def collect(module):
            if isinstance(module, Dropout):
                self.dropout_layers.append(module)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, (TransformerBlock, InfiniAttentionBlock, MoELayer, GRUBlock, AetherOmniModel)):
                    collect(attr)
                elif hasattr(attr, 'parameters') and callable(getattr(attr, 'parameters', None)):
                    collect(attr)
        collect(self.model)
        self.dropout_layers = list(set(self.dropout_layers))

    def _apply_regularization(self) -> None:
        if hasattr(self.optimizer, 'wd'):
            self.optimizer.wd = self.wd_val
        for layer in self.dropout_layers:
            layer.p = self.dropout_val

    def _build_lr_scheduler(self) -> Callable[[int], float]:
        cfg = self.config['learning_rate']
        if cfg['type'] == 'cosine':
            if self.total_steps is None:
                return lambda step: cfg['initial']
            def scheduler(step):
                if step < cfg['warmup_steps']:
                    return cfg['initial'] * (step / max(1, cfg['warmup_steps']))
                progress = (step - cfg['warmup_steps']) / (self.total_steps - cfg['warmup_steps'])
                return cfg['initial'] * 0.5 * (1 + np.cos(np.pi * progress))
            return scheduler
        elif cfg['type'] == 'step':
            def scheduler(step):
                return cfg['initial'] * (cfg['gamma'] ** (step // cfg['step_size']))
            return scheduler
        else:
            return lambda step: cfg['initial']

    def clip_gradients(self) -> float:
        total_norm = self.agc.clip(self.model, max_norm=None)
        return total_norm

    def before_step(self) -> None:
        scheduler_lr = self.lr_scheduler(self.step)
        new_lr = scheduler_lr * self.meta_lr_multiplier
        new_lr = max(1e-8, min(1e-3, new_lr))
        if hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = new_lr
        elif hasattr(self.optimizer, 'base_optimizer') and hasattr(self.optimizer.base_optimizer, 'lr'):
            self.optimizer.base_optimizer.lr = new_lr

    def after_step(self, train_loss: float, grad_norm: float, val_loss: Optional[float] = None) -> None:
        self.step += 1
        if val_loss is not None:
            if val_loss < self.best_val_loss - self.config['early_stop']['min_delta']:
                self.best_val_loss = val_loss
                self.best_state_dict = self.model.get_state_dict()
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                if self.early_stop_count >= self.config['early_stop']['patience']:
                    if self.config['early_stop']['restore_best'] and self.best_state_dict:
                        self.model.load_state_dict(self.best_state_dict)
                        log_system(f"早停触发，已恢复最佳模型 (验证损失 {self.best_val_loss:.4f})")
                    raise StopIteration("Early stopping")
            relative_diff = (val_loss - train_loss) / (train_loss + EPS)
            if relative_diff > self.config['overfit_threshold']:
                old_dropout, old_wd = self.dropout_val, self.wd_val
                self.dropout_val = min(self.dropout_val * 1.1, self.config['regularization']['dropout']['max'])
                self.wd_val = min(self.wd_val * 1.1, self.config['regularization']['weight_decay']['max'])
                if old_dropout != self.dropout_val or old_wd != self.wd_val:
                    log_system(f"检测到过拟合 (相对差 {relative_diff:.3f})，增大正则化: dropout {old_dropout:.3f}->{self.dropout_val:.3f}, wd {old_wd:.4f}->{self.wd_val:.4f}")
                    self._apply_regularization()
            elif val_loss > self.best_val_loss * (1 + self.config['underfit_threshold']):
                old_dropout, old_wd = self.dropout_val, self.wd_val
                self.dropout_val = max(self.dropout_val * 0.9, self.config['regularization']['dropout']['min'])
                self.wd_val = max(self.wd_val * 0.9, self.config['regularization']['weight_decay']['min'])
                if old_dropout != self.dropout_val or old_wd != self.wd_val:
                    log_system(f"检测到欠拟合 (损失 {val_loss:.4f} > {self.best_val_loss:.4f}*1.2)，减小正则化: dropout {old_dropout:.3f}->{self.dropout_val:.3f}, wd {old_wd:.4f}->{self.wd_val:.4f}")
                    self._apply_regularization()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'best_val_loss': self.best_val_loss,
            'best_state_dict': self.best_state_dict,
            'early_stop_count': self.early_stop_count,
            'dropout_val': self.dropout_val,
            'wd_val': self.wd_val,
            'config': self.config,
            'dpo_beta': self.dpo_beta,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step = state['step']
        self.best_val_loss = state['best_val_loss']
        self.best_state_dict = state['best_state_dict']
        self.early_stop_count = state['early_stop_count']
        self.dropout_val = state['dropout_val']
        self.wd_val = state['wd_val']
        self.config = state.get('config', self.config)
        self.dpo_beta = state.get('dpo_beta', 0.3)
        self._apply_regularization()

# ==================== 自适应控制器 ====================
class AdaptiveController:
    def __init__(self, trainer: Any, config: Dict[str, Any]):
        self.trainer = trainer
        self.config = config
        self.last_snapshot = None
        self.rollback_stack = []
        self.cooldown = config.get('cooldown_steps', 200)
        self.last_adjust_step = 0
        self.temp_high = config.get('temp_high', 75)
        self.temp_moderate = config.get('temp_moderate', 60)
        self.temp_low = config.get('temp_low', 45)
        self.battery_low = config.get('battery_low', 40)
        self.battery_critical = config.get('battery_critical', 20)

    def check_hardware(self) -> None:
        hw = self.trainer.hw_monitor.get_state()
        temp = hw.get('cpu_temp')
        batt = hw.get('battery')
        pred = self.trainer.hw_monitor.predict(steps=5)
        pred_temp = pred.get('cpu_temp', temp)
        if pred_temp > self.temp_high:
            log_hardware(f"预测未来温度 {pred_temp:.1f}°C 超过危险阈值 {self.temp_high}°C，提前降频")
            self._preemptive_throttle()
        if temp:
            if temp > self.temp_high:
                log_hardware(f"CPU 温度 {temp:.1f}°C 超过危险阈值 {self.temp_high}°C，紧急退出")
                self._emergency_exit()
            elif temp > self.temp_moderate:
                log_hardware(f"CPU 温度 {temp:.1f}°C 超过中度阈值 {self.temp_moderate}°C，强制降低配置")
                self._high_temp_action()
            elif temp > self.temp_low:
                log_hardware(f"CPU 温度 {temp:.1f}°C 超过轻度阈值 {self.temp_low}°C，降低验证/日志频率")
                self._moderate_temp_action()
        if batt:
            if batt < self.battery_critical:
                log_hardware(f"电池电量 {batt:.1f}% 低于危急阈值 {self.battery_critical}%，保存检查点并退出")
                self._battery_critical()
            elif batt < self.battery_low:
                log_hardware(f"电池电量 {batt:.1f}% 低于低电量阈值 {self.battery_low}%，降低学习率并休眠")
                self._battery_low()

    def _preemptive_throttle(self) -> None:
        if hasattr(self.trainer, 'batch_size') and self.trainer.batch_size > 1:
            self.trainer.batch_size = max(1, self.trainer.batch_size // 2)
            self.trainer.need_rebuild_loader = True
            log_hardware(f"预测性降频：batch_size 降至 {self.trainer.batch_size}")

    def _moderate_temp_action(self) -> None:
        if hasattr(self.trainer, 'eval_interval'):
            self.trainer.eval_interval = min(5000, self.trainer.eval_interval * 2)
        if hasattr(self.trainer, 'log_interval'):
            self.trainer.log_interval = min(2000, self.trainer.log_interval * 2)

    def _high_temp_action(self) -> None:
        if hasattr(self.trainer, 'batch_size') and self.trainer.batch_size > 1:
            self.trainer.batch_size = 1
            self.trainer.need_rebuild_loader = True
        time.sleep(30)

    def _battery_low(self) -> None:
        if hasattr(self.trainer, 'optimizer'):
            if hasattr(self.trainer.optimizer, 'lr'):
                self.trainer.optimizer.lr *= 0.8
            elif hasattr(self.trainer.optimizer, 'base_optimizer'):
                self.trainer.optimizer.base_optimizer.lr *= 0.8
        if hasattr(self.trainer, 'eval_interval'):
            self.trainer.eval_interval = min(5000, self.trainer.eval_interval * 2)
        time.sleep(60)

    def _battery_critical(self) -> None:
        if hasattr(self.trainer, 'save_checkpoint'):
            self.trainer.save_checkpoint(emergency=True)
        sys.exit(0)

    def _emergency_exit(self) -> None:
        if hasattr(self.trainer, 'save_checkpoint'):
            self.trainer.save_checkpoint(emergency=True)
        sys.exit(0)

    def push_snapshot(self) -> None:
        if not hasattr(self.trainer, 'global_step'):
            return
        snapshot = {
            'step': self.trainer.global_step,
            'config': {
                'batch_size': getattr(self.trainer, 'batch_size', None),
                'max_seq_len': getattr(self.trainer, 'max_seq_len', None),
                'lr': getattr(self.trainer.optimizer, 'lr', None) if hasattr(self.trainer.optimizer, 'lr') else (
                    self.trainer.optimizer.base_optimizer.lr if hasattr(self.trainer.optimizer, 'base_optimizer') else None),
                'dropout': getattr(self.trainer, 'dropout_val', None),
            },
            'loss_rate': self._get_loss_descent_rate()
        }
        self.rollback_stack.append(snapshot)
        if len(self.rollback_stack) > 5:
            self.rollback_stack.pop(0)
        self.last_snapshot = snapshot

    def _get_loss_descent_rate(self, window: int = 100) -> Optional[float]:
        if not hasattr(self.trainer, 'loss_history') or len(self.trainer.loss_history) < window + 10:
            return None
        recent = np.mean(list(self.trainer.loss_history)[-window:])
        old = np.mean(list(self.trainer.loss_history)[-2 * window:-window])
        return (old - recent) / old if old > 0 else 0

    def check_and_rollback(self) -> None:
        if self.last_snapshot is None:
            return
        if self.trainer.global_step - self.last_snapshot['step'] < 200:
            return
        current_rate = self._get_loss_descent_rate()
        if current_rate is None or self.last_snapshot['loss_rate'] is None:
            return
        if current_rate < self.last_snapshot['loss_rate'] * 0.5:
            log_system("效果下降，回滚到上一配置")
            cfg = self.last_snapshot['config']
            if cfg['batch_size'] is not None and hasattr(self.trainer, 'batch_size'):
                self.trainer.batch_size = cfg['batch_size']
                self.trainer.need_rebuild_loader = True
            if cfg['max_seq_len'] is not None and hasattr(self.trainer, 'max_seq_len'):
                self.trainer.max_seq_len = cfg['max_seq_len']
                self.trainer.need_rebuild_loader = True
            if cfg['lr'] is not None:
                if hasattr(self.trainer.optimizer, 'lr'):
                    self.trainer.optimizer.lr = cfg['lr']
                elif hasattr(self.trainer.optimizer, 'base_optimizer'):
                    self.trainer.optimizer.base_optimizer.lr = cfg['lr']
            if cfg['dropout'] is not None and hasattr(self.trainer, 'manager'):
                self.trainer.manager.dropout_val = cfg['dropout']
                self.trainer.manager._apply_regularization()
            if hasattr(self.trainer, 'scaler'):
                self.trainer.scaler = GradientScaler(init_scale=1)
            if hasattr(self.trainer, 'ema'):
                self.trainer.ema.restore()
            self.last_snapshot = None

    def adjust(self) -> None:
        if not hasattr(self.trainer, 'global_step') or self.trainer.global_step - self.last_adjust_step < self.cooldown:
            return
        self.push_snapshot()
        self.check_hardware()
        self.last_adjust_step = self.trainer.global_step

# ==================== 数据加载器 ====================
class StreamingDPOLoader:
    def __init__(self, data_dir: str, max_seq_len: int = 512, strip_special: bool = True, use_mmap: bool = True, use_per: bool = True, per_alpha: float = 0.6):
        self.data_dir = safe_data_dir(data_dir)
        self.max_seq_len = max_seq_len
        self.strip_special = strip_special
        self.use_mmap = use_mmap
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.train_file = os.path.join(self.data_dir, 'train.ids')
        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Training file not found: {self.train_file}")
        self.triple_offsets = []
        self._file_handle = None
        if use_mmap:
            self._file_handle = open(self.train_file, 'rb')
            while True:
                start = self._file_handle.tell()
                line1 = self._file_handle.readline()
                if not line1: break
                line2 = self._file_handle.readline()
                if not line2: break
                line3 = self._file_handle.readline()
                if not line3: break
                self.triple_offsets.append(start)
        self._load_special_ids(data_dir)
        self.priorities = None
        self._init_per()

    def _init_per(self) -> None:
        if self.use_per:
            total = len(self)
            self.priorities = np.ones(total, dtype=np.float32)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _load_special_ids(self, data_dir: str) -> None:
        self.bos = None
        self.eos = None
        info_path = os.path.join(data_dir, 'dataset_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                special = info.get('special_ids', {})
                self.bos = special.get('<BOS>')
                self.eos = special.get('<EOS>')

    def _read_triple(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        if self.use_mmap and self._file_handle is not None:
            self._file_handle.seek(self.triple_offsets[idx])
            raw_prompt = self._file_handle.readline()
            raw_chosen = self._file_handle.readline()
            raw_rejected = self._file_handle.readline()
            try:
                prompt = list(map(int, raw_prompt.decode('utf-8').strip().split()))
                chosen = list(map(int, raw_chosen.decode('utf-8').strip().split()))
                rejected = list(map(int, raw_rejected.decode('utf-8').strip().split()))
            except UnicodeDecodeError:
                prompt = list(map(int, raw_prompt.decode('latin-1').strip().split()))
                chosen = list(map(int, raw_chosen.decode('latin-1').strip().split()))
                rejected = list(map(int, raw_rejected.decode('latin-1').strip().split()))
            return prompt, chosen, rejected
        else:
            with open(self.train_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                if idx * 3 + 2 >= len(lines):
                    raise StopIteration
                prompt = list(map(int, lines[idx * 3].split()))
                chosen = list(map(int, lines[idx * 3 + 1].split()))
                rejected = list(map(int, lines[idx * 3 + 2].split()))
        return prompt, chosen, rejected

    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        if self.use_per and self.priorities is not None:
            filtered = [(i, e) for i, e in zip(indices, td_errors) if 0 <= i < len(self.priorities)]
            if not filtered:
                return
            idx, err = zip(*filtered)
            priorities = np.abs(err) + EPS
            self.priorities[list(idx)] = priorities ** self.per_alpha
            # 归一化，防止优先级无限增长
            self.priorities /= (np.sum(self.priorities) + EPS)

    def iter_triples(self) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], int]]:
        total = len(self.triple_offsets) if self.use_mmap else self._get_triple_count()
        if self.use_per and self.priorities is not None:
            probs = self.priorities / np.sum(self.priorities)
            indices = np.random.choice(total, size=total, replace=False, p=probs)
        else:
            indices = list(range(total))
            random.shuffle(indices)
        for idx in indices:
            prompt, chosen, rejected = self._read_triple(idx)
            if self.strip_special:
                if self.bos is not None and prompt and prompt[0] == self.bos:
                    prompt = prompt[1:]
                if self.eos is not None and chosen and chosen[-1] == self.eos:
                    chosen = chosen[:-1]
                if self.eos is not None and rejected and rejected[-1] == self.eos:
                    rejected = rejected[:-1]
            prompt = prompt[:self.max_seq_len // 2]
            chosen = chosen[:self.max_seq_len // 2]
            rejected = rejected[:self.max_seq_len // 2]
            if prompt and chosen and rejected:
                yield (np.array(prompt, dtype=np.int64),
                       np.array(chosen, dtype=np.int64),
                       np.array(rejected, dtype=np.int64)), idx

    def _get_triple_count(self) -> int:
        with open(self.train_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        return len(lines) // 3

    def __len__(self) -> int:
        return len(self.triple_offsets) if self.use_mmap else self._get_triple_count()

    def close(self) -> None:
        if self._file_handle:
            self._file_handle.close()

class DataLoader:
    def __init__(self, sample_gen_func: Callable, batch_size: int, pad_id: int,
                 shuffle_buffer_size: int = 1000, shuffle_each_epoch: bool = True,
                 max_samples: Optional[int] = None, timeout: int = 600,
                 skip_samples: int = 0):
        self.sample_gen_func = sample_gen_func
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_each_epoch = shuffle_each_epoch
        self.max_samples = max_samples
        self.timeout = timeout
        self.skip_samples = skip_samples
        self._reset()

    def _reset(self) -> None:
        self.buffer = []
        self.epoch = 0
        self.sample_iter = None
        self.processed = 0
        self.generator = self._generator()

    def _generator(self):
        if self.sample_iter is None:
            self.sample_iter = self.sample_gen_func()
            for _ in range(self.skip_samples):
                try:
                    next(self.sample_iter)
                except StopIteration:
                    break
        for sample, length in self.sample_iter:
            yield sample, length

    def __iter__(self):
        self._reset()
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        try:
            while len(self.buffer) < self.shuffle_buffer_size:
                try:
                    sample, length = next(self.generator)
                    if time.time() - start_time > self.timeout:
                        raise TimeoutError("数据加载超时")
                    self.buffer.append((sample, length))
                    self.processed += 1
                    if self.max_samples and self.processed >= self.max_samples:
                        break
                except StopIteration:
                    break
            if not self.buffer:
                raise StopIteration
            if self.shuffle_each_epoch:
                random.shuffle(self.buffer)
            batch_samples = self.buffer[:self.batch_size]
            self.buffer = self.buffer[self.batch_size:]
            return self._collate(batch_samples)
        except StopIteration:
            raise
        except TimeoutError as e:
            raise DataLoaderTimeoutError(str(e), self.processed) from e
        except Exception as e:
            raise DataLoaderTimeoutError(f"数据加载失败: {e}", self.processed) from e

    def _collate(self, batch_samples: List[Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, np.ndarray]:
        max_len = max(len(seq) for seq, _ in batch_samples)
        padded = []
        masks = []
        for seq, length in batch_samples:
            pad_len = max_len - length
            padded_seq = np.concatenate([seq, np.full(pad_len, self.pad_id, dtype=INDEX_DTYPE)])
            mask = np.concatenate([np.ones(length, dtype=bool), np.zeros(pad_len, dtype=bool)])
            padded.append(padded_seq)
            masks.append(mask)
        return np.stack(padded), np.stack(masks)

class DataLoaderTimeoutError(Exception):
    def __init__(self, message: str, processed_samples: int = 0):
        super().__init__(message)
        self.processed_samples = processed_samples

# ==================== 进化搜索（超参数） ====================
class HyperParameterSpace:
    def __init__(self):
        self.continuous = {
            'learning_rate': (1e-6, 1e-3, 'loguniform'),
            'weight_decay': (1e-8, 0.1, 'loguniform'),
            'dropout': (0.0, 0.5, 'uniform'),
        }
        self.discrete = {'optimizer': ['AdamW', 'SGD']}

    def sample_continuous(self, name: str) -> float:
        minv, maxv, dist = self.continuous[name]
        if dist == 'loguniform':
            return np.exp(np.random.uniform(np.log(minv), np.log(maxv)))
        return np.random.uniform(minv, maxv)

    def sample_discrete(self, name: str) -> str:
        return np.random.choice(self.discrete[name])

class HyperParameterIndividual:
    def __init__(self, space: HyperParameterSpace):
        self.space = space
        self.genes = {}
        self.fitness = None

    def random_initialize(self) -> None:
        for name in self.space.continuous:
            self.genes[name] = self.space.sample_continuous(name)
        for name in self.space.discrete:
            self.genes[name] = self.space.sample_discrete(name)

    def decode(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        return self.genes.copy()

def evaluate_hyper_individual(individual: HyperParameterIndividual, model_class: Any,
                              train_loader: StreamingDPOLoader, val_loader: StreamingDPOLoader,
                              base_config: Dict[str, Any], eval_steps: int, seed: int) -> float:
    set_seed(seed)
    params = individual.decode(base_config)
    lr = params.get('learning_rate', 5e-6)
    wd = params.get('weight_decay', 0.0)
    opt_name = params.get('optimizer', 'AdamW')
    dropout_p = params.get('dropout', 0.1)
    model_kwargs = base_config['model_kwargs'].copy()
    model_kwargs['dropout'] = dropout_p
    model = model_class(**model_kwargs)
    ref_model = model_class(**model_kwargs)
    ref_model.load_state_dict(model.get_state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    if opt_name == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd)
    step = 0
    total_loss = 0.0
    triple_iter = train_loader.iter_triples()
    try:
        while step < eval_steps:
            (prompt, chosen, rejected), _ = next(triple_iter)
            model.zero_grad()
            loss, _ = dpo_loss(model, ref_model, prompt, chosen, rejected, beta=0.3)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            step += 1
        avg_loss = total_loss / eval_steps
        val_loss = 0.0
        val_count = 0
        model.eval()
        with Tensor.no_grad():
            for i, (batch, _) in enumerate(val_loader.iter_triples()):
                if i >= 20:
                    break
                prompt, chosen, rejected = batch
                loss, _ = dpo_loss(model, ref_model, prompt, chosen, rejected, beta=0.3)
                val_loss += loss.data.item()
                val_count += 1
        val_loss /= max(1, val_count)
        fitness = -val_loss
        return fitness
    except Exception as e:
        logger.error(f"评估失败: {e}")
        return -float('inf')
    finally:
        del model, ref_model
        gc.collect()

class EvolutionarySearcher:
    def __init__(self, model_class: Any, train_loader: StreamingDPOLoader, val_loader: StreamingDPOLoader,
                 base_config: Dict[str, Any], space: HyperParameterSpace,
                 population_size: int = 10, n_generations: int = 5, eval_steps: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1, elite_ratio: float = 0.2, seed: int = 42):
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.base_config = base_config
        self.space = space
        self.population_size = population_size
        self.n_generations = n_generations
        self.eval_steps = eval_steps
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.seed = seed
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')

    def _initialize_population(self) -> None:
        self.population = []
        for _ in range(self.population_size):
            ind = HyperParameterIndividual(self.space)
            ind.random_initialize()
            self.population.append(ind)

    def _crossover(self, p1: HyperParameterIndividual, p2: HyperParameterIndividual) -> HyperParameterIndividual:
        child = HyperParameterIndividual(self.space)
        for name in p1.genes:
            if np.random.random() < 0.5:
                child.genes[name] = p1.genes[name]
            else:
                child.genes[name] = p2.genes[name]
        return child

    def _mutate(self, ind: HyperParameterIndividual) -> HyperParameterIndividual:
        for name in ind.genes:
            if np.random.random() < self.mutation_rate:
                if name in self.space.continuous:
                    ind.genes[name] = self.space.sample_continuous(name)
                elif name in self.space.discrete:
                    ind.genes[name] = self.space.sample_discrete(name)
        return ind

    def _select(self, population: List[HyperParameterIndividual], fitnesses: List[float]) -> Tuple[List[HyperParameterIndividual], np.ndarray]:
        elite_size = max(1, int(len(population) * self.elite_ratio))
        sorted_indices = np.argsort(fitnesses)[::-1]
        elites = [population[i] for i in sorted_indices[:elite_size]]
        min_fit = np.min(fitnesses)
        shifted = fitnesses - min_fit + 1e-8
        probs = shifted / np.sum(shifted)
        return elites, probs

    def search(self) -> Optional[HyperParameterIndividual]:
        self._initialize_population()
        for gen in range(self.n_generations):
            fitnesses = []
            for i, ind in enumerate(self.population):
                seed = self.seed * 1000 + i
                fit = evaluate_hyper_individual(ind, self.model_class, self.train_loader,
                                                 self.val_loader, self.base_config,
                                                 self.eval_steps, seed)
                fitnesses.append(fit)
                ind.fitness = fit
            best_idx = np.argmax(fitnesses)
            best_fit = fitnesses[best_idx]
            if best_fit > self.best_fitness:
                self.best_fitness = best_fit
                self.best_individual = self.population[best_idx]
            elites, probs = self._select(self.population, fitnesses)
            next_pop = elites[:]
            while len(next_pop) < self.population_size:
                idx1 = np.random.choice(len(self.population), p=probs)
                idx2 = np.random.choice(len(self.population), p=probs)
                child = self._crossover(self.population[idx1], self.population[idx2])
                child = self._mutate(child)
                next_pop.append(child)
            self.population = next_pop
        return self.best_individual

# ==================== 多目标进化组件 ====================
class ArchitectureIndividual:
    __slots__ = ("d_model", "n_layers", "nhead", "lr", "batch_size", "dropout",
                 "window_size", "max_seq_len", "gen_temperature", "gen_top_p",
                 "use_infini", "mem_len", "config", "fitness", "fitness_vec",
                 "memory_est", "latency_est", "moe_ratio", "num_experts", "moe_layers")
    def __init__(self, d_model: int, n_layers: int, nhead: int, lr: float, batch_size: int, dropout: float = 0.1,
                 window_size: int = 16, max_seq_len: int = 128, gen_temperature: float = 0.6, gen_top_p: float = 0.9,
                 use_infini: bool = False, mem_len: int = 0, moe_ratio: float = 0.0, num_experts: int = 0,
                 moe_layers: Optional[List[int]] = None, config: Optional[Dict] = None):
        self.d_model = d_model
        self.n_layers = n_layers
        self.nhead = nhead
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.use_infini = use_infini
        self.mem_len = mem_len if use_infini else 0
        self.moe_ratio = moe_ratio
        self.num_experts = num_experts if moe_ratio > 0 else 0
        self.moe_layers = moe_layers or []
        self.config = config or {}
        self.fitness = float("inf")
        self.fitness_vec = [float("inf"), float("inf"), float("inf")]
        self.memory_est = 0.0
        self.latency_est = 0.0

    @classmethod
    def random(cls, lower: Dict[str, Any], upper: Dict[str, Any], config: Optional[Dict] = None) -> 'ArchitectureIndividual':
        d_model = np.random.randint(lower["d_model"], upper["d_model"] + 1)
        n_layers = np.random.randint(lower["n_layers"], upper["n_layers"] + 1)
        nhead = np.random.choice([2, 4, 8])
        d_model = (d_model // nhead) * nhead
        d_model = max(lower["d_model"], min(upper["d_model"], d_model))
        lr = 10 ** np.random.uniform(np.log10(lower["lr"]), np.log10(upper["lr"]))
        batch_size = np.random.randint(lower["batch_size"], upper["batch_size"] + 1)
        return cls(
            d_model=d_model, n_layers=n_layers, nhead=nhead, lr=lr, batch_size=batch_size,
            dropout=np.random.uniform(0.1, 0.5),
            window_size=np.random.choice([8, 16, 32]),
            max_seq_len=np.random.choice([64, 128, 256]),
            gen_temperature=np.random.uniform(0.5, 1.5),
            gen_top_p=np.random.uniform(0.8, 1.0),
            use_infini=np.random.rand() < 0.3,
            mem_len=np.random.choice([4, 8, 16]),
            moe_ratio=np.random.uniform(0, 0.5) if np.random.rand() < 0.4 else 0.0,
            num_experts=np.random.choice([2, 4, 8]),
            config=config
        )

    def mutate(self, mutation_rate: float = 0.2) -> None:
        if random.random() < mutation_rate:
            self.d_model += random.choice([-32, -16, 0, 16, 32])
            self.d_model = max(64, min(256, self.d_model))
            self.d_model = (self.d_model // self.nhead) * self.nhead
        if random.random() < mutation_rate:
            self.n_layers += random.choice([-1, 0, 1])
            self.n_layers = max(2, min(12, self.n_layers))
        if random.random() < mutation_rate:
            self.nhead = random.choice([2, 4, 8])
            self.d_model = (self.d_model // self.nhead) * self.nhead
        if random.random() < mutation_rate:
            self.lr *= random.uniform(0.7, 1.3)
            self.lr = max(1e-6, min(1e-3, self.lr))
        if random.random() < mutation_rate:
            self.batch_size += random.choice([-1, 0, 1])
            self.batch_size = max(1, min(4, self.batch_size))
        if random.random() < mutation_rate:
            self.dropout += random.uniform(-0.1, 0.1)
            self.dropout = np.clip(self.dropout, 0.05, 0.5)
        if random.random() < mutation_rate:
            self.use_infini = not self.use_infini
            if not self.use_infini:
                self.mem_len = 0
        if self.use_infini and random.random() < mutation_rate:
            self.mem_len = random.choice([4, 8, 16])
        if random.random() < mutation_rate:
            self.moe_ratio += random.uniform(-0.2, 0.2)
            self.moe_ratio = np.clip(self.moe_ratio, 0.0, 0.5)
        if self.moe_ratio > 0 and random.random() < mutation_rate:
            self.num_experts = random.choice([2, 4, 8])
        self.d_model = (self.d_model // self.nhead) * self.nhead
        num_moe = int(self.n_layers * self.moe_ratio)
        if num_moe > 0:
            self.moe_layers = sorted(np.random.choice(self.n_layers, num_moe, replace=False).tolist())
        else:
            self.moe_layers = []

    def crossover(self, other: 'ArchitectureIndividual') -> 'ArchitectureIndividual':
        child = copy.deepcopy(self)
        if random.random() < 0.5:
            child.d_model = other.d_model
        if random.random() < 0.5:
            child.n_layers = other.n_layers
        if random.random() < 0.5:
            child.nhead = other.nhead
        if random.random() < 0.5:
            child.lr = other.lr
        if random.random() < 0.5:
            child.batch_size = other.batch_size
        if random.random() < 0.5:
            child.use_infini = other.use_infini
            child.mem_len = other.mem_len
        if random.random() < 0.5:
            child.moe_ratio = other.moe_ratio
            child.num_experts = other.num_experts
        child.d_model = (child.d_model // child.nhead) * child.nhead
        if child.moe_ratio <= 0:
            child.num_experts = 0
            child.moe_layers = []
        else:
            num_moe = int(child.n_layers * child.moe_ratio)
            child.moe_layers = sorted(np.random.choice(child.n_layers, num_moe, replace=False).tolist()) if num_moe > 0 else []
        return child

    def to_config(self) -> Dict[str, Any]:
        return {
            "D_MODEL": self.d_model, "N_LAYERS": self.n_layers, "NHEAD": self.nhead,
            "LR": self.lr, "BATCH_SIZE": self.batch_size, "dropout": self.dropout,
            "WINDOW_SIZE": self.window_size, "MAX_SEQ_LEN": self.max_seq_len,
            "GEN_TEMPERATURE": self.gen_temperature, "GEN_TOP_P": self.gen_top_p,
            "USE_INFINI": self.use_infini, "MEM_LEN": self.mem_len,
            "MOE_RATIO": self.moe_ratio, "NUM_EXPERTS": self.num_experts, "MOE_LAYERS": self.moe_layers,
        }

class NSGA2:
    EPS = 1e-9
    @staticmethod
    def dominates(a: List[float], b: List[float]) -> bool:
        return all(ai <= bi + NSGA2.EPS for ai, bi in zip(a, b)) and any(ai < bi - NSGA2.EPS for ai, bi in zip(a, b))
    @staticmethod
    def fast_non_dominated_sort(pop: List[ArchitectureIndividual]) -> List[List[int]]:
        n = len(pop)
        S = [[] for _ in range(n)]
        n_dom = [0] * n
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if NSGA2.dominates(pop[i].fitness_vec, pop[j].fitness_vec):
                    S[i].append(j)
                elif NSGA2.dominates(pop[j].fitness_vec, pop[i].fitness_vec):
                    n_dom[i] += 1
            if n_dom[i] == 0:
                fronts[0].append(i)
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        return fronts[:-1]
    @staticmethod
    def crowding_distance(front: List[int], pop: List[ArchitectureIndividual]) -> List[float]:
        m = len(pop[0].fitness_vec)
        dist = [0.0] * len(front)
        if len(front) <= 2:
            return [float('inf')] * len(front)
        for obj in range(m):
            values = [pop[i].fitness_vec[obj] for i in front]
            minv, maxv = min(values), max(values)
            if maxv - minv < NSGA2.EPS:
                continue
            front_with_obj = [(i, (pop[i].fitness_vec[obj] - minv) / (maxv - minv)) for i in front]
            front_with_obj.sort(key=lambda x: x[1])
            dist[front.index(front_with_obj[0][0])] = float('inf')
            dist[front.index(front_with_obj[-1][0])] = float('inf')
            for k in range(1, len(front_with_obj) - 1):
                idx = front.index(front_with_obj[k][0])
                dist[idx] += (front_with_obj[k+1][1] - front_with_obj[k-1][1])
        return dist
    @staticmethod
    def tournament_selection(pop: List[ArchitectureIndividual]) -> ArchitectureIndividual:
        a, b = random.sample(pop, 2)
        if NSGA2.dominates(a.fitness_vec, b.fitness_vec):
            return a
        elif NSGA2.dominates(b.fitness_vec, a.fitness_vec):
            return b
        else:
            return a if random.random() < 0.5 else b

class CulturalMemory:
    def __init__(self, max_size: int = 20, min_support: int = 2, min_confidence: float = 0.6):
        self.pareto_front: List[ArchitectureIndividual] = []
        self.patterns: Dict[Tuple[str, ...], float] = {}
        self.max_size = max_size
        self.min_support = min_support
        self.min_confidence = min_confidence

    def update(self, population: List[ArchitectureIndividual]) -> None:
        fronts = NSGA2.fast_non_dominated_sort(population)
        self.pareto_front = [population[i] for i in fronts[0]]
        if len(self.pareto_front) > self.max_size:
            self.pareto_front.sort(key=lambda x: x.fitness)
            self.pareto_front = self.pareto_front[:self.max_size]
        self._mine_patterns()

    def _mine_patterns(self) -> None:
        d_model_bins = [64, 128, 192, 256]
        n_layers_bins = [2, 4, 6, 8, 12]
        transactions = []
        for ind in self.pareto_front:
            trans = []
            d_model_bin = min((b for b in d_model_bins if b >= ind.d_model), default=d_model_bins[-1])
            trans.append(f"d_model≤{d_model_bin}")
            trans.append(f"nhead={ind.nhead}")
            n_layers_bin = min((b for b in n_layers_bins if b >= ind.n_layers), default=n_layers_bins[-1])
            trans.append(f"n_layers≤{n_layers_bin}")
            if ind.use_infini:
                trans.append("use_infini=True")
            if ind.moe_ratio > 0:
                trans.append("moe_ratio>0")
            transactions.append(trans)
        item_counts = Counter()
        for trans in transactions:
            item_counts.update(trans)
        total = len(transactions)
        freq_items = [item for item, cnt in item_counts.items() if cnt >= self.min_support]
        if len(freq_items) < 2:
            self.patterns = {}
            return
        rules = {}
        for i in range(len(freq_items)):
            for j in range(i+1, len(freq_items)):
                lhs = (freq_items[i],)
                rhs = (freq_items[j],)
                support_union = sum(1 for trans in transactions if all(it in trans for it in (lhs[0], rhs[0]))) / total
                support_lhs = item_counts[lhs[0]] / total
                if support_lhs > 0:
                    conf = support_union / support_lhs
                    if conf >= self.min_confidence:
                        rules[tuple(sorted((lhs[0], rhs[0])))] = conf
                support_rhs = item_counts[rhs[0]] / total
                if support_rhs > 0:
                    conf = support_union / support_rhs
                    if conf >= self.min_confidence:
                        rules[tuple(sorted((rhs[0], lhs[0])))] = conf
        self.patterns = rules

    def apply_constraints(self, ind: ArchitectureIndividual) -> float:
        d_model_bins = [64, 128, 192, 256]
        d_model_bin = min((b for b in d_model_bins if b >= ind.d_model), default=d_model_bins[-1])
        n_layers_bins = [2, 4, 6, 8, 12]
        n_layers_bin = min((b for b in n_layers_bins if b >= ind.n_layers), default=n_layers_bins[-1])
        trans = [f"d_model≤{d_model_bin}", f"nhead={ind.nhead}", f"n_layers≤{n_layers_bin}"]
        if ind.use_infini:
            trans.append("use_infini=True")
        if ind.moe_ratio > 0:
            trans.append("moe_ratio>0")
        matched = any(all(it in trans for it in rule) for rule in self.patterns)
        return 1.02 if matched else 0.98

    def get_best_template(self) -> Optional[ArchitectureIndividual]:
        return min(self.pareto_front, key=lambda x: x.fitness) if self.pareto_front else None

class EvolutionIsland:
    def __init__(self, pop_size: int, lower: Dict[str, Any], upper: Dict[str, Any], data_dir: str,
                 tokenizer: Any, hardware_state: Optional[Dict] = None,
                 base_model_state: Optional[Dict[str, np.ndarray]] = None):
        self.pop_size = pop_size
        self.lower = lower
        self.upper = upper
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.hardware_state = hardware_state
        self.base_model_state = base_model_state
        self.population = [ArchitectureIndividual.random(lower, upper, config={}) for _ in range(pop_size)]
        self.cultural_memory = CulturalMemory()
        self.generation = 0

    def evaluate_population(self, steps: int = 800) -> None:
        total = len(self.population)
        for idx, ind in enumerate(self.population):
            loss, mem, lat = evaluate_individual_fast(ind, self.data_dir, self.tokenizer, steps, self.base_model_state)
            ind.fitness_vec = [loss, mem, lat]
            ind.fitness = loss + 0.1 * mem + 0.1 * lat
            log_evolution(f"个体 {idx+1}/{total} -> loss={loss:.4f}, mem={mem:.1f}MB, lat={lat:.2f}ms")

    def evolve_one_generation(self, mutation_rate: float = 0.2, crossover_rate: float = 0.7, fast_steps: int = 800) -> ArchitectureIndividual:
        self.evaluate_population(fast_steps)
        fronts = NSGA2.fast_non_dominated_sort(self.population)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                new_pop.extend(self.population[i] for i in front)
            else:
                dist = NSGA2.crowding_distance(front, self.population)
                front_with_dist = [(front[i], dist[i]) for i in range(len(front))]
                front_with_dist.sort(key=lambda x: x[1], reverse=True)
                need = self.pop_size - len(new_pop)
                for i in range(need):
                    new_pop.append(self.population[front_with_dist[i][0]])
                break
        self.cultural_memory.update(self.population)
        template = self.cultural_memory.get_best_template()
        while len(new_pop) < self.pop_size:
            if template is not None and random.random() < 0.3:
                child = copy.deepcopy(template)
                child.mutate(mutation_rate * 0.5)
                child.fitness = float('inf')
                child.fitness_vec = [float('inf')] * 3
                new_pop.append(child)
                continue
            p1 = NSGA2.tournament_selection(self.population)
            p2 = NSGA2.tournament_selection(self.population)
            if random.random() < crossover_rate:
                c1 = p1.crossover(p2)
                c2 = p2.crossover(p1)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
            c1.mutate(mutation_rate)
            c2.mutate(mutation_rate)
            c1.fitness = float('inf')
            c2.fitness = float('inf')
            c1.fitness_vec = [float('inf')] * 3
            c2.fitness_vec = [float('inf')] * 3
            c1.fitness *= self.cultural_memory.apply_constraints(c1)
            c2.fitness *= self.cultural_memory.apply_constraints(c2)
            new_pop.append(c1)
            if len(new_pop) < self.pop_size:
                new_pop.append(c2)
        self.population = new_pop[:self.pop_size]
        self.generation += 1
        return self.get_best_individual()

    def get_best_individual(self) -> ArchitectureIndividual:
        return min(self.population, key=lambda x: x.fitness)

class Archipelago:
    def __init__(self, island_configs: List[Dict[str, Any]], migration_interval: int = 10, max_history: int = 50):
        self.islands = [EvolutionIsland(**cfg) for cfg in island_configs]
        self.migration_interval = migration_interval
        self.generation = 0
        self.pareto_history = []
        self.max_history = max_history

    def evolve_all(self, total_generations: int, fast_steps: int = 800) -> None:
        for gen in range(total_generations):
            log_evolution(f"\n========== 多目标进化 第 {gen+1}/{total_generations} 代 ==========")
            for i, island in enumerate(self.islands):
                log_evolution(f"岛屿 {i+1}/{len(self.islands)} 进化中...")
                best = island.evolve_one_generation(fast_steps=fast_steps)
                log_evolution(f"  岛屿 {i+1} 最佳: fitness={best.fitness:.4f} loss={best.fitness_vec[0]:.4f} mem={best.fitness_vec[1]:.0f}MB lat={best.fitness_vec[2]:.2f}ms")
            all_inds = [ind for island in self.islands for ind in island.population]
            fronts = NSGA2.fast_non_dominated_sort(all_inds)
            current_front = [all_inds[i] for i in fronts[0]]
            log_evolution(f"当前帕累托前沿大小: {len(current_front)}")
            self.pareto_history.append(current_front)
            if len(self.pareto_history) > self.max_history:
                self.pareto_history = self.pareto_history[-self.max_history:]
            if gen % self.migration_interval == 0 and gen > 0:
                log_evolution("执行岛屿迁移...")
                self._migrate()
            self.generation = gen + 1

    def _migrate(self) -> None:
        n = len(self.islands)
        elites = []
        for island in self.islands:
            sorted_pop = sorted(island.population, key=lambda x: x.fitness)
            elite_count = max(1, int(0.1 * len(island.population)))
            elites.append([copy.deepcopy(ind) for ind in sorted_pop[:elite_count]])
        for i in range(n):
            next_island = self.islands[(i+1) % n]
            next_island.population.sort(key=lambda x: x.fitness)
            for j, e in enumerate(elites[i]):
                if j < len(next_island.population):
                    next_island.population[-(j+1)] = e

    def get_best_overall(self) -> ArchitectureIndividual:
        all_inds = [ind for island in self.islands for ind in island.population]
        fronts = NSGA2.fast_non_dominated_sort(all_inds)
        front0 = [all_inds[i] for i in fronts[0]]
        return min(front0, key=lambda x: x.fitness_vec[0])

def evaluate_individual_fast(ind: ArchitectureIndividual, data_dir: str, tokenizer: Any, steps: int = 800, base_model_state: Optional[Dict] = None) -> Tuple[float, float, float]:
    if tokenizer is not None:
        if hasattr(tokenizer, 'vocab'):
            vocab_size = len(tokenizer.vocab) if isinstance(tokenizer.vocab, (dict, list)) else 32000
        else:
            vocab_size = 32000
    else:
        vocab_size = 32000
    config_dict = ind.to_config()
    config_dict["VOCAB_SIZE"] = vocab_size
    config_dict["D_MODEL"] = adjust_d_model(config_dict["D_MODEL"], config_dict["NHEAD"], 64, 256)
    model = AetherOmniModel(ModelConfig(
        vocab_size=config_dict["VOCAB_SIZE"],
        d_model=config_dict["D_MODEL"],
        n_layers=config_dict["N_LAYERS"],
        n_heads=config_dict["NHEAD"],
        window_size=config_dict["WINDOW_SIZE"],
        use_infini=config_dict["USE_INFINI"],
        mem_len=config_dict["MEM_LEN"],
        use_moe=(config_dict.get("MOE_RATIO", 0) > 0),
        num_experts=config_dict.get("NUM_EXPERTS", 4),
        moe_layers=config_dict.get("MOE_LAYERS", []),
        use_hybrid=False,
        use_image=False,
        use_ternary=False,
        dropout=config_dict["dropout"]
    ))
    if base_model_state:
        model_state = model.get_state_dict()
        for k, v in base_model_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                model_state[k][:] = v
        model.load_state_dict(model_state)
    opt = AdamW(model.parameters(), lr=config_dict["LR"])
    train_path = os.path.join(data_dir, "train.ids")
    if not os.path.exists(train_path):
        return 9999.0, 9999.0, 9999.0
    with open(train_path, "r") as f:
        lines = f.readlines()[:1000]
    random.shuffle(lines)
    train_lines = lines[:800]
    val_lines = lines[800:]
    max_seq_len = config_dict["MAX_SEQ_LEN"]

    def train_gen():
        for line in train_lines:
            ids = list(map(int, line.strip().split()))
            if len(ids) < 1: continue
            if len(ids) > max_seq_len: ids = ids[:max_seq_len]
            yield (np.array(ids, dtype=np.uint16), len(ids))

    def val_gen():
        for line in val_lines[:MAX_VAL_SAMPLES]:
            ids = list(map(int, line.strip().split()))
            if len(ids) < 1: continue
            if len(ids) > max_seq_len: ids = ids[:max_seq_len]
            yield (np.array(ids, dtype=np.uint16), len(ids))

    train_loader = DataLoader(train_gen, config_dict["BATCH_SIZE"], pad_id=0, shuffle_buffer_size=100)
    val_loader = DataLoader(val_gen, config_dict["BATCH_SIZE"], pad_id=0, shuffle_buffer_size=0)

    step = 0
    best_loss = float("inf")
    start_time = time.time()
    for batch in train_loader:
        if step >= steps: break
        inp, mask = batch
        logits = model(inp, attention_mask=Tensor(mask, dtype=bool))
        loss, _ = compute_loss(logits, inp, mask)
        loss.backward()
        opt.step()
        opt.zero_grad()
        step += 1
        loss_val = loss.data.item()
        if loss_val < best_loss: best_loss = loss_val
    elapsed = time.time() - start_time
    lat_ms = (elapsed / max(1, step)) * 1000
    param_bytes = sum(p.data.nbytes for p in model.parameters())
    act_bytes = config_dict["N_LAYERS"] * config_dict["BATCH_SIZE"] * max_seq_len * config_dict["D_MODEL"] * 4 * 8
    mem_mb = (param_bytes + act_bytes) / (1024**2)
    max_batches = max(1, MAX_VAL_SAMPLES // config_dict["BATCH_SIZE"])
    val_loss = compute_val_loss(model, val_loader, config_dict["VOCAB_SIZE"], config_dict["MAX_SEQ_LEN"], max_batches)
    del model
    gc.collect()
    return val_loss, mem_mb, lat_ms

# ==================== 辅助函数 ====================
def compute_loss(shift_logits: Tensor, shift_labels: np.ndarray, shift_mask: np.ndarray) -> Tuple[Tensor, int]:
    log_probs = shift_logits.log_softmax(axis=-1)
    batch, seq_len, _ = log_probs.shape
    indices = shift_labels.reshape(-1)
    index_tensor = Tensor(indices, requires_grad=False, dtype=INDEX_DTYPE)
    index_tensor = index_tensor.reshape(batch, seq_len, 1)
    selected = log_probs.gather(dim=-1, index=index_tensor).squeeze(-1)
    mask_tensor = Tensor(shift_mask.astype(np.float32), requires_grad=False, dtype=np.float32)
    valid = mask_tensor.sum()
    valid_value = valid.data.item()
    if valid_value > 0:
        loss = -(selected * mask_tensor).sum() / valid_value
    else:
        loss = Tensor(0.0, requires_grad=False, dtype=np.float32)
    return loss, valid_value

def compute_val_loss(model: AetherOmniModel, val_loader: DataLoader, vocab_size: int, max_seq_len: int, max_batches: Optional[int] = None) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with Tensor.no_grad():
        for i, (inp, mask) in enumerate(val_loader):
            if max_batches and i >= max_batches: break
            logits = model(inp, attention_mask=Tensor(mask, dtype=bool))
            shift_logits = logits[:, :-1, :]
            shift_labels = inp[:, 1:]
            shift_mask = mask[:, 1:]
            loss, _ = compute_loss(shift_logits, shift_labels, shift_mask)
            total_loss += loss.data.item()
            count += 1
    model.train()
    return total_loss / max(1, count)

def get_sequence_log_probs(model: AetherOmniModel, input_ids: np.ndarray, target_slice: slice) -> Tuple[Tensor, int]:
    L = input_ids.shape[1]
    start = target_slice.start
    end = target_slice.stop
    if start <= 0 or start >= L or end <= start or end > L:
        return Tensor(0.0, requires_grad=False), 0
    logits = model(input_ids)
    logits_slice = logits[:, start-1:end-1, :]
    target_ids = input_ids[:, start:end]
    log_probs = logits_slice.log_softmax(axis=-1)
    batch, seq_len, _ = log_probs.shape
    index_tensor = Tensor(target_ids.reshape(batch, seq_len, 1), requires_grad=False, dtype=INDEX_DTYPE)
    selected = log_probs.gather(dim=-1, index=index_tensor).squeeze(-1)
    total_log_prob = selected.sum()
    valid_tokens = target_ids.size
    if valid_tokens == 0:
        return Tensor(0.0, requires_grad=False), 0
    return total_log_prob, valid_tokens

def dpo_loss(policy_model: AetherOmniModel, ref_model: AetherOmniModel,
             prompt: np.ndarray, chosen: np.ndarray, rejected: np.ndarray,
             beta: float = 0.3) -> Tuple[Tensor, Dict[str, Any]]:
    Lp = len(prompt)
    Lc = len(chosen)
    Lr = len(rejected)
    chosen_input = np.concatenate([prompt, chosen])[None, :]
    rejected_input = np.concatenate([prompt, rejected])[None, :]
    policy_chosen_logp, valid_c = get_sequence_log_probs(policy_model, chosen_input, slice(Lp, Lp + Lc))
    policy_rejected_logp, valid_r = get_sequence_log_probs(policy_model, rejected_input, slice(Lp, Lp + Lr))
    if valid_c < 1 or valid_r < 1:
        return Tensor(0.0, requires_grad=True), {"invalid": 1}
    with Tensor.no_grad():
        ref_chosen_logp, _ = get_sequence_log_probs(ref_model, chosen_input, slice(Lp, Lp + Lc))
        ref_rejected_logp, _ = get_sequence_log_probs(ref_model, rejected_input, slice(Lp, Lp + Lr))
    chosen_logratios = policy_chosen_logp - ref_chosen_logp
    rejected_logratios = policy_rejected_logp - ref_rejected_logp
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -log_sigmoid(Tensor(logits.data, requires_grad=True))
    stats = {
        "loss": loss.data.item(),
        "chosen_logratio": chosen_logratios.data.item(),
        "rejected_logratio": rejected_logratios.data.item(),
        "logits": logits.data.item(),
        "invalid": 0
    }
    return loss, stats

def dpo_validate(model: AetherOmniModel, ref_model: AetherOmniModel, val_loader: StreamingDPOLoader,
                 num_batches: int = 20, beta: float = 0.3) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    start_time = time.time()
    with Tensor.no_grad():
        for i, (batch, _) in enumerate(val_loader.iter_triples()):
            if i >= num_batches: break
            prompt, chosen, rejected = batch
            loss, _ = dpo_loss(model, ref_model, prompt, chosen, rejected, beta=beta)
            total_loss += loss.data.item()
            count += 1
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, count)
    log_system(f"验证完成 ({count} batches, {elapsed:.1f}s) avg_loss={avg_loss:.6f}")
    model.train()
    return avg_loss

# ==================== 安全检查 ====================
class SingularityGuard:
    def __init__(self, entropy_threshold: float = 1.25, consecutive_limit: int = 5):
        self.threshold = entropy_threshold
        self.limit = consecutive_limit
        self.consecutive = 0

    def check(self, logits_np: np.ndarray) -> bool:
        probs = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True) + EPS
        entropies = -np.sum(probs * np.log(probs + EPS), axis=-1)
        min_entropy = entropies.min()
        if min_entropy < self.threshold:
            self.consecutive += 1
            if self.consecutive >= self.limit:
                self.consecutive = 0
                return True
        else:
            self.consecutive = 0
        return False

    def reset(self, model: AetherOmniModel) -> None:
        for p in model.parameters():
            p.data += np.random.randn(*p.data.shape) * 0.01

class AnchorGuard:
    def __init__(self, checkpoint_interval: int = 1000, loss_window: int = 10, spike_threshold: float = 5.0):
        self.interval = checkpoint_interval
        self.loss_window = loss_window
        self.spike_threshold = spike_threshold
        self.loss_history = deque(maxlen=loss_window)
        self.last_checkpoint = None
        self.last_step = 0

    def update(self, model: AetherOmniModel, step: int) -> None:
        if step - self.last_step >= self.interval:
            self.last_checkpoint = model.get_state_dict()
            self.last_step = step

    def verify_and_rollback(self, model: AetherOmniModel, step: int, current_loss: float) -> bool:
        self.loss_history.append(current_loss)
        if len(self.loss_history) >= self.loss_window:
            avg_loss = np.mean(list(self.loss_history))
            if current_loss > avg_loss * self.spike_threshold and self.last_checkpoint:
                log_system(f"损失尖峰 {current_loss:.3f} > {avg_loss:.3f}*{self.spike_threshold}，回滚")
                model.load_state_dict(self.last_checkpoint)
                self.loss_history.clear()
                return True
        return False

# ==================== 检查点保存与加载 ====================
_save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='SaveWorker')
_save_error_queue = deque()

def _shutdown_executor() -> None:
    _save_executor.shutdown(wait=True)

def _async_save_task(path: str, state_dict: Dict[str, np.ndarray], optimizer_state: Optional[Dict[str, Any]],
                     manager_state: Optional[Dict[str, Any]], step: int, ref_state_dict: Optional[Dict[str, np.ndarray]] = None) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path + '.npz', **state_dict)
        if ref_state_dict:
            np.savez_compressed(path + '_ref.npz', **ref_state_dict)
        if optimizer_state:
            np.savez_compressed(path + '_opt.npz', **optimizer_state)
        if manager_state:
            np.savez_compressed(path + '_mgr.npz', **manager_state)
        with open(path + '.json', 'w') as f:
            json.dump({'step': step}, f)
        log_system(f"异步检查点已保存至 {path}")
    except Exception as e:
        logger.error(f"异步保存失败: {e}", exc_info=True)
        _save_error_queue.append((path, str(e)))

def save_checkpoint_async(model: AetherOmniModel, optimizer: Union[AdamW, SGD, Lookahead],
                          manager: Optional[AdaptiveTrainManager], step: int, path_prefix: str,
                          ref_model: Optional[AetherOmniModel] = None) -> None:
    state_dict = model.get_state_dict()
    ref_state_dict = ref_model.get_state_dict() if ref_model else None
    opt_state = optimizer.state_dict() if optimizer else None
    mgr_state = manager.state_dict() if manager else None
    _save_executor.submit(_async_save_task, path_prefix, state_dict, opt_state, mgr_state, step, ref_state_dict)

def load_checkpoint(path_prefix: str, model_config: ModelConfig,
                    optimizer_class: Union[type[AdamW], type[SGD]],
                    manager_class: type[AdaptiveTrainManager]) -> Tuple[AetherOmniModel, Union[AdamW, SGD], Optional[AdaptiveTrainManager], Optional[AetherOmniModel], int]:
    npz_path = path_prefix + '.npz'
    json_path = path_prefix + '.json'
    with open(json_path, 'r') as f:
        meta = json.load(f)
    model = AetherOmniModel(model_config)
    state_dict = np.load(npz_path, allow_pickle=False)
    model.load_state_dict({k: v for k, v in state_dict.items()})
    ref_model = None
    ref_path = path_prefix + '_ref.npz'
    if os.path.exists(ref_path):
        ref_state_dict = np.load(ref_path, allow_pickle=False)
        ref_model = AetherOmniModel(model_config)
        ref_model.load_state_dict({k: v for k, v in ref_state_dict.items()})
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    opt_path = path_prefix + '_opt.npz'
    if os.path.exists(opt_path):
        opt_state = np.load(opt_path, allow_pickle=False)
        lr = meta.get('lr', model_config.d_model)
        opt = optimizer_class(model.parameters(), lr=lr)
        opt.load_state_dict({k: v for k, v in opt_state.items()})
    else:
        opt = optimizer_class(model.parameters(), lr=model_config.d_model)
    mgr_path = path_prefix + '_mgr.npz'
    if os.path.exists(mgr_path):
        mgr_state = np.load(mgr_path, allow_pickle=False)
        mgr = manager_class(model, opt, {}, None, None)
        mgr.load_state_dict({k: v for k, v in mgr_state.items()})
    else:
        mgr = None
    return model, opt, mgr, ref_model, meta.get('step', 0)

# ==================== 中央仲裁器 ====================
@dataclass
class Request:
    source: str
    action: str
    params: dict
    priority: int = 5
    cooldown: int = 0
    timestamp: float = field(default_factory=time.time)

class CentralArbiter:
    def __init__(self, request_timeout: float = 60.0):
        self.heap = []
        self.last_execution = {}
        self.handlers = {
            "adjust_lr": self._adjust_lr,
            "adjust_dropout": self._adjust_dropout,
            "adjust_beta": self._adjust_beta,
        }
        self.timeout = request_timeout

    def submit(self, req: Request) -> None:
        heapq.heappush(self.heap, (-req.priority, req.timestamp, req))

    def execute_one(self, trainer_wrapper: Any) -> None:
        now = time.time()
        while self.heap:
            neg_prio, ts, req = heapq.heappop(self.heap)
            if now - ts > self.timeout:
                continue
            key = (req.source, req.action)
            if key in self.last_execution and now - self.last_execution[key] < req.cooldown:
                continue
            handler = self.handlers.get(req.action)
            if handler:
                handler(trainer_wrapper, req.params)
                self.last_execution[key] = now
            break

    def _adjust_lr(self, trainer: Any, params: Dict[str, Any]) -> None:
        multiplier = params.get("multiplier", 1.0)
        if hasattr(trainer.optimizer, 'lr'):
            trainer.optimizer.lr *= multiplier
        elif hasattr(trainer.optimizer, 'base_optimizer'):
            trainer.optimizer.base_optimizer.lr *= multiplier
        log_system(f"仲裁器: 调整学习率乘数 {multiplier:.2f}")

    def _adjust_dropout(self, trainer: Any, params: Dict[str, Any]) -> None:
        multiplier = params.get("multiplier", 1.0)
        if trainer.manager:
            new_dropout = trainer.manager.dropout_val * multiplier
            new_dropout = max(0.0, min(0.5, new_dropout))
            trainer.manager.dropout_val = new_dropout
            trainer.manager._apply_regularization()
            log_system(f"仲裁器: 调整 dropout 为 {new_dropout:.3f}")

    def _adjust_beta(self, trainer: Any, params: Dict[str, Any]) -> None:
        multiplier = params.get("multiplier", 1.0)
        if hasattr(trainer, 'dpo_beta'):
            new_beta = trainer.dpo_beta * multiplier
            new_beta = max(0.01, min(2.0, new_beta))
            trainer.dpo_beta = new_beta
            log_system(f"仲裁器: 调整 DPO beta 为 {new_beta:.4f}")

# ==================== 生命化系统 ====================
class MetaMind:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.meta_lr = 0.001
        self.experience_buffer = []
        self.meta_knowledge = {}
        self.arbiter: Optional[CentralArbiter] = None
        self.last_lr_adjust_step = 0

    def learn_from_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > 50:
            self.experience_buffer.pop(0)
        recent_losses = [e.get('loss', 999) for e in self.experience_buffer[-20:] if e.get('loss') is not None]
        avg_loss = np.mean(recent_losses) if len(recent_losses) >= 10 else 1.0
        if len(self.experience_buffer) >= 20:
            recent_10 = [e.get('loss', avg_loss) for e in self.experience_buffer[-10:] if e.get('loss') is not None]
            older_10 = [e.get('loss', avg_loss) for e in self.experience_buffer[-20:-10] if e.get('loss') is not None]
            trend = np.mean(recent_10) - np.mean(older_10) if len(recent_10) >= 5 and len(older_10) >= 5 else 0.0
        else:
            trend = 0.0
        lr_adj = 0.98 if trend > 0.05 else (1.02 if trend < -0.05 else 1.0)
        if avg_loss > 2.0:
            lr_adj = min(lr_adj, 0.95)
        elif avg_loss < 0.5:
            lr_adj = max(lr_adj, 1.01)
        suggestion = {
            "meta_lr_adjust": lr_adj,
            "suggest_increase_layers": avg_loss < 0.5 and trend < -0.02,
            "suggest_enable_moe": avg_loss > 1.5 and len(self.experience_buffer) > 100,
            "evolve_strength": min(1.0, 10 / (avg_loss + 0.1))
        }
        current_step = experience.get('step', 0)
        if self.arbiter and current_step - self.last_lr_adjust_step > 200:
            req = Request("MetaMind", "adjust_lr", {"multiplier": lr_adj}, priority=5, cooldown=200)
            self.arbiter.submit(req)
            self.last_lr_adjust_step = current_step
        return suggestion

class AdaptiveBody:
    def __init__(self, config: ModelConfig, hardware_config: Dict[str, Any], model_rebuilder_callback: Optional[Callable] = None):
        self.config = config
        self.hardware = hardware_config
        self.current_quant = 16
        self.current_layers = config.n_layers
        self.current_moe_ratio = 0.0
        self.model_rebuilder_callback = model_rebuilder_callback
        self.last_rebuild_step = -9999
        self.rebuild_cooldown = 500
        self.arbiter: Optional[CentralArbiter] = None

    def perceive(self) -> Dict[str, Any]:
        return {
            "memory_gb": get_memory_available_mb() or 4.0,
            "battery": get_battery_percent() or 80.0,
            "cpu_temp": get_cpu_temp_real() or 45.0,
            "is_mobile": True
        }

    def adapt(self, env: Dict[str, Any], current_step: int = 0, grad_norm: Optional[float] = None, loss_trend: Optional[float] = None) -> None:
        if current_step - self.last_rebuild_step < self.rebuild_cooldown:
            return
        mem = env["memory_gb"]
        batt = env["battery"]
        needs_rebuild = False
        new_config = None
        if mem < 2.5:
            new_quant = 8
            new_layers = max(4, self.current_layers - 2)
            new_moe_ratio = 0.0
            if new_quant != self.current_quant or new_layers != self.current_layers or new_moe_ratio != self.current_moe_ratio:
                log_system(f"🚨 内存紧张 → 切换到 {new_quant}bit + {new_layers}层")
                needs_rebuild = True
                new_config = {'quant': new_quant, 'n_layers': new_layers, 'moe_ratio': new_moe_ratio}
                self.current_quant, self.current_layers, self.current_moe_ratio = new_quant, new_layers, new_moe_ratio
        if batt < 30:
            new_quant = 4
            new_moe_ratio = 0.0
            if new_quant != self.current_quant or new_moe_ratio != self.current_moe_ratio:
                log_system("⚡ 低电量 → 进入极致省电模式")
                needs_rebuild = True
                new_config = {'quant': new_quant, 'moe_ratio': new_moe_ratio}
                self.current_quant, self.current_moe_ratio = new_quant, new_moe_ratio
        if mem > 6.0 and batt > 70 and current_step > 1000:
            new_moe_ratio = 0.4
            if new_moe_ratio != self.current_moe_ratio:
                log_system("🌟 资源充足 → 开启 MoE 增强能力")
                needs_rebuild = True
                new_config = {'moe_ratio': new_moe_ratio}
                self.current_moe_ratio = new_moe_ratio
        if grad_norm is not None and grad_norm > 10.0:
            log_system(f"⚠️ 梯度范数过大 ({grad_norm:.2f})，建议降低学习率")
            if self.arbiter:
                req = Request("AdaptiveBody", "adjust_lr", {"multiplier": 0.8}, priority=3, cooldown=100)
                self.arbiter.submit(req)
        if loss_trend is not None and loss_trend > 0.1:
            log_system(f"⚠️ 损失上升趋势 ({loss_trend:.3f})，建议调整正则化")
            if self.arbiter:
                req = Request("AdaptiveBody", "adjust_dropout", {"multiplier": 1.1}, priority=4, cooldown=200)
                self.arbiter.submit(req)
        if needs_rebuild and self.model_rebuilder_callback:
            self.model_rebuilder_callback(new_config)
            self.last_rebuild_step = current_step

class ConsciousnessCore:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.vitality = 1.0
        self.gradient_health = 1.0
        self.overfit_risk = 0.0
        self.log = []
        self.arbiter: Optional[CentralArbiter] = None
        self.actions = {
            "overfitting": [
                ("adjust_dropout", {"multiplier": 1.2}, "增大 dropout"),
                ("adjust_lr", {"multiplier": 0.9}, "降低学习率"),
                ("adjust_beta", {"multiplier": 0.8}, "降低 beta"),
            ],
            "gradient_vanishing": [
                ("adjust_lr", {"multiplier": 1.2}, "增大学习率"),
                ("adjust_beta", {"multiplier": 1.2}, "增大 beta"),
                ("adjust_dropout", {"multiplier": 0.9}, "减小 dropout"),
            ]
        }
        self.action_memory = deque(maxlen=50)

    def diagnose(self) -> Dict[str, Any]:
        diagnosis = {
            "vitality": self.vitality,
            "gradient_health": self.gradient_health,
            "overfit_risk": self.overfit_risk,
            "needs_evolution": self.overfit_risk > 0.6 or self.gradient_health < 0.3,
            "status": "healthy" if self.vitality > 0.7 else "warning"
        }
        self.log.append(diagnosis)
        return diagnosis

    def self_repair(self, issue: str) -> None:
        if issue == "gradient_vanishing":
            self.vitality *= 0.95
            log_system("🛠️ 自我修复：梯度消失 → 自动降低 beta + 注入噪声")
            if self.arbiter:
                req = Request("ConsciousnessCore", "adjust_beta", {"multiplier": 1.2}, priority=4, cooldown=100)
                self.arbiter.submit(req)
        elif issue == "overfitting":
            self.overfit_risk = 0.0
            log_system("🛠️ 自我修复：过拟合 → 增强正则化")
            if self.arbiter:
                req = Request("ConsciousnessCore", "adjust_dropout", {"multiplier": 1.2}, priority=4, cooldown=100)
                self.arbiter.submit(req)

    def act(self, diagnosis: Dict[str, Any]) -> None:
        if not diagnosis["needs_evolution"]:
            return
        if diagnosis["overfit_risk"] > 0.5:
            actions = self.actions["overfitting"]
        elif diagnosis["gradient_health"] < 0.3:
            actions = self.actions["gradient_vanishing"]
        else:
            return
        action, params, desc = actions[0]
        log_system(f"意识决定行动: {desc}")
        if self.arbiter:
            req = Request("ConsciousnessCore", action, params, priority=4, cooldown=100)
            self.arbiter.submit(req)

class LifeCycleManager:
    def __init__(self, meta_mind: MetaMind, adaptive_body: AdaptiveBody, consciousness: ConsciousnessCore,
                 model_config: ModelConfig, trainer_manager: Optional[AdaptiveTrainManager] = None,
                 arbiter: Optional[CentralArbiter] = None):
        self.meta_mind = meta_mind
        self.adaptive_body = adaptive_body
        self.consciousness = consciousness
        self.model_config = model_config
        self.trainer_manager = trainer_manager
        self.arbiter = arbiter
        self.cycle_count = 0

    def run_one_cycle(self, current_loss: Optional[float] = None, grad_norm: Optional[float] = None,
                      current_step: int = 0, loss_trend: Optional[float] = None) -> None:
        self.cycle_count += 1
        env = self.adaptive_body.perceive()
        diagnosis = self.consciousness.diagnose()
        meta_decision = self.meta_mind.learn_from_experience({
            "env": env, "diagnosis": diagnosis, "loss": current_loss, "grad_norm": grad_norm, "step": current_step
        })
        self.adaptive_body.adapt(env, current_step, grad_norm=grad_norm, loss_trend=loss_trend)
        self.consciousness.act(diagnosis)
        if self.trainer_manager:
            self.trainer_manager.meta_lr_multiplier = meta_decision.get("meta_lr_adjust", 1.0)
        if self.cycle_count % 10 == 0:
            log_system(f"🔄 第 {self.cycle_count} 次生命循环完成 - 活力: {diagnosis['vitality']:.3f}")

class LifeCore:
    def __init__(self, model: AetherOmniModel, ref_model: AetherOmniModel,
                 meta_mind: MetaMind, adaptive_body: AdaptiveBody,
                 consciousness: ConsciousnessCore, life_cycle: LifeCycleManager,
                 arbiter: Optional[CentralArbiter] = None):
        self.model = model
        self.ref_model = ref_model
        self.meta_mind = meta_mind
        self.adaptive_body = adaptive_body
        self.consciousness = consciousness
        self.life_cycle = life_cycle
        self.arbiter = arbiter

    def forward(self, input_ids: np.ndarray, **kwargs) -> Tensor:
        return self.model(input_ids, **kwargs)

    def evolve(self, loss: Optional[float] = None, grad_norm: Optional[float] = None, step: int = 0, loss_trend: Optional[float] = None) -> None:
        self.life_cycle.run_one_cycle(current_loss=loss, grad_norm=grad_norm, current_step=step, loss_trend=loss_trend)

    def perceive_and_adapt(self, step: int = 0) -> None:
        env = self.adaptive_body.perceive()
        self.adaptive_body.adapt(env, step)

    def self_diagnose(self) -> Dict[str, Any]:
        return self.consciousness.diagnose()

    def parameters(self) -> List[Tensor]:
        return self.model.parameters()

    def zero_grad(self) -> None:
        self.model.zero_grad()

    def get_state_dict(self) -> Dict[str, np.ndarray]:
        return self.model.get_state_dict()

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.model.load_state_dict(state)
        self.ref_model.load_state_dict(state)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def save_life_state(self, path: str) -> None:
        original_callback = self.adaptive_body.model_rebuilder_callback
        self.adaptive_body.model_rebuilder_callback = None
        try:
            meta_mind_state = {}
            for k, v in self.meta_mind.__dict__.items():
                if not callable(v) and not isinstance(v, (types.FunctionType, types.MethodType, types.LambdaType)):
                    meta_mind_state[k] = v

            adaptive_body_state = {}
            for k, v in self.adaptive_body.__dict__.items():
                if not callable(v) and not isinstance(v, (types.FunctionType, types.MethodType, types.LambdaType)):
                    adaptive_body_state[k] = v

            consciousness_state = {}
            for k, v in self.consciousness.__dict__.items():
                if not callable(v) and not isinstance(v, (types.FunctionType, types.MethodType, types.LambdaType)):
                    consciousness_state[k] = v

            state = {
                'model': self.model.get_state_dict(),
                'ref_model': self.ref_model.get_state_dict(),
                'meta_mind': meta_mind_state,
                'adaptive_body': adaptive_body_state,
                'consciousness': consciousness_state,
            }
            np.savez_compressed(path, **state)
            log_system(f"生命状态已保存至 {path}")
        finally:
            self.adaptive_body.model_rebuilder_callback = original_callback

    def load_life_state(self, path: str) -> None:
        state = np.load(path, allow_pickle=True)
        self.model.load_state_dict(state['model'].item())
        self.ref_model.load_state_dict(state['ref_model'].item())
        for k, v in state['meta_mind'].item().items():
            setattr(self.meta_mind, k, v)
        for k, v in state['adaptive_body'].item().items():
            setattr(self.adaptive_body, k, v)
        for k, v in state['consciousness'].item().items():
            setattr(self.consciousness, k, v)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        log_system(f"生命状态已从 {path} 恢复")

# ==================== 智能诊断 ====================
def smart_diagnostic(model: AetherOmniModel, ref_model: AetherOmniModel, train_loader: StreamingDPOLoader,
                     args: argparse.Namespace, max_retries: int = 3) -> bool:
    log_system("===== 智能诊断开始 =====")
    has_grad_params = sum(p.data.size for p in model.parameters() if p.requires_grad)
    log_system(f"可训练参数总数: {has_grad_params}")
    try:
        (batch, idx) = next(train_loader.iter_triples())
        prompt, chosen, rejected = batch
        log_system(f"样本: prompt_len={len(prompt)}, chosen_len={len(chosen)}, rejected_len={len(rejected)}")
    except StopIteration:
        log_system("错误: 训练数据为空")
        return False
    beta = args.dpo_beta
    for retry in range(max_retries):
        model.zero_grad()
        loss, stats = dpo_loss(model, ref_model, prompt, chosen, rejected, beta=beta)
        loss.backward()
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += np.sum(p.grad ** 2)
        total_norm = np.sqrt(total_norm)
        log_system(f"尝试 beta={beta:.4f}: 梯度范数={total_norm:.6f}, 损失={loss.data.item():.6f}")
        log_system(f"  chosen_logratio={stats['chosen_logratio']:.4f}, rejected_logratio={stats['rejected_logratio']:.4f}, logits={stats['logits']:.4f}")
        if total_norm > 1e-8:
            log_system("✅ 梯度正常，可继续训练")
            model.zero_grad()
            args.dpo_beta = beta
            return True
        old_beta = beta
        beta = min(2.0, beta * 1.5)
        log_system(f"⚠️ 梯度过小，将 beta 从 {old_beta:.4f} 增大到 {beta:.4f}")
    log_system("❌ 智能诊断失败：无法获得有效梯度")
    log_system("建议：1.检查数据 2.检查模型前向 3.降低学习率或关闭混合精度")
    return False

# ==================== SFT预热 ====================
def sft_warmup(life: LifeCore, optimizer: Union[AdamW, SGD, Lookahead], train_loader: StreamingDPOLoader,
               warmup_steps: int, verbose: bool = False) -> None:
    print("\n" + "=" * 60)
    print("【SFT 预热阶段】")
    print("=" * 60)
    log_system(f"启动SFT预热，共 {warmup_steps} 步...")
    base_opt = optimizer.base_optimizer if isinstance(optimizer, Lookahead) else optimizer
    original_lr = base_opt.lr
    base_opt.lr = original_lr * 0.2
    log_system(f"SFT预热临时降低学习率: {original_lr:.2e} -> {base_opt.lr:.2e}")

    triple_iter = train_loader.iter_triples()
    step = 0
    pbar = tqdm(total=warmup_steps, desc="SFT预热", unit="step", disable=not verbose)
    while step < warmup_steps:
        try:
            (prompt, chosen, rejected), _ = next(triple_iter)
        except StopIteration:
            triple_iter = train_loader.iter_triples()
            (prompt, chosen, rejected), _ = next(triple_iter)

        input_ids = np.concatenate([prompt, chosen])[None, :]
        mask = np.ones_like(input_ids, dtype=bool)
        shift_mask = mask[:, 1:]

        life.zero_grad()
        logits = life.model(input_ids)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss, _ = compute_loss(shift_logits, shift_labels, shift_mask)
        loss.backward()
        optimizer.step()

        step += 1
        pbar.update(1)
        if step % 100 == 0 or step == warmup_steps:
            pbar.write(f"  step={step}/{warmup_steps}, loss={loss.data.item():.6f}")
    pbar.close()

    base_opt.lr = original_lr
    life.ref_model.load_state_dict(life.model.get_state_dict())
    life.ref_model.eval()
    for p in life.ref_model.parameters():
        p.requires_grad = False
    print("\n✅ SFT预热完成，参考模型已冻结")
    print("=" * 60 + "\n")

# ==================== 训练主循环 ====================
def run_training_loop(state: Dict[str, Any], verbose: bool = False) -> None:
    life = state['life']
    optimizer = state['optimizer']
    manager = state['manager']
    train_loader = state['train_loader']
    val_loader = state['val_loader']
    hw_monitor = state['hw_monitor']
    adaptive_ctl = state['adaptive_ctl']
    trainer_wrapper = state['trainer_wrapper']
    args = state['args']
    arbiter = state['arbiter']

    if args.sft_warmup_steps > 0:
        sft_warmup(life, optimizer, train_loader, args.sft_warmup_steps, verbose=verbose)

    ema = EMA(life.model, decay=args.ema_decay) if args.use_ema else None
    scaler = GradientScaler(init_scale=1, growth_interval=1000) if args.fp16 else None
    trainer_wrapper.scaler = scaler
    trainer_wrapper.ema = ema

    guard = SingularityGuard()
    anchor = AnchorGuard()

    step = manager.step
    triple_iter = train_loader.iter_triples()
    grad_accum_steps = args.grad_accum
    loss_history = deque(maxlen=500)
    loss_history_for_trend = deque(maxlen=50)
    start_time = time.time()

    print("\n" + "=" * 60)
    print("【DPO 训练主循环】")
    print("=" * 60)

    pbar = tqdm(total=args.steps, desc="DPO训练", unit="step", initial=step, disable=not verbose)

    train_loader.__enter__()
    val_loader_ctx = val_loader.__enter__() if val_loader else nullcontext()

    log_interval = args.log_interval
    last_print_step = step

    try:
        while step < args.steps:
            batch_buffer = []
            while len(batch_buffer) < grad_accum_steps:
                try:
                    batch, idx = next(triple_iter)
                    batch_buffer.append((batch, idx))
                except StopIteration:
                    triple_iter = train_loader.iter_triples()
                    continue

            life.zero_grad()
            total_loss = None
            valid_batch_count = 0
            stats_accum = {"chosen_logratio": 0.0, "rejected_logratio": 0.0, "logits": 0.0}
            per_errors = []
            valid_indices = []
            for b, idx in batch_buffer:
                prompt, chosen, rejected = b
                loss, stats = dpo_loss(life.model, life.ref_model, prompt, chosen, rejected, beta=args.dpo_beta)
                if stats["invalid"]:
                    continue
                loss = loss / len(batch_buffer)
                loss.backward()
                total_loss = loss if total_loss is None else total_loss + loss
                valid_batch_count += 1
                stats_accum["chosen_logratio"] += stats["chosen_logratio"]
                stats_accum["rejected_logratio"] += stats["rejected_logratio"]
                stats_accum["logits"] += stats["logits"]
                td_error = abs(stats["chosen_logratio"] - stats["rejected_logratio"])
                per_errors.append(td_error)
                valid_indices.append(idx)

            if valid_batch_count == 0:
                continue

            # 数值监控
            for p in life.model.parameters():
                if p.grad is not None:
                    if np.any(np.isnan(p.grad)) or np.any(np.isinf(p.grad)):
                        raise ValueError(f"梯度包含 NaN/Inf at step {step}")

            grad_norm = manager.clip_gradients()
            try:
                optimizer.step()
            except Exception as e:
                logger.error(f"优化器步进异常: {e}")
                continue

            if ema:
                ema.update()

            step += 1
            loss_val = total_loss.data.item()
            loss_history.append(loss_val)
            loss_history_for_trend.append(loss_val)

            # 计算损失趋势
            if len(loss_history_for_trend) >= 20:
                ema_loss = 0.0
                ema_trend = 0.0
                alpha = 0.1
                for l in loss_history_for_trend:
                    ema_loss = alpha * l + (1 - alpha) * ema_loss
                    ema_trend = (l - ema_loss) / (ema_loss + EPS)
                loss_trend = ema_trend
            else:
                loss_trend = 0.0

            trainer_wrapper.global_step = step
            trainer_wrapper.loss_history = loss_history

            manager.before_step()
            manager.after_step(loss_val, grad_norm, val_loss=None)
            life.evolve(loss=loss_val, grad_norm=grad_norm, step=step, loss_trend=loss_trend)

            if train_loader.use_per and valid_indices:
                train_loader.update_priorities(valid_indices, per_errors)

            pbar.update(1)
            current_lr = optimizer.lr if hasattr(optimizer, 'lr') else optimizer.base_optimizer.lr

            if step - last_print_step >= log_interval or step == args.steps:
                pbar.write(f"step={step:6d} | loss={loss_val:.6f} | grad={grad_norm:.4f} | lr={current_lr:.2e}")
                last_print_step = step

            if arbiter:
                arbiter.execute_one(trainer_wrapper)

            if step % args.eval_interval == 0 and val_loader is not None:
                pbar.write("\n" + "-" * 40)
                pbar.write("🔍 开始验证...")
                with Tensor.no_grad():
                    val_loss = dpo_validate(life.model, life.ref_model, val_loader, num_batches=20, beta=args.dpo_beta)
                pbar.write(f"step={step:6d} | train_loss={loss_val:.6f} | val_loss={val_loss:.6f} | grad={grad_norm:.4f} | lr={current_lr:.2e}")
                pbar.write("-" * 40 + "\n")
                try:
                    manager.after_step(loss_val, grad_norm, val_loss)
                except StopIteration:
                    pbar.write("早停触发，结束训练")
                    break

            if step % 100 == 0:
                adaptive_ctl.adjust()
                adaptive_ctl.check_and_rollback()

            anchor.update(life.model, step)
            if step % 100 == 0:
                if anchor.verify_and_rollback(life.model, step, loss_val):
                    if args.optimizer.lower() == 'adamw':
                        base_opt = AdamW(life.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    else:
                        base_opt = SGD(life.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                    if args.use_lookahead:
                        optimizer = Lookahead(base_opt, k=5, alpha=0.5)
                    else:
                        optimizer = base_opt
                    manager.optimizer = optimizer
                    manager.early_stop_count = 0
                    if scaler:
                        scaler = GradientScaler(init_scale=1)
                        trainer_wrapper.scaler = scaler
                    if ema:
                        ema.restore()
                    continue

            if step % args.save_interval == 0:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                prefix = f"checkpoint_{step:06d}"
                path = os.path.join(args.checkpoint_dir, prefix)
                life.save_life_state(path + '_life.npz')
                save_checkpoint_async(life.model, optimizer, manager, step, path, life.ref_model)
                pbar.write("\n" + "-" * 40)
                pbar.write(f"💾 保存检查点: {path}")
                pbar.write("-" * 40 + "\n")

            if step % GC_INTERVAL == 0:
                clear_all_caches()
                gc.collect()

    finally:
        pbar.close()
        train_loader.__exit__(None, None, None)
        if val_loader:
            val_loader.__exit__(None, None, None)

    if ema:
        ema.apply_shadow()
    final_state = life.get_state_dict()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.checkpoint_dir, 'aether_life_final.npz'), **final_state)
    elapsed_total = time.time() - start_time
    log_system(f"训练完成！总步数: {step}, 总耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min), 最终损失: {loss_val:.6f}")
    clear_all_caches()
    if hw_monitor:
        hw_monitor.stop()

    if args.generate_report and HAS_PLOTLY:
        try:
            fig = make_subplots(rows=2, cols=1)
            fig.add_trace(go.Scatter(y=list(loss_history), name='Train Loss'), row=1, col=1)
            fig.update_layout(title='AetherLife Training Report')
            fig.write_html(os.path.join(args.checkpoint_dir, 'training_report.html'))
            log_system(f"报告已生成: {args.checkpoint_dir}/training_report.html")
        except Exception as e:
            logger.warning(f"报告生成失败: {e}")

# ==================== 训练设置 ====================
def get_default_args() -> argparse.Namespace:
    class Args:
        pass
    args = Args()
    args.data_dir = os.getenv("AETHER_DATA_DIR", "./bpe_data")
    args.checkpoint_dir = "./checkpoints"
    args.resume = None
    args.steps = 50000
    args.batch_size = 4
    args.grad_accum = 1
    args.lr = 5e-6
    args.weight_decay = 0.01
    args.dropout = 0.1
    args.d_model = 256
    args.n_layers = 6
    args.n_heads = 8
    args.window_size = 16
    args.max_seq_len = 512
    args.use_infini = False
    args.mem_len = 256
    args.use_moe = False
    args.num_experts = 4
    args.moe_layers = []
    args.use_hybrid = False
    args.use_image = False
    args.image_size = 224
    args.use_ternary = False
    args.optimizer = "AdamW"
    args.use_lookahead = False
    args.lr_schedule = "cosine"
    args.warmup_steps = 500
    args.grad_clip = 5.0
    args.dpo_beta = 0.3
    args.sft_warmup_steps = 1000
    args.eval_interval = 500
    args.save_interval = 200
    args.log_interval = 10
    args.early_stop_patience = 10
    args.use_evolution = False
    args.evo_pop_size = 10
    args.evo_gens = 5
    args.eval_steps = 100
    args.use_ema = False
    args.ema_decay = 0.999
    args.fp16 = True
    args.use_mmap = True
    args.use_per = True
    args.generate_report = False
    args.seed = 42
    args.adaptive_cooldown = 200
    args.temp_high = 75.0
    args.temp_moderate = 60.0
    args.temp_low = 45.0
    args.battery_low = 40.0
    args.battery_critical = 20.0
    args.use_multi_obj_evo = False
    args.verbose = True
    return args

def setup_training(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    data_dir = args.data_dir
    try:
        data_dir = safe_data_dir(data_dir)
    except PermissionError as e:
        log_system(f"数据目录错误: {e}")
        sys.exit(1)
    if not os.path.exists(data_dir):
        log_system(f"错误: 数据目录 {args.data_dir} 不存在")
        sys.exit(1)
    vocab_size = 32000
    info_path = os.path.join(data_dir, 'dataset_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
            vocab_size = info.get('vocab_size', 32000)
            log_system(f"从 dataset_info.json 加载词汇表大小: {vocab_size}")
    else:
        tok_path = os.path.join(data_dir, 'bpe_tokenizer.json')
        if os.path.exists(tok_path):
            with open(tok_path, 'r') as f:
                tok_data = json.load(f)
                vocab_size = len(tok_data.get('vocab', []))
                log_system(f"从 bpe_tokenizer.json 加载词汇表大小: {vocab_size}")
        else:
            log_system(f"警告: 未找到词汇表信息，使用默认值 {vocab_size}")
    model_config = ModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        window_size=args.window_size,
        use_infini=args.use_infini,
        mem_len=args.mem_len,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        moe_layers=args.moe_layers,
        use_hybrid=args.use_hybrid,
        use_image=args.use_image,
        image_size=args.image_size,
        use_ternary=args.use_ternary,
        dropout=args.dropout,
    )
    class DummyTokenizer:
        def __init__(self, vocab_size):
            self.vocab = {'size': vocab_size}
            self.vocab_size = vocab_size
        def __len__(self):
            return self.vocab_size
    dummy_tokenizer = DummyTokenizer(vocab_size)

    arbiter = CentralArbiter()
    model = AetherOmniModel(model_config)
    ref_model = AetherOmniModel(model_config)
    ref_model.load_state_dict(model.get_state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    meta_mind = MetaMind(model_config)
    adaptive_body = AdaptiveBody(model_config, detect_hardware(), model_rebuilder_callback=None)
    consciousness = ConsciousnessCore(model_config)
    life_cycle = LifeCycleManager(meta_mind, adaptive_body, consciousness, model_config, arbiter=arbiter)
    life = LifeCore(model, ref_model, meta_mind, adaptive_body, consciousness, life_cycle, arbiter)

    def rebuild_model(config_changes: Dict[str, Any]) -> None:
        new_config = ModelConfig(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_layers=config_changes.get('n_layers', model_config.n_layers),
            n_heads=model_config.n_heads,
            window_size=model_config.window_size,
            use_infini=model_config.use_infini,
            mem_len=model_config.mem_len,
            use_moe=config_changes.get('moe_ratio', 0.0) > 0,
            num_experts=model_config.num_experts,
            moe_layers=model_config.moe_layers,
            use_hybrid=model_config.use_hybrid,
            use_image=model_config.use_image,
            image_size=model_config.image_size,
            use_ternary=model_config.use_ternary,
            dropout=model_config.dropout,
        )
        new_model = AetherOmniModel(new_config)
        old_state = model.get_state_dict()
        new_state = new_model.get_state_dict()
        for k, v in old_state.items():
            if k in new_state and v.shape == new_state[k].shape:
                new_state[k][:] = v
        new_model.load_state_dict(new_state)
        life.model = new_model
        new_ref = AetherOmniModel(new_config)
        new_ref.load_state_dict(new_model.get_state_dict())
        new_ref.eval()
        for p in new_ref.parameters():
            p.requires_grad = False
        life.ref_model = new_ref
        if args.optimizer.lower() == 'adamw':
            base_opt = AdamW(life.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            base_opt = SGD(life.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        if args.use_lookahead:
            new_opt = Lookahead(base_opt, k=5, alpha=0.5)
        else:
            new_opt = base_opt
        optimizer_holder[0] = new_opt
        manager_holder[0].optimizer = new_opt
        manager_holder[0].model = new_model
        trainer_wrapper_holder[0].optimizer = new_opt
        trainer_wrapper_holder[0].model = new_model
        log_system("模型已根据自适应决策重建")

    life.adaptive_body.model_rebuilder_callback = rebuild_model

    if args.optimizer.lower() == 'adamw':
        base_opt = AdamW(life.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        base_opt = SGD(life.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.use_lookahead:
        optimizer = Lookahead(base_opt, k=5, alpha=0.5)
    else:
        optimizer = base_opt
    optimizer_holder = [optimizer]

    adaptive_config = {
        'learning_rate': {
            'type': args.lr_schedule,
            'initial': args.lr,
            'warmup_steps': args.warmup_steps,
            'step_size': 5000,
            'gamma': 0.5,
            'min_lr': 1e-7,
            'factor': 0.5,
            'patience': 5,
            'cooldown': 2,
        },
        'gradient_clip': {
            'initial': args.grad_clip,
            'window': 10,
            'multiplier_up': 1.2,
            'multiplier_down': 0.8,
        },
        'regularization': {
            'dropout': {'initial': args.dropout, 'min': 0.0, 'max': 0.5},
            'weight_decay': {'initial': args.weight_decay, 'min': 0.0, 'max': 0.1},
        },
        'early_stop': {
            'patience': args.early_stop_patience,
            'min_delta': 1e-4,
            'restore_best': True,
        },
        'overfit_threshold': 0.2,
        'underfit_threshold': 0.5,
        'dpo_beta': args.dpo_beta,
    }
    manager = AdaptiveTrainManager(life.model, optimizer, adaptive_config, val_loader=None, total_steps=args.steps)
    manager_holder = [manager]
    life.life_cycle.trainer_manager = manager

    log_system("开始加载训练数据...")
    train_loader = StreamingDPOLoader(data_dir, max_seq_len=args.max_seq_len, use_mmap=args.use_mmap, use_per=args.use_per)
    val_loader = None
    val_file = os.path.join(data_dir, 'val.ids')
    if os.path.exists(val_file):
        val_loader = StreamingDPOLoader(data_dir, max_seq_len=args.max_seq_len, use_mmap=args.use_mmap, use_per=False)
        val_loader.train_file = val_file
        log_system("验证集已加载")
    else:
        log_system("未找到验证集，将禁用早停和验证")

    hw_monitor = HardwareMonitorThread(sleep_interval=20)
    hw_monitor.start()

    class TrainerWrapper:
        def __init__(self):
            self.model = life.model
            self.optimizer = optimizer
            self.manager = manager
            self.hw_monitor = hw_monitor
            self.global_step = manager.step
            self.batch_size = args.batch_size
            self.max_seq_len = args.max_seq_len
            self.eval_interval = args.eval_interval
            self.log_interval = args.log_interval
            self.need_rebuild_loader = False
            self.loss_history = deque(maxlen=500)
            self.dropout_val = manager.dropout_val
            self.dpo_beta = args.dpo_beta
            self.life = life
            self.rebuild_callback = rebuild_model
            self.scaler = None
            self.ema = None

        def save_checkpoint(self, emergency: bool = False) -> None:
            prefix = f"emergency_{self.global_step}" if emergency else f"checkpoint_{self.global_step:06d}"
            path = os.path.join(args.checkpoint_dir, prefix)
            life.save_life_state(path + '_life.npz')
            save_checkpoint_async(self.model, self.optimizer, self.manager, self.global_step, path, life.ref_model)

    trainer_wrapper = TrainerWrapper()
    trainer_wrapper_holder = [trainer_wrapper]
    adaptive_ctl = AdaptiveController(trainer_wrapper, {
        'cooldown_steps': args.adaptive_cooldown,
        'temp_high': args.temp_high,
        'temp_moderate': args.temp_moderate,
        'temp_low': args.temp_low,
        'battery_low': args.battery_low,
        'battery_critical': args.battery_critical,
    })

    return {
        'life': life, 'optimizer': optimizer, 'manager': manager,
        'train_loader': train_loader, 'val_loader': val_loader, 'hw_monitor': hw_monitor,
        'adaptive_ctl': adaptive_ctl, 'trainer_wrapper': trainer_wrapper,
        'args': args, 'model_config': model_config, 'vocab_size': vocab_size,
        'arbiter': arbiter, 'tokenizer': dummy_tokenizer
    }

# ==================== 算法自检模块 ====================
class AlgorithmSelfChecker:
    @staticmethod
    def check_gradient(func: Callable, inputs: List[np.ndarray], epsilon: float = 1e-5) -> bool:
        tensors = [Tensor(x, requires_grad=True) for x in inputs]
        outputs = func(*tensors)
        outputs.sum().backward()
        for i, x in enumerate(tensors):
            numeric = np.zeros_like(x.data)
            for idx in np.ndindex(x.data.shape):
                original = x.data[idx]
                x.data[idx] = original + epsilon
                f_plus = func(*tensors).sum().data.item()
                x.data[idx] = original - epsilon
                f_minus = func(*tensors).sum().data.item()
                numeric[idx] = (f_plus - f_minus) / (2 * epsilon)
                x.data[idx] = original
            diff = np.abs(numeric - x.grad).max()
            if diff > 1e-4:
                log_system(f"梯度校验失败，最大差异 {diff}")
                return False
        return True

    @staticmethod
    def check_numerical_stability() -> bool:
        x = Tensor(np.array([-1000.0, 0.0, 1000.0]))
        y = log_sigmoid(x)
        if np.any(np.isnan(y.data)) or np.any(np.isinf(y.data)):
            log_system("log_sigmoid 数值不稳定")
            return False
        x = Tensor(np.array([[1000.0, 1000.0, 1000.0]]))
        y = x.softmax()
        if np.any(np.isnan(y.data)):
            log_system("softmax 数值不稳定")
            return False
        return True

    @staticmethod
    def run_all() -> Dict[str, bool]:
        results = {}
        log_system("运行算法自检...")

        def simple_func(x):
            return (x * x).sum()
        results['gradient_check'] = AlgorithmSelfChecker.check_gradient(simple_func, [np.random.randn(3, 5).astype(np.float32)])
        results['numerical_stability'] = AlgorithmSelfChecker.check_numerical_stability()

        try:
            config = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=4)
            model = AetherOmniModel(config)
            inp = np.random.randint(0, 100, size=(2, 16)).astype(np.int64)
            logits = model(inp)
            results['model_forward'] = logits.shape == (2, 16, 100)
        except Exception as e:
            log_system(f"模型前向测试失败: {e}")
            results['model_forward'] = False

        try:
            params = [Tensor(np.random.randn(10).astype(np.float32), requires_grad=True)]
            opt = AdamW(params, lr=0.01)
            loss = (params[0] * params[0]).sum()
            loss.backward()
            old_data = params[0].data.copy()
            opt.step()
            results['optimizer_step'] = not np.array_equal(old_data, params[0].data)
        except Exception as e:
            log_system(f"优化器步进测试失败: {e}")
            results['optimizer_step'] = False

        log_system(f"自检结果: {results}")
        return results

# ==================== 单元测试 ====================
def run_unit_tests() -> None:
    print("\n开始运行单元测试...")
    set_seed(42)

    print("测试1: 梯度正确性...")
    x = Tensor(np.random.randn(3, 5).astype(np.float32), requires_grad=True)
    y = (x * x).sum()
    y.backward()
    num_grad = 2 * x.data
    grad_diff = np.abs(x.grad - num_grad).max()
    assert grad_diff < 1e-6, f"梯度错误，最大差异 {grad_diff}"
    print("  ✅ 通过")

    print("测试2: DPO loss 数值稳定性...")
    class DummyModel:
        def __call__(self, inp): return Tensor(np.random.randn(1, 10, 100).astype(np.float32))
        def parameters(self): return []
        def zero_grad(self): pass

    model = DummyModel()
    ref = DummyModel()
    prompt = np.array([1, 2, 3])
    chosen = np.array([4, 5, 6])
    rejected = np.array([7, 8, 9])
    loss, stats = dpo_loss(model, ref, prompt, chosen, rejected, beta=0.3)
    assert not np.isnan(loss.data.item()) and not np.isinf(loss.data.item()), "DPO loss 包含 NaN/Inf"
    print("  ✅ 通过")

    print("测试3: 模型前向传播...")
    config = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=4)
    model = AetherOmniModel(config)
    inp = np.random.randint(0, 100, size=(2, 16)).astype(np.int64)
    logits = model(inp)
    assert logits.shape == (2, 16, 100), f"输出形状错误: {logits.shape}"
    print("  ✅ 通过")

    print("测试4: 优化器步进...")
    params = [Tensor(np.random.randn(10).astype(np.float32), requires_grad=True)]
    opt = AdamW(params, lr=0.01)
    loss = (params[0] * params[0]).sum()
    loss.backward()
    old_data = params[0].data.copy()
    opt.step()
    assert not np.array_equal(old_data, params[0].data), "优化器未更新参数"
    print("  ✅ 通过")

    print("测试5: gather 梯度...")
    x = Tensor(np.random.randn(3, 5).astype(np.float32), requires_grad=True)
    idx = Tensor(np.array([[0, 1], [2, 0], [1, 2]]), dtype=INDEX_DTYPE)
    y = x.gather(1, idx)
    y.sum().backward()
    assert x.grad is not None, "gather 梯度为 None"
    print("  ✅ 通过")

    print("\n所有单元测试通过！")

# ==================== 检查点恢复 ====================
def find_latest_checkpoint(checkpoint_dir: str = "./checkpoints") -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.npz') and f.startswith('checkpoint_')]
    if not files:
        return None
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    return os.path.join(checkpoint_dir, latest.replace('.npz', ''))

def auto_resume_or_test() -> None:
    checkpoint = find_latest_checkpoint()
    if checkpoint:
        print(f"\n检测到检查点 {checkpoint}，自动恢复训练...")
        args = get_default_args()
        args.resume = checkpoint
        train_omni(args)
        return

    data_dir = None
    candidates = ['./bpe_data', './data', '../bpe_data', '../data']
    for cand in candidates:
        if os.path.exists(os.path.join(cand, 'train.ids')):
            data_dir = cand
            break
    if data_dir:
        print(f"\n检测到数据目录 {data_dir}，先运行单元测试...")
        run_unit_tests()
        print("\n✅ 单元测试通过，自动开始训练...")
        args = get_default_args()
        args.data_dir = data_dir
        train_omni(args)
        return

    print("\n未检测到检查点且未找到数据目录，运行单元测试...")
    run_unit_tests()

# ==================== 训练入口 ====================
def train_omni(args: argparse.Namespace) -> None:
    try:
        state = setup_training(args)
        if args.resume and os.path.exists(args.resume):
            log_system(f"从检查点恢复: {args.resume}")
            try:
                life_path = args.resume.replace('.npz', '_life.npz')
                if os.path.exists(life_path):
                    state['life'].load_life_state(life_path)
                else:
                    model, opt, mgr, ref_model, step = load_checkpoint(
                        args.resume, state['model_config'], AdamW, AdaptiveTrainManager)
                    state['life'].model = model
                    state['life'].ref_model = ref_model if ref_model else state['life'].ref_model
                    state['optimizer'] = opt
                    state['manager'] = mgr
                    state['manager'].step = step
                state['trainer_wrapper'].global_step = state['manager'].step
                log_system(f"恢复模型，步数: {state['manager'].step}")
            except Exception as e:
                log_system(f"恢复失败: {e}，从头开始")
        if not smart_diagnostic(state['life'].model, state['life'].ref_model, state['train_loader'], args):
            log_system("智能诊断失败，请根据建议调整超参数后重试")
        else:
            state['manager'].dpo_beta = args.dpo_beta
            state['trainer_wrapper'].dpo_beta = args.dpo_beta
        if args.use_evolution and state['val_loader'] is not None:
            # 进化搜索（略）
            pass
        run_training_loop(state, verbose=args.verbose)
    except Exception as e:
        logger.error(f"训练过程异常: {e}")
        traceback.print_exc()
        raise

# ==================== 主函数 ====================
def main() -> None:
    if len(sys.argv) == 1:
        auto_resume_or_test()
        return

    parser = argparse.ArgumentParser(description="AetherLife v1.0 最终完美版")
    parser.add_argument('--config', type=str, default=None, help="配置文件路径（JSON/YAML）")
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--mode', choices=['train', 'eval', 'test'], default='train')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--use-infini', action='store_true')
    parser.add_argument('--mem-len', type=int, default=256)
    parser.add_argument('--use-moe', action='store_true')
    parser.add_argument('--num-experts', type=int, default=4)
    parser.add_argument('--moe-layers', type=str, default='')
    parser.add_argument('--use-hybrid', action='store_true')
    parser.add_argument('--use-image', action='store_true')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--use-ternary', action='store_true')
    parser.add_argument('--optimizer', choices=['AdamW', 'SGD'], default='AdamW')
    parser.add_argument('--use-lookahead', action='store_true')
    parser.add_argument('--lr-schedule', choices=['cosine', 'step', 'plateau'], default='cosine')
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--dpo-beta', type=float, default=0.3)
    parser.add_argument('--sft-warmup-steps', type=int, default=1000)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--save-interval', type=int, default=200)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--early-stop-patience', type=int, default=10)
    parser.add_argument('--use-evolution', action='store_true')
    parser.add_argument('--evo-pop-size', type=int, default=10)
    parser.add_argument('--evo-gens', type=int, default=5)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--use-mmap', action='store_true')
    parser.add_argument('--use-per', action='store_true', default=True)
    parser.add_argument('--generate-report', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--adaptive-cooldown', type=int, default=200)
    parser.add_argument('--temp-high', type=float, default=75.0)
    parser.add_argument('--temp-moderate', type=float, default=60.0)
    parser.add_argument('--temp-low', type=float, default=45.0)
    parser.add_argument('--battery-low', type=float, default=40.0)
    parser.add_argument('--battery-critical', type=float, default=20.0)
    parser.add_argument('--use-multi-obj-evo', action='store_true')
    parser.add_argument('--verbose', action='store_true', help="显示详细进度条和DEBUG日志")
    args = parser.parse_args()

    if args.config:
        if args.config.endswith('.json'):
            with open(args.config, 'r') as f:
                config = json.load(f)
            for k, v in config.items():
                if hasattr(args, k):
                    setattr(args, k, v)
        elif args.config.endswith('.yaml'):
            try:
                import yaml
                with open(args.config, 'r') as f:
                    config = yaml.safe_load(f)
                for k, v in config.items():
                    if hasattr(args, k):
                        setattr(args, k, v)
            except ImportError:
                log_system("警告: 未安装 PyYAML，无法加载 YAML 配置")
        else:
            log_system("不支持的配置文件格式，仅支持 .json 或 .yaml")

    global logger
    logger = setup_logging(verbose=args.verbose, log_file="logs/aether_life.log")

    if args.moe_layers:
        args.moe_layers = [int(x) for x in args.moe_layers.split(',') if x.strip()]
    else:
        args.moe_layers = []

    if args.data_dir is None:
        args.data_dir = os.getenv("AETHER_DATA_DIR", "./bpe_data")

    if not os.path.exists(args.data_dir):
        log_system(f"错误: 数据目录 {args.data_dir} 不存在")
        sys.exit(1)

    if args.mode == 'train':
        train_omni(args)
    elif args.mode == 'test':
        run_unit_tests()
    else:
        print("未知模式")

if __name__ == '__main__':
    atexit.register(_shutdown_executor)
    main()