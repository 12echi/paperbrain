"""Embedding 客户端与向量检索 (v5.0 L4): 复用外部 OpenAI 兼容 /embeddings 源。

- 源: config.embed_base_url()/embed_api_key()/embed_model() (如远端 LM Studio, SSH 隧道).
- urllib 直调 embedding；float32 BLOB 仅作可重建缓存，召回由 DuckDB VSS/HNSW 执行。
- 内容哈希缓存: 同一文本不重复 embedding; 模型/维度写入键, 换模型自动失效.
- 任何失败都不抛出到主流程: available()/embed() 返回空, 检索自动降级 BM25.
"""
import array
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

from . import config

_LOCK = threading.Lock()
_Q_CACHE: Dict[Tuple[str, str, str], List[float]] = {}  # (base_url, model, text_hash) -> vec
_DOWN_UNTIL = 0.0  # 熔断: 连续失败后冷却期内直接返回 None, 避免逐查询硬等 (隧道断开时的关键保护)
_DOWN_KEY: Tuple[str, str] = ("", "")  # 熔断只针对出问题的 (base_url, model), 换端点立即恢复


def _reset_breaker() -> None:
    """测试/隧道恢复后重置熔断。"""
    global _DOWN_UNTIL, _DOWN_KEY
    _DOWN_UNTIL = 0.0
    _DOWN_KEY = ("", "")


def breaker_open() -> bool:
    return time.time() < _DOWN_UNTIL


def available() -> bool:
    return config.embed_allowed()


def model() -> str:
    return config.embed_model()


def storage_model() -> str:
    """数据库向量命名空间；同名模型切换端点时不得复用旧向量。"""
    base = config.embed_base_url()
    if not base:  # 允许离线 mock/旧库测试；真实端点存在时才需要隔离命名空间。
        return model()
    endpoint = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]
    return f"{model()}@{endpoint}"


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:24]


def _post(payload: dict, timeout: int = 15, retries: int = 1) -> Optional[dict]:
    global _DOWN_UNTIL, _DOWN_KEY
    base = config.embed_base_url()
    if not base:
        return None
    key_id = (base, model())
    if breaker_open() and _DOWN_KEY == key_id:
        return None
    url = base + "/embeddings"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = config.embed_api_key()
    if key:
        headers["Authorization"] = "Bearer " + key
    network_fail = False
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code >= 500:
                network_fail = True  # 服务端故障: 可熔断
            else:
                return None  # 4xx (超长/模型名错/鉴权): 客户端问题, 不熔断、不拖累其他请求
        except Exception:
            network_fail = True  # 连接拒绝/超时/DNS: 熔断
        if i < retries:
            time.sleep(min(2 ** i, 2))
    if network_fail:
        _DOWN_UNTIL = time.time() + 120  # 熔断 2 分钟: 期间所有向量请求立即跳过, 检索降级 BM25
        _DOWN_KEY = key_id
    return None


def embed_texts(texts: List[str], batch: int = 32) -> List[Optional[List[float]]]:
    """批量 embedding; 失败的位置返回 None (调用方跳过, 不阻塞)。"""
    out: List[Optional[List[float]]] = [None] * len(texts)
    if not texts or not available():
        return out
    m = model()
    base = config.embed_base_url()
    todo = []
    for idx, t in enumerate(texts):
        h = content_hash(t)
        with _LOCK:
            cached = _Q_CACHE.get((base, m, h))
        if cached is not None:
            out[idx] = cached
        else:
            todo.append(idx)
    for s in range(0, len(todo), batch):
        chunk_idx = todo[s:s + batch]
        payload = {"model": m, "input": [texts[i] for i in chunk_idx]}
        resp = _post(payload)
        data = (resp or {}).get("data") or []
        for j, item in enumerate(data):
            try:
                response_index = int(item.get("index", j))
            except (TypeError, ValueError):
                response_index = j
            if response_index < 0 or response_index >= len(chunk_idx):
                continue
            vec = item.get("embedding")
            if isinstance(vec, list) and vec:
                i = chunk_idx[response_index]
                out[i] = vec
                with _LOCK:
                    if len(_Q_CACHE) > 2048:  # 有界, 防长期驻留内存增长
                        _Q_CACHE.clear()
                    _Q_CACHE[(base, m, content_hash(texts[i]))] = vec
    return out


def embed_one(text: str) -> Optional[List[float]]:
    return embed_texts([text])[0]


# ---------------------------------------------------------------- BLOB 编解码
def to_blob(vec: List[float]) -> bytes:
    a = array.array("f", vec)
    return a.tobytes()


def from_blob(b: bytes) -> List[float]:
    a = array.array("f")
    a.frombytes(b)
    return a.tolist()


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def health() -> Dict:
    """轻量健康检查: 拿模型列表; 失败返回 ok=False。"""
    base = config.embed_base_url()
    if not base:
        return {"ok": False, "reason": "未配置 embed_base_url"}
    if not config.embed_allowed():
        return {"ok": False, "reason": "远端 embedding 未获云端发送授权"}
    try:
        url = base.rsplit("/v1", 1)[0] + "/v1/models"
        req = urllib.request.Request(url)
        key = config.embed_api_key()
        if key:
            req.add_header("Authorization", "Bearer " + key)
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read().decode("utf-8"))
        ids = [m.get("id") for m in d.get("data", [])]
        return {"ok": True, "model": model(), "available_models": ids[:20]}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:120]}
