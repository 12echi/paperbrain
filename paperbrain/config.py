"""全局配置 (v5.0): 所有环境变量/路径/限额的唯一来源。

原则: 运行时读取 (不缓存), 以便界面/`modelconf` 改环境变量后立即生效;
任何模块需要开关都从这里取, 不再各自 os.environ.get, 避免配置漂移。
"""
import json
import os
import sys
from urllib.parse import urlparse
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent

_ENVJSON = {"mtime": 0.0, "data": {}}  # {mtime,data}: env.json 兜底缓存 (mtime 变化才重读)


def _file_env() -> dict:
    """~/.config/paperbrain/env.json 里 PAPERBRAIN_* 键的兜底 (Key 现读 0600 文件, 不进日志)。
    CLI/工具 不经过 modelconf.load() 也能拿到 embed 等配置; 环境变量始终优先。
    PAPERBRAIN_TEST=1 时禁用兜底, =0 强制放行; 未设时 unittest 进程自动禁用
    (测试不得读用户配置/触网; 覆盖 discover 与 -m unittest 两种导入方式)。"""
    t = os.environ.get("PAPERBRAIN_TEST")
    if t == "1" or (t != "0" and "unittest" in sys.modules):
        return {}
    p = Path(os.environ.get("PAPERBRAIN_CONF_FILE")
             or (Path.home() / ".config" / "paperbrain" / "env.json"))
    try:
        mt = p.stat().st_mtime
        if mt != _ENVJSON["mtime"]:
            d = json.loads(p.read_text(encoding="utf-8"))
            _ENVJSON["data"] = {str(k): str(v) for k, v in d.items()
                                if str(k).startswith("PAPERBRAIN_") and v not in (None, "")}
            # 若标记了 opencode-go 且未设置 Key，动态现读 auth.json 注入内存字典 (不落盘)
            if d.get("opencode_go") and "PAPERBRAIN_API_KEY" not in _ENVJSON["data"]:
                try:
                    from .modelconf import read_opencode_key
                    k = read_opencode_key()
                    if k:
                        _ENVJSON["data"]["PAPERBRAIN_API_KEY"] = k
                except Exception:
                    pass
            _ENVJSON["mtime"] = mt
    except Exception:
        # 文件被删除/读取失败: 必须清空缓存, 否则旧 Key/旧地址仍在用
        _ENVJSON["data"] = {}
        _ENVJSON["mtime"] = 0.0
    return _ENVJSON["data"]


def env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is not None:
        return v
    if name.startswith("PAPERBRAIN_"):
        return _file_env().get(name, default)
    return default


def flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None and name.startswith("PAPERBRAIN_"):
        v = _file_env().get(name)  # 布尔开关同样读 env.json 兜底 (CLI 不经 modelconf.load 也生效)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def invalidate_file_cache() -> None:
    """配置变更钩子: modelconf.save/clear 后调用, 丢弃 env.json 缓存。"""
    _ENVJSON["mtime"] = 0.0
    _ENVJSON["data"] = {}


# ---------------------------------------------------------------- 模型/提供方
def provider() -> str:
    """https | opencode-cli | off"""
    return env("PAPERBRAIN_PROVIDER", "https")


def model() -> str:
    return env("PAPERBRAIN_MODEL", "muse-spark-1.3-contributor")


def vl_model() -> str:
    return env("PAPERBRAIN_VL_MODEL", "deepseek-v4-flash-vision-exp")


def base_url() -> str:
    return env("PAPERBRAIN_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def api_key() -> str:
    return env("PAPERBRAIN_API_KEY", "")


def has_key() -> bool:
    return bool(api_key())


def cloud_allowed() -> bool:
    """任何远端文本/视觉模型调用都要求显式授权，默认关闭。"""
    return flag("PAPERBRAIN_CLOUD_ALLOWED", False)


def all_model() -> bool:
    """全模型模式: 分章/图谱/NLI 也交给模型。"""
    return flag("PAPERBRAIN_ALL_MODEL", False)


def vision_enabled() -> bool:
    return flag("PAPERBRAIN_VISION", False)


# ---------------------------------------------------------------- Embedding 源
def embed_base_url() -> str:
    return env("PAPERBRAIN_EMBED_BASE_URL", "").rstrip("/")


def embed_api_key() -> str:
    return env("PAPERBRAIN_EMBED_API_KEY", "")


def embed_model() -> str:
    return env("PAPERBRAIN_EMBED_MODEL", "qwen/text-embedding-qwen3-embedding-4b")


def embed_enabled() -> bool:
    return bool(embed_base_url())


def endpoint_is_local(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
        return host in ("localhost", "127.0.0.1", "::1")
    except Exception:
        return False


def embed_allowed() -> bool:
    base = embed_base_url()
    return bool(base) and (endpoint_is_local(base) or cloud_allowed())


# ---------------------------------------------------------------- 工具链
def opencode_bin() -> str:
    return env("OPENCODE_BIN", str(Path.home() / ".opencode" / "bin" / "opencode"))


def opencode_run_dir() -> str:
    return env("OPENCODE_RUN_DIR", "/tmp/pb_occli")


def cli_cleanup() -> bool:
    return flag("PAPERBRAIN_CLI_CLEANUP", True)


def rps() -> float:
    try:
        return max(0.05, min(20.0, float(env("PAPERBRAIN_RPS", "2"))))
    except ValueError:
        return 2.0


# ---------------------------------------------------------------- 路径
def out_root() -> Path:
    return Path(env("PAPERBRAIN_OUT", str(ROOT / "out" / "web")))


def memory_db() -> str:
    return env("PAPERBRAIN_MEMORY_DB", str(ROOT / "out" / "memory.sqlite"))


def graph_policy_file() -> Path:
    """Versioned alias/blacklist policy used by graph normalization."""
    return Path(env("PAPERBRAIN_GRAPH_POLICY", str(ROOT / "graph_policy.json")))


def vector_db() -> Path:
    override = env("PAPERBRAIN_VECTOR_DB")
    if override:
        return Path(override)
    memory = Path(memory_db())
    return memory.with_suffix(memory.suffix + ".vectors.duckdb")


def conf_file() -> Path:
    ov = env("PAPERBRAIN_CONF_FILE")
    return Path(ov) if ov else Path.home() / ".config" / "paperbrain" / "env.json"


# ---------------------------------------------------------------- 限额
def context_chars() -> int:
    try:
        return int(env("PAPERBRAIN_CONTEXT_CHARS", "9000"))
    except ValueError:
        return 9000


def n_questions() -> int:
    try:
        return int(env("PAPERBRAIN_QUESTIONS", "7"))
    except ValueError:
        return 7


def max_images() -> int:
    try:
        return max(0, min(5, int(env("PAPERBRAIN_MAX_IMAGES", "3"))))
    except ValueError:
        return 3


def job_workers() -> int:
    try:
        return max(1, min(4, int(env("PAPERBRAIN_JOB_WORKERS", "2"))))
    except ValueError:
        return 2


def job_queue_limit() -> int:
    try:
        return max(job_workers(), min(64, int(env("PAPERBRAIN_JOB_QUEUE", "8"))))
    except ValueError:
        return 8


def note_link_th() -> float:
    try:
        return float(env("PAPERBRAIN_NOTE_LINK_TH", "0.28"))
    except ValueError:
        return 0.28


def reflect_enabled() -> bool:
    """反思层开关 (Generative Agents reflection): 默认开, 需模型可用。"""
    return flag("PAPERBRAIN_REFLECT", True)


def web_verify_enabled() -> bool:
    """Allow claim-only queries to public scholarly indexes for adjudication."""
    return flag("PAPERBRAIN_WEB_VERIFY", False)


def paper_suffixes() -> List[str]:
    return [".pdf", ".txt", ".md"]
