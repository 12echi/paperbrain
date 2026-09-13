"""模型接入配置 (v5.0): Key/地址/模型名管理.

- 落盘 ~/.config/paperbrain/env.json, 权限 0600, 永不进 git (仓库外).
- 内存即时生效 (写 os.environ), llm.py 直接可用, 重启自动加载.
- 对外只给掩码状态, Key 永不回显、不进日志/台账/报告.
- 测试路径可用 PAPERBRAIN_CONF_FILE 覆盖.
"""
import json
import os
from pathlib import Path
from typing import Dict

KEYS = ("PAPERBRAIN_API_KEY", "PAPERBRAIN_BASE_URL", "PAPERBRAIN_MODEL", "PAPERBRAIN_VL_MODEL",
        "PAPERBRAIN_PROVIDER", "PAPERBRAIN_CLOUD_ALLOWED", "PAPERBRAIN_WEB_VERIFY",
        "PAPERBRAIN_EMBED_BASE_URL", "PAPERBRAIN_EMBED_API_KEY", "PAPERBRAIN_EMBED_MODEL")

# opencode-go (本机 opencode 二进制内置 provider): OpenAI 兼容协议
OPENCODE_AUTH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
OPENCODE_BASE = "https://opencode.ai/zen/go/v1"
OPENCODE_DEFAULT_MODEL = "muse-spark-1.3-contributor"
OPENCODE_MODELS = ["deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp",
    "deepseek-v4-pro", "glm-5.1", "glm-5.2", "glm-5.3", "glm-5.3-flash", "gpt-5.6-luna",
    "grok-4.6", "hy3", "hy4-preview", "kimi-k2.6", "kimi-k2.7-code", "kimi-k3",
    "longcat-2.0", "mimo-v2.5", "mimo-v2.5-pro", "minimax-m2.7", "minimax-m3",
    "muse-spark-1.2-contributor", "muse-spark-1.3-contributor", "qwen3.6-plus",
    "qwen3.7-max", "qwen3.7-plus", "qwen3.8-flash", "qwen3.8-max"]


def conf_path() -> Path:
    ov = os.environ.get("PAPERBRAIN_CONF_FILE")
    if ov:
        return Path(ov)
    return Path.home() / ".config" / "paperbrain" / "env.json"


def load() -> Dict[str, str]:
    """启动时调一次: 落盘值注入环境变量 (不覆盖已有环境变量).
    若标记了 opencode-go, Key 每次现读 auth.json (不复制, 自动跟随轮换)."""
    data = _read_file()
    for k in KEYS:
        if data.get(k) and not os.environ.get(k):
            os.environ[k] = str(data[k])
    if data.get("opencode_go") and not os.environ.get("PAPERBRAIN_API_KEY"):
        key = read_opencode_key()
        if key:
            os.environ["PAPERBRAIN_API_KEY"] = key
            os.environ.setdefault("PAPERBRAIN_BASE_URL", str(data.get("PAPERBRAIN_BASE_URL", data.get("base_url", OPENCODE_BASE))))
            os.environ.setdefault("PAPERBRAIN_MODEL", str(data.get("PAPERBRAIN_MODEL", data.get("model", OPENCODE_DEFAULT_MODEL))))
    if data.get("opencode_go"):
        os.environ.setdefault("PAPERBRAIN_PROVIDER", str(data.get("PAPERBRAIN_PROVIDER", data.get("provider", "opencode-cli"))))
    return status()


def _read_file() -> Dict[str, str]:
    p = conf_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_opencode_key() -> str:
    """现读 opencode auth.json 的 opencode-go key (绝不打印/记录)."""
    try:
        d = json.loads(OPENCODE_AUTH.read_text(encoding="utf-8"))
        return str(d.get("opencode-go", {}).get("key", ""))
    except Exception:
        return ""


def import_opencode(model: str = "") -> Dict[str, str]:
    """一键导入 opencode-go 配置: 存标记 (不存 Key) + 即时生效.
    调用链默认走本机 opencode CLI (免网关逆向); 直连用作备用。
    导入凭据不等于授权发送论文内容；云端授权必须由用户单独开启。"""
    key = read_opencode_key()
    if not key:
        return {"ok": False, "error": "读不到 opencode auth.json 的 opencode-go key"}
    model = (model or "").strip() or OPENCODE_DEFAULT_MODEL
    os.environ["PAPERBRAIN_API_KEY"] = key
    os.environ["PAPERBRAIN_BASE_URL"] = OPENCODE_BASE
    os.environ["PAPERBRAIN_MODEL"] = model
    os.environ.setdefault("PAPERBRAIN_VL_MODEL", model)
    os.environ["PAPERBRAIN_PROVIDER"] = "opencode-cli"
    p = conf_path()
    cur = _read_file()
    cur.update({
        "opencode_go": True,
        "PAPERBRAIN_BASE_URL": OPENCODE_BASE,
        "PAPERBRAIN_MODEL": model,
        "PAPERBRAIN_VL_MODEL": model,
        "PAPERBRAIN_PROVIDER": "opencode-cli",
        "base_url": OPENCODE_BASE,
        "model": model,
        "provider": "opencode-cli"
    })
    cur.pop("PAPERBRAIN_API_KEY", None)  # Key 永不落本文件, 现读 auth.json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass
    return {"ok": True, **status()}


def save(fields: Dict[str, str]) -> Dict[str, str]:
    """保存非空字段并即时生效. 返回掩码状态."""
    p = conf_path()
    cur: Dict[str, str] = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    for k in KEYS:
        short = k.replace("PAPERBRAIN_", "").lower()
        v = (fields.get(k) or fields.get(short) or "").strip()
        if v:
            cur[k] = v
            os.environ[k] = v
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass
    _reset_dependent_caches()
    return status()


def _reset_dependent_caches() -> None:
    """配置变更后: 清 config 的 env.json 缓存 + embeddings 熔断/查询缓存 (防旧 Key/旧地址继续用)。"""
    try:
        from . import config
        config.invalidate_file_cache()
    except Exception:
        pass
    try:
        from . import embeddings
        embeddings._reset_breaker()
        embeddings._Q_CACHE.clear()
    except Exception:
        pass
    try:
        from . import llm_ops
        llm_ops.reset_cache()
    except Exception:
        pass


def clear() -> Dict[str, str]:
    p = conf_path()
    if p.exists():
        p.unlink()
    for k in KEYS:
        os.environ.pop(k, None)
    _reset_dependent_caches()
    return status()


def _mask(v: str) -> str:
    v = v or ""
    if len(v) <= 8:
        return "****" if v else ""
    return v[:4] + "****" + v[-4:]


def status() -> Dict[str, str]:
    data = _read_file()
    prov = (os.environ.get("PAPERBRAIN_PROVIDER")
            or data.get("PAPERBRAIN_PROVIDER")
            or ("opencode-go" if data.get("opencode_go") else "custom"))
    st = {"has_key": bool(os.environ.get("PAPERBRAIN_API_KEY")),
          "cloud_allowed": ((os.environ.get("PAPERBRAIN_CLOUD_ALLOWED") or
                              data.get("PAPERBRAIN_CLOUD_ALLOWED", "0")).strip().lower()
                             in ("1", "true", "yes", "on")),
          "web_verify_allowed": ((os.environ.get("PAPERBRAIN_WEB_VERIFY") or
                                   data.get("PAPERBRAIN_WEB_VERIFY", "0")).strip().lower()
                                  in ("1", "true", "yes", "on")),
          "api_key_masked": _mask(os.environ.get("PAPERBRAIN_API_KEY", "")),
          "base_url": os.environ.get("PAPERBRAIN_BASE_URL", data.get("PAPERBRAIN_BASE_URL", "https://api.openai.com/v1")),
          "model": os.environ.get("PAPERBRAIN_MODEL", data.get("PAPERBRAIN_MODEL", "gpt-4o-mini")),
          "vl_model": os.environ.get("PAPERBRAIN_VL_MODEL", data.get("PAPERBRAIN_VL_MODEL", "gpt-4o-mini")),
          "provider": prov,
          "opencode_available": bool(read_opencode_key()),
          "opencode_models": OPENCODE_MODELS,
          "conf_file": str(conf_path())}
    return st


def test_connection(timeout: int = 20) -> Dict[str, str]:
    """最小连通性: 让模型回一个词. 失败返回 error (不含 Key)."""
    from .llm import chat
    try:
        out = chat([{"role": "user", "content": "只回一个字: 好"}], max_tokens=10, timeout=timeout, retries=1)
        return {"ok": True, "reply": out[:50]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
