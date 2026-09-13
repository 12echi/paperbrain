"""图谱归一最小闭环 (v5.0 生产版).

入库前强制: 归一 -> 别名表 -> embedding去重(可注入相似度, 离线可mock)
            -> 泛词黑名单拦截计数. Schema 外直接拒收.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import config

ALLOWED_ENTITIES = {
    "Algorithm/Model", "Dataset/Benchmark", "Evaluation Metric",
    "Theoretical Component", "Problem/Task", "Limitation/Artifact",
}
ALLOWED_RELATIONS = {"Improves_Upon", "Evaluated_On", "Contradicts", "Vulnerable_To", "Requires"}

GENERIC_BLACKLIST = {"model", "data", "performance", "accuracy", "algorithm",
                     "method", "result", "paper", "experiment", "dataset", "metric"}

# 最小别名表, 生产用 JSON 文件维护, 这里只留种子
ALIAS_SEED = {
    "flashattention 2": "FlashAttention-v2",
    "flashattention-2": "FlashAttention-v2",
    "imagenet1k": "ImageNet-1k",
    "imageNet-1k": "ImageNet-1k",
    "bleu4": "BLEU-4",
    "bleu-4": "BLEU-4",
}

POLICY_SCHEMA = "paperbrain-graph-policy-v1"
POLICY_REVIEW_DAYS = 7
_POLICY_CACHE: Dict[str, object] = {"path": "", "mtime_ns": -1, "data": None}


def _parse_reviewed_at(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except ValueError:
        return None


def load_graph_policy(path: Optional[str] = None, now: Optional[datetime] = None) -> Dict:
    """读取 JSON 策略并报告复审状态；损坏文件不得被伪装成已复审。

    运行仍可使用内置种子以保持可用，但 release gate 会因 missing/invalid/stale 阻塞。
    """
    policy_path = Path(path) if path else config.graph_policy_file()
    try:
        mtime_ns = policy_path.stat().st_mtime_ns
        cache_hit = (_POLICY_CACHE["path"] == str(policy_path) and
                     _POLICY_CACHE["mtime_ns"] == mtime_ns and
                     isinstance(_POLICY_CACHE["data"], dict))
        if cache_hit:
            raw = dict(_POLICY_CACHE["data"] or {})
        else:
            raw = json.loads(policy_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("policy root must be an object")
            _POLICY_CACHE.update(path=str(policy_path), mtime_ns=mtime_ns, data=dict(raw))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"aliases": {}, "blacklist": set(), "valid": False,
                "review_due": True, "reviewed_at": None,
                "path": str(policy_path), "error": str(exc)[:200]}

    aliases = raw.get("aliases")
    blacklist = raw.get("blacklist")
    reviewed = _parse_reviewed_at(raw.get("reviewed_at"))
    valid = (raw.get("schema") == POLICY_SCHEMA and isinstance(aliases, dict) and
             all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
                 for k, v in (aliases or {}).items()) and
             isinstance(blacklist, list) and
             all(isinstance(x, str) and x.strip() for x in (blacklist or [])))
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age_days = ((current.astimezone(timezone.utc) - reviewed).total_seconds() / 86400
                if reviewed is not None else None)
    review_due = not valid or age_days is None or age_days < 0 or age_days > POLICY_REVIEW_DAYS
    return {
        "aliases": {str(k).strip().lower(): str(v).strip()
                    for k, v in (aliases or {}).items()} if valid else {},
        "blacklist": {str(x).strip().lower() for x in (blacklist or [])} if valid else set(),
        "valid": valid, "review_due": review_due,
        "reviewed_at": raw.get("reviewed_at"), "age_days": age_days,
        "path": str(policy_path), "error": None if valid else "schema or field validation failed",
    }


def normalize_entity(name: str) -> str:
    t = re.sub(r"\s+", " ", name.strip())
    return t


def apply_alias(name: str, alias: Optional[Dict[str, str]] = None,
                policy: Optional[Dict] = None) -> str:
    table = dict(ALIAS_SEED)
    table.update((policy or load_graph_policy())["aliases"])
    if alias:
        table.update({str(k).strip().lower(): str(v).strip() for k, v in alias.items()})
    return table.get(name.strip().lower(), name.strip())


def is_generic(name: str, policy: Optional[Dict] = None) -> bool:
    policy_blacklist = (policy or load_graph_policy())["blacklist"]
    return name.strip().lower() in (GENERIC_BLACKLIST | policy_blacklist)


def dedup_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def lexical_sim(a: str, b: str) -> float:
    """离线词法相似度 (ASCII 词 + 中文 2-gram Jaccard):
    ImageNet-1k vs ImageNet 1k -> 高分; 闪速放疗 vs 闪速放疗技术 -> 高分。
    embedding 注入时仍用 cos>0.93；词法回退链用 >=0.85，见 build_memory。"""
    def toks(s: str):
        s = s or ""
        out = set(re.findall(r"[a-z0-9]+", s.lower()))
        for run in re.findall(r"[\u4e00-\u9fff]{2,}", s):
            out.update(run[i:i + 2] for i in range(len(run) - 1))
        return out
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def sanitize_entities(items: List[Dict], alias: Optional[Dict[str, str]] = None,
                      sim: Optional[Callable[[str, str], float]] = None,
                      merge_th: float = 0.93) -> Tuple[List[Dict], Dict]:
    """清洗实体列表. item: {name, type}. 返回 (kept, stats).
    - type 不在白名单 -> 拒收计数
    - 泛词 -> 拦截计数
    - 别名合并 + (可选) embedding 相似合并
    """
    stats = {"rejected_schema": 0, "rejected_generic": 0, "merged": 0, "kept": 0,
             "canonical_map": {}}
    policy = load_graph_policy()
    kept: List[Dict] = []
    seen: Dict[str, Dict] = {}
    for it in items:
        name = normalize_entity(str(it.get("name", "")))
        typ = str(it.get("type", "")).strip()
        if typ not in ALLOWED_ENTITIES:
            stats["rejected_schema"] += 1
            continue
        name = apply_alias(name, alias, policy)
        if is_generic(name, policy):
            stats["rejected_generic"] += 1
            continue
        k = dedup_key(name)
        if k in seen:
            stats["merged"] += 1
            stats["canonical_map"][name] = seen[k]["name"]
            continue
        merged = False
        if sim is not None:
            for ek in list(seen.keys()):
                try:
                    if sim(name, seen[ek]["name"]) >= merge_th:
                        stats["merged"] += 1
                        stats["canonical_map"][name] = seen[ek]["name"]
                        merged = True
                        break
                except Exception:
                    continue
        if merged:
            continue
        rec = {"name": name, "type": typ}
        seen[k] = rec
        kept.append(rec)
        stats["canonical_map"][name] = name
    stats["kept"] = len(kept)
    return kept, stats


def sanitize_relation(rel: str) -> bool:
    return rel.strip() in ALLOWED_RELATIONS


def sanitize_relations(items: List[Dict], entity_names,
                       alias: Optional[Dict[str, str]] = None,
                       canonical_map: Optional[Dict[str, str]] = None) -> Tuple[List[Dict], Dict]:
    """关系白名单与端点完整性门禁；Schema 外和幽灵端点均拒收入库。"""
    names = set(entity_names)
    names_by_key = {dedup_key(name): name for name in names}
    kept: List[Dict] = []
    seen = set()
    stats = {"rejected_relation_schema": 0, "rejected_relation_endpoint": 0,
             "merged_relations": 0, "kept_relations": 0}
    policy = load_graph_policy()
    for item in items:
        rel = str(item.get("rel", "")).strip()
        src = apply_alias(normalize_entity(str(item.get("from", ""))), alias, policy)
        dst = apply_alias(normalize_entity(str(item.get("to", ""))), alias, policy)
        src = (canonical_map or {}).get(src, src)
        dst = (canonical_map or {}).get(dst, dst)
        src = names_by_key.get(dedup_key(src), src)
        dst = names_by_key.get(dedup_key(dst), dst)
        if not sanitize_relation(rel):
            stats["rejected_relation_schema"] += 1
            continue
        if not src or not dst or src not in names or dst not in names:
            stats["rejected_relation_endpoint"] += 1
            continue
        key = (src, rel, dst)
        if key in seen:
            stats["merged_relations"] += 1
            continue
        seen.add(key)
        rec = {"from": src, "rel": rel, "to": dst}
        if item.get("ev"):
            rec["ev"] = str(item["ev"])[:160]
        kept.append(rec)
    stats["kept_relations"] = len(kept)
    return kept, stats
