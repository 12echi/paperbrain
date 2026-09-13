"""全模型模式语义算子 (v5.0 最小闭环, 需 PAPERBRAIN_ALL_MODEL=1).

原则:
- 全部为高层语义算子 (结构解析/图谱抽取/NLI蕴含), 基础分句/公式质检仍用规则 (成本分层).
- 调用失败时返回 None / 走词法打分, 由上游安全回退规则式, 保证系统永不崩溃.
- 输出严格为纯数据结构 (JSON/float), 不得引入 markdown 杂音.
"""
import json
import hashlib
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config
from .graph import ALLOWED_ENTITIES, ALLOWED_RELATIONS


def enabled() -> bool:
    """全模型模式总开关: 需显式配置 PAPERBRAIN_ALL_MODEL=1."""
    return config.all_model()


def _extract_json(text: str) -> Optional[Any]:
    """从 LLM 返回文本中稳健提取 JSON (支持 ```json 块及裸 JSON 对象/数组)."""
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


def _chat(prompt: str, max_tokens: int = 1500) -> str:
    from .llm import chat
    return chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)


# ---------------------------------------------------------------- 篇章结构解析
def segment(text: str, paper_id: str) -> Optional[List[Dict]]:
    """用模型定位论文大章节起止。失败返回 None, 上游回退规则分章。"""
    heading_candidates = []
    for line in text.splitlines():
        s = line.strip()
        if 2 <= len(s) <= 160 and (re.match(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+", s) or
                                   re.match(r"^(?:abstract|introduction|background|methods?|methodology|"
                                            r"experiments?|results?|discussion|conclusions?|摘要|引言|方法|"
                                            r"实验|结果|讨论|结论)\b", s, re.I)):
            heading_candidates.append(s)
    excerpt = (text[:4000] + "\n\n=== 全文标题候选 ===\n" +
               "\n".join(heading_candidates[:240]) + "\n\n=== 文末节选 ===\n" + text[-2500:])
    prompt = ("你是学术文献结构解析器。从以下论文中识别所有主章节, 只输出严格 JSON 数组, 结构:\n"
              '[{"name":"abstract|intro|method|experiments|conclusion|full","sec":"数字编号(如0,1,2)",'
              '"title":"章节标题原文"}]\n'
              "不要解释, 不要输出除 JSON 以外的内容。\n\n"
              f"论文首尾与全文标题候选:\n{excerpt[:12000]}")
    try:
        out = _chat(prompt, max_tokens=600)
        # 提取 ```json ... ``` 或裸数组
        m = re.search(r"\[\s*\{.*\}\s*\]", out, re.S)
        if not m:
            return None
        raw = json.loads(m.group(0))
        if not isinstance(raw, list) or len(raw) < 2:
            return None
        # 用标题原文在全文中搜寻位置, 切分文本
        secs: List[Dict] = []
        positions: List[Tuple[int, Dict]] = []
        for item in raw:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            idx = text.find(title)
            if idx >= 0:
                positions.append((idx, item))
        if len(positions) < 2:
            return None
        positions.sort(key=lambda x: x[0])
        for i, (idx, item) in enumerate(positions):
            idx = 0 if i == 0 else idx  # 标题前摘要/元数据归入首节，禁止静默丢弃
            end_idx = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            body = text[idx:end_idx].strip()
            sec_no = str(item.get("sec", str(i))).strip()
            from .sections import recursive_semantic_chunk
            chunks = recursive_semantic_chunk(body, paper_id, sec_no)
            name = str(item.get("name", "full")).lower()
            aliases = {"methods": "method", "results": "experiments", "discussion": "experiments",
                       "theory": "method", "other": "full"}
            name = aliases.get(name, name)
            if name not in {"abstract", "intro", "method", "experiments", "conclusion", "full"}:
                name = "full"
            secs.append({
                "name": name,
                "sec": sec_no,
                "text": body,
                "confidence": 0.85,
                "downgrade_pass1_only": False,
                "chunks": chunks
            })
        return secs if len(secs) >= 2 else None
    except Exception:
        return None


# ---------------------------------------------------------------- 知识图谱抽取
_SCHEMA_ENTS = set(ALLOWED_ENTITIES)
_SCHEMA_RELS = set(ALLOWED_RELATIONS)
_GRAPH_LOCAL = threading.local()


def _set_graph_status(status: Dict) -> None:
    _GRAPH_LOCAL.status = dict(status)


def last_graph_status() -> Dict:
    return dict(getattr(_GRAPH_LOCAL, "status", {}) or {})


def _validate_graph_payload(payload: Any) -> Tuple[List[Dict], List[Dict]]:
    if not isinstance(payload, dict):
        raise ValueError("响应不是 JSON 对象")
    raw_entities = payload.get("entities")
    raw_relations = payload.get("relations")
    if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
        raise ValueError("entities/relations 必须是数组")
    if not raw_entities:
        raise ValueError("entities 为空，不能证明抽取成功")
    entities = []
    for i, item in enumerate(raw_entities):
        if not isinstance(item, dict):
            raise ValueError(f"entity[{i}] 不是对象")
        name, typ = str(item.get("name", "")).strip(), item.get("type")
        if not name or typ not in _SCHEMA_ENTS:
            raise ValueError(f"entity[{i}] 名称为空或类型越界")
        entities.append({"name": name, "type": typ})
    names = {item["name"] for item in entities}
    relations = []
    for i, item in enumerate(raw_relations):
        if not isinstance(item, dict):
            raise ValueError(f"relation[{i}] 不是对象")
        src = str(item.get("from", "")).strip()
        dst = str(item.get("to", "")).strip()
        rel = item.get("rel")
        if rel not in _SCHEMA_RELS:
            raise ValueError(f"relation[{i}] 类型越界")
        if not src or not dst or src not in names or dst not in names:
            raise ValueError(f"relation[{i}] 端点不在实体清单")
        relations.append({"from": src, "rel": rel, "to": dst})
    return entities, relations


def extract_graph(text: str) -> Tuple[List[Dict], List[Dict]]:
    """原子校验图谱；失败重抽一次，再失败进入人工队列并由上游规则回退。"""
    schema_hint = (f"实体类型必须且只能属于: {sorted(_SCHEMA_ENTS)}\n"
                   f"关系类型必须且只能属于: {sorted(_SCHEMA_RELS)}")
    prompt = ("你是学术知识图谱构建专家。从以下论文中抽取核心实体与关系。\n"
              f"{schema_hint}\n"
              '只输出严格 JSON 对象: {"entities":[{"name":"...","type":"..."}],'
              '"relations":[{"from":"...","rel":"...","to":"..."}]}\n'
              "实体数量 5-15, 关系数量 5-20。不要解释。\n\n"
              f"论文文本:\n{text[:15000]}")
    reasons = []
    for attempt in range(2):
        try:
            corrective = ("\n上一次响应被 Schema 门禁整批拒收。必须修正所有类型和关系端点，"
                          "仍只输出严格 JSON。" if attempt else "")
            out = _chat(prompt + corrective, max_tokens=1200)
            payload = _extract_json(out)
            entities, relations = _validate_graph_payload(payload)
            _set_graph_status({"status": "ACCEPTED" if attempt == 0 else "RETRY_ACCEPTED",
                               "attempts": attempt + 1, "reasons": reasons})
            return entities, relations
        except Exception as exc:
            reasons.append(str(exc)[:160])
    _set_graph_status({"status": "MANUAL_REVIEW", "attempts": 2, "reasons": reasons})
    return [], []


# ---------------------------------------------------------------- NLI 打分
_NLI_PROMPT_VERSION = "nli-v2"
_NLI_CACHE: Dict[Tuple[str, str, str, str, str, str], float] = {}
_NLI_LOCK = threading.Lock()


def nli_score(claim: str, source: str) -> float:
    """claim 是否被 source 支持, 返回 [0,1]；模型失败时抛错并由验证器复核。

    不能在 NLI 标定阈值下静默替换成 TF-IDF：那会让评分器身份与标定记录失配。
    """
    claim, source = (claim or "").strip(), (source or "").strip()
    if not claim or not source:
        return 0.0
    # 缓存必须绑定完整内容、模型端点和提示版本；不得把同前缀长文本或切换后的模型混用。
    k = (_NLI_PROMPT_VERSION, config.provider(), config.base_url(), config.model(),
         hashlib.sha256(claim.encode("utf-8")).hexdigest(),
         hashlib.sha256(source.encode("utf-8")).hexdigest())
    with _NLI_LOCK:
        if k in _NLI_CACHE:
            return _NLI_CACHE[k]
    prompt = ("判定下面句子是否被参考文本支持。只回一个 0-100 的整数(100=完全支持, 0=无关/矛盾), 不要解释。\n"
              f"句子: {claim[:600]}\n参考文本: {source[:1200]}")
    try:
        out = _chat(prompt, max_tokens=8)
        m = re.search(r"\d{1,3}", out)
        if not m:
            raise ValueError("no number")
        score = max(0.0, min(100.0, float(m.group(0)))) / 100.0
        with _NLI_LOCK:
            _NLI_CACHE[k] = score
        return score
    except Exception as exc:
        # 瞬时失败不入缓存；下一次调用仍可恢复，但本次必须进入 NEEDS_REVIEW。
        raise RuntimeError(f"NLI 模型打分失败: {exc}") from exc


def make_nli_scorer() -> Callable[[str, str], float]:
    return nli_score


def reset_cache():
    """重置 NLI 缓存 (线程安全)."""
    with _NLI_LOCK:
        _NLI_CACHE.clear()
