"""记忆沉淀最小闭环: 规则式实体抽取 + graph.sanitize + SQLite 落库.

生产替换 extract_fn 即可接 LLM 约束解码; 存储用 sqlite3 标准库,
向量列预留 embedding_json (云端embedding, 离线可空).
"""
import re
import sqlite3
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .graph import sanitize_entities, sanitize_relation, sanitize_relations

# 白名单触发词 (最小种子, 生产用 LLM 抽取替换; 领域种子放 RADIOTHERAPY_SEED)
TRIGGERS = [
    ("FlashAttention-v2", "Algorithm/Model"),
    ("Transformer", "Algorithm/Model"),
    ("HumanEval", "Dataset/Benchmark"),
    ("ImageNet-1k", "Dataset/Benchmark"),
    ("BLEU-4", "Evaluation Metric"),
    ("Latency", "Evaluation Metric"),
    ("Cross-Entropy with Label Smoothing", "Theoretical Component"),
    ("Long-sequence Hallucination", "Problem/Task"),
    ("Quadratic Memory Footprint", "Limitation/Artifact"),
]

# 放疗/蒙卡领域种子 (用户主业方向; 类型经人工核定, 生产转 JSON 维护)
RADIOTHERAPY_SEED = [
    ("DBSCAN", "Algorithm/Model"),
    ("Monte Carlo", "Algorithm/Model"),
    ("Geant4", "Algorithm/Model"),
    ("DNA damage", "Problem/Task"),
    ("LET", "Evaluation Metric"),
    ("RBE", "Evaluation Metric"),
    ("Microdosimetry", "Theoretical Component"),
    ("Bragg peak", "Theoretical Component"),
    ("FLASH", "Problem/Task"),
    ("Beam Quality", "Evaluation Metric"),
]

TRIGGERS = TRIGGERS + RADIOTHERAPY_SEED

REL_HINTS = [("Improves_Upon", r"improv|outperform|优于|提升"),
             ("Evaluated_On", r"evaluat|benchmark|在.*上评估|实验"),
             ("Contradicts", r"contradict|conflict|矛盾|不一致"),
             ("Vulnerable_To", r"vulnerable|fail|limitation|局限|失效"),
             ("Requires", r"requir|depend|需要|依赖")]


def _name_in(name: str, low_text: str) -> bool:
    """实体名是否出现在文本中. ASCII 名用词边界 (防 LET 命中 complete / FLASH 命中
    flashing); 中文名用子串 (中文无词边界)."""
    n = name.lower()
    if re.search(r"[\u4e00-\u9fff]", n):
        return n in low_text
    return re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", low_text) is not None


def rule_extract(text: str) -> Tuple[List[Dict], List[Dict]]:
    low_all = text.lower()
    ents: List[Dict] = []
    for name, typ in TRIGGERS:
        if _name_in(name, low_all):
            ents.append({"name": name, "type": typ})
    # 关系只连同句共现实体 (防整篇乱连), hint 必须出现在该句
    from .text import split_sentences
    rels: List[Dict] = []
    for sent in split_sentences(text):
        low = sent.lower()
        present = [e["name"] for e in ents if _name_in(e["name"], low)]
        if len(present) < 2:
            continue
        for i in range(len(present)):
            for j in range(len(present)):
                if i == j:
                    continue
                for rel, pat in REL_HINTS:
                    if re.search(pat, sent, re.I):
                        rels.append({"from": present[i], "to": present[j], "rel": rel,
                                     "ev": sent[:160]})
                        break
            if len(rels) >= 12:
                break
        if len(rels) >= 12:
            break
    # 去重
    seen, uniq = set(), []
    for r in rels:
        k = (r["from"], r["rel"], r["to"])
        if k not in seen and sanitize_relation(r["rel"]):
            seen.add(k)
            uniq.append(r)
    return ents, uniq[:12]


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY, paper_id TEXT, sec TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS entities(name TEXT, type TEXT, paper_id TEXT, PRIMARY KEY(name, paper_id));
CREATE TABLE IF NOT EXISTS relations(src TEXT, rel TEXT, dst TEXT, paper_id TEXT);
"""


def save_sqlite(db_path: str, paper_id: str, sections: List[Dict],
                entities: List[Dict], relations: List[Dict]) -> Dict:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    n_c = n_e = 0
    valid_names = {e["name"] for e in entities}
    safe_relations, _ = sanitize_relations(relations, valid_names)
    # A rerun replaces the complete per-paper snapshot. Deleting only relations leaves
    # ghost chunks/entities when a newer parser emits fewer records.
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA)
        con.execute("DELETE FROM relations WHERE paper_id=?", (paper_id,))
        con.execute("DELETE FROM entities WHERE paper_id=?", (paper_id,))
        con.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        for s in sections:
            for c in s.get("chunks", []):
                con.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?)",
                            (c["chunk_id"], paper_id, s.get("sec", "0"), c["text"][:2000]))
                n_c += 1
        for e in entities:
            con.execute("INSERT OR REPLACE INTO entities VALUES(?,?,?)",
                        (e["name"], e["type"], paper_id))
            n_e += 1
        for r in safe_relations:
            con.execute("INSERT INTO relations VALUES(?,?,?,?)",
                        (r["from"], r["rel"], r["to"], paper_id))
    return {"chunks": n_c, "entities": n_e, "relations": len(safe_relations)}


def build_memory(sections: List[Dict], paper_id: str, db_path: str,
                 extract_fn: Optional[Callable[[str], Tuple[List[Dict], List[Dict]]]] = None,
                 sim: Optional[Callable[[str, str], float]] = None) -> Dict:
    from .graph import lexical_sim
    full = "\n".join(s.get("text", "") for s in sections)
    ents, rels = (extract_fn(full) if extract_fn else rule_extract(full))
    # 离线默认词法去重 (阈值0.85); embedding 注入时调用方传 sim + 0.93
    kept, stats = sanitize_entities(ents, sim=lexical_sim if sim is None else sim,
                                    merge_th=0.85 if sim is None else 0.93)
    canonical_map = stats.get("canonical_map", {})
    # 自定义抽取器返回非空但全被 Schema 拒收时，也必须回退，不能留下空图谱。
    if extract_fn is not None and ents and not kept:
        ents, rels = rule_extract(full)
        kept, fallback_stats = sanitize_entities(
            ents, sim=lexical_sim if sim is None else sim,
            merge_th=0.85 if sim is None else 0.93)
        for key, value in fallback_stats.items():
            stats[f"fallback_{key}"] = value
        stats["fallback_used"] = 1
        canonical_map = fallback_stats.get("canonical_map", {})
    rels, rel_stats = sanitize_relations(
        rels, {e["name"] for e in kept}, canonical_map=canonical_map)
    stats.update(rel_stats)
    saved = save_sqlite(db_path, paper_id, sections, kept, rels)
    return {"entities": kept, "relations": rels, "stats": stats, "saved": saved}
