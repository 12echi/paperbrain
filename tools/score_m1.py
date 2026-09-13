"""M1 解析评分脚本 (v5.0 生产门禁, 离线可跑).

用法:
  python3 tools/score_m1.py --golden golden.json --pred pred.json
输入均为 JSON list。golden/pred 的 tables 必须含唯一 table_id 与 cells 二维数组；
captions 必须含唯一 figure_id（或唯一 caption）与 bound_contexts 字符串数组。
输出: 还原率/表格精度/Caption绑定率 + PASS/FAIL (阈值见 v5.0 M1).

硬门禁要求 golden 至少 30 个唯一 ID（two_column/scanned/complex_figures 各至少10篇），
且预测覆盖全部 golden；缺样本不能靠交集平均绕过。

文本还原率 = max(0, 1 - Levenshtein(norm(gold), norm(pred)) / max(len(gold),1))。
短样本用内置精确 DP；长论文必须安装 RapidFuzz，禁止用另一种相似度冒充。
"""
import argparse
import json
import re
import sys

try:
    from rapidfuzz.distance import Levenshtein as _LEVENSHTEIN  # type: ignore
except ImportError:
    _LEVENSHTEIN = None

REQUIRED_PAPERS = 30
REQUIRED_CATEGORIES = {"two_column": 10, "scanned": 10, "complex_figures": 10}


def norm_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t


def text_score(gold: str, pred: str) -> float:
    g, p = norm_text(gold), norm_text(pred)
    if not g:
        return 1.0 if not p else 0.0
    gold_len = len(g)
    if _LEVENSHTEIN is not None:
        distance = int(_LEVENSHTEIN.distance(g, p))
    else:
        # Exact two-row DP is acceptable for unit fixtures, not entire papers.
        if len(g) * len(p) > 4_000_000:
            raise RuntimeError("长文本 M1 精确评分需要 rapidfuzz==3.13.0")
        if len(g) < len(p):
            g, p = p, g
        previous = list(range(len(p) + 1))
        for i, cg in enumerate(g, 1):
            current = [i]
            for j, cp in enumerate(p, 1):
                current.append(min(current[-1] + 1, previous[j] + 1,
                                   previous[j - 1] + (cg != cp)))
            previous = current
        distance = previous[-1]
    return max(0.0, 1.0 - distance / gold_len)


def _validate_payload(row: dict, label: str, prediction: bool = False) -> None:
    if not isinstance(row.get("text", ""), str):
        raise ValueError(f"{label}.text 必须是字符串")
    for field in ("formulas", "tables", "captions"):
        value = row.get(field, [])
        if not isinstance(value, list):
            raise ValueError(f"{label}.{field} 必须是数组")
        if any(not isinstance(item, dict) for item in value):
            raise ValueError(f"{label}.{field} 每项必须是对象")
    for index, formula in enumerate(row.get("formulas", [])):
        if not isinstance(formula.get("tex"), str):
            raise ValueError(f"{label}.formulas[{index}].tex 必须是字符串")
        if prediction and type(formula.get("katex_ok")) is not bool:
            raise ValueError(f"{label}.formulas[{index}].katex_ok 必须是 JSON boolean")


def score_pair(g: dict, p: dict) -> dict:
    _validate_payload(g, "golden")
    _validate_payload(p, "pred", prediction=True)
    ts = text_score(g.get("text", ""), p.get("text", ""))
    gf = g.get("formulas", [])
    pf = {f.get("tex"): f for f in p.get("formulas", [])}
    fok = sum(1 for f in gf if pf.get(f.get("tex"), {}).get("katex_ok"))
    frate = (fok / len(gf)) if gf else 1.0
    tc_ok, tc_tot_raw, _ = _score_tables(g.get("tables", []), p.get("tables", []))
    trate = tc_ok / tc_tot_raw if tc_tot_raw else 0.0
    gc = g.get("captions", [])
    b2, caption_total, _ = _score_captions(gc, p.get("captions", []))
    brate = (b2 / caption_total) if caption_total else 0.0
    return {"text": round(ts, 4), "formula": round(frate, 4),
            "table": round(trate, 4), "caption_bind": round(brate, 4)}


def _table_key(row: dict) -> str:
    return str(row.get("table_id") or row.get("id") or "").strip().lower()


def _caption_key(row: dict) -> str:
    return str(row.get("figure_id") or row.get("id") or norm_text(row.get("caption", ""))).strip().lower()


def _valid_cells(value) -> bool:
    return (isinstance(value, list) and bool(value) and
            all(isinstance(row, list) and bool(row) for row in value))


def _context_matches(expected: str, actual: str) -> bool:
    expected = norm_text(expected).casefold()
    actual = norm_text(actual).casefold()
    if not expected or not actual:
        return False
    if expected == actual:
        return True
    # Human goldens may mark a paragraph while PDF extraction exposes its reference line.
    shorter, longer = sorted((expected, actual), key=len)
    return len(shorter) >= 20 and shorter in longer


def _score_tables(golden, pred):
    errors = []
    pmap = {}
    for i, row in enumerate(pred if isinstance(pred, list) else []):
        key = _table_key(row) if isinstance(row, dict) else ""
        if not key or key in pmap or not _valid_cells(row.get("cells")):
            errors.append(f"pred table[{i}] 缺唯一 table_id 或 cells")
            continue
        pmap[key] = row
    correct = total = 0
    seen = set()
    for i, row in enumerate(golden if isinstance(golden, list) else []):
        key = _table_key(row) if isinstance(row, dict) else ""
        cells = row.get("cells") if isinstance(row, dict) else None
        if not key or key in seen or not _valid_cells(cells):
            errors.append(f"golden table[{i}] 缺唯一 table_id 或 cells")
            continue
        seen.add(key)
        predicted = pmap.get(key, {}).get("cells", [])
        golden_coords = {(rno, cno): value for rno, grow in enumerate(cells)
                         for cno, value in enumerate(grow)}
        predicted_coords = {(rno, cno): value for rno, prow in enumerate(predicted)
                            for cno, value in enumerate(prow)}
        for coordinate in set(golden_coords) | set(predicted_coords):
            total += 1
            if coordinate in golden_coords and coordinate in predicted_coords and \
                    norm_text(str(golden_coords[coordinate] or "")) == \
                    norm_text(str(predicted_coords[coordinate] or "")):
                correct += 1
    for key, row in pmap.items():
        if key not in seen:
            total += sum(len(prow) for prow in row.get("cells", []))
    return correct, total, errors


def _score_captions(golden, pred):
    errors = []
    pmap = {}
    for i, row in enumerate(pred if isinstance(pred, list) else []):
        key = _caption_key(row) if isinstance(row, dict) else ""
        contexts = row.get("bound_contexts") if isinstance(row, dict) else None
        if not key or key in pmap or not isinstance(contexts, list):
            errors.append(f"pred caption[{i}] 缺唯一 ID/caption 或 bound_contexts 数组")
            continue
        pmap[key] = row
    correct = total = 0
    seen = set()
    for i, row in enumerate(golden if isinstance(golden, list) else []):
        key = _caption_key(row) if isinstance(row, dict) else ""
        contexts = row.get("bound_contexts") if isinstance(row, dict) else None
        if not key or key in seen or not isinstance(contexts, list) or not contexts:
            errors.append(f"golden caption[{i}] 缺唯一 ID/caption 或非空 bound_contexts")
            continue
        seen.add(key)
        total += 1
        expected = [norm_text(str(x)) for x in contexts if norm_text(str(x))]
        predicted = pmap.get(key, {}).get("bound_contexts", [])
        actual = [norm_text(str(x)) for x in predicted if norm_text(str(x))]
        if any(_context_matches(e, a) for e in expected for a in actual):
            correct += 1
    total += sum(1 for key in pmap if key not in seen)
    return correct, total, errors


def _index(rows, label: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError(f"{label} 必须是 JSON list")
    out = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not str(row.get("id", "")).strip():
            raise ValueError(f"{label}[{i}] 缺有效 id")
        key = str(row["id"]).strip()
        if key in out:
            raise ValueError(f"{label} 存在重复 id: {key}")
        out[key] = row
    return out


def evaluate(gold_rows, pred_rows, min_text: float = 0.98,
             min_table: float = 0.99, min_caption: float = 0.99,
             required_papers: int = REQUIRED_PAPERS,
             required_categories=None) -> dict:
    """按完整 golden 集评分；返回机器可读结果，不隐藏缺失预测。"""
    gold = _index(gold_rows, "golden")
    pred = _index(pred_rows, "pred")
    for paper_id, row in gold.items():
        _validate_payload(row, f"golden[{paper_id}]")
    for paper_id, row in pred.items():
        _validate_payload(row, f"pred[{paper_id}]", prediction=True)
    ids = sorted(gold)
    missing = [paper_id for paper_id in ids if paper_id not in pred]
    extra = sorted(set(pred) - set(gold))
    if not ids:
        raise ValueError("golden 为空")

    # Keep rounded per-paper values for display only; release decisions use raw values.
    raw_text = {paper_id: text_score(gold[paper_id].get("text", ""),
                                     pred.get(paper_id, {}).get("text", ""))
                for paper_id in ids}
    per_paper = {paper_id: score_pair(gold[paper_id], pred.get(paper_id, {}))
                 for paper_id in ids}
    text_avg = sum(raw_text.values()) / len(ids)

    source_errors = []
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    golden_hashes = []
    for paper_id in ids:
        expected_sha = str(gold[paper_id].get("source_sha256", "")).lower()
        actual_sha = str(pred.get(paper_id, {}).get("source_sha256", "")).lower()
        if not sha_pattern.fullmatch(expected_sha):
            source_errors.append(f"{paper_id}: golden 缺有效 source_sha256")
        else:
            golden_hashes.append(expected_sha)
            if actual_sha != expected_sha:
                source_errors.append(f"{paper_id}: pred source_sha256 不匹配")
    if len(set(golden_hashes)) != len(golden_hashes):
        source_errors.append("golden source_sha256 必须逐篇唯一，禁止重复论文刷样本数")

    table_total = table_ok = 0
    structure_errors = []
    for paper_id in ids:
        ok_cells, total_cells, errors = _score_tables(
            gold[paper_id].get("tables", []), pred.get(paper_id, {}).get("tables", []))
        table_ok += ok_cells
        table_total += total_cells
        structure_errors.extend(f"{paper_id}: {error}" for error in errors)
    table_rate = table_ok / table_total if table_total else 0.0

    caption_total = caption_ok = 0
    for paper_id in ids:
        ok_caps, total_caps, errors = _score_captions(
            gold[paper_id].get("captions", []), pred.get(paper_id, {}).get("captions", []))
        caption_ok += ok_caps
        caption_total += total_caps
        structure_errors.extend(f"{paper_id}: {error}" for error in errors)
    caption_rate = caption_ok / caption_total if caption_total else 0.0

    formula_total = sum(len(g.get("formulas", [])) for g in gold.values())
    formula_ok = 0
    for paper_id in ids:
        pforms = {f.get("tex"): bool(f.get("katex_ok"))
                  for f in pred.get(paper_id, {}).get("formulas", [])}
        formula_ok += sum(1 for f in gold[paper_id].get("formulas", [])
                          if pforms.get(f.get("tex")))
    formula_rate = formula_ok / formula_total if formula_total else 1.0

    required_categories = (REQUIRED_CATEGORIES if required_categories is None
                           else required_categories)
    category_counts = {}
    dataset_errors = []
    for row in gold.values():
        category = str(row.get("category", "")).strip()
        category_counts[category] = category_counts.get(category, 0) + 1
        if not row.get("text", "").strip():
            dataset_errors.append(f"{row['id']}: golden text 为空")
        if category == "complex_figures" and not row.get("captions"):
            dataset_errors.append(f"{row['id']}: complex_figures 缺 Caption 绑定金标")
    categories_complete = all(category_counts.get(name, 0) >= count
                              for name, count in required_categories.items())
    structure_complete = table_total > 0 and caption_total > 0 and not structure_errors
    source_bound = not source_errors
    dataset_complete = (len(ids) >= required_papers and not missing and categories_complete and
                        structure_complete and source_bound and not dataset_errors)
    avg = {"text": round(text_avg, 4), "formula": round(formula_rate, 4),
           "table": round(table_rate, 4), "caption_bind": round(caption_rate, 4)}
    ok = (dataset_complete and text_avg >= min_text and
          table_rate >= min_table and caption_rate >= min_caption)
    return {
        "n_golden": len(ids), "n_pred": len(pred), "required_papers": required_papers,
        "category_counts": category_counts, "required_categories": required_categories,
        "missing_pred": missing, "extra_pred": extra,
        "source_errors": source_errors,
        "dataset_errors": dataset_errors,
        "structure_errors": structure_errors,
        "avg": avg,
        "counts": {"table_cells_ok": table_ok, "table_cells_total": table_total,
                   "captions_ok": caption_ok, "captions_total": caption_total,
                   "formulas_ok": formula_ok, "formulas_total": formula_total},
        "gate": {"min_text": min_text, "min_table": min_table,
                 "min_caption": min_caption, "structure_complete": structure_complete,
                 "source_bound": source_bound,
                 "golden_semantics_complete": not bool(dataset_errors),
                 "dataset_complete": dataset_complete},
        "result": "PASS" if ok else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--min-text", type=float, default=0.98)
    ap.add_argument("--min-table", type=float, default=0.99)
    ap.add_argument("--min-caption", type=float, default=0.99)
    a = ap.parse_args()
    try:
        with open(a.golden, encoding="utf-8") as f:
            gold_rows = json.load(f)
        with open(a.pred, encoding="utf-8") as f:
            pred_rows = json.load(f)
        result = evaluate(gold_rows, pred_rows, a.min_text, a.min_table, a.min_caption)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"M1 输入无效: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
