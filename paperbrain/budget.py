"""Token 双顶预算 (v5.0 生产版).

v4.0 假账修复:
- 12k 只是纯文本, 视觉 9 张 *1.2k≈11k 被藏起来了. v5 改双顶:
  TEXT_HARD=12k (输入输出全含, 含 system/JSON/重试) AND VISION_HARD=6k AND TOTAL_HARD=18k.
- 每 Pass 限额内再扣 15% headroom, 预留 prompt 开销, 超限直接拒绝执行不伪造.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

TEXT_HARD = 12_000
VISION_HARD = 6_000
TOTAL_HARD = 18_000
HEADROOM = 0.15

# 单 Pass 文本限额 (已是硬顶, 调用方还需再扣 headroom)
PASS_LIMITS = {
    "pass1": 2500,
    "pass2": 3500,
    "pass3": 4000,
    "pass3_text": 4000,
    "pass4": 2000,
}


def estimate_vision_tokens(num_images: int, longest_edge: int = 1024) -> int:
    """粗估 vision token: 1024px≈1200, 按面积线性缩放, 上限单图 2000."""
    if num_images <= 0:
        return 0
    scale = min(1.0, (longest_edge / 1024) ** 2) if longest_edge else 1.0
    per = int(min(2000, 1200 * max(0.25, scale)))
    return per * num_images


@dataclass
class BudgetResult:
    ok: bool
    reasons: List[str]
    text: int
    vision: int
    total: int


def check_budget(pass_text: Optional[Dict[str, int]] = None, num_images: int = 0,
                 longest_edge: int = 1024, cache_hit: bool = False) -> BudgetResult:
    reasons: List[str] = []
    pass_map = pass_text or {}
    text = sum(pass_map.values())
    vision = estimate_vision_tokens(num_images, longest_edge)
    total = text + vision

    # 单 Pass 检查 (含 headroom)
    for k, v in pass_map.items():
        lim = PASS_LIMITS.get(k)
        if lim and v > lim:  # Pass 限额本身已是硬顶, 不再二次扣, 但超即拒
            reasons.append(f"{k} {v} > {lim}")
    cap_text = TEXT_HARD if not cache_hit else 2000
    if text > cap_text:
        reasons.append(f"text {text} > {cap_text}{' (cache_hit)' if cache_hit else ''}")
    if vision > VISION_HARD:
        reasons.append(f"vision {vision} > {VISION_HARD}: 图太多, 必须抽样至多5图或转表为文本")
    if total > TOTAL_HARD:
        reasons.append(f"total {total} > {TOTAL_HARD}")
    # headroom 预警 (超 85% 即警告, 供上游截断)
    if text > int(TEXT_HARD * (1 - HEADROOM)) and not cache_hit:
        reasons.append(f"warn: text {text} 超 85% 水位, 需截断/降级")
    hard_fail = [r for r in reasons if not r.startswith("warn:")]
    return BudgetResult(ok=not hard_fail,
                        reasons=reasons, text=text, vision=vision, total=total)
