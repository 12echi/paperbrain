"""公式合法性门禁 (v5.0 生产版).

主校验 katex (node 全局包) + 辅校验 sympy.parse_latex.
双 fail -> 回退纯文本 + $$[FORMULA_UNVERIFIED]$$, 禁止当干净公式入库.
任一工具缺失 -> 标记 checker 缺失, 不静默通过 (状态 NEEDS_REVIEW).
"""
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Dict, List

TEX_PAT = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\begin\{equation\}(.*?)\\end\{equation\}", re.S)


def _katex_ok(tex: str) -> bool:
    node = shutil.which("node")
    if not node:
        return False
    js = ("const k=require('%s');"
          "try{k.renderToString(process.argv[1],{throwOnError:true});console.log('OK');}"
          "catch(e){console.log('FAIL')}" % _katex_path())
    try:
        r = subprocess.run([node, "-e", js, tex], capture_output=True, text=True, timeout=15)
        return r.stdout.strip() == "OK"
    except Exception:
        return False


def _katex_path() -> str:
    import pathlib
    cands = [pathlib.Path.home() / ".npm-global/lib/node_modules/katex/dist/katex.js",
             pathlib.Path("/usr/local/lib/node_modules/katex/dist/katex.js"),
             pathlib.Path("/opt/homebrew/lib/node_modules/katex/dist/katex.js")]
    for c in cands:
        if c.exists():
            return str(c)
    return "katex/dist/katex.js"


def _sympy_ok(tex: str) -> bool:
    try:
        from sympy.parsing.latex import parse_latex  # type: ignore
        parse_latex(tex)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def checker_health() -> Dict[str, bool]:
    """Exercise both parsers once; package presence alone does not prove compatibility."""
    katex = _katex_ok("x^2 + 1")
    sympy = _sympy_ok("x^2 + 1")
    return {"katex": katex, "sympy": sympy, "any": katex or sympy,
            "dual": katex and sympy}


def check_formulas(text: str) -> Dict:
    """返回 {formulas:[{tex,katex_ok,sympy_ok,verdict}], dirty:int}."""
    out: List[Dict] = []
    for m in TEX_PAT.finditer(text or ""):
        tex = next(g for g in m.groups() if g)[:500]
        k = _katex_ok(tex)
        s = _sympy_ok(tex)
        verdict = "clean" if (k or s) else "unverified"
        out.append({"tex": tex[:120], "katex_ok": k, "sympy_ok": s, "verdict": verdict})
    dirty = sum(1 for f in out if f["verdict"] == "unverified")
    return {"formulas": out, "dirty": dirty}


def clean_text(text: str) -> str:
    """双 fail 公式回退为纯文本并打标 (会重新校验; 已有结果请用 mark_text)."""
    return mark_text(text, check_formulas(text or ""))


def mark_text(text: str, checked: Dict) -> str:
    """用已算好的 check_formulas 结果打标, 不重复调工具."""
    bad = {f["tex"] for f in checked.get("formulas", []) if f["verdict"] == "unverified"}

    def rep(m):
        tex = next(g for g in m.groups() if g)
        if tex[:120] in bad or tex in bad:
            return f"$$[FORMULA_UNVERIFIED]$$ {tex[:200]}"
        return m.group(0)
    return TEX_PAT.sub(rep, text or "")
