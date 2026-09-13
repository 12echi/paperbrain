"""LLM 真实调用层 (v5.0 生产版, stdlib only urllib).

- OpenAI-compatible /chat/completions, 无第三方依赖.
- 凭据全走环境变量, 永不落盘: PAPERBRAIN_API_KEY (必填才真调),
  PAPERBRAIN_BASE_URL (默认 https://api.openai.com/v1),
  PAPERBRAIN_MODEL (默认 gpt-4o-mini), PAPERBRAIN_VL_MODEL (默认 gpt-4o-mini).
- 无 key 时所有调用抛 NoKeyError, 上游自动回退规则式 (绝不伪造成功).
- 内置 token-bucket 限流 + 指数退避 + 超时, 触发 TPM/RPM 降并发不丢任务.
"""
import json
import shutil
import subprocess
import threading
import time
import urllib.request
from typing import Dict, List, Optional

from . import config


class NoKeyError(RuntimeError):
    pass


def has_key() -> bool:
    return config.has_key() and config.cloud_allowed()


def provider() -> str:
    """https(默认, 需 Key) | opencode-cli(走本机 opencode 二进制, 用 auth.json) | off"""
    return config.provider()


OCCLI_BIN = config.opencode_bin()
OCCLI_DIR = config.opencode_run_dir()

# 调用计数 (报告用) + CLI 会话复用 (线程局部存储, 防跨并发线程上下文串扰与会话销毁竞态)
CALL_COUNT = 0
_COUNT_LOCK = threading.Lock()
_OCCLI_LOCAL = threading.local()
_USAGE_LOCAL = threading.local()


def _estimate_tokens(value) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return max(1, int(len(text) / 3.5)) if text else 0


def reset_usage() -> None:
    """重置当前任务线程的模型 Token 台账。"""
    _USAGE_LOCAL.data = {"input": 0, "output": 0, "calls": 0,
                         "vision_images": 0, "by_stage": {}}


def _record_usage(stage: str, input_tokens: int, output_tokens: int) -> None:
    data = getattr(_USAGE_LOCAL, "data", None)
    if data is None:
        reset_usage()
        data = _USAGE_LOCAL.data
    inp, out = max(0, int(input_tokens)), max(0, int(output_tokens))
    data["input"] += inp
    data["output"] += out
    data["calls"] += 1
    data["by_stage"][stage] = data["by_stage"].get(stage, 0) + inp + out


def usage_snapshot() -> Dict:
    data = getattr(_USAGE_LOCAL, "data", None) or {"input": 0, "output": 0,
                                                    "calls": 0, "vision_images": 0,
                                                    "by_stage": {}}
    return {"input": int(data["input"]), "output": int(data["output"]),
            "total": int(data["input"] + data["output"]), "calls": int(data["calls"]),
            "vision_images": int(data.get("vision_images", 0)),
            "by_stage": dict(data["by_stage"])}


def record_external_call(stage: str, prompt, output="", vision_images: int = 0) -> None:
    """记录由视觉等外部适配器发起、但未经过 chat() 的一次模型调用。"""
    global CALL_COUNT
    with _COUNT_LOCK:
        CALL_COUNT += 1
    _record_usage(stage, _estimate_tokens(prompt), _estimate_tokens(output))
    _USAGE_LOCAL.data["vision_images"] = int(
        _USAGE_LOCAL.data.get("vision_images", 0)) + max(0, int(vision_images))


def _budget_output_cap(stage: str, input_tokens: int, requested: int) -> int:
    """在发送前按累计输入/重试账本收紧最大输出，避免先超支再报错。"""
    from .budget import PASS_LIMITS, TEXT_HARD
    data = usage_snapshot()
    global_remaining = TEXT_HARD - data["total"] - input_tokens
    stage_limit = PASS_LIMITS.get(stage)
    if stage_limit is None:
        stage_remaining = requested
    else:
        stage_used = int(data["by_stage"].get(stage, 0))
        stage_remaining = stage_limit - stage_used - input_tokens
    allowed = min(int(requested), global_remaining, stage_remaining)
    if allowed < 1:
        raise RuntimeError(
            f"模型调用预算不足: stage={stage} input={input_tokens} "
            f"global_remaining={global_remaining} stage_remaining={stage_remaining}")
    return allowed


def _get_occli_session() -> Optional[str]:
    return getattr(_OCCLI_LOCAL, "session", None)


def _set_occli_session(sid: Optional[str]):
    _OCCLI_LOCAL.session = sid


def reset_occli_session():
    """结束一篇时调用: 删除当前线程复用的 scratch 会话并清空 (防会话列表堆积与跨线程干扰)。
    注意: 必须"用完再删", 不能在每次调用后就删, 否则下次 -s 指向已删会话而快速失败。"""
    sid = _get_occli_session()
    _set_occli_session(None)
    if sid:
        try:
            from .vision import _cleanup_session
            _cleanup_session(sid)
        except Exception:
            pass


def close_occli_session():
    reset_occli_session()


class Bucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.cap = capacity
        self.tokens = capacity
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def take(self, n: float = 1.0):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.cap, self.tokens + (now - self.ts) * self.rate)
                self.ts = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            time.sleep(min(wait, 5.0))


_BUCKET = Bucket(rate=config.rps(), capacity=4.0)


def _post(url: str, body: bytes, key: str, timeout: int) -> str:
    """优先 curl (部分网关 Cloudflare 拦 urllib 指纹), 回退 urllib."""
    if shutil.which("curl"):
        try:
            r = subprocess.run(["curl", "-sS", "-m", str(timeout), "-w", "\n%{http_code}",
                                "-X", "POST", url, "-H", "Authorization: Bearer " + key,
                                "-H", "Content-Type: application/json",
                                "--data-binary", "@-"],
                               input=body, capture_output=True, timeout=timeout + 5)
            out = r.stdout.decode("utf-8", errors="replace")
            payload, _, code = out.rpartition("\n")
            if code.strip() == "200":
                return payload
            raise RuntimeError(f"HTTP {code.strip()} {payload[:200]}")
        except RuntimeError:
            raise
        except Exception:
            pass  # 掉回 urllib
    req = urllib.request.Request(url, data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def chat(messages: List[Dict], model: Optional[str] = None, max_tokens: int = 1200,
         timeout: int = 60, retries: int = 3, usage_bucket: str = "other") -> str:
    if not config.cloud_allowed():
        raise NoKeyError("云端模型调用未授权；请显式开启 PAPERBRAIN_CLOUD_ALLOWED")
    if provider() == "opencode-cli":
        return occli_chat(messages, model, timeout=max(timeout, 240), usage_bucket=usage_bucket)
    key = config.api_key()
    if not key:
        raise NoKeyError("缺 PAPERBRAIN_API_KEY, 已回退离线规则式")
    base = config.base_url()
    model = model or config.model()
    last: Exception = RuntimeError("unreachable")
    input_est = _estimate_tokens(messages)
    for i in range(retries):
        allowed_output = _budget_output_cap(usage_bucket, input_est, max_tokens)
        body = json.dumps({"model": model, "messages": messages,
                           "max_tokens": allowed_output, "temperature": 0.2}).encode()
        try:
            _BUCKET.take()
            global CALL_COUNT
            with _COUNT_LOCK:
                CALL_COUNT += 1
            payload = json.loads(_post(base + "/chat/completions", body, key, timeout))
            content = payload["choices"][0]["message"]["content"].strip()
            usage = payload.get("usage") or {}
            _record_usage(usage_bucket,
                          usage.get("prompt_tokens", input_est),
                          usage.get("completion_tokens", _estimate_tokens(content)))
            return content
        except Exception as e:
            _record_usage(usage_bucket, input_est, 0)
            last = e
            time.sleep(min(2 ** i, 8))
    raise RuntimeError(f"LLM 调用 {retries} 次均失败: {last}")


def occli_chat(messages: List[Dict], model: Optional[str] = None, timeout: int = 300,
               usage_bucket: str = "other") -> str:
    """经由本机 opencode 二进制调用。优先复用线程会话加速; 若失败(会话上下文膨胀/超限),
    自动用全新会话重试一次, 兼顾速度与可靠。"""
    global CALL_COUNT
    import os as _os
    model = model or config.model()
    if "/" not in model:
        model = "opencode-go/" + model
    prompt = "\n".join(f"[{m.get('role', 'user')}] {m.get('content', '')}" for m in messages)
    binp, rundir = config.opencode_bin(), config.opencode_run_dir()
    _os.makedirs(rundir, exist_ok=True)

    def _once(use_session: Optional[str]):
        global CALL_COUNT
        input_est = _estimate_tokens(prompt)
        _budget_output_cap(usage_bucket, input_est, 1)
        cmd = [binp, "run", "--format", "json", "-m", model]
        if use_session:
            cmd += ["-s", use_session]
        cmd += [prompt]
        _BUCKET.take()
        with _COUNT_LOCK:
            CALL_COUNT += 1
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=rundir)
        texts, sid = [], ""
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            sid = sid or ev.get("sessionID", "") or ev.get("sessionId", "")
            if ev.get("type") == "text" and isinstance(ev.get("part"), dict):
                t = ev["part"].get("text", "")
                if t:
                    texts.append(t)
        out = "".join(texts).strip()
        _record_usage(usage_bucket, input_est, _estimate_tokens(out))
        if r.returncode != 0 or not out:
            raise RuntimeError(f"opencode CLI rc={r.returncode}: {(r.stderr or '')[-200:]}")
        return out, sid

    curr_sid = _get_occli_session()
    try:
        out, sid = _once(curr_sid)
    except Exception:
        # 任何失败都丢弃当前线程会话, 用全新会话重试一次 (会话膨胀/网关抖动均可恢复)
        _set_occli_session(None)
        time.sleep(1.0)
        out, sid = _once(None)
    if sid:
        _set_occli_session(sid)
    return out


def summarize(kind: str, text: str) -> str:
    """Pass 纪要真实版. 无 key 抛 NoKeyError 由调用方回退."""
    prompts = {
        "pass1": "你是审稿人。用中文总结以下论文的宏观骨架: 问题、核心贡献(分点)、主要结论, ≤300字, 只基于原文不臆测:\n",
        "pass2": "你是审稿人。用中文总结以下方法章节: 模型结构、关键公式思想、训练目标, ≤300字, 只基于原文:\n",
        "pass3": "你是审稿人。用中文总结以下实验章节: 数据集、指标、主结果数字、消融结论, ≤300字, 数字必须来自原文:\n",
        "pass4": "你是苛刻审稿人。基于以下三遍纪要做批判: 3个优点、3个实质性质疑、1个拒稿风险点, ≤300字:\n",
    }
    return chat([{"role": "user", "content": prompts.get(kind, prompts["pass1"]) + text[:8000]}],
                max_tokens=800, usage_bucket=kind)


def draft_section(h1: str, h2: str, claim: str, entities: str) -> str:
    return chat([{"role": "user", "content":
                  f"写学术综述小节 [{h1}/{h2}], 围绕 {entities}。必须紧扣以下原文依据改写, 不得引入依据外事实, ≤250字:\n{claim[:1500]}"}],
                max_tokens=600, usage_bucket="draft")


def draft_review_section(h1: str, h2: str, evidence: str, entities: str,
                         paper_id: str) -> str:
    """Generate an evidence-bound peer-review section, not a renamed summary."""
    prompt = (
        f"你是严格的同行评审人。审查 [{h1}/{h2}]，相关实体为 {entities}。"
        "仅依据下方带 [Sec X] 的材料，分别给出：有证据支持的优点、实质性局限或效度威胁、材料无法回答的问题。"
        "不得把作者主张当成已证实事实，不得臆造对照、统计显著性或拒稿理由。"
        f"每条事实性判断末尾必须写 [Ref: {paper_id}, Sec X]，X 必须来自材料；"
        "推演必须标 [推断]，证据不足明确写“材料未提供”。中文，≤250字。\n"
        f"=== 审稿证据 ===\n{evidence[:4200]}"
    )
    return chat([{"role": "user", "content": prompt}], max_tokens=700,
                usage_bucket="draft_review")


# ---------------------------------------------------------------- 深度解读 (A)
def deep_analyze(context: str, focus: str = "", paper_id: str = "") -> str:
    """穿透式深度解读: 重建论证链, 非泛泛摘要。返回 Markdown。"""
    focus_line = (f"\n【用户特别关注】请在相关小标题下额外回应: {focus}\n" if focus else "")
    prompt = (
        "你是资深科研审稿人, 正在做一次穿透式深度解读。只基于给定材料, 不得引入材料外事实; "
        "每个论点都要锚定原文(给出章节号/具体数字/公式/术语), 禁止空话套话。用中文输出, 严格按以下小标题:\n"
        "## 一句话主张\n用一句话说清这篇文章到底主张什么(不是它做了什么, 而是它想让读者相信什么)。\n"
        "## 研究问题与空白 (Gap)\n先前做法为何不够? 作者要补的缺口是什么?\n"
        "## 核心洞见 (Key Insight)\n打开这个问题的关键想法是什么, 为什么它成立。\n"
        "## 方法与关键设计\n方法主线, 关键公式(用 LaTeX), 以及每个设计选择的动机。\n"
        "## 证据强度\n主要结果(带具体数字)、对照基线、消融、统计显著性; 证据是否足以支撑主张。\n"
        "## 隐藏假设\n方法成立所依赖但未明说的前提。\n"
        "## 局限与威胁效度\n作者自述局限 + 你识别出的效度威胁。\n"
        "## 可复现性\n数据/代码/超参/环境是否可得, 复现障碍。\n"
        "## 延伸设想 (If I were to extend)\n3 条具体可行的下一步研究设想。\n"
        f"每个事实性段落末尾必须使用规范来源 [Ref: {paper_id}, Sec X]，X 必须取自材料中的 Sec 标记；"
        "证据不足写“材料未提供”，不得用占位符 X。推演性内容必须明确标为[推断]并给出所依据章节。\n"
        f"{focus_line}\n=== 论文材料 ===\n{context[:8000]}"
    )
    return chat([{"role": "user", "content": prompt}], max_tokens=1800, timeout=300,
                usage_bucket="deepread")


# ---------------------------------------------------------------- 提问驱动 (B)
def gen_questions(context: str, n: int = 5) -> str:
    prompt = (
        "你是多视角研讨主持人。针对论文材料, 从【方法学家】【统计学家】【实践者】【怀疑者】四种视角中选取最关键的, "
        f"提出 {n} 个能逼近论文核心的尖锐问题(宁少勿滥, 每个都要能戳中要害)。只输出 JSON 数组, 元素形如 "
        '{"perspective":"统计学家","q":"..."}。不要解释, 只输出 JSON:\n' + context[:7000])
    return chat([{"role": "user", "content": prompt}], max_tokens=700, timeout=300)


def answer_questions(context: str, questions_json: str, paper_id: str) -> str:
    prompt = (
        "针对下列问题逐个作答, 每个回答控制在 2 句以内、直击要点, 末尾标注来源 [Ref: "
        f"{paper_id}, Sec X]; 证据不足就写“材料未提供”, 不得编造。用中文 Markdown, 问题原文加粗, 回答用短段落。\n"
        f"问题(JSON): {questions_json[:2000]}\n\n=== 论文材料 ===\n{context[:7000]}")
    return chat([{"role": "user", "content": prompt}], max_tokens=1100, timeout=300)


# ---------------------------------------------------------------- 全局综合 (C)
def global_synthesis(context: str, paper_id: str = "") -> str:
    prompt = (
        "基于论文材料, 输出一个 JSON 对象(只输出 JSON, 不要解释), 结构:\n"
        '{"argument_map":[{"claim":"...","evidence":"...","limitation":"..."}],'
        '"positioning":{"improves_upon":"...","contradicts":"...","gap_left":"..."},'
        '"field_view":"这篇文章在其研究方向中的位置与意义, 80字内"}\n'
        f"argument_map 给 3-5 条核心主张及其证据与局限；evidence 必须包含材料中的规范 [Ref: {paper_id}, Sec X]；"
        "无则填“材料未提供”。材料:\n" + context[:7000])
    return chat([{"role": "user", "content": prompt}], max_tokens=1200, timeout=300)


# ---------------------------------------------------------------- 记忆层 (L2/L5)
def answer_memory(question: str, notes_ctx: str) -> str:
    """就"学习记忆"作答: 只依据给定笔记, 用 [M<id>] 标注来源, 不得引入外部事实。"""
    prompt = (
        "你是用户的学习记忆助手。仅依据下面的记忆条目回答用户问题; 每条都有编号 [M<id>]。"
        "综合多条给出结论, 句末用 [M<id>] 标注依据; 若记忆不足以回答就直说“记忆中没有相关信息”, 不得编造。"
        "若某条目标记为 contested(存在争议), 必须同时指出该结论有分歧, 不得当作定论。"
        "中文, 150 字内, 直接给答案不要复述问题。\n\n"
        f"用户问题：{question[:300]}\n\n=== 记忆条目 ===\n{notes_ctx[:6000]}")
    return chat([{"role": "user", "content": prompt}], max_tokens=600, timeout=300)


def memory_card(context: str, existing_concepts: str = "") -> str:
    """写时记忆卡: 规范概念+别名+主张立场, 供概念消解与矛盾标记。只输出 JSON。"""
    prompt = (
        "从论文材料提炼一张'记忆卡', 只输出 JSON 对象:\n"
        '{"concepts":[{"name":"规范概念名","aliases":["别名1","别名2"]}],'
        '"claims":[{"text":"一句话主张","stance":"supports|contradicts|extends","about":"对应概念"}]}\n'
        "概念用规范写法(如 FlashAttention-v2), 别名给常见异写; 主张 3-6 条, stance 表示该主张相对既有认知的关系。"
        + (f"\n已有概念(请复用同名): {existing_concepts[:600]}" if existing_concepts else "")
        + "\n材料:\n" + context[:7000])
    return chat([{"role": "user", "content": prompt}], max_tokens=1200, timeout=300)


def verify_claims(claims: List[str], source: str) -> str:
    """批量质检: 逐条判断主张是否被原文支持。只输出 JSON 数组 [{"i":0,"ok":true}]。"""
    listing = "\n".join(f"{i}. {c[:200]}" for i, c in enumerate(claims))
    prompt = (
        "判断下列每条主张是否被'原文'支持(可包括直接陈述或严格蕴含)。只输出 JSON 数组, "
        '元素形如 {"i":0,"ok":true}; 不解释。宁严勿松, 不确定即 false。\n'
        f"主张:\n{listing}\n\n原文:\n{source[:6000]}")
    return chat([{"role": "user", "content": prompt}], max_tokens=500, timeout=300)


def expand_query(q: str) -> str:
    """查询扩展: 同义词/英文对照/相关术语, 只输出 JSON 数组字符串。"""
    prompt = (
        "把用户问题扩展成 2-4 个不同措辞的检索式(含同义表达、英文术语、相关概念), 便于词法与向量检索。"
        '只输出 JSON 数组, 如 ["DBSCAN 参数标定","DBSCAN parameter tuning"]。问题: ' + q[:300])
    return chat([{"role": "user", "content": prompt}], max_tokens=300, timeout=300)


def reflect_notes(notes_ctx: str, n: int = 3) -> str:
    """反思: 从给定记忆条目 (编号 [M#]) 综合出更高层洞见。只输出 JSON 数组。"""
    prompt = (
        f"以下是关于同一研究的若干记忆条目(每条编号 [M<id>])。请综合出 {n}-5 条**更高层次**的洞见: "
        "跨条目的规律、矛盾、隐含前提、或可迁移到其他问题的结论; 不要复述单条内容, 不要引入条目外事实。"
        '只输出 JSON 数组, 元素形如 {"insight":"...","evidence":[12,15]} (evidence 为支撑该洞见的条目编号)。\n'
        f"记忆条目:\n{notes_ctx[:6000]}")
    return chat([{"role": "user", "content": prompt}], max_tokens=900, timeout=300)


def gen_eval_question(note: str, lang: str = "zh") -> str:
    """评测问题生成: 依据笔记写一个自然的用户提问 (口语化, 不复述术语), 用于模糊检索评测。"""
    style = ("用中文口语化的问法" if lang == "zh" else
             "用英文提问 (English question, 可保留必要专名)")
    prompt = (
        f"根据下面这段研究笔记, 写 1 个科研人员可能提出的自然问题, {style}。"
        "要求: 只输出问题本身, 不要解释; 不得直接复制笔记原句; "
        "尽量用日常说法复述关键概念, 避免照抄术语。\n"
        f"笔记:\n{note[:900]}")
    return chat([{"role": "user", "content": prompt}], max_tokens=200, timeout=120).strip()


def rerank(query: str, candidates: str) -> str:
    """对候选记忆按与问题的相关性重排。只输出 JSON 数组 [id,...] (最相关在前)。"""
    prompt = (
        "按与问题的相关性给候选记忆排序, 只输出 JSON 数组(元素为编号, 最相关在前), 不解释。\n"
        f"问题: {query[:300]}\n候选:\n{candidates[:5000]}")
    return chat([{"role": "user", "content": prompt}], max_tokens=300, timeout=300)
