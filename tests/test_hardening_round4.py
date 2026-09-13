"""第四轮审计反例：只覆盖此前全量测试未触达的确定性缺陷。"""
import json
import os
import shutil
import struct
import tempfile
import unittest
import zlib
from pathlib import Path


class TestOCRRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(dir="/tmp")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_preflight_exposes_same_low_confidence_pages_used_by_hybrid(self):
        import fitz
        from paperbrain.preflight import preflight
        pdf = Path(self.tmp, "mixed.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 550, 740),
                            "This born-digital page contains extractable evidence. " * 20,
                            fontsize=10)
        doc.new_page()  # a scan-like page with no extractable text layer
        doc.save(str(pdf))
        doc.close()
        result = preflight(str(pdf), "P")
        self.assertEqual(result.route, "hybrid")
        self.assertEqual(result.low_conf_pages, [1])
        self.assertGreaterEqual(result.page_rates[0], 0.9)
        self.assertEqual(result.page_rates[1], 0.0)

    def test_ocr_selected_pages_preserve_original_page_numbers_and_confidence(self):
        import fitz
        from unittest.mock import patch
        from paperbrain import ocr
        pdf = Path(self.tmp, "two-pages.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.new_page()
        doc.save(str(pdf))
        doc.close()
        tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
               "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t95\tRecovered\n")
        class R:
            returncode = 0
            stdout = tsv.encode()
            stderr = b""
        with patch.object(ocr, "_select_language", return_value=("eng", [])), \
                patch.object(ocr.subprocess, "run", return_value=R()) as run:
            result = ocr.ocr_pdf(str(pdf), workdir=self.tmp, pages=[1])
        self.assertTrue(result["ok"])
        self.assertEqual(result["pages"], [{"page": 1, "text": "Recovered", "conf": 0.95}])
        self.assertEqual(run.call_count, 1)

    def test_table_caption_without_detectable_lines_routes_hybrid(self):
        import fitz
        from paperbrain.preflight import preflight
        pdf = Path(self.tmp, "borderless-table.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(60, 60, 540, 700),
            "Table 1. Results\n" +
            "This born-digital page has abundant extractable text and a borderless table. " * 25,
            fontsize=10)
        doc.save(str(pdf))
        doc.close()
        result = preflight(str(pdf), "T")
        self.assertGreater(result.page_rates[0], 0.9)
        self.assertEqual(result.route, "hybrid")
        self.assertEqual(result.low_conf_pages, [0])
        self.assertEqual(result.table_line_missing_pages, [0])


class TestOutlineHistory(unittest.TestCase):
    def test_append_diff_and_rollback_preserve_history(self):
        from paperbrain.outline import (diff_outline_versions, list_outline_versions,
                                        rollback_outline, save_outline_version)
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            base = {"paper_id": "P", "sections": [{"h1": "Method", "chunks": ["C1"]}],
                    "entities": []}
            v1 = save_outline_version(td, base, source="generated")
            edited = json.loads(json.dumps(base))
            edited["sections"][0]["h1"] = "Revised Method"
            v2 = save_outline_version(td, edited, source="confirmed")
            self.assertEqual((v1["version"], v2["version"]), ("v1", "v2"))
            self.assertIn("Revised Method", diff_outline_versions(td, "v1", "v2"))
            v3 = rollback_outline(td, "v1")
            self.assertEqual(v3["version"], "v3")
            self.assertEqual(v3["parent_version"], "v2")
            self.assertEqual(v3["source"], "rollback:v1")
            self.assertEqual([x["version"] for x in list_outline_versions(td)],
                             ["v1", "v2", "v3"])

    def test_generation_outline_validation_rejects_malformed_nested_shape(self):
        from paperbrain.outline import validate_generation_outline
        with self.assertRaisesRegex(ValueError, "h2"):
            validate_generation_outline({
                "paper_id": "P", "entities": [],
                "sections": [{"h1": "Method", "chunks": ["P_Sec2_C001"]}],
            })
        with self.assertRaisesRegex(ValueError, "chunks"):
            validate_generation_outline({
                "paper_id": "P", "entities": [],
                "sections": [{"h1": "Method", "h2": "Model", "chunks": "P_Sec2_C001"}],
            })


class TestPipelineAndSchema(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(dir="/tmp")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_injected_semantic_scorer_does_not_crash(self):
        from paperbrain.pipeline import run_outline, generate_from_outline
        out = Path(self.tmp, "run")
        first = run_outline("demo/sample_paper.txt", "2024_NeurIPS_01", str(out), use_llm=False)
        result = generate_from_outline(
            str(out), "2024_NeurIPS_01", first["outline"],
            semantic_scorer=lambda claim, source: 0.99, use_llm=False)
        self.assertIn(result["verify"], ("CLEAN", "DIRTY", "NEEDS_REVIEW"))

    def test_model_graph_uses_canonical_schema_and_filters_relations(self):
        import paperbrain.llm_ops as ops
        from paperbrain.graph import ALLOWED_ENTITIES, ALLOWED_RELATIONS
        self.assertEqual(ops._SCHEMA_ENTS, ALLOWED_ENTITIES)
        self.assertEqual(ops._SCHEMA_RELS, ALLOWED_RELATIONS)

        from paperbrain.memory import build_memory
        sections = [{"name": "method", "sec": "2", "text": "DBSCAN and DNA damage.",
                     "chunks": [{"chunk_id": "P_Sec2_C001", "text": "DBSCAN and DNA damage."}]}]
        def bad_extract(_):
            return ([{"name": "DBSCAN", "type": "Algorithm/Model"},
                     {"name": "DNA damage", "type": "Problem/Task"}],
                    [{"from": "DBSCAN", "rel": "Built_On", "to": "DNA damage"},
                     {"from": "DBSCAN", "rel": "Requires", "to": "DNA damage"}])
        result = build_memory(sections, "P", str(Path(self.tmp, "g.sqlite")), extract_fn=bad_extract)
        self.assertEqual([r["rel"] for r in result["relations"]], ["Requires"])
        self.assertEqual(result["stats"]["rejected_relation_schema"], 1)

    def test_relation_alias_endpoints_follow_entity_normalization(self):
        from paperbrain.graph import sanitize_relations
        kept, stats = sanitize_relations(
            [{"from": "flashattention 2", "rel": "Requires", "to": "ImageNet1k"}],
            {"FlashAttention-v2", "ImageNet-1k"})
        self.assertEqual(kept[0]["from"], "FlashAttention-v2")
        self.assertEqual(kept[0]["to"], "ImageNet-1k")
        self.assertEqual(stats["rejected_relation_endpoint"], 0)

    def test_per_paper_graph_rerun_replaces_the_complete_snapshot(self):
        import sqlite3
        from paperbrain.memory import save_sqlite
        db = str(Path(self.tmp, "snapshot.sqlite"))
        first_sections = [{"sec": "1", "chunks": [
            {"chunk_id": "P_Sec1_C001", "text": "old one"},
            {"chunk_id": "P_Sec1_C002", "text": "old two"}]}]
        first_entities = [
            {"name": "A", "type": "Algorithm/Model"},
            {"name": "B", "type": "Dataset/Benchmark"}]
        save_sqlite(db, "P", first_sections, first_entities,
                    [{"from": "A", "rel": "Evaluated_On", "to": "B"}])
        save_sqlite(db, "P", [{"sec": "1", "chunks": [
            {"chunk_id": "P_Sec1_C001", "text": "new one"}]}],
                    [{"name": "A", "type": "Algorithm/Model"}], [])
        with sqlite3.connect(db) as con:
            chunks = con.execute(
                "SELECT chunk_id,text FROM chunks WHERE paper_id='P'").fetchall()
            entities = con.execute(
                "SELECT name FROM entities WHERE paper_id='P'").fetchall()
            relations = con.execute(
                "SELECT rel FROM relations WHERE paper_id='P'").fetchall()
        self.assertEqual(chunks, [("P_Sec1_C001", "new one")])
        self.assertEqual(entities, [("A",)])
        self.assertEqual(relations, [])

    def test_similarity_merged_entity_keeps_relations_on_canonical_endpoint(self):
        from paperbrain.memory import build_memory
        sections = [{"name": "method", "sec": "2", "text": "source",
                     "chunks": [{"chunk_id": "P_Sec2_C001", "text": "source"}]}]
        def extracted(_):
            return ([{"name": "Canonical Model", "type": "Algorithm/Model"},
                     {"name": "Model Alias", "type": "Algorithm/Model"},
                     {"name": "Benchmark", "type": "Dataset/Benchmark"}],
                    [{"from": "Model Alias", "rel": "Evaluated_On", "to": "Benchmark"}])
        def similarity(a, b):
            return 0.99 if {a, b} == {"Canonical Model", "Model Alias"} else 0.0
        result = build_memory(sections, "P", str(Path(self.tmp, "canonical.sqlite")),
                              extract_fn=extracted, sim=similarity)
        self.assertEqual(result["relations"], [{
            "from": "Canonical Model", "rel": "Evaluated_On", "to": "Benchmark"}])
        self.assertEqual(result["stats"]["rejected_relation_endpoint"], 0)

    def test_graph_policy_reports_missing_review_and_loads_json(self):
        from datetime import datetime, timezone
        from unittest.mock import patch
        from paperbrain.graph import apply_alias, is_generic, load_graph_policy
        policy = Path(self.tmp, "policy.json")
        policy.write_text(json.dumps({
            "schema": "paperbrain-graph-policy-v1",
            "reviewed_at": "2026-09-10T00:00:00Z",
            "aliases": {"custom alias": "Canonical"},
            "blacklist": ["generic custom"],
        }), encoding="utf-8")
        with patch.dict(os.environ, {"PAPERBRAIN_GRAPH_POLICY": str(policy)}, clear=False):
            status = load_graph_policy(now=datetime(2026, 9, 12, tzinfo=timezone.utc))
            self.assertTrue(status["valid"])
            self.assertFalse(status["review_due"])
            self.assertEqual(apply_alias("CUSTOM ALIAS"), "Canonical")
            self.assertTrue(is_generic("Generic Custom"))

    def test_two_model_graph_rejections_create_manual_review_artifact(self):
        from unittest.mock import patch
        from paperbrain.pipeline import run_outline
        out = Path(self.tmp, "manual-queue")
        with patch.dict(os.environ, {"PAPERBRAIN_ALL_MODEL": "1"}, clear=False):
            os.environ.pop("PAPERBRAIN_CLOUD_ALLOWED", None)
            result = run_outline("demo/sample_paper.txt", "P", str(out), use_llm=True)
        queue = json.loads((out / "graph_review_queue.json").read_text(encoding="utf-8"))
        self.assertEqual(queue[0]["status"], "MANUAL_REVIEW")
        self.assertEqual(queue[0]["attempts"], 2)
        self.assertTrue(any("人工复核队列" in warning for warning in result["warnings"]))

    def test_known_contradiction_blocks_clean_release(self):
        from paperbrain.pipeline import run_outline, generate_from_outline
        out = Path(self.tmp, "contradiction")
        first = run_outline("demo/sample_paper.txt", "2024_NeurIPS_01", str(out), use_llm=False)
        mem_path = out / "memory.json"
        mem = json.loads(mem_path.read_text(encoding="utf-8"))
        mem["relations"] = [{"from": "FlashAttention-v2", "rel": "Contradicts",
                             "to": "Transformer", "ev": "conflicting assumptions"}]
        mem_path.write_text(json.dumps(mem), encoding="utf-8")
        draft = ("FlashAttention-v2 and Transformer are discussed together with source evidence "
                 "[Ref: 2024_NeurIPS_01, Sec 1].")
        result = generate_from_outline(
            str(out), "2024_NeurIPS_01", first["outline"], draft_override=draft,
            semantic_scorer=lambda claim, source: 0.99, use_llm=False)
        self.assertFalse(result["is_clean"])
        verify = json.loads((out / "verify.json").read_text(encoding="utf-8"))
        self.assertTrue(any(r["citation"] == "[CONTRADICTION]" for r in verify["report"]))

    def test_low_confidence_extraction_blocks_release_even_with_high_scorer(self):
        from paperbrain.pipeline import run_outline, generate_from_outline
        out = Path(self.tmp, "low-confidence")
        first = run_outline("demo/sample_paper.txt", "LOW", str(out), use_llm=False)
        state_path = out / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["downgrade"] = True
        state_path.write_text(json.dumps(state), encoding="utf-8")

        result = generate_from_outline(
            str(out), "LOW", first["outline"], semantic_scorer=lambda claim, source: 1.0,
            use_llm=False)

        verify = json.loads((out / "verify.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verify"], "DIRTY")
        self.assertFalse(result["is_clean"])
        self.assertTrue(any(row["citation"] == "[LOW_CONFIDENCE_EXTRACTION]"
                            for row in verify["report"]))

    def test_outline_identity_and_partial_unknown_chunks_are_not_silently_ignored(self):
        from paperbrain.pipeline import run_outline, generate_from_outline
        out = Path(self.tmp, "bad-outline")
        first = run_outline("demo/sample_paper.txt", "RIGHT", str(out), use_llm=False)
        outline = dict(first["outline"])
        outline["paper_id"] = "WRONG"
        sections = [dict(first["outline"]["sections"][0])]
        sections[0]["chunks"] = list(sections[0]["chunks"]) + ["RIGHT_Sec999_C999"]
        outline["sections"] = sections

        result = generate_from_outline(
            str(out), "RIGHT", outline, semantic_scorer=lambda claim, source: 1.0,
            use_llm=False)

        verify = json.loads((out / "verify.json").read_text(encoding="utf-8"))
        markers = {row["citation"] for row in verify["report"]}
        self.assertEqual(result["verify"], "DIRTY")
        self.assertIn("[INVALID_CHUNK_REFERENCE]", markers)
        self.assertIn("[OUTLINE_PAPER_MISMATCH]", markers)


class TestCacheCorrectness(unittest.TestCase):
    def test_retrieval_cache_uses_complete_text(self):
        from paperbrain.retrieval import make_scorer
        prefix = "shared " * 32
        alpha = prefix + ("alpha " * 20)
        omega = prefix + ("omega " * 20)
        self.assertEqual(len(alpha), len(omega))
        scorer = make_scorer({"a": alpha, "o": omega})
        high = scorer("alpha " * 8, alpha)
        low = scorer("alpha " * 8, omega)
        self.assertGreater(high, low)

    def test_nli_cache_uses_complete_text_and_does_not_cache_failure(self):
        import paperbrain.llm_ops as ops
        ops.reset_cache()
        original = ops._chat
        calls = []
        prefix = "x" * 400
        try:
            def different(prompt, max_tokens=1500):
                calls.append(prompt)
                return "91" if (prefix + "A") in prompt else "12"
            ops._chat = different
            self.assertEqual(ops.nli_score(prefix + "A", "source"), 0.91)
            self.assertEqual(ops.nli_score(prefix + "B", "source"), 0.12)
            self.assertEqual(len(calls), 2)

            ops.reset_cache()
            calls.clear()
            def transient(prompt, max_tokens=1500):
                calls.append(prompt)
                if len(calls) == 1:
                    raise RuntimeError("temporary")
                return "88"
            ops._chat = transient
            with self.assertRaisesRegex(RuntimeError, "NLI 模型打分失败"):
                ops.nli_score("claim with enough terms", "source with enough terms")
            self.assertEqual(ops.nli_score("claim with enough terms", "source with enough terms"), 0.88)
            self.assertEqual(len(calls), 2)
        finally:
            ops._chat = original
            ops.reset_cache()

    def test_embedding_response_index_and_dimension_guard(self):
        from paperbrain import embeddings
        old_post, old_available = embeddings._post, embeddings.available
        embeddings._Q_CACHE.clear()
        try:
            embeddings.available = lambda: True
            embeddings._post = lambda payload: {"data": [
                {"index": 1, "embedding": [2.0, 2.0]},
                {"index": 0, "embedding": [1.0, 1.0]},
            ]}
            self.assertEqual(embeddings.embed_texts(["first", "second"]),
                             [[1.0, 1.0], [2.0, 2.0]])
            self.assertEqual(embeddings.cosine([1.0], [1.0, 2.0]), 0.0)
        finally:
            embeddings._post, embeddings.available = old_post, old_available
            embeddings._Q_CACHE.clear()

    def test_llm_ledger_prefers_provider_usage(self):
        from paperbrain import llm
        original_post = llm._post
        old = {k: os.environ.get(k) for k in
               ("PAPERBRAIN_PROVIDER", "PAPERBRAIN_API_KEY", "PAPERBRAIN_BASE_URL",
                "PAPERBRAIN_CLOUD_ALLOWED")}
        os.environ["PAPERBRAIN_PROVIDER"] = "https"
        os.environ["PAPERBRAIN_API_KEY"] = "test-only"
        os.environ["PAPERBRAIN_BASE_URL"] = "http://example.invalid/v1"
        os.environ["PAPERBRAIN_CLOUD_ALLOWED"] = "1"
        try:
            llm._post = lambda *args, **kwargs: json.dumps({
                "choices": [{"message": {"content": "result"}}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 5}
            })
            llm.reset_usage()
            self.assertEqual(llm.chat([{"role": "user", "content": "prompt"}],
                                      usage_bucket="pass1"), "result")
            usage = llm.usage_snapshot()
            self.assertEqual(usage["total"], 22)
            self.assertEqual(usage["by_stage"]["pass1"], 22)
            self.assertEqual(usage["calls"], 1)
        finally:
            llm._post = original_post
            llm.reset_usage()
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_llm_refuses_before_sending_when_text_budget_is_exhausted(self):
        from unittest.mock import patch
        from paperbrain import llm
        old = {k: os.environ.get(k) for k in
               ("PAPERBRAIN_PROVIDER", "PAPERBRAIN_API_KEY", "PAPERBRAIN_CLOUD_ALLOWED")}
        os.environ["PAPERBRAIN_PROVIDER"] = "https"
        os.environ["PAPERBRAIN_API_KEY"] = "test-only"
        os.environ["PAPERBRAIN_CLOUD_ALLOWED"] = "1"
        try:
            llm.reset_usage()
            llm._record_usage("other", 11999, 0)
            with patch.object(llm, "_post") as post:
                with self.assertRaisesRegex(RuntimeError, "预算不足"):
                    llm.chat([{"role": "user", "content": "more input"}], retries=1)
            post.assert_not_called()
        finally:
            llm.reset_usage()
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_remote_models_require_explicit_cloud_authorization(self):
        from paperbrain import config, llm
        keys = ("PAPERBRAIN_API_KEY", "PAPERBRAIN_CLOUD_ALLOWED",
                "PAPERBRAIN_EMBED_BASE_URL")
        old = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["PAPERBRAIN_API_KEY"] = "test-only"
            os.environ.pop("PAPERBRAIN_CLOUD_ALLOWED", None)
            with self.assertRaises(llm.NoKeyError):
                llm.chat([{"role": "user", "content": "must not send"}])
            os.environ["PAPERBRAIN_EMBED_BASE_URL"] = "https://remote.invalid/v1"
            self.assertFalse(config.embed_allowed())
            os.environ["PAPERBRAIN_EMBED_BASE_URL"] = "http://127.0.0.1:5010/v1"
            self.assertTrue(config.embed_allowed())
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_embedding_health_does_not_probe_unauthorized_remote(self):
        from unittest.mock import patch
        from paperbrain import embeddings
        old = {k: os.environ.get(k) for k in
               ("PAPERBRAIN_EMBED_BASE_URL", "PAPERBRAIN_CLOUD_ALLOWED")}
        os.environ["PAPERBRAIN_EMBED_BASE_URL"] = "https://remote.invalid/v1"
        os.environ.pop("PAPERBRAIN_CLOUD_ALLOWED", None)
        try:
            with patch.object(embeddings.urllib.request, "urlopen") as call:
                result = embeddings.health()
            self.assertFalse(result["ok"])
            call.assert_not_called()
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestMemoryIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(dir="/tmp")
        os.environ["PAPERBRAIN_MEMORY_DB"] = str(Path(self.tmp, "memory.sqlite"))

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _artifacts(self):
        out = Path(self.tmp, "run")
        out.mkdir(exist_ok=True)
        (out / "deepread.md").write_text(
            "# 深读\n\n## 局限\nDBSCAN threshold evidence remains uncertain in this study.\n",
            encoding="utf-8")
        (out / "deepread_synthesis.json").write_text("{}", encoding="utf-8")
        (out / "memory.json").write_text("{\"entities\": []}", encoding="utf-8")
        (out / "sections.json").write_text(json.dumps([{
            "text": "This study reports DBSCAN threshold evidence remains uncertain."
        }]), encoding="utf-8")
        return out

    def test_rebuild_preserves_manual_notes_and_is_idempotent(self):
        from paperbrain import memory_store as ms
        ms.add_notes("P", "full_read", [{"kind": "manual", "content": "人工保留的重要判断 manual-anchor"}])
        out = self._artifacts()
        ms.build_notes_from_out(str(out), "P", "full_read")
        ms.build_notes_from_out(str(out), "P", "full_read")
        con = ms._conn()
        manual = con.execute("SELECT COUNT(*) FROM notes WHERE kind='manual'").fetchone()[0]
        generated = con.execute("SELECT COUNT(*) FROM notes WHERE tags LIKE ?",
                                ('%"paperbrain:generated"%',)).fetchone()[0]
        con.close()
        self.assertEqual(manual, 1)
        self.assertEqual(generated, 1)

    def test_pending_notes_do_not_build_concepts(self):
        from paperbrain import memory_store as ms
        ms.add_notes("P", "t", [
            {"kind": "claim", "content": "verifiedquasar mechanism evidence", "status": "active"},
            {"kind": "claim", "content": "forbiddennebula unsupported assertion", "status": "candidate"},
        ])
        ms.build_concepts("P")
        con = ms._conn()
        concepts = {r[0] for r in con.execute("SELECT concept FROM concepts WHERE paper_id='P'")}
        con.close()
        self.assertTrue(any("verifiedquasar" in c for c in concepts))
        self.assertFalse(any("forbiddennebula" in c for c in concepts))

    def test_unverified_artifact_cannot_seed_active_memory(self):
        from paperbrain import memory_store as ms
        out = self._artifacts()
        (out / "deepread_verify.json").write_text(json.dumps({
            "status": "NEEDS_REVIEW", "is_clean": False, "report": [],
            "artifacts": {
                "brief": {"status": "NEEDS_REVIEW", "is_clean": False, "report": []},
                "full": {"status": "CLEAN", "is_clean": True, "report": []},
            },
        }), encoding="utf-8")

        ms.build_notes_from_out(str(out), "P", "full_read")

        notes = ms.search_notes("", include_pending=True)
        generated = [n for n in notes if n["kind"] == "局限"]
        self.assertTrue(generated)
        self.assertEqual(generated[0]["status"], "candidate")
        self.assertFalse(ms.search_notes("DBSCAN threshold"))

    def test_clean_artifact_and_supported_claim_can_seed_active_memory(self):
        from paperbrain import memory_store as ms
        out = self._artifacts()
        (out / "deepread_verify.json").write_text(json.dumps({
            "status": "CLEAN", "is_clean": True, "report": [],
            "artifacts": {
                "brief": {"status": "CLEAN", "is_clean": True, "report": []},
                "full": {"status": "CLEAN", "is_clean": True, "report": []},
            },
        }), encoding="utf-8")

        ms.build_notes_from_out(str(out), "P", "full_read")

        notes = ms.search_notes("DBSCAN threshold")
        generated = [n for n in notes if n["kind"] == "局限"]
        self.assertTrue(generated)
        self.assertEqual(generated[0]["status"], "active")

    def test_current_graph_policy_and_source_binding_are_required_for_active_entity(self):
        from datetime import datetime, timezone
        from unittest.mock import patch
        from paperbrain import memory_store as ms
        out = self._artifacts()
        (out / "memory.json").write_text(json.dumps({
            "entities": [{"name": "DBSCAN", "type": "Algorithm/Model"},
                         {"name": "BERT", "type": "Algorithm/Model"}],
        }), encoding="utf-8")
        (out / "sections.json").write_text(json.dumps([{
            "text": "This study reports DBSCAN evidence and compares RoBERTa baselines."
        }]), encoding="utf-8")
        (out / "deepread_verify.json").write_text(json.dumps({
            "artifacts": {
                "brief": {"status": "CLEAN"}, "full": {"status": "CLEAN"},
            },
        }), encoding="utf-8")
        policy = Path(self.tmp, "current-policy.json")
        policy.write_text(json.dumps({
            "schema": "paperbrain-graph-policy-v1",
            "reviewed_at": "2026-09-12T00:00:00Z", "aliases": {}, "blacklist": [],
        }), encoding="utf-8")
        with patch.dict(os.environ, {"PAPERBRAIN_GRAPH_POLICY": str(policy)}, clear=False):
            with patch("paperbrain.graph.datetime") as clock:
                clock.now.return_value = datetime(2026, 9, 12, 1, tzinfo=timezone.utc)
                clock.fromisoformat.side_effect = datetime.fromisoformat
                ms.build_notes_from_out(str(out), "P", "full_read")

        entities = {n["content"].split("（", 1)[0]: n["status"]
                    for n in ms.search_notes("", include_pending=True)
                    if n["kind"] == "entity"}
        self.assertEqual(entities["DBSCAN"], "active")
        self.assertEqual(entities["BERT"], "candidate")

    def test_link_missing_note_fails_closed(self):
        from paperbrain import memory_store as ms
        result = ms.link_notes(99991, 99992, "supports")
        self.assertFalse(result["ok"])


class TestContracts(unittest.TestCase):
    def test_cache_key_rejects_missing_fifth_field(self):
        from paperbrain.ids import make_cache_key
        with self.assertRaises(ValueError):
            make_cache_key("pdf", "section", "prompt", "model")

    def test_multiple_sections_in_one_reference_are_rejected(self):
        from paperbrain.ids import parse_loc
        from paperbrain.verifier import CitationVerifierV5
        self.assertTrue(parse_loc("Sec 1, Sec 2")[3])
        result = CitationVerifierV5(
            {"P_1": "one", "P_2": "two"}, semantic_scorer=lambda a, b: 1.0
        ).verify_draft("Ambiguous source [Ref: P, Sec 1, Sec 2].")
        self.assertEqual(result["status"], "DIRTY")

    def test_outline_retains_all_section_chunks(self):
        from paperbrain.outline import build_outline
        chunks = {f"P_Sec2_C{i:03d}": f"method evidence {i}" for i in range(1, 7)}
        passes = {"ground_truth": chunks, "name_sec": {"method": ["2"]},
                  "pass1": "", "pass2": "method", "pass3": "", "pass4": ""}
        outline = build_outline(passes, {"entities": []}, "P")
        method = next(s for s in outline["sections"] if s["h1"] == "Method")
        self.assertEqual(method["chunks"], list(chunks))

    def test_downgrade_outline_deduplicates_evidence_and_does_not_invent_sections(self):
        from paperbrain.outline import build_outline
        chunks = {f"P_Sec{i}_C001": f"evidence {i}" for i in range(3)}
        coarse = {f"P_{i}": f"evidence {i}" for i in range(3)}
        passes = {"ground_truth": {**chunks, **coarse},
                  "name_sec": {"intro": ["0"], "method": ["1"], "experiments": ["2"]},
                  "pass1": "low confidence overview", "pass2": "", "pass3": "", "pass4": "",
                  "downgrade_pass1_only": True}

        outline = build_outline(passes, {"entities": []}, "P")

        self.assertEqual(len(outline["sections"]), 1)
        section = outline["sections"][0]
        self.assertEqual(section["chunks"], list(chunks))
        self.assertEqual(section["status"], "NEEDS_REVIEW")
        self.assertIn("Low-confidence", section["h1"])

    def test_even_reduce_preserves_tail_with_many_windows(self):
        from paperbrain.passes import _even_reduce
        parts = [f"{i:03d} segment evidence" for i in range(50)]
        reduced = _even_reduce(parts, 200)
        self.assertIn("000", reduced)
        self.assertIn("025", reduced)
        self.assertIn("049", reduced)
        self.assertLessEqual(len(reduced), 200)

    def test_deepread_context_can_reach_late_chunks(self):
        from paperbrain.context import build_paper_context
        temp = Path(tempfile.mkdtemp(dir="/tmp"))
        try:
            sections = [{"name": "method", "sec": "2", "text": "early only",
                         "chunks": [
                             {"chunk_id": "P_Sec2_C001", "text": "early generic material."},
                             {"chunk_id": "P_Sec2_C002",
                              "text": "latechunk_unique_target explains calibration evidence."},
                         ]}]
            (temp / "sections.json").write_text(json.dumps(sections), encoding="utf-8")
            (temp / "memory.json").write_text("{}", encoding="utf-8")
            context = build_paper_context(str(temp), budget=300, focus="latechunk_unique_target")
            self.assertIn("latechunk_unique_target", context)
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def test_brief_reads_top_level_field_view(self):
        from paperbrain.deepread import _brief
        text = _brief("P", "## 一句话主张\n结论。", "", {
            "positioning": {}, "field_view": "Top-level field view"
        }, [], "")
        self.assertIn("Top-level field view", text)

    def test_long_uncited_paragraph_is_dirty(self):
        from paperbrain.verifier import CitationVerifierV5
        gt = {"P_1": "supported source"}
        draft = ("Supported statement [Ref: P, Sec 1].\n\n" +
                 "这是一个没有任何引用的长段落。" * 25)
        result = CitationVerifierV5(gt, semantic_scorer=lambda a, b: 0.99).verify_draft(draft)
        self.assertEqual(result["status"], "DIRTY")
        self.assertTrue(any(r["citation"] == "[UNCITED_LONG_PARAGRAPH]"
                            for r in result["report"]))

    def test_term_consistency_ratio(self):
        from paperbrain.consistency import term_consistency_ratio, unify_terms
        dirty = "flashattention 2 and FlashAttention-v2"
        self.assertLess(term_consistency_ratio(dirty), 0.99)
        clean, _ = unify_terms(dirty)
        self.assertEqual(term_consistency_ratio(clean), 1.0)


class TestBatchAndVisionLimits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(dir="/tmp")

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MAX_IMAGES", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_basename_gets_distinct_batch_ids(self):
        from paperbrain.batch import run_batch
        paths = []
        for name in ("a", "b"):
            d = Path(self.tmp, name)
            d.mkdir()
            p = d / "paper.txt"
            p.write_text("Abstract\nEnough source text.\n1. Introduction\nDBSCAN evidence.",
                         encoding="utf-8")
            paths.append(str(p))
        result = run_batch([{"path": p} for p in paths], str(Path(self.tmp, "out")),
                           task="outline", depth="fast", use_llm=False)
        ids = [row["paper_id"] for row in result["results"]]
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(all((Path(self.tmp, "out") / pid).exists() for pid in ids))

    def test_figure_task_honors_per_paper_cap_of_five(self):
        import paperbrain.vision as vision
        from paperbrain.tasks import _run_figures
        original = vision.extract_images
        seen = []
        os.environ["PAPERBRAIN_MAX_IMAGES"] = "9"
        try:
            vision.extract_images = lambda src, out, max_images: seen.append(max_images) or []
            out = Path(self.tmp, "figures")
            out.mkdir()
            _run_figures(str(Path(self.tmp, "fake.pdf")), "P", str(out), False)
        finally:
            vision.extract_images = original
        self.assertEqual(seen, [5])

    def test_figure_task_sends_caption_hint_with_each_selected_image(self):
        import paperbrain.vision as vision
        from paperbrain.tasks import _run_figures
        original_extract = vision.extract_images
        original_describe = vision.describe_figure
        hints = []
        try:
            vision.extract_images = lambda *a, **k: [{
                "image": "/tmp/private-safe.png", "page": 0, "pixels": 90000,
                "caption": "Figure 2. Ablation across datasets."
            }]
            vision.describe_figure = lambda image, hint="", **k: (
                hints.append((image, hint)) or {"analysis": "caption-aware"})
            out = Path(self.tmp, "caption-hint")
            out.mkdir()
            _run_figures(str(Path(self.tmp, "fake.pdf")), "P", str(out), True)
        finally:
            vision.extract_images = original_extract
            vision.describe_figure = original_describe
        self.assertEqual(hints, [("/tmp/private-safe.png",
                                  "Figure 2. Ablation across datasets.")])

    def test_server_job_submission_is_bounded(self):
        import paperbrain.server as server
        original_limit = server.config.job_queue_limit
        original_submitted = server._SUBMITTED
        try:
            server.config.job_queue_limit = lambda: 2
            server._SUBMITTED = 2
            jid = server._job_new("P")
            self.assertFalse(server._submit_job(jid, lambda: None))
            with server.JOBS_LOCK:
                self.assertTrue(server.JOBS[jid]["done"])
                self.assertFalse(server.JOBS[jid]["ok"])
        finally:
            server.config.job_queue_limit = original_limit
            server._SUBMITTED = original_submitted
            with server.JOBS_LOCK:
                server.JOBS.clear()

    def test_max_images_is_hard_capped_at_five(self):
        from paperbrain import config
        os.environ["PAPERBRAIN_MAX_IMAGES"] = "999"
        self.assertEqual(config.max_images(), 5)


class TestProductionGates(unittest.TestCase):
    def test_m1_uses_exact_levenshtein_not_sequence_matcher(self):
        from tools.score_m1 import text_score
        self.assertAlmostEqual(text_score("aaaaab", "baaaaa"), 2 / 3, places=6)

    def test_m1_long_text_fails_closed_without_fast_exact_backend(self):
        from unittest.mock import patch
        import tools.score_m1 as score_m1
        with patch.object(score_m1, "_LEVENSHTEIN", None):
            with self.assertRaisesRegex(RuntimeError, "rapidfuzz"):
                score_m1.text_score("a" * 2100, "b" * 2100)

    def test_m1_rejects_self_reported_table_counts_and_boolean_caption(self):
        from tools.score_m1 import evaluate
        categories = (["two_column"] * 10 + ["scanned"] * 10 +
                      ["complex_figures"] * 10)
        gold, pred = [], []
        for i, category in enumerate(categories):
            base = {"id": f"P{i}", "category": category, "text": "same",
                    "tables": [], "captions": []}
            gold.append(base)
            pred.append(dict(base))
        gold[0]["tables"] = [{"table_id": "tab1", "cells": [["A", "B"]]}]
        pred[0]["tables"] = [{"table_id": "tab1", "cells_ok": 2, "cells_total": 2}]
        gold[0]["captions"] = [{"figure_id": "fig1", "caption": "Figure 1",
                                "bound_contexts": ["correct body context"]}]
        pred[0]["captions"] = [{"figure_id": "fig1", "caption": "Figure 1",
                                "bound_contexts": True}]
        result = evaluate(gold, pred)
        self.assertEqual(result["result"], "FAIL")
        self.assertFalse(result["gate"]["structure_complete"])
        self.assertTrue(result["structure_errors"])

    def test_production_scorers_reject_malformed_container_types(self):
        from tools.score_m1 import evaluate as score_m1
        from tools.score_m3 import evaluate as score_m3
        from tools.score_m5 import evaluate as score_m5
        sha = "a" * 64
        with self.assertRaisesRegex(ValueError, "captions"):
            score_m1([{"id": "P", "source_sha256": sha, "text": "x",
                       "captions": None}],
                     [{"id": "P", "source_sha256": sha, "text": "x"}],
                     required_papers=1, required_categories={})
        with self.assertRaisesRegex(ValueError, "JSON list"):
            score_m3([{"mention": "x", "canonical": "x",
                       "type": "Algorithm/Model"}], {})
        with self.assertRaisesRegex(ValueError, "JSON list"):
            score_m5([{"id": "x", "expected": True}], {})

    def test_m1_scores_actual_cells_and_caption_context(self):
        from tools.score_m1 import score_pair
        gold = {"text": "same", "formulas": [],
                "tables": [{"table_id": "tab1", "cells": [["A", "B"], ["1", "2"]]}],
                "captions": [{"figure_id": "fig1", "caption": "Figure 1",
                              "bound_contexts": ["expected context"]}]}
        pred = {"text": "same", "formulas": [],
                "tables": [{"table_id": "tab1", "cells": [["A", "wrong"], ["1", "2"]]}],
                "captions": [{"figure_id": "fig1", "caption": "Figure 1",
                              "bound_contexts": ["wrong context"]}]}
        result = score_pair(gold, pred)
        self.assertEqual(result["table"], 0.75)
        self.assertEqual(result["caption_bind"], 0.0)

    def test_m1_table_precision_penalizes_hallucinated_cells_and_tables(self):
        from tools.score_m1 import score_pair
        gold = {"text": "same", "formulas": [],
                "tables": [{"table_id": "tab1", "cells": [["A", "B"]]}],
                "captions": []}
        pred = {"text": "same", "formulas": [],
                "tables": [{"table_id": "tab1", "cells": [["A", "B", "extra"]]},
                           {"table_id": "tab2", "cells": [["fake"]]}],
                "captions": []}
        self.assertEqual(score_pair(gold, pred)["table"], 0.5)

    def test_m1_caption_context_accepts_long_line_inside_human_paragraph(self):
        from tools.score_m1 import score_pair
        line = "As shown in Figure 2, accuracy rises by ten points on the held-out set."
        gold = {"text": "same", "formulas": [], "tables": [],
                "captions": [{"figure_id": "fig2", "caption": "Figure 2",
                              "bound_contexts": ["Prior sentence. " + line + " Next sentence."]}]}
        pred = {"text": "same", "formulas": [], "tables": [],
                "captions": [{"figure_id": "fig2", "caption": "Figure 2",
                              "bound_contexts": [line]}]}
        self.assertEqual(score_pair(gold, pred)["caption_bind"], 1.0)

    def test_m1_caption_binding_penalizes_extra_predicted_figure(self):
        from tools.score_m1 import score_pair
        gold = {"text": "same", "formulas": [], "tables": [],
                "captions": [{"figure_id": "fig1", "caption": "Figure 1",
                              "bound_contexts": ["correct body context for figure one"]}]}
        pred = {"text": "same", "formulas": [], "tables": [],
                "captions": [{"figure_id": "fig1", "caption": "Figure 1",
                              "bound_contexts": ["correct body context for figure one"]},
                             {"figure_id": "fig2", "caption": "Figure 2",
                              "bound_contexts": ["hallucinated context for figure two"]}]}
        self.assertEqual(score_pair(gold, pred)["caption_bind"], 0.5)

    def test_m1_release_decision_uses_unrounded_text_score_and_source_hash(self):
        from unittest.mock import patch
        from tools import score_m1
        sha = "a" * 64
        gold = [{"id": "P", "source_sha256": sha, "text": "gold",
                 "tables": [{"table_id": "tab1", "cells": [["A"]]}],
                 "captions": [{"figure_id": "fig1", "caption": "Figure 1",
                               "bound_contexts": ["context"]}]}]
        pred = [{"id": "P", "source_sha256": sha, "text": "pred",
                 "tables": [{"table_id": "tab1", "cells": [["A"]]}],
                 "captions": [{"figure_id": "fig1", "caption": "Figure 1",
                               "bound_contexts": ["context"]}]}]
        with patch.object(score_m1, "text_score", return_value=0.97996):
            result = score_m1.evaluate(gold, pred, required_papers=1,
                                       required_categories={})
        self.assertEqual(result["avg"]["text"], 0.98)
        self.assertEqual(result["result"], "FAIL")
        self.assertTrue(result["gate"]["source_bound"])

    def test_m1_prediction_is_generated_by_pipeline_and_hides_source_path(self):
        from tools.predict_m1 import generate
        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td, "pred.json")
            rows = generate([{"id": "DEMO", "path": "demo/sample_paper.txt"}],
                            str(pred_path), str(Path(td, "runs")))
            saved = json.loads(pred_path.read_text(encoding="utf-8"))
        self.assertEqual(rows, saved)
        self.assertEqual(rows[0]["id"], "DEMO")
        self.assertEqual(len(rows[0]["source_sha256"]), 64)
        self.assertIn("text", rows[0])
        self.assertNotIn("path", rows[0])

    def test_m1_prediction_forces_offline_mode_and_restores_environment(self):
        from unittest.mock import patch
        from tools.predict_m1 import generate
        with tempfile.TemporaryDirectory() as td, patch.dict(
                os.environ, {"PAPERBRAIN_VISION": "1", "PAPERBRAIN_CLOUD_ALLOWED": "1",
                             "PAPERBRAIN_ALL_MODEL": "1", "PAPERBRAIN_PROVIDER": "https"},
                clear=False):
            observed = []

            def vision_enabled():
                observed.append(os.environ.get("PAPERBRAIN_VISION"))
                return os.environ.get("PAPERBRAIN_VISION") == "1"

            with patch("paperbrain.pipeline.config.vision_enabled",
                       side_effect=vision_enabled):
                generate([{"id": "OFFLINE", "path": "demo/sample_paper.txt"}],
                         str(Path(td, "pred.json")))
            self.assertEqual(observed, ["0"])
            self.assertEqual(os.environ["PAPERBRAIN_VISION"], "1")
            self.assertEqual(os.environ["PAPERBRAIN_CLOUD_ALLOWED"], "1")

    def test_vector_retrieval_never_uses_python_blob_scan(self):
        import inspect
        from paperbrain import memory_store
        source = inspect.getsource(memory_store._vector_ids)
        self.assertIn("vector_store.search", source)
        self.assertNotIn("embeddings.cosine", source)
        import paperbrain.vector_store as vector_store
        upsert_source = inspect.getsource(vector_store.upsert)
        self.assertIn("duckdb_indexes()", upsert_source)
        self.assertNotIn("CREATE INDEX IF NOT EXISTS", upsert_source)

    def test_embed_notes_excludes_nonretrievable_and_entity_notes(self):
        from unittest.mock import patch
        from paperbrain import embeddings, memory_store, vector_store
        with tempfile.TemporaryDirectory(dir="/tmp") as td, patch.dict(
                os.environ, {"PAPERBRAIN_MEMORY_DB": str(Path(td, "memory.sqlite")),
                             "PAPERBRAIN_VECTOR_DB": str(Path(td, "vectors.duckdb"))},
                clear=False):
            memory_store.add_notes("PV", "t", [
                {"kind": "claim", "content": "active scientific claim"},
                {"kind": "claim", "content": "candidate scientific claim", "status": "candidate"},
                {"kind": "entity", "content": "Entity Scientific Name"},
            ])
            indexed = []
            with patch.object(embeddings, "available", return_value=True), \
                    patch.object(embeddings, "storage_model", return_value="mock"), \
                    patch.object(embeddings, "embed_texts",
                                 side_effect=lambda texts, batch=32: [[1.0, 0.0] for _ in texts]), \
                    patch.object(vector_store, "upsert",
                                 side_effect=lambda model, rows: indexed.extend(list(rows)) or
                                 {"ok": True, "indexed": 1}):
                result = memory_store.embed_notes("PV")
        self.assertEqual(result["embedded"], 1)
        self.assertEqual(len(indexed), 1)

    def test_embed_notes_repairs_partial_vss_from_sqlite_without_remote_reembedding(self):
        from unittest.mock import patch
        from paperbrain import embeddings, memory_store, vector_store
        with tempfile.TemporaryDirectory(dir="/tmp") as td, patch.dict(
                os.environ, {"PAPERBRAIN_MEMORY_DB": str(Path(td, "memory.sqlite")),
                             "PAPERBRAIN_VECTOR_DB": str(Path(td, "vectors.duckdb"))},
                clear=False):
            memory_store.add_notes("PV", "t", [
                {"kind": "claim", "content": "active scientific claim for repair"}])
            with patch.object(embeddings, "available", return_value=True), \
                    patch.object(embeddings, "storage_model", return_value="mock"), \
                    patch.object(embeddings, "embed_texts", return_value=[[1.0, 0.0]]), \
                    patch.object(vector_store, "available", return_value=False), \
                    patch.object(vector_store, "upsert",
                                 return_value={"ok": False, "indexed": 0}):
                first = memory_store.embed_notes("PV")
            repaired = []
            with patch.object(embeddings, "available", return_value=True), \
                    patch.object(embeddings, "storage_model", return_value="mock"), \
                    patch.object(embeddings, "embed_texts") as remote, \
                    patch.object(vector_store, "available", return_value=True), \
                    patch.object(vector_store, "content_hashes", return_value={}), \
                    patch.object(vector_store, "upsert",
                                 side_effect=lambda model, rows: repaired.extend(list(rows)) or
                                 {"ok": True, "indexed": 1}):
                second = memory_store.embed_notes("PV")
        self.assertEqual(first["embedded"], 1)
        remote.assert_not_called()
        self.assertEqual(second["embedded"], 0)
        self.assertEqual(second["vector_index"]["indexed"], 1)
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0][2], [1.0, 0.0])

    def test_release_gate_fails_closed_when_external_evidence_is_missing(self):
        from tools.release_gate import evaluate_release
        result = evaluate_release(run_tests=False)
        self.assertEqual(result["result"], "FAIL")
        self.assertEqual(result["passed"], ["M2", "M7"])
        self.assertEqual(result["failed"], ["M1", "M3", "M4", "M5", "M6", "M8"])

    def test_m8_requires_hash_bound_independent_human_review(self):
        import hashlib
        from tools.validate_output_quality import evaluate

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = []
            tasks = ["full_read"] * 10 + ["method"] * 5 + ["review"] * 5
            for index, task in enumerate(tasks):
                source = root / f"source-{index}.txt"
                artifact = root / f"artifact-{index}.md"
                source.write_text(f"source evidence {index} " * 30, encoding="utf-8")
                artifact.write_text(f"# reviewed output {index}\n\n" +
                                    (f"Evidence-bound analysis {index}. " * 20), encoding="utf-8")
                scores = {name: 4 for name in (
                    "factual_accuracy", "evidence_traceability", "learning_objective_quality",
                    "insight_depth", "actionability", "language_coherence")}
                reviews = [{"reviewer_id": f"reviewer-{reviewer}", "attestation": "human",
                            "reviewed_at": "2026-09-12T10:00:00+08:00",
                            "scores": scores, "blocking_issues": [],
                            "notes": f"独立检查产物 {index} 的证据和学习目标。"}
                           for reviewer in (1, 2)]
                cases.append({"id": f"case-{index}", "task": task,
                              "source_path": str(source), "artifact_path": str(artifact),
                              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                              "reviews": reviews})
            report = {"schema": "paperbrain-output-quality-v1",
                      "protocol": "independent-double-review-v1", "cases": cases}
            passed = evaluate(report, base_dir=root)
            self.assertEqual(passed["result"], "PASS", passed)

            cases[0]["reviews"][1]["reviewer_id"] = "reviewer-1"
            cases[1]["artifact_sha256"] = "0" * 64
            failed = evaluate(report, base_dir=root)
            self.assertEqual(failed["result"], "FAIL")
            self.assertTrue(any("reviewer_id" in error for error in failed["errors"]))
            self.assertTrue(any("artifact_sha256" in error for error in failed["errors"]))

    def test_m8_initializer_binds_files_without_faking_human_scores(self):
        import hashlib
        from tools.validate_output_quality import evaluate, initialize
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.txt"
            artifact = root / "artifact.md"
            source.write_text("source evidence " * 30, encoding="utf-8")
            artifact.write_text("reviewable output " * 30, encoding="utf-8")
            report = initialize([{"id": "one", "task": "full_read",
                                  "source_path": "source.txt", "artifact_path": "artifact.md"}],
                                base_dir=root)
            case = report["cases"][0]
            self.assertEqual(case["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertIsNone(case["reviews"][0]["scores"]["factual_accuracy"])
            self.assertEqual(case["reviews"][0]["blocking_issues"], ["REVIEW_REQUIRED"])
            self.assertIsNot(case["reviews"][0]["scores"], case["reviews"][1]["scores"])
            result = evaluate(report, min_cases=1, min_distinct_sources=1,
                              min_tasks={"full_read": 1})
            self.assertEqual(result["result"], "FAIL")

    def test_release_gate_regenerates_m5_and_ignores_forged_external_prediction(self):
        from tools.release_gate import evaluate_release
        cases = []
        for i in range(20):
            cases.append({"id": f"p{i}", "expected": True,
                          "draft": f"Apos{i} and Bpos{i} are compared.",
                          "relations": [{"from": f"Apos{i}", "to": f"Bpos{i}",
                                         "rel": "Contradicts"}]})
        for i in range(10):
            cases.append({"id": f"n{i}", "expected": False,
                          "draft": f"Aneg{i} appears alone.",
                          "relations": [{"from": f"Aneg{i}", "to": f"Bneg{i}",
                                         "rel": "Contradicts"}]})
        with tempfile.TemporaryDirectory() as td:
            golden = Path(td, "m5.json")
            golden.write_text(json.dumps(cases), encoding="utf-8")
            result = evaluate_release(m5_golden=str(golden),
                                      m5_pred=str(Path(td, "forged-does-not-exist.json")),
                                      run_tests=False)
        self.assertEqual(result["gates"]["M5"]["result"], "PASS")
        self.assertEqual(result["gates"]["M5"]["prediction_source"],
                         "generated-current-code")

    def test_release_gate_regenerates_m1_from_source_and_ignores_external_prediction(self):
        from paperbrain.preflight import preflight
        from tools.release_gate import evaluate_release
        source = Path("demo/sample_paper.txt").resolve()
        parsed = preflight(str(source), "P")
        golden_row = {"id": "P", "path": str(source), "source_sha256": parsed.sha,
                      "category": "two_column", "text": parsed.text,
                      "formulas": [], "tables": [], "captions": []}
        with tempfile.TemporaryDirectory() as td:
            golden = Path(td, "m1.json")
            golden.write_text(json.dumps([golden_row]), encoding="utf-8")
            result = evaluate_release(m1_golden=str(golden),
                                      m1_pred=str(Path(td, "forged-does-not-exist.json")),
                                      run_tests=False)
        gate = result["gates"]["M1"]
        self.assertEqual(gate["prediction_source"], "generated-current-code")
        self.assertEqual(gate["source_errors"], [])
        self.assertEqual(gate["n_pred"], 1)

    def test_pass4_receives_real_chunk_citation_chain(self):
        from paperbrain.passes import run_passes
        calls = {}

        def summarize(kind, text):
            calls[kind] = text
            return f"{kind} summary"

        sections = [{"name": "method", "sec": "2", "text": "method evidence",
                     "confidence": 1.0, "downgrade_pass1_only": False,
                     "chunks": [{"chunk_id": "P_Sec2_C001", "text": "method evidence"}]}]
        result = run_passes(sections, "P", summarize_fn=summarize)
        self.assertIn("[Source: P_Sec2_C001]", calls["pass4"])
        self.assertTrue(result["citation_chain"])

    def test_downgrade_really_runs_pass1_only(self):
        from paperbrain.passes import run_passes
        calls = []
        sections = [{"name": "full", "sec": "0", "text": "source text",
                     "confidence": 0.2, "downgrade_pass1_only": True,
                     "chunks": [{"chunk_id": "P_Sec0_C001", "text": "source text"}]}]
        result = run_passes(
            sections, "P", summarize_fn=lambda kind, text: calls.append(kind) or "summary")
        self.assertEqual(calls, ["pass1"])
        self.assertEqual(result["pass4"], "")

    def test_low_confidence_section_forces_whole_document_downgrade(self):
        from paperbrain.passes import run_passes
        sections = [{"name": "full", "sec": "0", "text": "uncertain source",
                     "confidence": 0.49, "downgrade_pass1_only": False,
                     "chunks": [{"chunk_id": "P_Sec0_C001", "text": "uncertain source"}]}]
        result = run_passes(sections, "P")
        self.assertTrue(result["downgrade_pass1_only"])
        self.assertEqual(result["pass2"], "")
        self.assertEqual(result["pass3"], "")
        self.assertEqual(result["pass4"], "")

    def test_m1_gate_cannot_score_only_the_overlap(self):
        from tools.score_m1 import evaluate
        gold = [{"id": "A", "text": "alpha"}, {"id": "B", "text": "beta"}]
        pred = [{"id": "A", "text": "alpha"}]
        result = evaluate(gold, pred, required_papers=2, required_categories={})
        self.assertEqual(result["result"], "FAIL")
        self.assertEqual(result["missing_pred"], ["B"])
        self.assertEqual(result["avg"]["text"], 0.5)

    def test_m1_gate_requires_full_dataset_size(self):
        from tools.score_m1 import evaluate
        row = {"id": "A", "text": "alpha"}
        result = evaluate([row], [row], required_papers=2, required_categories={})
        self.assertEqual(result["result"], "FAIL")
        self.assertFalse(result["gate"]["dataset_complete"])

    def test_m1_gate_requires_all_three_document_categories(self):
        from tools.score_m1 import evaluate
        rows = [{"id": f"P{i}", "text": "same", "category": "two_column"}
                for i in range(30)]
        result = evaluate(rows, rows)
        self.assertEqual(result["result"], "FAIL")
        self.assertFalse(result["gate"]["dataset_complete"])

    def test_m1_gate_rejects_repeated_sources_and_empty_complex_figure_labels(self):
        from tools.score_m1 import evaluate
        categories = (["two_column"] * 10 + ["scanned"] * 10 +
                      ["complex_figures"] * 10)
        rows = [{"id": f"P{i}", "source_sha256": "a" * 64, "text": "paper text",
                 "category": category, "tables": [], "captions": []}
                for i, category in enumerate(categories)]
        result = evaluate(rows, rows)
        self.assertEqual(result["result"], "FAIL")
        self.assertFalse(result["gate"]["source_bound"])
        self.assertEqual(len(result["dataset_errors"]), 10)

    def test_calibration_record_requires_50_traceable_pairs(self):
        from paperbrain.calibration import load_record
        from tools.calibrate import calibrate
        rows = []
        for i in range(25):
            rows.append({"claim": f"supported claim {i}", "source": f"supported source {i}",
                         "label": True, "score": 0.9})
            rows.append({"claim": f"unsupported claim {i}", "source": f"different source {i}",
                         "label": False, "score": 0.1})
        record = calibrate(rows, "tfidf-corpus", "v5.0")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "thresholds.json")
            path.write_text(json.dumps(record), encoding="utf-8")
            checked = load_record("tfidf-corpus", path)
        self.assertTrue(checked["verified"])
        self.assertEqual(len(record["dataset_sha256"]), 64)
        self.assertEqual(record["roc_auc"], 1.0)

    def test_uninformative_calibration_data_cannot_authorize_clean_release(self):
        from paperbrain.calibration import load_record
        from tools.calibrate import calibrate
        rows = ([{"claim": f"positive {i}", "source": "same", "label": True,
                  "score": 0.5} for i in range(25)] +
                [{"claim": f"negative {i}", "source": "same", "label": False,
                  "score": 0.5} for i in range(25)])
        record = calibrate(rows, "tfidf-corpus", "v5.0")
        self.assertFalse(record["calibrated"])
        self.assertEqual(record["roc_auc"], 0.5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "thresholds.json")
            path.write_text(json.dumps(record), encoding="utf-8")
            checked = load_record("tfidf-corpus", path)
        self.assertFalse(checked["verified"])

    def test_repository_demo_threshold_is_not_production_calibration(self):
        from paperbrain.calibration import load_record
        self.assertFalse(load_record("tfidf-corpus")["verified"])

    def test_m3_and_m5_scorers_count_missing_as_failures(self):
        from tools.score_m3 import evaluate as score_m3
        from tools.score_m5 import evaluate as score_m5
        m3 = score_m3(
            [{"mention": "FA2", "canonical": "FlashAttention-v2", "type": "Algorithm/Model"}],
            [])
        self.assertEqual(m3["result"], "FAIL")
        m5 = score_m5(
            [{"id": "c1", "expected": True}, {"id": "c2", "expected": True}],
            [{"id": "c1", "flagged": True}])
        self.assertEqual(m5["result"], "FAIL")
        self.assertEqual(m5["recall"], 0.5)

    def test_m3_and_m5_small_or_type_ambiguous_goldens_cannot_pass(self):
        from tools.score_m3 import evaluate as score_m3
        from tools.score_m5 import evaluate as score_m5
        m3 = score_m3(
            [{"mention": "FA2", "canonical": "FlashAttention-v2",
              "type": "Algorithm/Model"}],
            [{"mention": "FA2", "canonical": "FlashAttention-v2",
              "type": "Algorithm/Model"}])
        self.assertEqual(m3["result"], "FAIL")
        self.assertFalse(m3["gate"]["dataset_complete"])
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            score_m5([{"id": "x", "expected": "false"}],
                     [{"id": "x", "flagged": "false"}])

    def test_m3_prediction_is_bound_to_cases_and_current_policy(self):
        from tools.predict_m3 import generate
        from tools.score_m3 import evaluate
        aliases = [("flashattention 2", "FlashAttention-v2"),
                   ("flashattention-2", "FlashAttention-v2"),
                   ("imagenet1k", "ImageNet-1k"),
                   ("imagenet-1k", "ImageNet-1k"),
                   ("bleu4", "BLEU-4"), ("bleu-4", "BLEU-4")]
        cases = [{"mention": mention, "input_type": "Algorithm/Model",
                  "canonical": canonical, "type": "Algorithm/Model"}
                 for mention, canonical in aliases]
        cases.extend({"mention": f"ArchitectureVariant{i}",
                      "input_type": "Algorithm/Model",
                      "canonical": f"ArchitectureVariant{i}",
                      "type": "Algorithm/Model"} for i in range(44))
        pred = generate(cases)
        result = evaluate(cases, pred)
        self.assertEqual(result["result"], "PASS")
        self.assertTrue(result["gate"]["source_bound"])
        tampered = [dict(row) for row in pred]
        tampered[0]["policy_sha256"] = "0" * 64
        self.assertEqual(evaluate(cases, tampered)["result"], "FAIL")
        extra = pred + [{"mention": "unrequested", "canonical": "Extra",
                         "type": "Algorithm/Model", "case_sha256": "0" * 64,
                         "policy_sha256": pred[0]["policy_sha256"]}]
        self.assertEqual(evaluate(cases, extra)["result"], "FAIL")

    def test_m5_all_positive_trivial_detector_fails_false_positive_gate(self):
        from tools.score_m5 import evaluate
        golden = ([{"id": f"p{i}", "expected": True} for i in range(20)] +
                  [{"id": f"n{i}", "expected": False} for i in range(10)])
        predicted = [{"id": row["id"], "flagged": True} for row in golden]
        result = evaluate(golden, predicted)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["false_positive_rate"], 1.0)
        self.assertEqual(result["result"], "FAIL")

    def test_m5_prediction_is_case_bound_and_runs_real_detector(self):
        from tools.predict_m5 import generate
        from tools.score_m5 import evaluate
        cases = []
        for i in range(20):
            cases.append({"id": f"p{i}", "expected": True,
                          "draft": f"MethodA{i} is compared directly with MethodB{i}.",
                          "relations": [{"from": f"MethodA{i}", "to": f"MethodB{i}",
                                         "rel": "Contradicts"}]})
        for i in range(10):
            cases.append({"id": f"n{i}", "expected": False,
                          "draft": f"MethodAneg{i} is discussed without its counterpart.",
                          "relations": [{"from": f"MethodAneg{i}", "to": f"MethodBneg{i}",
                                         "rel": "Contradicts"}]})
        pred = generate(cases)
        result = evaluate(cases, pred)
        self.assertEqual(result["result"], "PASS")
        self.assertTrue(result["gate"]["source_bound"])
        self.assertEqual((result["recall"], result["false_positive_rate"]), (1.0, 0.0))
        forged = [dict(row) for row in pred]
        forged[0]["case_sha256"] = "0" * 64
        self.assertEqual(evaluate(cases, forged)["result"], "FAIL")

    def test_contradiction_detector_ignores_malformed_relation_instead_of_crashing(self):
        from paperbrain.consistency import check_contradicts
        memory = {"relations": [None, {}, {"rel": "Contradicts", "from": "A"},
                                {"rel": "Contradicts", "from": "A", "to": "B"}]}
        self.assertEqual(len(check_contradicts(memory, "A and B")), 1)

    def test_contradiction_detector_does_not_match_entity_inside_longer_name(self):
        from paperbrain.consistency import check_contradicts
        memory = {"relations": [{"rel": "Contradicts", "from": "BERT",
                                 "to": "RoBERTa"}]}
        self.assertEqual(check_contradicts(memory, "RoBERTa is evaluated."), [])
        self.assertEqual(len(check_contradicts(memory, "BERT and RoBERTa are compared.")), 1)

    def test_png_private_metadata_is_removed(self):
        from paperbrain.vision import strip_png_metadata

        def chunk(kind, data):
            return (struct.pack(">I", len(data)) + kind + data +
                    struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

        raw = (b"\x89PNG\r\n\x1a\n" +
               chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)) +
               chunk(b"eXIf", b"camera-serial-private") +
               chunk(b"iTXt", b"XML:com.adobe.xmp\x00private") +
               chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\xff")) +
               chunk(b"IEND", b""))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "x.png")
            path.write_bytes(raw)
            removed = strip_png_metadata(str(path))
            clean = path.read_bytes()
        self.assertEqual(removed, ["eXIf", "iTXt"])
        self.assertNotIn(b"camera-serial-private", clean)
        self.assertNotIn(b"com.adobe.xmp", clean)

    def test_figure_index_binds_body_context(self):
        from paperbrain.passes import build_bidirectional_fig_index
        sections = [{"sec": "3", "text": "As shown in Fig. 2, accuracy rises by ten points."}]
        figures = [{"id": "fig2", "num": "2", "caption": "Fig. 2 Accuracy"}]
        index = build_bidirectional_fig_index(sections, "P", figures=figures)
        self.assertIn("accuracy rises", index["_by_id"]["fig2"]["bound_contexts"][0])

    def test_figure_caption_line_alone_is_not_body_context(self):
        from paperbrain.passes import build_bidirectional_fig_index
        sections = [{"sec": "3", "text": "Fig. 2. Accuracy by method"}]
        figures = [{"id": "fig2", "num": "2", "caption": "Fig. 2. Accuracy by method"}]
        index = build_bidirectional_fig_index(sections, "P", figures=figures)
        self.assertEqual(index["_by_id"]["fig2"]["bound_contexts"], [])


if __name__ == "__main__":
    unittest.main()
