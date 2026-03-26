"""Tests for run_loop.py — the autonomous experiment loop."""

import json
import os
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pytest

# Add parent dir so we can import run_loop
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_loop import (
    apply_code_patch,
    apply_insights,
    apply_proposal,
    load_json,
    prompt_approval,
    save_json,
    store_run,
    train,
)
from qkv.orchestration import ExperimentGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp(tmp_path):
    """Provide tmp_path and create common test files."""
    return tmp_path


@pytest.fixture
def graph(tmp_path):
    g = ExperimentGraph(str(tmp_path / "test.db"))
    yield g
    g.close()


@pytest.fixture
def config_file(tmp_path):
    path = str(tmp_path / "config.json")
    data = {"model_dim": 128, "num_layers": 9, "mlp_mult": 2}
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def train_script(tmp_path):
    """A fake training script that writes run_summary.json."""
    path = str(tmp_path / "train.py")
    with open(path, "w") as f:
        f.write(textwrap.dedent("""\
            import json, os, sys
            summary = {
                "run_id": os.environ.get("RUN_ID", "test"),
                "val_bpb": 2.45,
                "val_loss": 3.12,
                "train_time_ms": 5000,
                "steps": 200,
                "artifact_bytes": 1000000,
                "model_params": 500000,
            }
            with open("run_summary.json", "w") as f:
                json.dump(summary, f)
        """))
    return path


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_save_and_load(self, tmp):
        path = str(tmp / "test.json")
        save_json(path, {"a": 1, "b": [2, 3]})
        data = load_json(path)
        assert data == {"a": 1, "b": [2, 3]}

    def test_save_overwrites(self, tmp):
        path = str(tmp / "test.json")
        save_json(path, {"old": True})
        save_json(path, {"new": True})
        assert load_json(path) == {"new": True}


# ---------------------------------------------------------------------------
# apply_proposal — config changes
# ---------------------------------------------------------------------------

class TestApplyProposalConfig:
    def test_applies_config_changes(self, config_file, tmp):
        proposal = {
            "type": "config",
            "changes": {"mlp_mult": 3, "num_heads": 8},
        }
        result = apply_proposal(proposal, config_file, "unused.py")
        assert result is True

        updated = load_json(config_file)
        assert updated["mlp_mult"] == 3
        assert updated["num_heads"] == 8
        # Original keys preserved
        assert updated["model_dim"] == 128
        assert updated["num_layers"] == 9

    def test_empty_changes(self, config_file, tmp):
        original = load_json(config_file)
        proposal = {"type": "config", "changes": {}}
        result = apply_proposal(proposal, config_file, "unused.py")
        assert result is True
        assert load_json(config_file) == original

    def test_none_proposal(self, config_file, tmp):
        assert apply_proposal(None, config_file, "unused.py") is False

    def test_unknown_type(self, config_file, tmp):
        proposal = {"type": "magic", "changes": {}}
        assert apply_proposal(proposal, config_file, "unused.py") is False


# ---------------------------------------------------------------------------
# apply_proposal — code patches
# ---------------------------------------------------------------------------

class TestApplyProposalCode:
    def test_code_patch_no_diff(self, tmp):
        script = str(tmp / "train.py")
        with open(script, "w") as f:
            f.write("x = 1\n")
        proposal = {"type": "code", "diff": ""}
        assert apply_proposal(proposal, "unused.json", script) is False

    def test_code_patch_restores_on_failure(self, tmp):
        script = str(tmp / "train.py")
        original = "x = 1\n"
        with open(script, "w") as f:
            f.write(original)

        # Bad patch that won't apply
        proposal = {"type": "code", "diff": "garbage that isn't a patch"}
        result = apply_proposal(proposal, "unused.json", script)
        assert result is False
        # Original file restored
        with open(script) as f:
            assert f.read() == original

    def test_code_patch_applies(self, tmp):
        script = str(tmp / "train.py")
        with open(script, "w") as f:
            f.write("x = 1\ny = 2\n")

        diff = textwrap.dedent("""\
            --- a/train.py
            +++ b/train.py
            @@ -1,2 +1,2 @@
            -x = 1
            +x = 42
             y = 2
        """)
        proposal = {"type": "code", "diff": diff}
        result = apply_proposal(proposal, "unused.json", script)
        assert result is True
        with open(script) as f:
            content = f.read()
        assert "x = 42" in content


# ---------------------------------------------------------------------------
# store_run
# ---------------------------------------------------------------------------

class TestStoreRun:
    def test_stores_run_in_graph(self, graph):
        summary = {
            "run_id": "test_run",
            "val_bpb": 2.45,
            "val_loss": 3.12,
            "steps": 200,
            "train_time_ms": 5000,
            "artifact_bytes": 1000000,
            "model_params": 500000,
            "wall_time_s": 120.0,
        }
        config = {"model_dim": 128, "num_layers": 9}

        run_id = store_run(graph, config, summary, "import torch", "baseline")

        run = graph.get_run(run_id)
        assert run is not None
        assert run["metrics"]["val_bpb"] == 2.45
        assert run["config"]["model_dim"] == 128
        assert run["what_changed"] == "baseline"
        assert run["train_code"] == "import torch"

    def test_stores_with_parent(self, graph):
        from qkv.distill.schema import RunRecord
        rec = RunRecord(run_id="p", num_steps=10, num_layers=8,
                        wall_time_s=10, loss={}, gradient_flow={})
        parent_id = graph.add_run(config={}, record=rec, metrics={"val_bpb": 3.0})

        summary = {"run_id": "child", "val_bpb": 2.5, "steps": 100, "wall_time_s": 60}
        child_id = store_run(graph, {"num_layers": 8}, summary, "", "test",
                             parent_ids=[parent_id])

        run = graph.get_run(child_id)
        assert parent_id in run["parent_ids"]

    def test_filters_none_metrics(self, graph):
        summary = {"run_id": "x", "val_bpb": 2.0, "val_loss": None,
                    "steps": 50, "wall_time_s": 30}
        run_id = store_run(graph, {"num_layers": 4}, summary, "", "test")
        run = graph.get_run(run_id)
        assert "val_loss" not in run["metrics"]


# ---------------------------------------------------------------------------
# apply_insights
# ---------------------------------------------------------------------------

class TestApplyInsights:
    def test_adds_insights_to_graph(self, graph):
        insights = [
            {"insight": "SwiGLU > relu^2", "run_ids": ["abc"]},
            {"insight": "grad clip helps"},
        ]
        apply_insights(graph, insights)
        stored = graph.get_insights()
        assert len(stored) == 2
        assert stored[0]["insight"] == "SwiGLU > relu^2"
        assert stored[1]["insight"] == "grad clip helps"

    def test_empty_insights(self, graph):
        apply_insights(graph, [])
        assert graph.get_insights() == []


# ---------------------------------------------------------------------------
# prompt_approval
# ---------------------------------------------------------------------------

class TestPromptApproval:
    def test_auto_all_approves_everything(self):
        proposal = {"type": "code", "diff": "..."}
        assert prompt_approval(proposal, "all") is True

    def test_auto_config_approves_config(self):
        proposal = {"type": "config", "changes": {}}
        assert prompt_approval(proposal, "config") is True

    def test_auto_config_does_not_approve_code(self):
        proposal = {"type": "code", "diff": "..."}
        with patch("builtins.input", return_value="n"):
            assert prompt_approval(proposal, "config") is False

    def test_none_proposal_rejected(self):
        assert prompt_approval(None, "all") is False

    def test_interactive_approve(self):
        proposal = {"type": "config", "changes": {"lr": 0.01}}
        with patch("builtins.input", return_value="y"):
            assert prompt_approval(proposal, "none") is True

    def test_interactive_reject(self):
        proposal = {"type": "config", "changes": {"lr": 0.01}}
        with patch("builtins.input", return_value="n"):
            assert prompt_approval(proposal, "none") is False


# ---------------------------------------------------------------------------
# train (with fake script)
# ---------------------------------------------------------------------------

class TestTrain:
    def test_runs_training_script(self, config_file, train_script, tmp, monkeypatch):
        monkeypatch.chdir(tmp)
        # Create config in the tmp dir
        config_path = str(tmp / "config.json")
        save_json(config_path, {"model_dim": 128})

        summary = train(config_path, train_script, "test_run_001")
        assert summary["val_bpb"] == 2.45
        assert summary["run_id"] == "test_run_001"
        assert "wall_time_s" in summary

    def test_training_failure(self, tmp, monkeypatch):
        monkeypatch.chdir(tmp)
        bad_script = str(tmp / "bad.py")
        with open(bad_script, "w") as f:
            f.write("import sys; sys.exit(1)\n")

        config_path = str(tmp / "config.json")
        save_json(config_path, {})

        with pytest.raises(RuntimeError, match="exit code"):
            train(config_path, bad_script, "fail_run")

    def test_missing_summary(self, tmp, monkeypatch):
        monkeypatch.chdir(tmp)
        # Script that exits 0 but doesn't write summary
        script = str(tmp / "nosummary.py")
        with open(script, "w") as f:
            f.write("pass\n")

        config_path = str(tmp / "config.json")
        save_json(config_path, {})

        with pytest.raises(RuntimeError, match="run_summary.json not found"):
            train(config_path, script, "no_summary")


# ---------------------------------------------------------------------------
# Integration: store → insights → proposal → apply cycle
# ---------------------------------------------------------------------------

class TestIntegrationCycle:
    def test_full_cycle(self, graph, config_file):
        """Simulate one full iteration: store run, add insights, apply config proposal."""
        # 1. Store a run
        summary = {
            "run_id": "baseline_001",
            "val_bpb": 2.45,
            "val_loss": 3.12,
            "steps": 200,
            "wall_time_s": 120.0,
        }
        config = load_json(config_file)
        run_id = store_run(graph, config, summary, "import torch", "baseline")

        # 2. Agent returns analysis
        agent_result = {
            "analysis": "Baseline is 2.45 BPB",
            "insights": [
                {"insight": "baseline dim=128 is 2.45 BPB", "run_ids": [run_id]}
            ],
            "research_state": "# Updated state\n",
            "proposal": {
                "description": "try swiglu",
                "type": "config",
                "changes": {"activation": "swiglu", "mlp_mult": 3},
                "rationale": "SwiGLU commonly helps",
                "parent_ids": [run_id],
                "expected_impact": "~0.03 BPB",
            },
        }

        # 3. Apply insights
        apply_insights(graph, agent_result["insights"])
        assert len(graph.get_insights()) == 1

        # 4. Apply proposal
        result = apply_proposal(agent_result["proposal"], config_file, "unused.py")
        assert result is True

        updated = load_json(config_file)
        assert updated["activation"] == "swiglu"
        assert updated["mlp_mult"] == 3

        # 5. Store next run with parent
        summary2 = {
            "run_id": "swiglu_001",
            "val_bpb": 2.42,
            "steps": 200,
            "wall_time_s": 125.0,
        }
        run_id2 = store_run(
            graph, updated, summary2, "import torch",
            "try swiglu", parent_ids=[run_id],
        )

        # Verify lineage
        lineage = graph.get_lineage(run_id2)
        assert len(lineage) == 2
        assert lineage[0]["run_id"] == run_id2
        assert lineage[1]["run_id"] == run_id

        # Best should be the new run
        best = graph.get_best()
        assert best["metrics"]["val_bpb"] == 2.42
