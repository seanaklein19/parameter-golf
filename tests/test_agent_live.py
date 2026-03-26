#!/usr/bin/env python3
"""Live integration test: runs the agentic loop against real Opus.

Requires ANTHROPIC_API_KEY or AWS credentials for Bedrock.
Uses a fake experiment graph with 2 runs so the agent has something
to investigate. Costs ~$0.10-0.30 in API credits.

Usage:
    # With Anthropic API key:
    ANTHROPIC_API_KEY=sk-... python tests/test_agent_live.py

    # With AWS Bedrock:
    python tests/test_agent_live.py

    # Use a cheaper model for testing:
    ORCHESTRATOR_MODEL=claude-haiku-4-5 python tests/test_agent_live.py
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qkv.orchestration.graph import ExperimentGraph
from qkv.orchestration.agent import analyze_and_propose
from qkv.distill.schema import RunRecord


def make_record(run_id, **kwargs):
    return RunRecord(
        run_id=run_id,
        num_steps=200,
        num_layers=9,
        wall_time_s=120.0,
        loss={"initial": 9.0, "final": 3.5, "trend": "improving"},
        gradient_flow={"flow_type": "healthy", "uniformity": 0.8,
                       "ratio_first_last": 0.95},
        **kwargs,
    )


def main():
    print("=" * 60)
    print("  LIVE AGENT INTEGRATION TEST")
    print("=" * 60)

    # Check credentials
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"  Using: Anthropic API (direct)")
    else:
        # Check for AWS credentials (env vars, profile, or ~/.aws/credentials)
        has_aws = (
            os.environ.get("AWS_ACCESS_KEY_ID")
            or os.environ.get("AWS_PROFILE")
            or os.path.exists(os.path.expanduser("~/.aws/credentials"))
        )
        if has_aws:
            print(f"  Using: AWS Bedrock")
        else:
            print("  ERROR: No API credentials found.")
            print("  Set ANTHROPIC_API_KEY or configure AWS credentials.")
            sys.exit(1)

    model = os.environ.get("ORCHESTRATOR_MODEL")
    if model:
        print(f"  Model override: {model}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        rs_path = os.path.join(tmpdir, "research_state.md")

        # Seed research state
        with open(rs_path, "w") as f:
            f.write(
                "# Research State\n\n"
                "## Beliefs\n"
                "- Baseline dim=128 gives ~2.50 BPB\n"
                "- Higher LR might help\n\n"
                "## Open Questions\n"
                "- Does SwiGLU help at this scale?\n"
                "- What's the optimal LR?\n\n"
                "## Experiment Queue\n"
                "1. Try higher learning rate\n"
                "2. Try SwiGLU activation\n"
                "3. Try deeper network\n"
            )

        graph = ExperimentGraph(db_path)

        # Seed two runs so the agent has something to investigate
        baseline_config = {
            "model_dim": 128,
            "num_layers": 9,
            "num_heads": 4,
            "mlp_mult": 2,
            "activation": "relu_sq",
            "lr": 0.001,
            "iterations": 200,
        }
        r1 = make_record("baseline_001")
        id1 = graph.add_run(
            config=baseline_config,
            record=r1,
            metrics={"val_bpb": 2.50, "val_loss": 3.20},
            what_changed="baseline",
        )
        print(f"  Seeded run {id1}: baseline, val_bpb=2.50")

        higher_lr_config = {**baseline_config, "lr": 0.003}
        r2 = make_record("higher_lr_001")
        id2 = graph.add_run(
            config=higher_lr_config,
            record=r2,
            metrics={"val_bpb": 2.42, "val_loss": 3.05},
            what_changed="higher learning rate (0.001 -> 0.003)",
            parent_id=id1,
        )
        print(f"  Seeded run {id2}: higher LR, val_bpb=2.42")
        print()

        # The "latest run" the agent will analyze
        latest_summary = {
            "run_id": id2,
            "val_bpb": 2.42,
            "val_loss": 3.05,
            "steps": 200,
            "wall_time_s": 118.0,
            "artifact_bytes": 1_200_000,
            "model_params": 450_000,
        }

        # Previous proposal (for prediction tracking)
        previous_proposal = {
            "description": "higher learning rate (0.001 -> 0.003)",
            "predicted_bpb": 2.45,
            "confidence": "low",
        }

        tool_calls = []

        def on_tool(name, inp, result):
            tool_calls.append(name)
            # Compact display
            if name in ("read_training_code",):
                print(f"    [{name}] -> (code)")
            elif name == "propose_experiment":
                desc = inp.get("description", "?")
                bpb = inp.get("predicted_bpb", "?")
                print(f"    [{name}] -> \"{desc}\" (predicted_bpb={bpb})")
            elif name == "update_research_state":
                print(f"    [{name}] -> ({len(inp.get('content', ''))} chars)")
            else:
                preview = result.split("\n")[0][:80]
                print(f"    [{name}] -> {preview}")

        print("-" * 60)
        print("  Agent tool calls:")
        t0 = time.time()

        result = analyze_and_propose(
            run_summary=latest_summary,
            graph=graph,
            config=higher_lr_config,
            train_code="# (fake training code for testing)\nimport torch\n...",
            research_state=open(rs_path).read(),
            previous_proposal=previous_proposal,
            research_state_path=rs_path,
            on_tool_call=on_tool,
        )

        elapsed = time.time() - t0
        print("-" * 60)
        print()

        # --- Validate results ---
        errors = []

        # 1. Agent should have used some tools
        if len(tool_calls) < 2:
            errors.append(
                f"FAIL: Agent only used {len(tool_calls)} tool calls "
                f"(expected >= 2)"
            )
        else:
            print(f"  OK: Agent used {len(tool_calls)} tool calls")

        # 2. Agent should have proposed an experiment
        proposal = result.get("proposal")
        if proposal is None:
            errors.append("FAIL: No proposal returned")
        else:
            print(f"  OK: Proposal: \"{proposal.get('description', '?')}\"")

            # 3. Proposal should have required fields
            for field in ("description", "type", "rationale",
                          "predicted_bpb", "confidence"):
                if field not in proposal:
                    errors.append(f"FAIL: Proposal missing '{field}'")
                else:
                    print(f"  OK: Proposal has '{field}': {proposal[field]}")

            # 4. predicted_bpb should be a number
            if isinstance(proposal.get("predicted_bpb"), (int, float)):
                print(f"  OK: predicted_bpb is numeric: {proposal['predicted_bpb']}")
            else:
                errors.append(
                    f"FAIL: predicted_bpb is not numeric: "
                    f"{proposal.get('predicted_bpb')}"
                )

            # 5. type should be valid
            if proposal.get("type") in ("config", "code"):
                print(f"  OK: type is valid: {proposal['type']}")
            else:
                errors.append(f"FAIL: invalid type: {proposal.get('type')}")

            # 6. If config type, should have changes
            if proposal.get("type") == "config":
                changes = proposal.get("changes", {})
                if isinstance(changes, dict):
                    print(f"  OK: config changes: {changes}")
                else:
                    errors.append(f"FAIL: changes is not a dict: {changes}")

        # 7. Check observations were recorded
        obs = graph.get_observations()
        if obs:
            print(f"  OK: {len(obs)} observation(s) recorded in graph")
        else:
            print(f"  NOTE: No observations recorded (not required)")

        # 8. Check if research state was updated
        with open(rs_path) as f:
            new_rs = f.read()
        if new_rs != (
            "# Research State\n\n"
            "## Beliefs\n"
            "- Baseline dim=128 gives ~2.50 BPB\n"
            "- Higher LR might help\n\n"
            "## Open Questions\n"
            "- Does SwiGLU help at this scale?\n"
            "- What's the optimal LR?\n\n"
            "## Experiment Queue\n"
            "1. Try higher learning rate\n"
            "2. Try SwiGLU activation\n"
            "3. Try deeper network\n"
        ):
            print(f"  OK: Research state was updated ({len(new_rs)} chars)")
        else:
            print(f"  NOTE: Research state unchanged")

        # 9. Check error field
        if result.get("error"):
            errors.append(f"FAIL: Agent returned error: {result['error']}")

        print()
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Tool calls: {' -> '.join(tool_calls)}")
        print()

        graph.close()

        # --- Final verdict ---
        if errors:
            print("=" * 60)
            print("  FAILURES:")
            for e in errors:
                print(f"    {e}")
            print("=" * 60)
            sys.exit(1)
        else:
            print("=" * 60)
            print("  ALL CHECKS PASSED")
            print("=" * 60)


if __name__ == "__main__":
    main()
