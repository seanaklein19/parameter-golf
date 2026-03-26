#!/usr/bin/env python3
"""Autonomous experiment loop for parameter golf.

Runs train → analyze → propose → apply in a loop.
Each iteration:
  1. Train with current config (subprocess, A6000)
  2. Read run_summary.json
  3. Store run in ExperimentGraph
  4. Call orchestrator agent (Bedrock/Anthropic) for analysis + next proposal
  5. Write insights, update research_state.md
  6. Apply proposal (config change or code patch)
  7. Repeat

Usage:
    # Interactive (approve each proposal):
    python run_loop.py

    # Auto-approve config changes only:
    python run_loop.py --auto-config

    # Fully autonomous:
    python run_loop.py --auto

    # Limit iterations:
    python run_loop.py --auto --max-runs 20
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure qkv is importable (installed via pip install -e ../logging)
from qkv.orchestration import ExperimentGraph, analyze_and_propose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_text(path: str) -> str:
    with open(path) as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def apply_code_patch(train_script: str, diff: str) -> bool:
    """Apply a unified diff patch to the training script.

    Returns True on success, False on failure.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(diff)
        patch_path = f.name

    try:
        result = subprocess.run(
            ["patch", "--forward", "--reject-file=-", train_script, patch_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info("Code patch applied successfully")
            return True
        else:
            logger.error(f"Patch failed: {result.stderr}")
            return False
    finally:
        os.unlink(patch_path)


# ---------------------------------------------------------------------------
# Core loop steps
# ---------------------------------------------------------------------------

def print_banner(text: str, char: str = "=") -> None:
    """Print a visible banner to stdout."""
    width = max(len(text) + 4, 60)
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_run_result(iteration: int, summary: dict, run_id: str) -> None:
    """Print training results."""
    print(f"\n  Run {run_id} complete:")
    print(f"    val_bpb  = {summary.get('val_bpb', '?')}")
    print(f"    val_loss = {summary.get('val_loss', '?')}")
    print(f"    steps    = {summary.get('steps', '?')}")
    print(f"    time     = {summary.get('wall_time_s', 0):.0f}s")
    artifact = summary.get("artifact_bytes")
    if artifact:
        print(f"    artifact = {artifact / 1e6:.2f} MB / 16.00 MB")


def print_agent_result(result: dict) -> None:
    """Print the agent's analysis and proposal."""
    print(f"\n  Agent analysis:")
    print(f"    {result.get('analysis', '(none)')}")

    insights = result.get("insights", [])
    if insights:
        print(f"\n  New insights ({len(insights)}):")
        for ins in insights:
            print(f"    - {ins.get('insight', '?')}")

    proposal = result.get("proposal")
    if proposal:
        print(f"\n  Proposed next experiment:")
        print(f"    {proposal.get('description', 'unnamed')}")
        print(f"    Type: {proposal.get('type', '?')}")
        print(f"    Rationale: {proposal.get('rationale', '?')}")
        print(f"    Expected: {proposal.get('expected_impact', '?')}")
        if proposal.get("type") == "config":
            print(f"    Changes: {json.dumps(proposal.get('changes', {}))}")
    else:
        print("\n  Agent has no proposal.")


def print_scoreboard(graph: ExperimentGraph) -> None:
    """Print a scoreboard of all runs."""
    runs = graph.get_all_runs()
    best = graph.get_best()
    best_id = best["run_id"] if best else None

    print(f"\n  Scoreboard ({len(runs)} runs):")
    print(f"  {'ID':>8}  {'val_bpb':>8}  {'what_changed'}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*30}")
    for run in runs:
        bpb = run["metrics"].get("val_bpb", "?")
        bpb_str = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else str(bpb)
        marker = " <-- BEST" if run["run_id"] == best_id else ""
        what = run.get("what_changed", "?") or "?"
        print(f"  {run['run_id']:>8}  {bpb_str:>8}  {what[:40]}{marker}")

    insights = graph.get_insights()
    if insights:
        print(f"\n  Confirmed insights ({len(insights)}):")
        for ins in insights:
            print(f"    - {ins['insight']}")


def train(config_path: str, train_script: str, run_id: str) -> dict:
    """Run training and return the run summary."""
    print_banner(f"TRAINING: {run_id}")

    env = os.environ.copy()
    env["RUN_ID"] = run_id

    # Remove run_summary.json from previous run
    summary_path = "run_summary.json"
    if os.path.exists(summary_path):
        os.remove(summary_path)

    cmd = [sys.executable, train_script, "--config", config_path]
    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    if not os.path.exists(summary_path):
        raise RuntimeError("Training completed but run_summary.json not found")

    summary = load_json(summary_path)
    summary["wall_time_s"] = elapsed
    logger.info(
        f"Training done in {elapsed:.0f}s — "
        f"val_bpb={summary.get('val_bpb', '?')}"
    )
    return summary


def store_run(
    graph: ExperimentGraph,
    config: dict,
    summary: dict,
    train_code: str,
    description: str,
    parent_ids: list[str] | None = None,
    code_diff: str | None = None,
) -> str:
    """Store a completed run in the experiment graph."""
    from qkv.distill.schema import RunRecord

    # Build a minimal RunRecord for the graph
    record = RunRecord(
        run_id=summary.get("run_id", "unknown"),
        num_steps=summary.get("steps", 0),
        num_layers=config.get("num_layers", 0),
        wall_time_s=summary.get("wall_time_s", 0.0),
        loss={"final": summary.get("val_loss", 0.0)},
        gradient_flow={},
    )

    metrics = {
        "val_bpb": summary.get("val_bpb"),
        "val_loss": summary.get("val_loss"),
        "artifact_bytes": summary.get("artifact_bytes"),
        "model_params": summary.get("model_params"),
        "train_time_ms": summary.get("train_time_ms"),
    }
    # Remove None values
    metrics = {k: v for k, v in metrics.items() if v is not None}

    run_id = graph.add_run(
        config=config,
        record=record,
        metrics=metrics,
        primary_metric="val_bpb",
        parent_id=parent_ids,
        what_changed=description,
        train_code=train_code,
        code_diff=code_diff,
    )
    logger.info(f"Stored run {run_id} in graph")
    return run_id


def orchestrate(
    graph: ExperimentGraph,
    summary: dict,
    config: dict,
    train_code: str,
    research_state: str,
) -> dict:
    """Call the orchestrator agent and return its response."""
    logger.info("Calling orchestrator agent...")
    result = analyze_and_propose(
        run_summary=summary,
        graph=graph,
        config=config,
        train_code=train_code,
        research_state=research_state,
    )
    logger.info(f"Agent analysis: {result.get('analysis', '?')[:120]}")
    return result


def apply_insights(graph: ExperimentGraph, insights: list[dict]) -> None:
    """Write new insights to the graph."""
    for ins in insights:
        iid = graph.add_insight(ins["insight"], ins.get("run_ids"))
        logger.info(f"Added insight #{iid}: {ins['insight'][:80]}")


def apply_proposal(proposal: dict, config_path: str, train_script: str) -> bool:
    """Apply a proposal. Returns True if applied successfully."""
    if proposal is None:
        logger.warning("No proposal from agent")
        return False

    ptype = proposal.get("type", "config")

    if ptype == "config":
        config = load_json(config_path)
        changes = proposal.get("changes", {})
        config.update(changes)
        save_json(config_path, config)
        logger.info(f"Applied config changes: {changes}")
        return True

    elif ptype == "code":
        diff = proposal.get("diff", "")
        if not diff:
            logger.warning("Code proposal has no diff")
            return False
        # Back up the training script before patching
        backup = train_script + ".bak"
        shutil.copy2(train_script, backup)
        if apply_code_patch(train_script, diff):
            return True
        else:
            # Restore backup on failure
            shutil.copy2(backup, train_script)
            logger.warning("Patch failed, restored backup")
            return False

    else:
        logger.warning(f"Unknown proposal type: {ptype}")
        return False


def prompt_approval(proposal: dict, auto_mode: str) -> bool:
    """Check if the proposal should proceed.

    auto_mode: "none" (always ask), "config" (auto-approve config), "all" (auto-approve everything)
    """
    if proposal is None:
        return False

    ptype = proposal.get("type", "config")

    if auto_mode == "all":
        return True
    if auto_mode == "config" and ptype == "config":
        return True

    # Interactive approval
    print(f"\n{'='*60}")
    print(f"Proposal: {proposal.get('description', 'unnamed')}")
    print(f"Type: {ptype}")
    print(f"Rationale: {proposal.get('rationale', '?')}")
    print(f"Expected impact: {proposal.get('expected_impact', '?')}")
    if ptype == "config":
        print(f"Changes: {json.dumps(proposal.get('changes', {}), indent=2)}")
    elif ptype == "code":
        print(f"Diff:\n{proposal.get('diff', '(empty)')}")
    print(f"{'='*60}")

    while True:
        response = input("Approve? [y/n/q] ").strip().lower()
        if response == "y":
            return True
        if response in ("n", "q"):
            return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autonomous experiment loop")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--train-script", default="train_gpt.py", help="Training script")
    parser.add_argument("--research-state", default="research_state.md", help="Research state file")
    parser.add_argument("--db", default="experiments.db", help="Experiment graph database")
    parser.add_argument("--max-runs", type=int, default=0, help="Max iterations (0=unlimited)")
    parser.add_argument("--auto-config", action="store_true",
                        help="Auto-approve config changes, prompt for code changes")
    parser.add_argument("--auto", action="store_true",
                        help="Fully autonomous, no human approval")
    args = parser.parse_args()

    if args.auto:
        auto_mode = "all"
    elif args.auto_config:
        auto_mode = "config"
    else:
        auto_mode = "none"

    graph = ExperimentGraph(args.db)
    config_path = args.config
    train_script = args.train_script
    research_state_path = args.research_state

    print_banner("PARAMETER GOLF — AUTONOMOUS EXPERIMENT LOOP", char="*")
    print(f"  Config:         {config_path}")
    print(f"  Train script:   {train_script}")
    print(f"  Research state: {research_state_path}")
    print(f"  Database:       {args.db}")
    print(f"  Mode:           {auto_mode}")
    if args.max_runs:
        print(f"  Max runs:       {args.max_runs}")

    # Check if this is first run (no experiments yet) or resuming
    all_runs = graph.get_all_runs()
    parent_ids = None
    description = "baseline"

    if all_runs:
        # Resuming: use the most recent run as parent
        last_run = all_runs[-1]
        parent_ids = [last_run["run_id"]]
        description = "resumed"
        logger.info(f"Resuming from {len(all_runs)} existing runs")
        logger.info(f"Best so far: val_bpb={graph.get_best().get('metrics', {}).get('val_bpb', '?')}")

        # If resuming, skip straight to orchestration with the last run
        # to get a proposal for the next experiment
        config = load_json(config_path)
        train_code = read_text(train_script)
        research_state = read_text(research_state_path)

        # Use the last run's metrics as the summary
        last_summary = {
            "run_id": last_run["run_id"],
            "val_bpb": last_run["metrics"].get("val_bpb"),
            "val_loss": last_run["metrics"].get("val_loss"),
        }

        result = orchestrate(graph, last_summary, config, train_code, research_state)
        apply_insights(graph, result.get("insights", []))

        if result.get("research_state"):
            write_text(research_state_path, result["research_state"])

        proposal = result.get("proposal")
        if proposal and prompt_approval(proposal, auto_mode):
            apply_proposal(proposal, config_path, train_script)
            parent_ids = proposal.get("parent_ids") or parent_ids
            description = proposal.get("description", "agent proposal")
        else:
            logger.info("No proposal approved, running with current config")
            description = "manual run (no proposal)"

    iteration = 0
    while True:
        if args.max_runs and iteration >= args.max_runs:
            logger.info(f"Reached max runs ({args.max_runs})")
            break

        try:
            config = load_json(config_path)
            train_code = read_text(train_script)
            run_id_tag = f"run_{iteration:03d}_{int(time.time())}"

            # 1. Train
            summary = train(config_path, train_script, run_id_tag)

            # 2. Store in graph
            run_id = store_run(
                graph, config, summary, train_code, description,
                parent_ids=parent_ids,
            )
            print_run_result(iteration, summary, run_id)

            # 3. Orchestrate
            print_banner("OPUS THINKING...", char="-")
            research_state = read_text(research_state_path)
            result = orchestrate(graph, summary, config, train_code, research_state)
            print_agent_result(result)

            # 4. Write insights
            apply_insights(graph, result.get("insights", []))

            # 5. Update research state
            if result.get("research_state"):
                write_text(research_state_path, result["research_state"])

            # 6. Scoreboard
            print_scoreboard(graph)

            # 7. Propose next experiment
            proposal = result.get("proposal")
            if proposal is None:
                print("\n  Agent has no more ideas. Stopping.")
                break

            if not prompt_approval(proposal, auto_mode):
                print("\n  Proposal not approved. Stopping.")
                break

            # 8. Apply proposal
            if not apply_proposal(proposal, config_path, train_script):
                print("\n  Failed to apply proposal. Stopping.")
                break

            # Set up next iteration
            parent_ids = proposal.get("parent_ids") or [run_id]
            description = proposal.get("description", "agent proposal")
            iteration += 1

            best_bpb = graph.get_best().get("metrics", {}).get("val_bpb", "?")
            print_banner(
                f"ITERATION {iteration} COMPLETE | Best BPB: {best_bpb}",
                char="=",
            )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)
            break

    # Final summary
    print_banner("FINAL RESULTS", char="*")
    print_scoreboard(graph)
    best = graph.get_best()
    if best:
        print(f"\n  BEST: {best['run_id']} — val_bpb = {best['metrics'].get('val_bpb')}")
    print(f"\n  Check research_state.md for the agent's current thinking.")
    print(f"  Check experiments.db for full history.\n")
    graph.close()


if __name__ == "__main__":
    main()
