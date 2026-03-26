#!/usr/bin/env python3
"""Autonomous experiment loop for parameter golf.

Runs train -> analyze -> propose -> apply in a loop.
Each iteration:
  1. Train with current config (subprocess, A6000)
  2. Read run_summary.json
  3. Store run in ExperimentGraph
  4. Call orchestrator agent (agentic tool-use loop on Bedrock/Anthropic)
     - Agent investigates graph, records observations/insights,
       updates research state, proposes next experiment
  5. Apply proposal (config change or code patch)
  6. Repeat

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
# Display helpers
# ---------------------------------------------------------------------------

def append_log(path: str, entry: dict) -> None:
    """Append a structured log entry to the iteration log (JSONL)."""
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


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


def print_tool_call(tool_name: str, tool_input: dict, result: str) -> None:
    """Display an agent tool call in real-time."""
    # Compact input for display
    if tool_name in ("read_training_code", "read_config", "read_research_state",
                     "read_feedback", "get_insights", "get_observations"):
        input_str = ""
    elif tool_name == "get_run":
        input_str = f" {tool_input.get('run_id', '?')}"
    elif tool_name == "compare_runs":
        input_str = (
            f" {tool_input.get('run_id_a', '?')} vs "
            f"{tool_input.get('run_id_b', '?')}"
        )
    elif tool_name == "get_lineage":
        input_str = f" {tool_input.get('run_id', '?')}"
    elif tool_name == "list_runs":
        limit = tool_input.get("limit", 0)
        input_str = f" (limit={limit})" if limit else ""
    elif tool_name == "add_observation":
        input_str = f" \"{tool_input.get('observation', '?')[:60]}...\""
    elif tool_name == "add_insight":
        input_str = f" \"{tool_input.get('insight', '?')[:60]}...\""
    elif tool_name == "supersede_insight":
        input_str = f" #{tool_input.get('old_insight_id', '?')}"
    elif tool_name == "update_research_state":
        input_str = f" ({len(tool_input.get('content', ''))} chars)"
    elif tool_name == "propose_experiment":
        input_str = f" \"{tool_input.get('description', '?')}\""
    else:
        input_str = ""

    # Short result preview
    result_lines = result.strip().split("\n")
    result_preview = result_lines[0][:80] if result_lines else ""

    print(f"    [{tool_name}]{input_str}")
    if tool_name not in ("update_research_state", "read_training_code"):
        print(f"      -> {result_preview}")


def print_agent_result(result: dict) -> None:
    """Print a summary of the agent's work."""
    obs = result.get("observations", [])
    ins = result.get("insights", [])
    tool_calls = result.get("tool_calls", [])

    print(f"\n  Agent used {len(tool_calls)} tool calls")

    if obs:
        print(f"\n  Observations recorded ({len(obs)}):")
        for o in obs:
            print(f"    - {o[:80]}")

    if ins:
        print(f"\n  Insights recorded ({len(ins)}):")
        for i in ins:
            print(f"    - {i[:80]}")

    proposal = result.get("proposal")
    if proposal:
        print(f"\n  Proposed next experiment:")
        print(f"    {proposal.get('description', 'unnamed')}")
        print(f"    Type: {proposal.get('type', '?')}")
        print(f"    Rationale: {proposal.get('rationale', '?')}")
        if proposal.get("type") == "config":
            print(f"    Changes: {json.dumps(proposal.get('changes', {}))}")
        if proposal.get("predicted_bpb") is not None:
            print(
                f"    Predicted BPB: {proposal['predicted_bpb']} "
                f"(confidence: {proposal.get('confidence', '?')})"
            )
    else:
        print("\n  Agent has no proposal.")

    if result.get("error"):
        print(f"\n  ERROR: {result['error']}")


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


# ---------------------------------------------------------------------------
# Core loop steps
# ---------------------------------------------------------------------------

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
        f"Training done in {elapsed:.0f}s -- "
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
    parquet_path = summary.get("parquet_path")
    if parquet_path and os.path.exists(parquet_path):
        from qkv.distill import distill_run
        record = distill_run(parquet_path, config=config)
        logger.info(f"Distilled {parquet_path}: {record.num_steps} steps, "
                     f"{len(record.layer_health)} layer health entries, "
                     f"gradient_flow={record.gradient_flow.get('flow_type', '?')}")
    else:
        from qkv.distill.schema import RunRecord
        logger.warning("No parquet_path in summary, using minimal RunRecord")
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
    previous_proposal: dict | None = None,
    feedback_path: str = "feedback.md",
    research_state_path: str = "research_state.md",
) -> dict:
    """Call the orchestrator agent (agentic tool-use loop)."""
    print_banner("OPUS INVESTIGATING...", char="-")
    print("  Agent tool calls:")

    feedback = None
    if os.path.exists(feedback_path):
        feedback = read_text(feedback_path)
        if feedback.strip():
            logger.info(f"Including operator feedback from {feedback_path}")

    result = analyze_and_propose(
        run_summary=summary,
        graph=graph,
        config=config,
        train_code=train_code,
        research_state=research_state,
        previous_proposal=previous_proposal,
        feedback=feedback,
        research_state_path=research_state_path,
        on_tool_call=print_tool_call,
    )
    return result


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
        backup = train_script + ".bak"
        shutil.copy2(train_script, backup)
        if apply_code_patch(train_script, diff):
            return True
        else:
            shutil.copy2(backup, train_script)
            logger.warning("Patch failed, restored backup")
            return False

    else:
        logger.warning(f"Unknown proposal type: {ptype}")
        return False


def prompt_approval(proposal: dict, auto_mode: str) -> bool:
    """Check if the proposal should proceed."""
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
    if proposal.get("predicted_bpb") is not None:
        print(
            f"Predicted BPB: {proposal['predicted_bpb']} "
            f"(confidence: {proposal.get('confidence', '?')})"
        )
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
    parser.add_argument("--train-script", default="train_gpt.py",
                        help="Training script")
    parser.add_argument("--research-state", default="research_state.md",
                        help="Research state file")
    parser.add_argument("--db", default="experiments.db",
                        help="Experiment graph database")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Max iterations (0=unlimited)")
    parser.add_argument("--auto-config", action="store_true",
                        help="Auto-approve config changes, prompt for code")
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

    log_path = "loop_log.jsonl"

    print_banner("PARAMETER GOLF -- AUTONOMOUS EXPERIMENT LOOP", char="*")
    print(f"  Config:         {config_path}")
    print(f"  Train script:   {train_script}")
    print(f"  Research state: {research_state_path}")
    print(f"  Database:       {args.db}")
    print(f"  Iteration log:  {log_path}")
    print(f"  Mode:           {auto_mode}")
    if args.max_runs:
        print(f"  Max runs:       {args.max_runs}")

    # Check if resuming from existing experiments
    all_runs = graph.get_all_runs()
    parent_ids = None
    description = "baseline"

    if all_runs:
        last_run = all_runs[-1]
        parent_ids = [last_run["run_id"]]
        description = "resumed"
        logger.info(f"Resuming from {len(all_runs)} existing runs")
        best = graph.get_best()
        logger.info(
            f"Best so far: val_bpb="
            f"{best.get('metrics', {}).get('val_bpb', '?') if best else '?'}"
        )

        config = load_json(config_path)
        train_code = read_text(train_script)
        research_state = read_text(research_state_path)

        last_summary = {
            "run_id": last_run["run_id"],
            "val_bpb": last_run["metrics"].get("val_bpb"),
            "val_loss": last_run["metrics"].get("val_loss"),
        }

        result = orchestrate(
            graph, last_summary, config, train_code, research_state,
            research_state_path=research_state_path,
        )
        print_agent_result(result)

        proposal = result.get("proposal")
        if proposal and prompt_approval(proposal, auto_mode):
            apply_proposal(proposal, config_path, train_script)
            parent_ids = proposal.get("parent_ids") or parent_ids
            description = proposal.get("description", "agent proposal")
        else:
            logger.info("No proposal approved, running with current config")
            description = "manual run (no proposal)"

    iteration = 0
    previous_proposal = None
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5
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

            # 3. Check prediction from previous iteration
            actual_bpb = summary.get("val_bpb")
            if (previous_proposal
                    and previous_proposal.get("predicted_bpb") is not None):
                predicted = previous_proposal["predicted_bpb"]
                delta = actual_bpb - predicted
                direction = "WORSE" if delta > 0 else "BETTER"
                print(f"\n  Prediction vs reality:")
                print(f"    Predicted: {predicted:.4f}")
                print(f"    Actual:    {actual_bpb:.4f}")
                print(
                    f"    Delta:     {delta:+.4f} "
                    f"({direction} than predicted)"
                )

            # 4. Orchestrate (agentic tool-use loop) — retry on failure
            result = None
            for agent_attempt in range(3):
                try:
                    research_state = read_text(research_state_path)
                    result = orchestrate(
                        graph, summary, config, train_code, research_state,
                        previous_proposal=previous_proposal,
                        research_state_path=research_state_path,
                    )
                    break
                except Exception as agent_err:
                    wait = 30 * (2 ** agent_attempt)
                    logger.error(
                        f"Agent failed (attempt {agent_attempt + 1}/3): "
                        f"{agent_err}"
                    )
                    if agent_attempt < 2:
                        logger.info(f"Retrying agent in {wait}s...")
                        time.sleep(wait)

            if result is None:
                logger.error(
                    "Agent failed 3 times. Skipping analysis, "
                    "re-running with same config."
                )
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        f"{MAX_CONSECUTIVE_ERRORS} consecutive errors, "
                        f"giving up."
                    )
                    break
                continue

            consecutive_errors = 0
            print_agent_result(result)

            # 5. Scoreboard
            print_scoreboard(graph)

            # 6. Propose next experiment
            proposal = result.get("proposal")
            if proposal is None:
                print("\n  Agent has no more ideas. Stopping.")
                break

            if not prompt_approval(proposal, auto_mode):
                print("\n  Proposal not approved. Stopping.")
                break

            # 7. Apply proposal
            if not apply_proposal(proposal, config_path, train_script):
                print("\n  Failed to apply proposal. Stopping.")
                break

            # 8. Log this iteration for meta-analysis
            log_entry = {
                "iteration": iteration,
                "timestamp": time.time(),
                "run_id": run_id,
                "description": description,
                "config": config,
                "actual_bpb": actual_bpb,
                "actual_loss": summary.get("val_loss"),
                "train_time_s": summary.get("wall_time_s"),
                "predicted_bpb": (
                    previous_proposal.get("predicted_bpb")
                    if previous_proposal else None
                ),
                "prediction_confidence": (
                    previous_proposal.get("confidence")
                    if previous_proposal else None
                ),
                "prediction_error": (
                    actual_bpb - previous_proposal["predicted_bpb"]
                    if previous_proposal
                    and previous_proposal.get("predicted_bpb") is not None
                    and actual_bpb is not None
                    else None
                ),
                "observations": result.get("observations", []),
                "insights": result.get("insights", []),
                "tool_calls_count": len(result.get("tool_calls", [])),
                "next_proposal": (
                    proposal.get("description") if proposal else None
                ),
                "next_predicted_bpb": (
                    proposal.get("predicted_bpb") if proposal else None
                ),
                "next_confidence": (
                    proposal.get("confidence") if proposal else None
                ),
                "best_bpb_so_far": (
                    graph.get_best().get("metrics", {}).get("val_bpb")
                ),
            }
            append_log(log_path, log_entry)

            # Set up next iteration
            previous_proposal = proposal
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
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error(
                    f"{MAX_CONSECUTIVE_ERRORS} consecutive errors, giving up."
                )
                break
            wait = 60 * consecutive_errors
            logger.info(
                f"Retrying iteration in {wait}s "
                f"({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})..."
            )
            time.sleep(wait)

    # Final summary
    print_banner("FINAL RESULTS", char="*")
    print_scoreboard(graph)
    best = graph.get_best()
    if best:
        print(
            f"\n  BEST: {best['run_id']} -- "
            f"val_bpb = {best['metrics'].get('val_bpb')}"
        )
    print(f"\n  Check research_state.md for the agent's current thinking.")
    print(f"  Check experiments.db for full history (runs + observations + insights).")
    print(f"  Check loop_log.jsonl for meta-analysis (bring to Claude Code).\n")
    graph.close()


if __name__ == "__main__":
    main()
