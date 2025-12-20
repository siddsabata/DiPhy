#!/usr/bin/env python3
"""Run a single tumor job from the manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from sistem import GrowthSimulator, Parameters
except ImportError:  # pragma: no cover - fall back to explicit modules for older layouts
    from sistem.growth_sim import GrowthSimulator
    from sistem.parameters import Parameters

from sistem import anatomy as anatomy_module
from sistem import selection as selection_module

import inspect

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = ("gs.pkl", "SNV_events.tsv", "clone_tree.nwk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single manifest job")
    parser.add_argument("--jobs", type=Path, required=True, help="Path to jobs.jsonl")
    parser.add_argument("--task-id", type=int, required=True, help="0-based index into jobs.jsonl")
    return parser.parse_args()


def stable_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def read_job_line(path: Path, index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Job index {index} out of range for {path}")


def ensure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def filter_sistem_params(params: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(Parameters)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return params
    allowed = {name for name in signature.parameters if name != "self"}
    return {key: value for key, value in params.items() if key in allowed}


def load_selection_library(params: Parameters, spec: Dict[str, Any]):
    class_name = spec["class"]
    library_cls = getattr(selection_module, class_name)
    library = library_cls(params=params)
    init_conf = spec.get("init", {})
    if hasattr(library, "initialize"):
        if init_conf.get("from_params"):
            library.initialize(params=params)
        elif init_conf:
            library.initialize(**init_conf)
        else:
            library.initialize(params=params)
    return library


def apply_anatomy_config(anatomy, spec: Dict[str, Any], params: Parameters) -> None:
    distances = spec.get("distances")
    if distances and distances.get("method") == "random":
        if hasattr(anatomy, "initialize_distances"):
            anatomy.initialize_distances(method="random")
        else:
            raise ValueError("Anatomy does not support initialize_distances")

    metastatic = spec.get("metastatic_libraries")
    if metastatic and metastatic.get("enabled"):
        if hasattr(anatomy, "create_random_metastatic_libraries"):
            kwargs = {
                "method": metastatic.get("method", "random"),
                "params": params,
            }
            if not metastatic.get("from_params", False):
                if "alter_prop" in metastatic:
                    kwargs["alter_prop"] = metastatic["alter_prop"]
                if "CN_coeff" in metastatic:
                    kwargs["CN_coeff"] = metastatic["CN_coeff"]
            anatomy.create_random_metastatic_libraries(**kwargs)
        else:
            raise ValueError("Anatomy does not support create_random_metastatic_libraries")


def load_anatomy(library, params: Parameters, spec: Dict[str, Any]):
    class_name = spec["class"]
    anatomy_cls = getattr(anatomy_module, class_name)
    anatomy = anatomy_cls(library, params=params)
    apply_anatomy_config(anatomy, spec, params)
    return anatomy


def missing_outputs(attempt_dir: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (attempt_dir / name).exists()]


def is_success(attempt_dir: Path) -> bool:
    return not missing_outputs(attempt_dir)


def main() -> None:
    args = parse_args()
    job = read_job_line(args.jobs, args.task_id)

    output_root = Path(job["output_root"])
    output_dir = (ROOT / output_root).resolve()
    regime_id = job["regime_id"]
    tumor_index = int(job["tumor_index"])

    tumor_dir = output_dir / "regimes" / regime_id / "tumors" / f"tumor_{tumor_index:05d}"
    tumor_dir.mkdir(parents=True, exist_ok=True)

    success_path = tumor_dir / "success.json"
    if success_path.exists():
        try:
            payload = json.loads(success_path.read_text())
            attempt_dir = tumor_dir / payload["attempt_dir"]
            if is_success(attempt_dir):
                return
        except Exception:
            pass

    (tumor_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")

    attempts = int(job["attempts_per_tumor"])
    keep_policy = job["keep_policy"]

    for attempt_index in range(1, attempts + 1):
        attempt_dir = tumor_dir / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        seed = stable_seed(job["run_id"], regime_id, tumor_index, attempt_index)
        ensure_seed(seed)

        try:
            sistem_params = filter_sistem_params(job["sistem_parameters"])
            dropped = sorted(set(job["sistem_parameters"]) - set(sistem_params))
            params = Parameters(out_dir=str(attempt_dir), **sistem_params)
            library = load_selection_library(params, job["selection_library"])
            anatomy = load_anatomy(library, params, job["anatomy"])

            gs = GrowthSimulator(anatomy)
            gs.simulate_agents(params=params)
            gs.save_checkpoint(str(attempt_dir / "gs.pkl"))

            gs.sample_cells(params=params)
            gs.simulate_clonal_lineage(params=params, out_dir=str(attempt_dir))
        except Exception as exc:  # noqa: BLE001
            # Keep retrying within the same array task.
            failure_log = tumor_dir / "failures.log"
            with failure_log.open("a", encoding="utf-8") as handle:
                if "dropped" in locals() and dropped:
                    handle.write(f"attempt={attempt_index} seed={seed} dropped={','.join(dropped)}\n")
                handle.write(f"attempt={attempt_index} seed={seed} error={exc}\n")
                handle.write(traceback.format_exc())
            continue

        missing = missing_outputs(attempt_dir)
        if not missing:
            success_payload = {
                "run_id": job["run_id"],
                "regime_id": regime_id,
                "tumor_index": tumor_index,
                "attempt_index": attempt_index,
                "attempt_dir": attempt_dir.name,
                "seed": seed,
            }
            success_path.write_text(json.dumps(success_payload, indent=2) + "\n")
            if keep_policy == "first_success":
                break
        else:
            failure_log = tumor_dir / "failures.log"
            with failure_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"attempt={attempt_index} seed={seed} missing={','.join(missing)}\n"
                )


if __name__ == "__main__":
    main()
