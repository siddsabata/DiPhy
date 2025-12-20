#!/usr/bin/env python3
"""Expand regime configs into per-tumor job definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build regime job manifest")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "regimes.yaml",
        help="Path to regimes.yaml",
    )
    return parser.parse_args()


def stable_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def round_sigfigs(value: float, sigfigs: int) -> float:
    if value == 0:
        return 0.0
    return round(value, sigfigs - int(math.floor(math.log10(abs(value)))) - 1)


def normalize_spec(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict) and "dist" in value:
        return value
    return {"dist": "fixed", "value": value}


def lhs_samples(rng: np.random.Generator, count: int, dims: int) -> np.ndarray:
    if dims == 0:
        return np.empty((count, 0))
    cut = np.linspace(0, 1, count + 1)
    u = rng.random((count, dims))
    samples = cut[:count, None] + u * (1.0 / count)
    for col in range(dims):
        rng.shuffle(samples[:, col])
    return samples


def eval_constraint(expr: str, values: Dict[str, Any]) -> bool:
    safe_globals = {"__builtins__": {}, "min": min, "max": max, "math": math}
    try:
        return bool(eval(expr, safe_globals, values))
    except NameError:
        return True


def flatten_for_key(value: Any, sigfigs: int) -> Tuple[Any, ...]:
    if isinstance(value, list):
        flattened: List[Any] = []
        for item in value:
            flattened.extend(flatten_for_key(item, sigfigs))
        return tuple(flattened)
    if isinstance(value, float):
        return (round_sigfigs(value, sigfigs),)
    return (value,)


def build_dedup_key(values: Dict[str, Any], sigfigs: int) -> Tuple[Any, ...]:
    key: List[Any] = []
    for param in sorted(values.keys()):
        key.append(param)
        key.extend(flatten_for_key(values[param], sigfigs))
    return tuple(key)


def sample_distribution(
    rng: np.random.Generator,
    spec: Dict[str, Any],
    lhs_value: float | None = None,
) -> Any:
    dist = spec["dist"]
    def as_float(value: Any) -> float:
        return float(value) if isinstance(value, str) else float(value)

    def as_int(value: Any) -> int:
        return int(float(value)) if isinstance(value, str) else int(value)

    if dist == "fixed":
        return spec["value"]
    if dist == "uniform":
        u = lhs_value if lhs_value is not None else rng.random()
        min_val = as_float(spec["min"])
        max_val = as_float(spec["max"])
        return min_val + (max_val - min_val) * u
    if dist == "log_uniform":
        u = lhs_value if lhs_value is not None else rng.random()
        lo = math.log10(as_float(spec["min"]))
        hi = math.log10(as_float(spec["max"]))
        return 10 ** (lo + (hi - lo) * u)
    if dist == "log_uniform_int":
        u = lhs_value if lhs_value is not None else rng.random()
        lo = math.log10(as_float(spec["min"]))
        hi = math.log10(as_float(spec["max"]))
        return int(round(10 ** (lo + (hi - lo) * u)))
    if dist == "int_range":
        min_val = as_int(spec["min"])
        max_val = as_int(spec["max"])
        return int(rng.integers(min_val, max_val + 1))
    if dist == "choice":
        return rng.choice(spec["values"]).item()
    raise ValueError(f"Unsupported dist: {dist}")


def prepare_param_specs(
    defaults: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Any] = {**defaults, **overrides}
    return {key: normalize_spec(value) for key, value in merged.items()}


def sample_params(
    rng: np.random.Generator,
    specs: Dict[str, Dict[str, Any]],
    lhs_row: Dict[str, float],
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}

    if "nsites" in specs:
        values["nsites"] = sample_distribution(rng, specs["nsites"], lhs_row.get("nsites"))

    unresolved = set(specs.keys()) - {"nsites"}

    for _ in range(5):
        progress = False
        for param in list(unresolved):
            spec = specs[param]
            dist = spec["dist"]
            if dist == "fraction_of":
                base_name = spec["of"]
                if base_name not in values:
                    continue
                base_val = values[base_name]
                base_num = min(base_val) if isinstance(base_val, list) else base_val
                fraction = sample_distribution(rng, spec["fraction"], lhs_row.get(f"fraction::{param}"))
                values[param] = int(round(base_num * fraction))
                unresolved.remove(param)
                progress = True
            elif dist == "per_site_from_base":
                if "nsites" not in values:
                    continue
                base = sample_distribution(rng, normalize_spec(spec["base"]), lhs_row.get(f"base::{param}"))
                multipliers = []
                for _ in range(int(values["nsites"])):
                    multipliers.append(sample_distribution(rng, normalize_spec(spec["multiplier"])))
                capacities = [int(round(base * mult)) for mult in multipliers]
                values[param] = capacities if values["nsites"] > 1 else capacities[0]
                unresolved.remove(param)
                progress = True
            else:
                values[param] = sample_distribution(rng, spec, lhs_row.get(param))
                unresolved.remove(param)
                progress = True
        if not unresolved:
            break
        if not progress:
            raise ValueError(f"Could not resolve parameters: {sorted(unresolved)}")

    return values


def build_jobs() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    experiment = config["experiment"]
    sampler = config.get("sampler", {})
    defaults = config.get("defaults", {})

    run_id = experiment["run_id"]
    output_root = Path(experiment["output_root"])
    output_dir = (ROOT / output_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs_path = output_dir / "jobs.jsonl"
    index_path = output_dir / "jobs_index.json"
    summary_path = output_dir / "summary.json"

    constraints = sampler.get("constraints", [])
    dedup_enabled = sampler.get("deduplication", {}).get("enabled", False)
    sigfigs = sampler.get("deduplication", {}).get("key_rounding_sigfigs", 4)

    jobs: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, int]] = {}

    global_seed = experiment.get("rng", {}).get("global_seed", 0)

    for regime in config.get("regimes", []):
        regime_id = regime["id"]
        n_tumors = int(regime["n_tumors"])
        rng = np.random.default_rng(stable_seed(global_seed, run_id, regime_id))

        defaults_params = defaults.get("sistem_parameters", {})
        overrides = regime.get("parameters", {})
        specs = prepare_param_specs(defaults_params, overrides)

        lhs_params = [
            name for name, spec in specs.items() if spec["dist"] in {"uniform", "log_uniform"}
        ]
        if "fraction_of" in {spec["dist"] for spec in specs.values()}:
            for name, spec in specs.items():
                if spec["dist"] == "fraction_of":
                    lhs_params.append(f"fraction::{name}")
        if "per_site_from_base" in {spec["dist"] for spec in specs.values()}:
            for name, spec in specs.items():
                if spec["dist"] == "per_site_from_base":
                    lhs_params.append(f"base::{name}")

        lhs_matrix = lhs_samples(rng, n_tumors * 3, len(lhs_params))

        seen = set()
        generated = 0
        cursor = 0
        while generated < n_tumors:
            if cursor >= lhs_matrix.shape[0]:
                lhs_matrix = np.vstack([lhs_matrix, lhs_samples(rng, n_tumors, len(lhs_params))])

            lhs_row = {
                lhs_params[idx]: lhs_matrix[cursor, idx]
                for idx in range(len(lhs_params))
            }
            cursor += 1

            params = sample_params(rng, specs, lhs_row)

            if dedup_enabled:
                key = build_dedup_key(params, sigfigs)
                if key in seen:
                    continue
                seen.add(key)

            constraint_ok = True
            for constraint in constraints:
                if not eval_constraint(constraint["expr"], params):
                    constraint_ok = False
                    break
            if not constraint_ok:
                continue

            generated += 1
            job = {
                "run_id": run_id,
                "regime_id": regime_id,
                "tumor_index": generated,
                "attempts_per_tumor": experiment["attempts_per_tumor"],
                "keep_policy": experiment["keep_policy"],
                "output_root": str(output_root),
                "sistem_parameters": params,
                "selection_library": regime["sistem"]["selection_library"],
                "anatomy": regime["sistem"]["anatomy"],
                "filters": {**defaults.get("filters", {}), **regime.get("filters", {})},
            }
            jobs.append(job)

        index[regime_id] = {
            "count": n_tumors,
            "start": len(jobs) - n_tumors,
            "end": len(jobs) - 1,
        }

    jobs_path.write_text("\n".join(json.dumps(job) for job in jobs) + "\n")
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "output_root": str(output_root),
                "total_jobs": len(jobs),
                "splits": experiment.get("splits", {}),
                "regimes": {k: v["count"] for k, v in index.items()},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    build_jobs()
