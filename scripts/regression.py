#!/usr/bin/env python3
# Requires Python 3.6+ (f-strings) and Python 3.2+ (datetime.timezone)
"""
GPU Architecture Simulator Enhancement – Regression Script
===========================================================

Compares IPC (Instructions Per Cycle) and warp-occupancy metrics from a
set of benchmark simulations against a baseline configuration, and outputs
a Markdown report.

Usage
-----
    python3 regression.py [--baseline <json>] [--current <json>] [--output <md>]

JSON format (baseline and current)
-----------------------------------
{
    "config_name": "Volta-baseline",
    "benchmarks": [
        {
            "name": "rodinia_srad",
            "ipc": 2.15,
            "occupancy": 0.78,
            "memory_efficiency": 0.65,
            "l1_hit_rate": 0.82,
            "dram_row_hit_rate": 0.71,
            "wall_clock_s": 12.4
        },
        ...
    ]
}

If no JSON files are provided the script uses built-in synthetic data that
exercises all report sections.

Exit codes
----------
0  – all metrics within tolerance
1  – one or more regressions detected
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

IPC_REGRESSION_THRESHOLD       = 0.05   # 5 % IPC drop is a regression
OCCUPANCY_REGRESSION_THRESHOLD = 0.03   # 3 % occupancy drop
MEMORY_EFF_REGRESSION_THRESHOLD= 0.05   # 5 % memory-efficiency drop

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    name: str
    ipc: float
    occupancy: float                # 0.0 – 1.0
    memory_efficiency: float = 0.0  # 0.0 – 1.0
    l1_hit_rate: float = 0.0        # 0.0 – 1.0
    dram_row_hit_rate: float = 0.0  # 0.0 – 1.0
    wall_clock_s: float = 0.0


@dataclass
class SimConfig:
    config_name: str
    benchmarks: List[BenchmarkResult] = field(default_factory=list)


@dataclass
class BenchmarkComparison:
    name: str
    baseline_ipc: float
    current_ipc: float
    baseline_occ: float
    current_occ: float
    baseline_mem_eff: float
    current_mem_eff: float
    ipc_delta_pct: float        # positive = improvement
    occ_delta_pct: float
    mem_eff_delta_pct: float
    ipc_regressed: bool
    occ_regressed: bool
    mem_eff_regressed: bool

    @property
    def any_regression(self) -> bool:
        return self.ipc_regressed or self.occ_regressed or self.mem_eff_regressed

    @property
    def status_emoji(self) -> str:
        if self.any_regression:
            return "❌"
        if self.ipc_delta_pct > 0 or self.occ_delta_pct > 0:
            return "✅"
        return "➖"

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> SimConfig:
    with open(path) as f:
        raw = json.load(f)
    benchmarks = [BenchmarkResult(**b) for b in raw.get("benchmarks", [])]
    return SimConfig(config_name=raw.get("config_name", path), benchmarks=benchmarks)


def build_lookup(cfg: SimConfig) -> dict[str, BenchmarkResult]:
    return {b.name: b for b in cfg.benchmarks}


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def pct_delta(base: float, curr: float) -> float:
    """Return percentage change from base to curr (positive = improvement)."""
    if base == 0.0:
        return 0.0
    return (curr - base) / base * 100.0


def compare(baseline: SimConfig, current: SimConfig) -> List[BenchmarkComparison]:
    base_lut = build_lookup(baseline)
    curr_lut = build_lookup(current)

    comparisons: List[BenchmarkComparison] = []
    all_names = sorted(set(base_lut) | set(curr_lut))

    for name in all_names:
        b = base_lut.get(name)
        c = curr_lut.get(name)
        if b is None or c is None:
            continue  # skip benchmarks present in only one config

        ipc_d   = pct_delta(b.ipc, c.ipc)
        occ_d   = pct_delta(b.occupancy, c.occupancy)
        mem_d   = pct_delta(b.memory_efficiency, c.memory_efficiency)

        comparisons.append(BenchmarkComparison(
            name=name,
            baseline_ipc=b.ipc,       current_ipc=c.ipc,
            baseline_occ=b.occupancy, current_occ=c.occupancy,
            baseline_mem_eff=b.memory_efficiency,
            current_mem_eff=c.memory_efficiency,
            ipc_delta_pct=ipc_d,
            occ_delta_pct=occ_d,
            mem_eff_delta_pct=mem_d,
            ipc_regressed=     ipc_d   < -IPC_REGRESSION_THRESHOLD * 100,
            occ_regressed=     occ_d   < -OCCUPANCY_REGRESSION_THRESHOLD * 100,
            mem_eff_regressed= mem_d   < -MEMORY_EFF_REGRESSION_THRESHOLD * 100,
        ))

    return comparisons


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def _fmt_delta(delta: float, regressed: bool) -> str:
    """Format a percentage delta with colour-hint symbols."""
    sign = "+" if delta >= 0 else ""
    tag  = " ⚠️" if regressed else ""
    return f"{sign}{delta:.2f}%{tag}"


def generate_report(baseline: SimConfig,
                    current: SimConfig,
                    comparisons: List[BenchmarkComparison]) -> str:
    total   = len(comparisons)
    regressions = sum(1 for c in comparisons if c.any_regression)
    improvements= sum(1 for c in comparisons if c.ipc_delta_pct > 0 and not c.any_regression)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: List[str] = []
    lines += [
        "# GPU Architecture Simulator — Regression Report",
        "",
        f"**Generated:** {now}  ",
        f"**Baseline:** `{baseline.config_name}`  ",
        f"**Current:** `{current.config_name}`  ",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"| Total benchmarks | Regressions | Improvements | Neutral |",
        f"|-----------------|-------------|--------------|---------|",
        f"| {total} | {regressions} | {improvements} | {total - regressions - improvements} |",
        "",
    ]

    if regressions > 0:
        lines += [
            "> ⚠️ **Regression detected.** One or more benchmarks show a significant",
            "> performance drop relative to the baseline configuration.",
            "",
        ]
    else:
        lines += [
            "> ✅ **No regressions detected.** All benchmarks meet or exceed baseline.",
            "",
        ]

    # ---- Per-benchmark IPC table ------------------------------------------
    lines += [
        "## IPC Comparison",
        "",
        "| Benchmark | Baseline IPC | Current IPC | Delta | Status |",
        "|-----------|-------------|------------|-------|--------|",
    ]
    for c in comparisons:
        lines.append(
            f"| `{c.name}` | {c.baseline_ipc:.3f} | {c.current_ipc:.3f} "
            f"| {_fmt_delta(c.ipc_delta_pct, c.ipc_regressed)} | {c.status_emoji} |"
        )
    lines.append("")

    # ---- Occupancy table --------------------------------------------------
    lines += [
        "## Warp Occupancy",
        "",
        "| Benchmark | Baseline | Current | Delta | Status |",
        "|-----------|---------|---------|-------|--------|",
    ]
    for c in comparisons:
        lines.append(
            f"| `{c.name}` | {c.baseline_occ*100:.1f}% | {c.current_occ*100:.1f}% "
            f"| {_fmt_delta(c.occ_delta_pct, c.occ_regressed)} | {c.status_emoji} |"
        )
    lines.append("")

    # ---- Memory efficiency table ------------------------------------------
    lines += [
        "## Memory Efficiency",
        "",
        "| Benchmark | Baseline | Current | Delta | Status |",
        "|-----------|---------|---------|-------|--------|",
    ]
    for c in comparisons:
        lines.append(
            f"| `{c.name}` | {c.baseline_mem_eff*100:.1f}% | {c.current_mem_eff*100:.1f}% "
            f"| {_fmt_delta(c.mem_eff_delta_pct, c.mem_eff_regressed)} | {c.status_emoji} |"
        )
    lines.append("")

    # ---- Regressions detail -----------------------------------------------
    if regressions > 0:
        lines += [
            "## ⚠️ Regression Details",
            "",
        ]
        for c in comparisons:
            if not c.any_regression:
                continue
            lines.append(f"### `{c.name}`")
            if c.ipc_regressed:
                lines.append(
                    f"- **IPC**: {c.baseline_ipc:.3f} → {c.current_ipc:.3f} "
                    f"({_fmt_delta(c.ipc_delta_pct, True)}) — exceeds "
                    f"{IPC_REGRESSION_THRESHOLD*100:.0f}% regression threshold"
                )
            if c.occ_regressed:
                lines.append(
                    f"- **Occupancy**: {c.baseline_occ*100:.1f}% → {c.current_occ*100:.1f}% "
                    f"({_fmt_delta(c.occ_delta_pct, True)})"
                )
            if c.mem_eff_regressed:
                lines.append(
                    f"- **Memory efficiency**: {c.baseline_mem_eff*100:.1f}% → "
                    f"{c.current_mem_eff*100:.1f}% "
                    f"({_fmt_delta(c.mem_eff_delta_pct, True)})"
                )
            lines.append("")

    # ---- Methodology note -------------------------------------------------
    lines += [
        "---",
        "",
        "## Methodology",
        "",
        "Metrics are collected via the `SimulationEngine` sampling framework",
        "(SimPoint-style, configurable warmup + sample windows).  Thresholds:",
        "",
        f"| Metric | Regression threshold |",
        f"|--------|---------------------|",
        f"| IPC | {IPC_REGRESSION_THRESHOLD*100:.0f}% drop |",
        f"| Warp occupancy | {OCCUPANCY_REGRESSION_THRESHOLD*100:.0f}% drop |",
        f"| Memory efficiency | {MEMORY_EFF_REGRESSION_THRESHOLD*100:.0f}% drop |",
        "",
        "Benchmarks: Rodinia (srad, lud, hotspot), Parboil (sgemm, stencil),",
        "PolyBench (gemm, heat-3d).  See `tests/validation_suite.cpp` for the",
        "C++ driver that exercises the new GTO scheduler, Tensor Core, Sector",
        "Cache, and DRAM timing model.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic demo data (used when no JSON files are provided)
# ---------------------------------------------------------------------------

BASELINE_DATA = {
    "config_name": "baseline-LRU-RoundRobin",
    "benchmarks": [
        {"name": "rodinia_srad",     "ipc": 1.82, "occupancy": 0.72,
         "memory_efficiency": 0.61, "l1_hit_rate": 0.74, "dram_row_hit_rate": 0.58, "wall_clock_s": 18.2},
        {"name": "rodinia_lud",      "ipc": 2.41, "occupancy": 0.85,
         "memory_efficiency": 0.79, "l1_hit_rate": 0.88, "dram_row_hit_rate": 0.71, "wall_clock_s": 9.6},
        {"name": "rodinia_hotspot",  "ipc": 1.67, "occupancy": 0.68,
         "memory_efficiency": 0.55, "l1_hit_rate": 0.69, "dram_row_hit_rate": 0.49, "wall_clock_s": 24.1},
        {"name": "parboil_sgemm",    "ipc": 3.12, "occupancy": 0.91,
         "memory_efficiency": 0.88, "l1_hit_rate": 0.93, "dram_row_hit_rate": 0.82, "wall_clock_s": 5.3},
        {"name": "parboil_stencil",  "ipc": 1.44, "occupancy": 0.63,
         "memory_efficiency": 0.48, "l1_hit_rate": 0.61, "dram_row_hit_rate": 0.42, "wall_clock_s": 31.7},
        {"name": "polybench_gemm",   "ipc": 2.95, "occupancy": 0.89,
         "memory_efficiency": 0.84, "l1_hit_rate": 0.91, "dram_row_hit_rate": 0.78, "wall_clock_s": 6.8},
        {"name": "polybench_heat3d", "ipc": 1.58, "occupancy": 0.66,
         "memory_efficiency": 0.52, "l1_hit_rate": 0.65, "dram_row_hit_rate": 0.45, "wall_clock_s": 27.4},
    ],
}

CURRENT_DATA = {
    "config_name": "enhanced-GTO-SectorCache-TensorCore",
    "benchmarks": [
        {"name": "rodinia_srad",     "ipc": 2.06, "occupancy": 0.76,
         "memory_efficiency": 0.68, "l1_hit_rate": 0.81, "dram_row_hit_rate": 0.67, "wall_clock_s": 15.1},
        {"name": "rodinia_lud",      "ipc": 2.58, "occupancy": 0.87,
         "memory_efficiency": 0.82, "l1_hit_rate": 0.91, "dram_row_hit_rate": 0.75, "wall_clock_s": 8.9},
        {"name": "rodinia_hotspot",  "ipc": 1.91, "occupancy": 0.74,
         "memory_efficiency": 0.63, "l1_hit_rate": 0.77, "dram_row_hit_rate": 0.58, "wall_clock_s": 20.4},
        {"name": "parboil_sgemm",    "ipc": 3.47, "occupancy": 0.94,
         "memory_efficiency": 0.92, "l1_hit_rate": 0.96, "dram_row_hit_rate": 0.87, "wall_clock_s": 4.7},
        {"name": "parboil_stencil",  "ipc": 1.61, "occupancy": 0.69,
         "memory_efficiency": 0.55, "l1_hit_rate": 0.68, "dram_row_hit_rate": 0.51, "wall_clock_s": 28.3},
        {"name": "polybench_gemm",   "ipc": 3.28, "occupancy": 0.92,
         "memory_efficiency": 0.89, "l1_hit_rate": 0.94, "dram_row_hit_rate": 0.83, "wall_clock_s": 6.0},
        {"name": "polybench_heat3d", "ipc": 1.79, "occupancy": 0.71,
         "memory_efficiency": 0.60, "l1_hit_rate": 0.72, "dram_row_hit_rate": 0.54, "wall_clock_s": 24.0},
    ],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU Simulator Regression Reporter"
    )
    p.add_argument("--baseline", metavar="JSON",
                   help="Baseline results JSON (default: built-in synthetic data)")
    p.add_argument("--current",  metavar="JSON",
                   help="Current results JSON (default: built-in synthetic data)")
    p.add_argument("--output",   metavar="MD", default="regression_report.md",
                   help="Output Markdown file (default: regression_report.md)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.baseline:
        baseline = load_config(args.baseline)
    else:
        baseline = SimConfig(
            config_name=BASELINE_DATA["config_name"],
            benchmarks=[BenchmarkResult(**b) for b in BASELINE_DATA["benchmarks"]],
        )

    if args.current:
        current = load_config(args.current)
    else:
        current = SimConfig(
            config_name=CURRENT_DATA["config_name"],
            benchmarks=[BenchmarkResult(**b) for b in CURRENT_DATA["benchmarks"]],
        )

    comparisons = compare(baseline, current)
    report_md   = generate_report(baseline, current, comparisons)

    out_path = Path(args.output)
    out_path.write_text(report_md, encoding="utf-8")
    print(f"Report written to: {out_path}")

    regressions = sum(1 for c in comparisons if c.any_regression)
    if regressions:
        print(f"⚠️  {regressions} regression(s) detected — returning exit code 1")
        return 1
    print("✅ No regressions detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
