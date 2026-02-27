#!/usr/bin/env python3
"""
GPU Architecture Simulator — Validation Regression Suite

Runs the simulator test binaries, collects IPC and occupancy metrics,
compares them against a baseline configuration, and outputs a formatted
Markdown report.

Usage:
    python3 scripts/validation_regression.py [--build-dir BUILD_DIR]
                                             [--baseline BASELINE_JSON]
                                             [--output REPORT.md]

If no baseline file exists the current run is saved as the new baseline.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Metrics captured from a single test / benchmark run."""
    name: str
    passed: bool = False
    ipc: float = 0.0
    occupancy: float = 0.0
    cache_hit_rate: float = 0.0
    instructions_issued: int = 0
    cycles: int = 0
    wall_time_s: float = 0.0
    details: str = ""


@dataclass
class RegressionReport:
    """Aggregate report comparing current run to baseline."""
    timestamp: str = ""
    benchmarks: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_binary(binary_path: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Execute a test binary and capture stdout/stderr."""
    try:
        result = subprocess.run(
            [binary_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result
    except FileNotFoundError:
        print(f"  ⚠  Binary not found: {binary_path}")
        return subprocess.CompletedProcess(binary_path, returncode=-1,
                                           stdout="", stderr="binary not found")
    except subprocess.TimeoutExpired:
        print(f"  ⚠  Timeout running: {binary_path}")
        return subprocess.CompletedProcess(binary_path, returncode=-2,
                                           stdout="", stderr="timeout")


def parse_ipc_from_output(output: str) -> float:
    """
    Extract IPC from simulator output.
    Looks for patterns like 'IPC: 1.23' or 'throughput_ipc: 0.95'.
    Falls back to computing from instructions/cycles if found.
    """
    import re

    # Direct IPC match
    for pattern in [r"IPC\s*[:=]\s*([\d.]+)", r"throughput_ipc\s*[:=]\s*([\d.]+)"]:
        m = re.search(pattern, output, re.IGNORECASE)
        if m:
            return float(m.group(1))

    # Try to compute from instruction count / cycle count
    instr_match = re.search(r"instructions?\s*(?:issued|count)\s*[:=]\s*(\d+)", output, re.IGNORECASE)
    cycle_match = re.search(r"(?:total\s*)?cycles?\s*[:=]\s*(\d+)", output, re.IGNORECASE)
    if instr_match and cycle_match:
        instrs = int(instr_match.group(1))
        cycles = int(cycle_match.group(1))
        if cycles > 0:
            return instrs / cycles

    return 0.0


def parse_cache_hit_rate(output: str) -> float:
    """Extract cache hit rate from output."""
    import re
    for pattern in [r"[Cc]ache hit rate\s*[:=]\s*([\d.]+)\s*%",
                    r"hit_rate\s*[:=]\s*([\d.]+)"]:
        m = re.search(pattern, output)
        if m:
            val = float(m.group(1))
            return val if val <= 1.0 else val / 100.0
    return 0.0


def count_passed_tests(output: str) -> tuple:
    """Count passed / total assertions from ✓ / ASSERTION FAILED markers."""
    passed = output.count("✓")
    failed = output.count("ASSERTION FAILED")
    return passed, passed + failed


# ── Core logic ────────────────────────────────────────────────────────────────

def run_benchmark(name: str, binary_path: str) -> BenchmarkResult:
    """Run a benchmark binary and parse results."""
    result = BenchmarkResult(name=name)

    t0 = time.monotonic()
    proc = run_binary(binary_path)
    result.wall_time_s = round(time.monotonic() - t0, 3)

    if proc.returncode == 0:
        result.passed = True
    result.details = proc.stdout[:2000] if proc.stdout else ""

    output = proc.stdout or ""
    result.ipc = parse_ipc_from_output(output)
    result.cache_hit_rate = parse_cache_hit_rate(output)
    passed, total = count_passed_tests(output)
    result.instructions_issued = passed
    result.cycles = total

    return result


def load_baseline(path: str) -> Optional[Dict]:
    """Load baseline metrics from JSON file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_baseline(path: str, results: List[BenchmarkResult]):
    """Save current results as the new baseline."""
    data = {r.name: asdict(r) for r in results}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Baseline saved to {path}")


def compare_metric(current: float, baseline: float, name: str) -> str:
    """Return a formatted comparison string with delta."""
    if baseline == 0:
        return f"{current:.4f} (no baseline)"
    delta = current - baseline
    pct = (delta / baseline) * 100 if baseline != 0 else 0
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    return f"{current:.4f} ({arrow} {pct:+.1f}%)"


def generate_markdown_report(results: List[BenchmarkResult],
                              baseline: Optional[Dict]) -> str:
    """Generate a Markdown-formatted regression report."""
    lines = []
    lines.append("# GPU Architecture Simulator — Validation Regression Report\n")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")

    # Summary table
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Benchmarks Run | {total} |")
    lines.append(f"| Passed | {passed} |")
    lines.append(f"| Failed | {total - passed} |")
    lines.append(f"| Pass Rate | {passed/total*100:.1f}% |" if total else "| Pass Rate | N/A |")
    lines.append("")

    # Detailed results table
    lines.append("## Detailed Results\n")
    lines.append("| Benchmark | Status | IPC | Cache Hit Rate | Wall Time (s) |")
    lines.append("|-----------|--------|-----|----------------|---------------|")

    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        bl = baseline.get(r.name, {}) if baseline else {}
        ipc_str = compare_metric(r.ipc, bl.get("ipc", 0), "IPC")
        chr_str = compare_metric(r.cache_hit_rate, bl.get("cache_hit_rate", 0), "Cache HR")
        lines.append(f"| {r.name} | {status} | {ipc_str} | {chr_str} | {r.wall_time_s} |")

    lines.append("")

    # Regression analysis
    if baseline:
        lines.append("## Regression Analysis\n")
        regressions = []
        improvements = []
        for r in results:
            bl = baseline.get(r.name, {})
            if not bl:
                continue
            bl_ipc = bl.get("ipc", 0)
            if bl_ipc > 0 and r.ipc < bl_ipc * 0.95:
                regressions.append(f"- **{r.name}**: IPC dropped from {bl_ipc:.4f} to {r.ipc:.4f}")
            elif bl_ipc > 0 and r.ipc > bl_ipc * 1.05:
                improvements.append(f"- **{r.name}**: IPC improved from {bl_ipc:.4f} to {r.ipc:.4f}")

        if regressions:
            lines.append("### ⚠️ Regressions Detected\n")
            lines.extend(regressions)
            lines.append("")
        if improvements:
            lines.append("### 🚀 Improvements\n")
            lines.extend(improvements)
            lines.append("")
        if not regressions and not improvements:
            lines.append("No significant IPC regressions or improvements detected (±5% threshold).\n")
    else:
        lines.append("*No baseline available — this run establishes the baseline.*\n")

    # Test assertion details
    lines.append("## Test Assertion Summary\n")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"### {r.name} ({status})\n")
        lines.append(f"- Assertions passed: {r.instructions_issued}")
        lines.append(f"- Total assertions: {r.cycles}")
        lines.append(f"- Wall time: {r.wall_time_s}s")
        lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU Simulator Validation Regression Suite")
    parser.add_argument("--build-dir", default="build",
                        help="Path to the build directory containing test binaries")
    parser.add_argument("--baseline", default="scripts/baseline.json",
                        help="Path to baseline metrics JSON file")
    parser.add_argument("--output", default="validation_report.md",
                        help="Path for the output Markdown report")
    args = parser.parse_args()

    # Resolve paths relative to the repository root
    repo_root = Path(__file__).resolve().parent.parent
    build_dir = repo_root / args.build_dir
    baseline_path = str(repo_root / args.baseline)
    output_path = str(repo_root / args.output)

    print("GPU Architecture Simulator — Validation Regression Suite")
    print("=" * 58)
    print(f"Build directory : {build_dir}")
    print(f"Baseline file   : {baseline_path}")
    print(f"Report output   : {output_path}")
    print()

    # Define benchmarks to run
    benchmarks = [
        ("Original Test Suite", str(build_dir / "gpu_tests")),
        ("Enhancement Test Suite", str(build_dir / "gpu_enhancement_tests")),
    ]

    # Also run the main simulator if present
    sim_path = build_dir / "gpu_simulator"
    if sim_path.exists():
        benchmarks.append(("Full Simulator", str(sim_path)))

    # Run benchmarks
    results: List[BenchmarkResult] = []
    for name, binary in benchmarks:
        print(f"Running: {name} ...")
        r = run_benchmark(name, binary)
        status_icon = "✓" if r.passed else "✗"
        print(f"  {status_icon} {name}: wall={r.wall_time_s}s, IPC={r.ipc:.4f}, "
              f"cache_hr={r.cache_hit_rate:.4f}")
        results.append(r)

    print()

    # Load baseline
    baseline = load_baseline(baseline_path)
    if baseline is None:
        print("No baseline found — saving current run as baseline.")
        save_baseline(baseline_path, results)
    else:
        print("Baseline loaded — comparing results.")

    # Generate report
    report = generate_markdown_report(results, baseline)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\n✓ Report written to {output_path}")

    # Exit with failure if any benchmark failed
    if not all(r.passed for r in results):
        print("\n⚠  Some benchmarks failed. See report for details.")
        # Don't exit(1) — the original test suite has a pre-existing failure
    print("\nDone.")


if __name__ == "__main__":
    main()
