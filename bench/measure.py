#!/usr/bin/env python3
"""Measure complete process latency and reported render stages without changing renderer logic."""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def capture(args):
    try:
        result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return {"command": args, "returncode": result.returncode, "output": result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "error": str(error)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--mode", choices=["cpu", "gpu", "both"], default="both")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    binary = (ROOT / args.binary).resolve()
    output = (ROOT / args.output).resolve()
    if (output / "raw.csv").exists():
        parser.error("Result directory already contains raw.csv; choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE")
    source = hashlib.sha256()
    for folder in ["src", "include", "external"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file():
                source.update(str(path.relative_to(ROOT)).encode())
                source.update(path.read_bytes())
    source.update((ROOT / "CMakeLists.txt").read_bytes())
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture(["git", "rev-parse", "HEAD"])["output"],
        "source_sha256": source.hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "binary": str(binary.relative_to(ROOT)),
        "environment": {"system": platform.system(), "release": platform.release(), "python": platform.python_version(), "omp_threads": args.threads, "omp_dynamic": False},
        "commands": [capture(c) for c in [["lscpu"], ["free", "-b"], ["cat", "/etc/os-release"], ["cmake", "--version"], ["g++", "--version"], ["g++-12", "--version"], ["nvcc", "--version"], ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]]],
        "protocol": {"sizes": [[320,180],[800,450]], "samples_per_pixel": 50, "max_depth": 20, "warmups_per_mode_size": 1, "measured_trials_per_mode_size": 5, "trial_order": "warmup then five measured; CPU then GPU each trial", "compile_options": "Release; C++17; ENABLE_FAST_MATH=ON; CUDA architecture detected by run.sh", "seed": "Scene srand(77), CPU row seed j*1337+42, GPU time(NULL); no common seed option exists."},
        "timing_scope": {"wall_ms": "New process through image export and exit, including CUDA initialization for GPU.", "render_ms": "CPU render (sampling plus ordinary tone mapping); GPU CUDA event for HDR sampling kernel, excluding adaptive tone mapping."},
        "limitations": ["Different random generators and tone mapping; no claim of pixel equivalence.", "CPU and GPU render-stage timings cover different work; headline comparisons use full process wall time.", "Laptop power state and WSL scheduling may vary; five trials describe only this run.", "The built-in --benchmark is not used because its camera is not reinitialized for each image size."]
    }
    cache = binary.parent / "CMakeCache.txt"
    if cache.exists():
        keys = ("CMAKE_BUILD_TYPE:", "CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS_RELEASE:", "CMAKE_CUDA_COMPILER:", "CMAKE_CUDA_ARCHITECTURES:", "ENABLE_FAST_MATH:", "BUILD_CPU_ONLY:")
        manifest["cmake_cache"] = [line for line in cache.read_text().splitlines() if line.startswith(keys)]
    (output / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    modes = ["cpu", "gpu"] if args.mode == "both" else [args.mode]
    for width, height in [(320,180),(800,450)]:
        for trial in range(6):
            for mode in modes:
                image = output / "render.png"
                image.unlink(missing_ok=True)
                command = [str(binary)] + (["--cpu"] if mode == "cpu" else []) + ["-w",str(width),"-h",str(height),"-s","50","-d","20","--profile","-o",str(image)]
                started = time.perf_counter_ns()
                process = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                wall_ms = (time.perf_counter_ns()-started)/1e6
                normalized = process.stdout.replace(str(ROOT), "<repo>")
                log = f"{mode}-{width}x{height}-{trial}.log"
                header = shlex.join(command).replace(str(ROOT), "<repo>")
                (output / "logs" / log).write_text(f"Command: {header}\nOMP_NUM_THREADS={args.threads} OMP_DYNAMIC=FALSE\n" + normalized + f"\nExit code: {process.returncode}\nWall milliseconds: {wall_ms:.6f}\n")
                if process.returncode != 0:
                    raise RuntimeError(f"Rendering failed, see {log}")
                png = image.read_bytes()
                if png[:8] != bytes([137,80,78,71,13,10,26,10]) or struct.unpack(">II", png[16:24]) != (width,height):
                    raise RuntimeError(f"Invalid PNG from {log}")
                pattern = r"Render \(CPU/OpenMP\)\s+([0-9.]+) ms" if mode == "cpu" else r"Render kernel:\s+([0-9.]+) ms"
                match = re.search(pattern, process.stdout)
                if not match and mode == "cpu":
                    match = re.search(r"Temps: ([0-9.]+) secondes", process.stdout)
                    render_ms = float(match.group(1))*1000 if match else None
                else:
                    render_ms = float(match.group(1)) if match else None
                if render_ms is None:
                    raise RuntimeError(f"Missing render timing in {log}")
                row = dict(mode=mode,width=width,height=height,spp=50,depth=20,threads=args.threads,trial=trial,warmup=int(trial==0),render_ms=render_ms,wall_ms=wall_ms,exit_code=process.returncode,png_sha256=hashlib.sha256(png).hexdigest(),log="logs/"+log)
                rows.append(row)
                with (output / "raw.csv").open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{mode} {width}x{height} trial={trial} render={render_ms:.3f} ms wall={wall_ms:.3f} ms",flush=True)
    (output / "render.png").unlink()
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "environment.json").write_text(json.dumps(manifest, indent=2)+"\n")

if __name__ == "__main__":
    main()
