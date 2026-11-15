import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
BENCHMARK_RUNS = 3
MODES = ("serial", "openmp")
PREDICTION_LIMITS = {
    "steps": (22, 1, 60),
    "iterations": (5000, 100, 20000),
    "threads": (4, 1, 6),
}
BENCHMARK_LOCK = Lock()
DEMO_STEPS = 10
THREAD_CHOICES = (1, 2, 4, 6)
CPP_SOURCE = "cpp/usd_idr_prediction.cpp"
SOURCE_FILES = (CPP_SOURCE, "app.py", "templates/index.html")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4096


def build_models():
    compiler = Path(shutil.which("g++") or r"C:\msys64\ucrt64\bin\g++.exe")
    if not compiler.is_file():
        raise FileNotFoundError("Install MSYS2 UCRT64 g++ or add it to PATH.")

    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    source_file = ROOT / CPP_SOURCE
    flags = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic"]
    env = os.environ.copy()
    env["PATH"] = str(compiler.parent) + os.pathsep + env.get("PATH", "")
    for mode in MODES:
        output = build / f"{mode}.exe"
        command = [str(compiler), *flags, str(source_file), "-o", str(output)]
        if mode == "openmp":
            command.append("-fopenmp")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        print(f"Built {output}")

    # VS Code can run the programs even if MSYS2 is outside its PATH.
    for name in ("libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll", "libgomp-1.dll"):
        library = compiler.parent / name
        if library.is_file():
            shutil.copy2(library, build)


def load_rates():
    rows = []
    with (ROOT / "data.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            day = date.fromisoformat(row["Date"])
            rate = float(row["USD_IDR_Rate"])
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Every rate in data.csv must be positive and finite.")
            rows.append((day, rate))
    rows.sort()
    if not 3 <= len(rows) <= 10000:
        raise ValueError("data.csv needs 3–10,000 observations.")
    if any(left[0] == right[0] for left, right in zip(rows, rows[1:])):
        raise ValueError("data.csv contains duplicate dates.")
    if all(rate == rows[0][1] for _, rate in rows):
        raise ValueError("data.csv needs at least two distinct rates.")
    return rows


def run_model(mode, input_text, *, demo=False):
    executable = ROOT / "build" / f"{mode}.exe"
    if not executable.is_file():
        raise RuntimeError("Build the C++ models in VS Code with Ctrl+Shift+B, then try again.")
    command = [str(executable)]
    if demo:
        command.append("--demo")
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=ROOT,
        timeout=20,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"The {mode} program failed. Rebuild with Ctrl+Shift+B.")
    return json.loads(completed.stdout)


def run_benchmark(rows, steps, iterations, threads):
    rates = " ".join(str(rate) for _, rate in rows)
    input_header = f"{len(rows)} {iterations} {steps}"
    model_inputs = {
        "serial": f"{input_header} 1\n{rates}\n",
        "openmp": f"{input_header} {threads}\n{rates}\n",
    }
    runs = {mode: [] for mode in MODES}
    for trial in range(BENCHMARK_RUNS):
        # Alternate which mode goes first so the same mode doesn't always run cold.
        order = MODES if trial % 2 == 0 else MODES[::-1]
        for mode in order:
            runs[mode].append(run_model(mode, model_inputs[mode]))

    timings = {}
    for mode, mode_runs in runs.items():
        samples = [run["training_ms"] for run in mode_runs]
        timings[mode] = {
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
            "workers": mode_runs[0]["workers"],
        }
    return runs["serial"][0], runs["openmp"][0], timings


def run_demo(rows, mode, step, threads):
    rates = " ".join(str(rate) for _, rate in rows)
    results = {}
    for version in MODES if mode == "compare" else (mode,):
        workers = 1 if version == "serial" else threads
        model_input = f"{len(rows)} {step} 1 {workers}\n{rates}\n"
        trace = run_model(version, model_input, demo=True)
        if "threads" not in trace:
            raise RuntimeError("Rebuild both versions with Ctrl+Shift+B to use the training demo.")
        results[version] = trace
    return results


def make_forecast(rows, steps, iterations, threads):
    last_day = rows[-1][0]
    dates = [last_day + timedelta(days=offset) for offset in range(1, steps + 1)]
    serial, openmp, timings = run_benchmark(rows, steps, iterations, threads)
    forecast = []
    for day, serial_rate, openmp_rate in zip(dates, serial["forecast"], openmp["forecast"], strict=True):
        forecast.append({"date": day.isoformat(), "serial": serial_rate, "openmp": openmp_rate})
    openmp_ms = timings["openmp"]["median_ms"]
    return {
        "settings": {"steps": steps, "iterations": iterations, "threads": threads},
        "observations": len(rows),
        "windows": len(rows) - 2,
        "history": [{"date": day.isoformat(), "rate": rate} for day, rate in rows],
        "forecast": forecast,
        "timings": timings,
        "speedup": timings["serial"]["median_ms"] / openmp_ms if openmp_ms > 0 else None,
        "rmse": serial["rmse"],
    }


def make_chart(rows, forecast):
    rates = [rate for _, rate in rows] + [point["serial"] for point in forecast]
    low, high = min(rates), max(rates)
    left, right = 50, 610
    top, bottom = 18, 160

    def y_at(rate):
        return bottom - (bottom - top) * (rate - low) / (high - low)

    points = [
        f"{left + (right - left) * index / (len(rates) - 1):.2f},{y_at(rate):.2f}"
        for index, rate in enumerate(rates)
    ]
    last_day = date.fromisoformat(forecast[-1]["date"]) if forecast else rows[-1][0]
    return {
        "history": " ".join(points[:len(rows)]),
        "forecast": " ".join(points[len(rows) - 1:]) if forecast else "",
        "ticks": [(y_at(rate), f"{rate:,.0f}") for rate in (low, (low + high) / 2, high)],
        "first_day": rows[0][0].strftime("%d %b"),
        "last_day": last_day.strftime("%d %b"),
    }


def group_calculations(report):
    groups = [[] for _ in range(report["workers"])]
    jobs = ("a + h", "a − h", "b + h", "b − h", "c + h", "c − h")
    for thread, job in zip(report["threads"], jobs, strict=True):
        groups[thread].append(job)
    return groups


@app.route("/", methods=["GET", "POST"])
def index():
    default_days, minimum_days, maximum_days = PREDICTION_LIMITS["steps"]
    default_threads = PREDICTION_LIMITS["threads"][0]
    days = request.form.get("steps", default_days, type=int)
    threads = request.form.get("threads", default_threads, type=int)
    mode = request.form.get("mode", "serial")
    demo_step = min(DEMO_STEPS, max(0, request.form.get("demo_step", 0, type=int)))
    previous_mode = request.form.get("previous_mode", "")
    previous_threads = request.form.get("previous_threads", 0, type=int)
    action = request.form.get("action")
    error = ""
    rows = []
    report = None
    demo_results = {}

    try:
        rows = load_rates()
        if request.method == "POST":
            if request.form.get("steps", type=int) is None or not minimum_days <= days <= maximum_days:
                raise ValueError(f"Enter a whole number from {minimum_days} to {maximum_days} for Days ahead.")
            if request.form.get("threads", type=int) not in THREAD_CHOICES:
                raise ValueError("Choose 1, 2, 4, or 6 OpenMP threads.")
            if mode not in (*MODES, "compare"):
                raise ValueError("Choose Serial, OpenMP, or Compare.")

            if action == "reset":
                demo_step = 0
            elif action in ("demo", "forecast"):
                if not BENCHMARK_LOCK.acquire(blocking=False):
                    raise RuntimeError("A run is already in progress. Try again in a moment.")
                try:
                    if action == "demo":
                        same_mode = (mode, threads) == (previous_mode, previous_threads)
                        step = demo_step + 1 if same_mode else 1
                        if step > DEMO_STEPS:
                            raise ValueError("You reached the last training step. Click Start over.")
                        demo_results = run_demo(rows, mode, step, threads)
                        demo_step = step
                        previous_mode, previous_threads = mode, threads
                        for result in demo_results.values():
                            result["groups"] = group_calculations(result)
                    else:
                        report = make_forecast(rows, days, PREDICTION_LIMITS["iterations"][0], threads)
                finally:
                    BENCHMARK_LOCK.release()
    except subprocess.TimeoutExpired:
        error = "The model took too long. Try again."
    except (OSError, ValueError, KeyError, RuntimeError) as problem:
        error = str(problem)

    chart = make_chart(rows, report["forecast"] if report else []) if rows else None
    return render_template(
        "index.html",
        days=days,
        forecast_days=(minimum_days, maximum_days),
        threads=threads,
        thread_choices=THREAD_CHOICES,
        mode=mode,
        demo_step=demo_step,
        max_demo_steps=DEMO_STEPS,
        previous_mode=previous_mode,
        previous_threads=previous_threads,
        demo_results=demo_results,
        report=report,
        chart=chart,
        error=error,
        action=action,
    )


@app.get("/source")
def source():
    files = [(name, (ROOT / name).read_text(encoding="utf-8")) for name in SOURCE_FILES]
    return render_template("index.html", source_files=files)


@app.post("/api/demo")
def demo():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Choose a mode and step."), 400
    mode = payload.get("mode")
    step = payload.get("step")
    default, minimum, maximum = PREDICTION_LIMITS["threads"]
    threads = payload.get("threads", default)
    if mode not in (*MODES, "compare"):
        return jsonify(error="Choose Serial, OpenMP, or Compare."), 400
    if type(step) is not int or not 1 <= step <= DEMO_STEPS:
        return jsonify(error=f"Choose a step from 1 to {DEMO_STEPS}."), 400
    if type(threads) is not int or not minimum <= threads <= maximum:
        return jsonify(error=f"Choose {minimum} to {maximum} OpenMP threads."), 400
    if not BENCHMARK_LOCK.acquire(blocking=False):
        return jsonify(error="A run is already in progress. Try again in a moment."), 409
    try:
        return jsonify(step=step, modes=run_demo(load_rates(), mode, step, threads))
    except subprocess.TimeoutExpired:
        return jsonify(error="The training step took too long."), 504
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return jsonify(error=str(error)), 500
    finally:
        BENCHMARK_LOCK.release()


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the benchmark settings as a JSON object."), 400
    settings = {}
    for name, (default, lower, upper) in PREDICTION_LIMITS.items():
        value = payload.get(name, default)
        if type(value) is not int or not lower <= value <= upper:
            return jsonify(error=f"{name.capitalize()} must be a whole number from {lower} to {upper}."), 400
        settings[name] = value

    if not BENCHMARK_LOCK.acquire(blocking=False):
        return jsonify(error="A benchmark is already running. Try again when it finishes."), 409
    try:
        return jsonify(make_forecast(load_rates(), **settings))
    except subprocess.TimeoutExpired:
        return jsonify(error="Training exceeded the time limit. Try fewer iterations."), 504
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return jsonify(error=str(error)), 500
    finally:
        BENCHMARK_LOCK.release()


if __name__ == "__main__":
    if sys.argv[1:] == ["--build"]:
        try:
            build_models()
        except (OSError, subprocess.CalledProcessError) as error:
            sys.exit(str(error))
    else:
        app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
