import csv
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.datastructures import ImmutableMultiDict
from werkzeug.exceptions import RequestEntityTooLarge

ROOT = Path(__file__).resolve().parent
BENCHMARK_RUNS = 3
MODES = ("serial", "openmp")
DEMO_MODES = ("serial", "openmp", "compare")
PREDICTION_LIMITS = {
    "steps": (22, 1, 365),
    "iterations": (5000, 100, 20000),
    "threads": (4, 1, 6),
}
BENCHMARK_LOCK = Lock()
DEMO_STEPS = 10
TEST_DAYS = 7
UPLOAD_NAME = "uploaded-rates.csv"
UPLOAD_LIMIT = 2 * 1024 * 1024
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


def read_rates(file):
    sample = file.readline(4096)
    file.seek(0)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.DictReader(file, delimiter=delimiter)
    columns = {name.strip().casefold().replace(" ", "_"): name for name in reader.fieldnames or []}
    date_column = columns.get("date") or columns.get("observation_date")
    rate_column = None
    for name in ("usd_idr_rate", "usd_idr", "usdidr", "rate", "close", "price"):
        if name in columns:
            rate_column = columns[name]
            break
    if not date_column or not rate_column:
        raise ValueError("Use a CSV with a Date column and a USD_IDR_Rate, Rate, or Close column.")

    rows = []
    for row in reader:
        value = (row[rate_column] or "").strip()
        if value.lower() in ("", ".", "n/a", "null"):
            continue
        try:
            day = date.fromisoformat((row[date_column] or "").strip())
            rate = float(value)
        except ValueError:
            raise ValueError(f"Line {reader.line_num} needs a YYYY-MM-DD date and a numeric rate.") from None
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"The rate on line {reader.line_num} must be positive and finite.")
        rows.append((day, rate))
        if len(rows) > 10000:
            raise ValueError("The CSV can contain up to 10,000 rates.")
    rows.sort()
    if len(rows) < 3:
        raise ValueError("The CSV needs at least three dated rates.")
    if any(left[0] == right[0] for left, right in zip(rows, rows[1:])):
        raise ValueError("The CSV contains duplicate dates. Keep one rate per date.")
    if all(rate == rows[0][1] for _, rate in rows):
        raise ValueError("The CSV needs at least two different rates.")
    return rows


def load_rates():
    try:
        file = (ROOT / UPLOAD_NAME).open(encoding="utf-8-sig", newline="")
    except FileNotFoundError:
        file = (ROOT / "data.csv").open(encoding="utf-8-sig", newline="")
    with file:
        return read_rates(file)


def save_upload(rows):
    temp = NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=ROOT, delete=False)
    path = Path(temp.name)
    try:
        with temp as file:
            writer = csv.writer(file)
            writer.writerow(("Date", "USD_IDR_Rate"))
            writer.writerows(rows)
        path.replace(ROOT / UPLOAD_NAME)
    finally:
        path.unlink(missing_ok=True)


def model_input(rows, iterations, steps, threads):
    # C++ reads the run settings first, then the rates.
    rates = " ".join(str(rate) for _, rate in rows)
    return f"{len(rows)} {iterations} {steps} {threads}\n{rates}\n"


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
    model_inputs = {
        "serial": model_input(rows, iterations, steps, 1),
        "openmp": model_input(rows, iterations, steps, threads),
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
    results = {}
    for version in MODES if mode == "compare" else (mode,):
        workers = 1 if version == "serial" else threads
        trace = run_model(version, model_input(rows, step, 1, workers), demo=True)
        if "threads" not in trace:
            raise RuntimeError("Rebuild both versions with Ctrl+Shift+B to use the training demo.")
        results[version] = trace
    return results


def uses_weekdays(rows):
    if not rows or (rows[-1][0] - rows[0][0]).days < 7:
        return False
    return all(day.weekday() < 5 for day, _ in rows)


def forecast_dates(rows, steps):
    day = rows[-1][0]
    weekdays_only = uses_weekdays(rows)
    dates = []
    while len(dates) < steps:
        if day == date.max:
            raise ValueError("The end of the forecast is outside the supported date range.")
        day += timedelta(days=1)
        if not weekdays_only or day.weekday() < 5:
            dates.append(day)
    return dates


def make_forecast(rows, steps, iterations, threads):
    dates = forecast_dates(rows, steps)
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


def test_week(rows, cutoff):
    training_rows = [row for row in rows if row[0] <= cutoff]
    first_test = len(training_rows)
    test_rows = rows[first_test:first_test + TEST_DAYS]
    if first_test < 3 or training_rows[-1][0] != cutoff or len(test_rows) != TEST_DAYS:
        raise ValueError("Pick a date in the CSV with at least two earlier rates and seven later rates.")

    last_known_rate = training_rows[-1][1]
    iterations = PREDICTION_LIMITS["iterations"][0]
    # Train on data through the cutoff before comparing the predictions with the held-out week.
    predictions = run_model("serial", model_input(training_rows, iterations, TEST_DAYS, 1))["forecast"]
    days = [
        {"date": day, "predicted": prediction, "actual": rate}
        for (day, rate), prediction in zip(test_rows, predictions, strict=True)
    ]
    return {
        "cutoff": cutoff,
        "days": days,
        "model_error": statistics.mean(abs(day["predicted"] - day["actual"]) for day in days),
        "baseline_error": statistics.mean(abs(last_known_rate - day["actual"]) for day in days),
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


def make_daily_chart(forecast):
    left, margin, date_spacing = 78, 42, 66
    top, bottom = 16, 141
    width = max(330, left + date_spacing * (len(forecast) - 1) + margin)
    rates = [point["serial"] for point in forecast]
    low, high = min(rates), max(rates)
    padding = max((high - low) * 0.08, 0.01)
    low, high = low - padding, high + padding

    def y_at(rate):
        return bottom - (bottom - top) * (rate - low) / (high - low)

    days = []
    for index, point in enumerate(forecast):
        x = left + (width - left - margin) * index / max(1, len(forecast) - 1)
        days.append({
            "label": date.fromisoformat(point["date"]).strftime("%d %b"),
            "x": round(x, 2),
            "y": round(y_at(point["serial"]), 2),
        })
    return {
        "width": width,
        "days": days,
        "guides": " ".join(f"M{day['x']} {top}V{bottom + 5}" for day in days),
        "dots": " ".join(f"M{day['x']} {day['y']}h.01" for day in days),
        "line": " ".join(f"{day['x']},{day['y']}" for day in days),
        "ticks": [(round(y_at(rate), 2), f"{rate:,.2f}") for rate in (low, (low + high) / 2, high)],
    }


def group_calculations(report):
    groups = [[] for _ in range(report["workers"])]
    jobs = ("a + h", "a − h", "b + h", "b − h", "c + h", "c − h")
    for thread, job in zip(report["threads"], jobs, strict=True):
        groups[thread].append(job)
    return groups


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return show_dashboard(request.form)
    return show_dashboard()


def show_dashboard(form=None, *, dataset_error=""):
    submitted = form is not None
    if not submitted:
        form = ImmutableMultiDict()

    default_days, minimum_days, maximum_days = PREDICTION_LIMITS["steps"]
    default_threads = PREDICTION_LIMITS["threads"][0]
    entered_days = form.get("steps", type=int)
    entered_threads = form.get("threads", type=int)
    days = default_days if entered_days is None else entered_days
    threads = default_threads if entered_threads is None else entered_threads
    mode = form.get("mode", "serial")
    demo_step = min(DEMO_STEPS, max(0, form.get("demo_step", 0, type=int)))
    previous_mode = form.get("previous_mode", "")
    previous_threads = form.get("previous_threads", 0, type=int)
    action = form.get("action")
    cutoff = form.get("cutoff", "")
    error = ""
    rows = []
    cutoffs = []
    report = None
    week = None
    demo_results = {}

    try:
        rows = load_rates()
        cutoffs = [day for day, _ in rows[2:-TEST_DAYS]]
        if submitted and action != "week":
            if entered_days is None or not minimum_days <= days <= maximum_days:
                raise ValueError(f"Enter a whole number from {minimum_days} to {maximum_days} for Days ahead.")
            if entered_threads not in THREAD_CHOICES:
                raise ValueError("Choose 1, 2, 4, or 6 OpenMP threads.")
            if mode not in DEMO_MODES:
                raise ValueError("Choose Serial, OpenMP, or Compare.")

        if action == "reset":
            demo_step = 0
        elif action in ("demo", "forecast", "week"):
            if not BENCHMARK_LOCK.acquire(blocking=False):
                raise RuntimeError("A run is already in progress. Try again in a moment.")
            try:
                if action == "demo":
                    step = 1
                    if mode == previous_mode and threads == previous_threads:
                        step = demo_step + 1
                    if step > DEMO_STEPS:
                        raise ValueError("You reached the last training step. Click Start over.")
                    demo_results = run_demo(rows, mode, step, threads)
                    demo_step = step
                    previous_mode, previous_threads = mode, threads
                    for result in demo_results.values():
                        result["groups"] = group_calculations(result)
                elif action == "week":
                    if not cutoffs:
                        raise ValueError("Use a CSV with at least ten rates to run this test.")
                    try:
                        cutoff_day = date.fromisoformat(cutoff) if cutoff else cutoffs[-1]
                    except ValueError:
                        raise ValueError("Choose a valid cutoff date.") from None
                    week = test_week(rows, cutoff_day)
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
        weekdays_only=uses_weekdays(rows),
        observations=len(rows),
        first_date=rows[0][0] if rows else None,
        last_date=rows[-1][0] if rows else None,
        using_upload=(ROOT / UPLOAD_NAME).is_file(),
        dataset_error=dataset_error,
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
        week=week,
        cutoffs=cutoffs,
        test_start=rows[-TEST_DAYS][0] if cutoffs else None,
        cutoff=cutoff or (cutoffs[-1].isoformat() if cutoffs else ""),
        chart=chart,
        daily_chart=make_daily_chart(report["forecast"]) if report else None,
        error=error,
        action=action,
    )


@app.post("/dataset")
def change_dataset():
    request.max_content_length = UPLOAD_LIMIT
    try:
        action = request.form.get("action")
        if action == "reset":
            (ROOT / UPLOAD_NAME).unlink(missing_ok=True)
        else:
            uploaded = request.files.get("rates")
            if not uploaded or not uploaded.filename:
                raise ValueError("Choose a CSV file to upload.")
            content = uploaded.read().decode("utf-8-sig")
            save_upload(read_rates(io.StringIO(content, newline="")))
    except RequestEntityTooLarge:
        return show_dashboard(dataset_error="Choose a CSV under 2 MB."), 413
    except UnicodeDecodeError:
        return show_dashboard(dataset_error="Save your file as a UTF-8 CSV and try again."), 400
    except (csv.Error, ValueError) as error:
        return show_dashboard(dataset_error=str(error)), 400
    except OSError:
        return show_dashboard(dataset_error="Could not save the dataset. Close the CSV if it is open and try again."), 500
    return redirect(url_for("index", _anchor="dataset"), code=303)


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
    if mode not in DEMO_MODES:
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
