# USD/IDR Forecast

This app forecasts USD/IDR rates and compares Serial and OpenMP training times.

## First-time setup on Windows

Install Python 3.10 or newer and MSYS2 UCRT64 GCC. In VS Code, install the Microsoft Python and Python Debugger extensions. (if not yet had)

Open the folder containing `app.py` in VS Code. Run these commands in the terminal:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

1. Press **F5** and choose **Run USD/IDR dashboard** if asked.
2. Open the address shown in the terminal, usually [http://127.0.0.1:5000](http://127.0.0.1:5000).
3. Click **Train one step** to compare how the threads work, or set **Days ahead** (1–365) and click **Run forecast**.

The app starts with `data.csv`. To use your own CSV, include `Date` and `USD_IDR_Rate` columns and use `YYYY-MM-DD` for dates.

Use **Test a past week** to compare seven predictions with the recorded rates.

Press **Shift+F5** to stop the app.
