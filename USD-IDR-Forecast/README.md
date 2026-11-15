# USD/IDR Forecast

This app reads data.csv and compares Serial and OpenMP training times.

## Setup on Windows

Install Python 3.10 or newer and MSYS2 UCRT64 GCC. In VS Code, install the Microsoft Python and Python Debugger extensions.

Open this folder in VS Code and run these commands in the terminal:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

1. Press F5 and choose **Run USD/IDR dashboard** if asked.
2. Open http://127.0.0.1:5000.
3. Pick the number of threads and click **Train one step**, or set **Days ahead** and click **Run forecast**.

Press Shift+F5 to stop the app.
