# USD/IDR Forecast

1. Open the web thingymajigy folder in VS Code.
2. Press F5. If asked, choose Run USD/IDR dashboard.
3. Go to http://127.0.0.1:5000.

Choose an OpenMP thread count, pick a mode, and click Train one step.
Set Days ahead (1 to 365) and click Run forecast.

The app reads data.csv and shows how long Serial and OpenMP take to train.
To use your own data, upload a CSV with Date and USD_IDR_Rate columns. Use YYYY-MM-DD dates.

Use Test a past week to compare seven predicted rates with the CSV.

Press Shift+F5 to stop the app.
