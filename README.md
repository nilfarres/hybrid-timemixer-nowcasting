# Hybrid-TimeMixer for single-station nowcasting

Code accompanying the manuscript *Hybrid probabilistic nowcasting of temperature and rainfall from a single weather station*. The method uses the previous 48 hours of observations to forecast the next three hours at 10-minute intervals. Two configurations are provided, one using surface observations and one adding radiosonde-derived convective indices.

## Data and access

The University of Barcelona owns the meteorological observations used in the study. The authors do not have permission to redistribute them. Access is subject to authorisation from the University of Barcelona. This repository contains software and configuration only. The MIT licence covers the code and its documentation. The observations, processed datasets, checkpoints, prediction caches, calibration outputs and figures generated from the study data are excluded.

Reproduction requires the original authorised inputs or the equivalent processed datasets. The surface table uses 10-minute records from the Faculty of Physics rooftop station. Key fields are `TIMESTAMP`, `Temp_1_Avg` (°C), `Lluvia_Tot` (mm per 10 minutes), the humidity, pressure, wind and radiation predictors, and `segment_id` and `valid_t_end` for the forecast windows. The radiosonde variant also requires the processed sounding features or authorised profiles in the `YYMMDDHH.txt` format. Preserve the original timestamp labels and segmentation.

## Installation

Use a Python environment with PyTorch appropriate for your CPU or GPU. Python 3.12 was used for packaging checks. Install the direct dependencies from the repository root.

~~~sh
python -m pip install -r requirements.txt
~~~

The requirements describe the public scripts. The original training environment was not fully pinned. FFmpeg is only needed for MP4 animations.

## Reproduction with authorised inputs

Keep local data and outputs outside the repository when possible. The examples below use the ignored `private/` directory. Run all commands from the repository root.

### Prepare the datasets

If you have access to the original processed CSVs, use them directly for training or verification. To prepare compatible inputs from authorised raw files, the following commands illustrate how the preprocessing scripts connect.

~~~sh
python code/csv_clean.py --data-dir private/logger --output-dir private/csv
python code/imputa_NaN.py --input private/csv/CR3000_Tabla1_2011_2024_clean.csv --output-dir private/csv --train-end 2023-01-01 --pressure-train-method iqr --pressure-iqr-k 10
python code/corr_var.py --input private/csv/CR3000_Tabla1_2011_2024_imputed.csv --output-dir private/csv --train-end 2023-01-01
python code/dataset_nowcasting.py --input private/csv/CR3000_Tabla1_2011_2024_features_nowcasting.csv --output private/surface.csv --report private/surface_report.json --data-start 2011-01-01
python code/dataset_nowcasting.py --input private/csv/CR3000_Tabla1_2011_2024_features_nowcasting.csv --output private/radiosonde.csv --report private/radiosonde_report.json --data-start 2012-01-01 --radiosonde-dir private/profiles
~~~

### Train the ensembles

~~~sh
bash launch/train_surface.sh private/surface.csv private/runs/surface
bash launch/train_radiosonde.sh private/radiosonde.csv private/runs/radiosonde
~~~

These launchers transcribe the study's training arguments without the original cluster configuration. They fit ten ensemble members for each variant and use 2023 for validation and calibration and 2024 for testing. The `code/model_nowcasting_hibrid_timemixer_FINAL_*.py` scripts implement both models, including their baselines, calibration and evaluation. Use `--help` to see the `train`, `evaluate-ensemble`, `evaluate-advanced` and `predict` commands.

### Generate diagnostics and figures

- `export_prediction_diagnostics.py` reads run metrics and can export selected predictions using the matching model script, local checkpoints and authorised CSV.
- `animate_temperature_nowcasting_v9_2024_v4_clean_window_fixed.py` and `animate_precipitation_nowcasting_v9_2024.py` generate the prediction caches. Use `--no-render` to generate a cache without a video. Provide the model script with `--model-script`, the local run with `--model-results` and the authorised CSV with `--dataset`.
- `plot_best_worst_nowcasting_cases_2024.py` uses the two prediction caches and the authorised dataset to select and plot examples.
- `genera_figures_resultats_v9.py` reads the temperature and rainfall evaluation outputs from the two run archives and uses the diagnostics archive for the intense-rain example.
- `evaluate_daily_tmax_tmin_nowcasting_2024.py` calculates daily extrema from local forecasts. `compare_tmax_tmin_operational_nwp_hourly_homogeneous_2024.py` also compares the local temperature cache with hourly numerical weather predictions. Use the original forecast cache and the original Open-Meteo cache via `--nwp-cache-dir` to check this comparison. A fresh download may yield different files. The supplementary comparison uses valid-hour labels. Matching physical time, issue time and forecast lead across products was not established.
- `plot_tmax_tmin_mae_rmse_nwp.py` redraws a summary plot using aggregate error scores from the manuscript. It reads no station observations.
- `tools/audit_reference_forecasts.py` and `tools/audit_solar_alignment.py` check the deterministic references and timestamp labels. These optional historical audit scripts expect the authorised surface ZIP at `data/CR3000_Tabla1_2011_2024_dataset_nowcasting_no_rs.zip` and write to the ignored `generated_audit/` directory.

Use `python code/SCRIPT.py --help` to see the options for scripts with a command-line interface. Check the original results with the original authorised data and fitted artefacts. This repository does not supply those artefacts.

## Citation and licence

Use [CITATION.cff](CITATION.cff) to cite the software. The [MIT licence](LICENSE) is the one already used for the original TFM code.
