# IFRS 9 LGD Modeling Framework via Advance Workout Approach

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-purple?logo=pandas&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-green?logo=numpy&style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-blue?logo=scipy&style=for-the-badge)
![statsmodels](https://img.shields.io/badge/statsmodels-Statistical%20Modeling-red?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal?style=for-the-badge)
![MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## Overview

## Project Structure
```
lgd_advance_workout_model/
├── models/          #Trainned model and parameters (pkl.)
│   ├── unbias_aft_model.pkl
│   ├── unbias_res_type_model.pkl
│   ├── unbias_res_type_encoder.pkl
│   ├── unbias_cf_hazard_model.pkl
│   ├── unbias_cf_amount_model.pkl
│   ├── unbias_cumulative_odr.pkl
│   ├── fwl_aft_model.pkl
│   ├── fwl_res_type_model.pkl
│   ├── fwl_res_type_encoder.pkl
│   ├── fwl_cf_hazard_model.pkl
│   └── fwl_cf_amount_model.pkl          #Not tracked by git
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_unbias_lgd.ipynb
│   ├── 03_fwl_lgd.ipynb
│   └── 04_residual_lgd.ipynb
├── src/
│   ├── data_prep.py
│   ├── lgd_builder.py
│   ├── fwl_model.py
│   ├── ....py
│   └── plot_function.py
├── data/          
│   ├── processed/
|   |   ├── default_account.parquet          #Not tracked by git
|   |   ├── default_cashflow.parquet          #Not tracked by git
|   |   ├── unbias_lgd.parquet          #Not tracked by git
|   |   └── fwl_lgd.parquet          #Not tracked by git
│   └── raw/
|   |   ├── main_transaction_data.parquet          #Not tracked by git
|   |   ├── cashflow_data.parquet          #Not tracked by git
|   |   ├── mev_transformed.parquet          #Not tracked by git
|   |   └── mev_sign_transformed.parquet          #Not tracked by git
├── requirements.txt
└── README.md
```
