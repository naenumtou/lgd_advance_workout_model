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

This repository implements a **Workout-based Loss Given Default (LGD)** model aligned with IFRS 9 Expected Credit Loss (ECL) requirements. The framework estimates **unbias LGD, forward-looking LGD and Residual LGD** by modeling recovery behavior, cashflow timing, resolution pathways, and macroeconomic relationships throughout the post-default lifecycle. It is designed for Stage 1, Stage 2, and Stage 3 impairment calculations and supports transparent, auditable, and production ready credit risk modeling suitable for regulatory and financial reporting purposes.

<p align="center">
<img width="1983" height="793" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/58112a29-7b07-4a0e-bcb7-3232f193611e" />
</p>

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
│   ├── residual_curve.py
│   └── plot_function.py
├── data/          
│   ├── processed/
|   |   ├── default_account.parquet          #Not tracked by git
|   |   ├── default_cashflow.parquet          #Not tracked by git
|   |   ├── unbias_lgd.parquet          #Not tracked by git
|   |   ├── fwl_lgd.parquet          #Not tracked by git
|   |   └── residual_lgd.parquet          #Not tracked by git
│   └── raw/
|   |   ├── main_transaction_data.parquet          #Not tracked by git
|   |   ├── cashflow_data.parquet          #Not tracked by git
|   |   ├── mev_transformed.parquet          #Not tracked by git
|   |   └── mev_sign_transformed.parquet          #Not tracked by git
├── requirements.txt
└── README.md
```

## Project Details
### 0. Data Preparation
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/245d23e5-ee68-4562-ade0-a82071921940" />
</p>

This framework transforms raw account-level default and recovery data into a structured analytical dataset for IFRS 9 LGD modeling. The process is divided into two major components; **1) Default Population** and **2) Monthly Recovery Panel**. Together, these steps create a clean, standardized, and model-ready dataset that supports survival analysis, recovery estimation, cashflow modeling, and residual LGD computation.

The methodology is designed to ensure consistency across default events, remove ambiguous observations, and preserve the chronological behavior of recovery cashflows over time. It also standardizes the treatment of resolved and ongoing collections to support unbiased recovery analysis and forward-looking LGD estimation.

#### 0.1 Default Population
The Default Population Engine constructs the master default account dataset used throughout the LGD modeling framework. The objective is to identify the first valid default event per account, determine the latest resolution status, and calculate the time to resolution metric required for recovery modeling.

**Key Processes**


**Business Logic**
- Accounts under ongoing collection statuses use the latest observable modeling period as the effective resolution date.
- Resolved accounts use the actual resolution date.
- Time to resolution is capped at a minimum of one month for modeling stability.
- Each account can only contribute one default observation to the modeling population.

#### 0.2 Monthly Recovery Panel
The Monthly Recovery Panel converts account-level default observations into a monthly longitudinal recovery panel. This panel captures the full recovery timeline for each account and enables month-by-month recovery analysis. The process creates a continuous monthly observation structure from default date until resolution or latest observable period. Actual cashflows are then mapped into this timeline to support recovery probability and recovery amount modeling.

**Business Logic**
- Month 0 represents the default month and cannot contain recovery cashflows.
- Missing monthly recoveries are treated as zero recovery observations.
- Ongoing accounts remain observable until the latest modeling period.
- Resolved accounts terminate at the actual resolution month.
- The panel preserves the sequential behavior of post-default recovery dynamics.

### 1. Unbias LGD
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/4146f938-d142-4249-83c2-9dce093159e2" />
</p>

### 2. Forward-looking LGD
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/9875a6af-5090-4766-beb2-e067c3ed9d23" />
</p>


### 3. Residual LGD
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/a7a46938-cdca-4a97-b66c-6c1225ac9d81" />
</p>


## License
MIT · Built for learning purposes













