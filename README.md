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
This project implements a Workout-based Loss Given Default (LGD) model designed to support IFRS 9 Expected Credit Loss (ECL) calculation. The framework estimates Unbias, Forward-looking LGD and Residual LGD by modeling post-default recovery behavior, recovery timing, resolution pathways, and discounted recovery cashflows throughout the collection lifecycle. By tracking how recoveries evolve over time since default and incorporating macroeconomic drivers, the approach provides a transparent and interpretable framework aligned with IFRS 9 requirements for forward-looking credit risk estimation.

The implementation emphasizes:
- Workout level transparency for auditability and model governance
- Cashflow based recovery estimation using discounted recovery cashflows and effective interest rates (EIR)
- Behavioral recovery modeling through separate modeling of:
  - Time to resolution model
  - Resolution type model
  - Cashflow recieve probability model and
  - Recovery amount model
- Forward-looking of macroeconomics integration using statistically selection
- Vectorized numerical computation for efficiency and scalability
- Flexible probability weighted recovery aggregation across multiple resolution pathways
- Residual LGD estimation for Stage 3 accounts based on remaining expected recoveries conditional on time already spent in default

The resulting LGD term structures can be directly used in Stage 1, Stage 2, and Stage 3 IFRS 9 ECL calculation. The project is intended to serve as a practical reference implementation for credit risk practitioners, model developers, and validators rather than a black-box model. All calculations, assumptions, and recovery mechanics are explicitly designed to support validation, backtesting, monitoring, and model explainability.

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

<p align="center">
<img width="1210" height="537" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/370f979e-2506-4199-914a-1e30cfad51ad" />
</p>

**Business Logic**
- Accounts under ongoing collection statuses use the latest observable modeling period as the effective resolution date.
- Resolved accounts use the actual resolution date.
- Time to resolution is capped at a minimum of one month for modeling stability.
- Each account can only contribute one default observation to the modeling population.

<p align="center">
<img width="481" height="504" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/a029b259-f9c3-445a-9b9f-dff237f207b4" />
</p>

#### 0.2 Monthly Recovery Panel
The Monthly Recovery Panel converts account-level default observations into a monthly longitudinal recovery panel. This panel captures the full recovery timeline for each account and enables month-by-month recovery analysis. The process creates a continuous monthly observation structure from default date until resolution or latest observable period. Actual cashflows are then mapped into this timeline to support recovery probability and recovery amount modeling.

**Business Logic**
- Month 0 represents the default month and cannot contain recovery cashflows.
- Missing monthly recoveries are treated as zero recovery observations.
- Ongoing accounts remain observable until the latest modeling period.
- Resolved accounts terminate at the actual resolution month.
- The panel preserves the sequential behavior of post-default recovery dynamics.

<p align="center">
<img width="989" height="590" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/484f5e06-004b-4f64-8e2b-0d3b96553da8" />
</p>

### 1. Unbias LGD
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/4146f938-d142-4249-83c2-9dce093159e2" />
</p>

#### 1.1 Resolved Cases
**Actual LGD Calculation for resolved accounts:** This first process computes **Actual LGD** for fully resolved default cases under the IFRS 9 Framework using a discounted cashflow recovery approach and deduct direct cost. The implementation follows a workout based LGD Methodology, where realized post default recoveries are discounted back to present value (PV) using the account level Effective Interest Rate (EIR). The LGD is finally calculated as the remaining loss relative to EAD. The LGD results of seperated resolved cases are shown below: 

```
[INFO]: Actual LGD for resolved cases
=== Result ===
Resolution type   0 - LGD: 0.45%
Resolution type 202 - LGD: 55.68%
Resolution type 204 - LGD: 62.41%
Resolution type 205 - LGD: 90.48%
```

#### 1.2 Unsolved Cases
**Estimated LGD Calculation for unsolved accounts:** The unsolved workout LGD Framework decomposes the post default recovery process into multiple behavioral sub-models. Each model estimates a different dimension of the recovery lifecycle, allowing the framework to capture both the timing and severity of recoveries under IFRS 9 Requirements.

**Time-to-Resolution Model:** A survival analysis using Weibull Accelerated Failure Time (AFT), this model estimates how long a defaulted account is expected to remain unresolved before reaching a final recovery outcome. The framework applies a Weibull AFT Model using both resolved and unresolved default accounts. Unsolved accounts are treated as right-censored observations, allowing the model to estimate expected remaining recovery duration even when the recovery process is still ongoing.

<p align="center">
<img width="989" height="590" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/4b5836e7-7641-46e8-b8d8-bb74224a5ad0" />
</p>

**Resolution Type Model:** A Multinomial Logistic Classification, this model estimates the probability of each recovery resolution pathway for unsolved accounts. Only resolved accounts are used during training because the final recovery outcome is observable.

<p align="center">
<img width="989" height="590" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/31fdaf67-1e83-4d06-8b59-49dec929a8ae" />
</p>

**Cashflow Receive Model:** A Logistic Hazard Model for Recovery Occurrence, this model estimates the probability that a recovery cashflow will occur at each month since default. The framework treats recovery timing as a monthly event process and applies a binary logistic classification model. 

<p align="center">
<img width="989" height="592" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/0a66fefe-a52f-4326-bfb1-ba4ab6a8d017" />
</p>

**Cashflow Amount Model:** A Recovery Severity Regression Model, this model estimates the expected recovery amount conditional on a recovery event occurring. The framework uses regression modeling separately by resolution type to capture different recovery severity patterns.

<p align="center">
<img width="989" height="590" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/d484e99b-5760-4120-83ca-5f2dcd7971ad" />
</p>

#### 1.3 Unbias Result

The final unsolved workout LGD estimate combines all model components together:
1. Time-to-Resolution Model estimates recovery horizon
2. Resolution Type Model estimates recovery pathway probabilities
3. Cashflow Receipt Model estimates monthly recovery timing
4. Cashflow Amount Model estimates recovery severity

The final portfolio workout LGD is a combination of resolved and unsolved cases. The result can be segmented into default status and resolved status as shown below:

```
======================================================================
Unbias LGD Model
======================================================================
Total default accounts        : 24,808
    Resolved cases            : 23,481
    Unsolved cases            : 1,327
======================================================================
Resolved portfolio LGD        : 57.84%
    Resolution type 0         : 0.45%
    Resolution type 202       : 55.68%
    Resolution type 204       : 62.41%
    Resolution type 205       : 90.48%
======================================================================
Unsolved portfolio LGD        : 40.52%
    Resolution type 0         : 40.44%
    Resolution type 202       : 29.96%
    Resolution type 204       : 39.21%
    Resolution type 205       : 45.75%
======================================================================
Unbias portfolio LGD          : 57.03%
    Default status 100        Resolution type 0          : 18.51%
    Default status 100        Resolution type 202        : 45.57%
    Default status 100        Resolution type 204        : 61.27%
    Default status 100        Resolution type 205        : 44.29%
    Default status 202        Resolution type 202        : 57.84%
    Default status 204        Resolution type 204        : 75.08%
    Default status 205        Resolution type 205        : 98.52%
```

### 2. Forward-looking LGD
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/9875a6af-5090-4766-beb2-e067c3ed9d23" />
</p>

The forward-looking considers additional relationship of macroeconomic variables (MEV) with the based models. The process is leveraged several statistical analysis to perform the features selection that having strong relationship in the model.

#### 2.1 Spearman Rank Correlation Analysis
The Spearman Rank Correlation Analysis is used to evaluate whether MEV(s) exhibit a monotonic relationship with key LGD behavioral drivers such as time to resolution and recovery cashflow severity. Unlike linear correlation methods, Spearman correlation operates on ranked observations, making it more robust to outliers, skewed distributions, and non-linear relationships commonly observed in default recovery data.

For the time-to-resolution analysis, the methodology evaluates whether worsening or improving macroeconomic conditions are associated with longer or shorter recovery processes. For recovery amount analysis, only positive recovery cashflows are considered, and the recovery rate is transformed into logarithmic scale relative to Exposure at Default (EAD) to stabilize variance and reduce extreme-value distortion.

The resulting coefficients provide directional sensitivity between each MEV(s) and recovery dynamics, while statistical significance is assessed using p-values derived from t-statistics. This analysis serves as an initial screening framework to identify economically meaningful forward-looking variables suitable for incorporation into IFRS 9 LGD Modeling.

#### 2.2 Point-Biserial Correlation Analysis
The Point-Biserial Correlation Analysis is designed to measure the relationship between continuous MEV(s) and binary recovery events, specifically whether a recovery cashflow occurs during a given observation period. Within the LGD framework, this analysis helps assess whether certain economic environments influence the likelihood of observing recovery activity after default. The target variable is transformed into a binary indicator where periods with non-zero cashflow are classified as recovery events and periods without recovery are classified as non-events.

Since the point-biserial correlation is mathematically equivalent to Pearson correlation with a binary target, the methodology quantifies both the direction and magnitude of association between macroeconomic conditions and recovery occurrence probability. This analysis is particularly useful for validating forward-looking drivers used in hazard-based recovery models and supports the identification of MEV that influence recovery timing behavior under different economic cycles.

#### 2.3 Kruskal–Wallis Analysis
The Kruskal–Wallis Analysis is applied to evaluate whether MEV(s) differ significantly across resolution outcomes or recovery segments. As a non-parametric alternative to one-way ANOVA, the methodology does not assume normality and is therefore well suited for distressed credit data, which typically contains heavy skewness, heterogeneity, and non-linear distributions.

The analysis ranks MEV(s) observations across all groups and compares whether the median distributions differ between resolution types such as normal (0), repossessed (202), or written-off (205) accounts. The resulting H-statistic is adjusted for ties and transformed into an effect-size measure to quantify explanatory strength. Statistical significance is evaluated using the chi-square distribution. Within the IFRS 9 LGD framework, this analysis provides evidence that macroeconomic conditions materially influence resolution pathways, supporting the use of forward-looking segmentation and differentiated recovery assumptions across workout strategies.

#### 2.4 Forward-looking Result

The t-statistics are transformed into the composit score to find the significant relationship for the forward-looking model. For this repository, the `BROMO_MA9M` and `GDP_MA12M_LAG12M` are selected.

```
========================================
MEV(s) Significant Score
========================================
BROMO_MA9M          : 92.15
BROMO_MA6M          : 91.76
BROMO_MA12M         : 91.37
BROMO_MA6M_LAG3M    : 91.11
BROMO_MA9M_LAG3M    : 90.93
BROMO_MA3M          : 90.78
BROMO_MA6M_LAG6M    : 90.66
BROMO_MA12M_LAG3M   : 90.60
BROMO_MA3M_LAG6M    : 90.58
BROMO               : 90.56
BROMO_LAG6M         : 90.52
BROMO_MA9M_LAG6M    : 89.95
BROMO_MA12M_LAG6M   : 89.91
BROMO_MA3M_LAG3M    : 89.80
BROMO_LN            : 89.78
BROMO_LN_LAG6M      : 89.78
BROMO_MA12M_LAG12M  : 89.78
BROMO_MA9M_LAG12M   : 89.70
BROMO_MA12M_LAG9M   : 89.62
BROMO_LAG3M         : 89.54
BROMO_MA6M_LAG9M    : 89.52
BROMO_MA9M_LAG9M    : 89.50
GDP_MA12M_LAG12M    : 89.48
BROMO_MA6M_LAG12M   : 88.95
MLR_MA12M           : 88.66
BROMO_LN_LAG3M      : 88.64
MLR_MA9M            : 88.50
MLR_MA6M            : 88.05
MLR_MA3M            : 87.85
BROMO_MA3M_LAG12M   : 87.81
```

The model results are compared with/wothout MEV(s) effects from the based model.

<p align="center">
<img width="1920" height="640" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/691cadb0-90b3-4ec3-b271-cec76381a232" />
</p>

<p align="center">
 <img width="1920" height="640" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/29df2d12-365a-407f-90f7-96479b5b35fe" />
</p>

<p align="center">
<img width="1920" height="640" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/2f44367e-c89c-42ba-8aa5-c606469f9533" />
</p>

<p align="center">
<img width="1920" height="640" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/3c0a1a71-275c-4572-a467-fbb51f5b2cda" />
</p>

The LGD computation processes are the same as unbias model but incorporated the MEV(s) effects and changed the output from the model.

```
======================================================================
FWL LGD Model
======================================================================
Total default accounts        : 24,808
    Resolved cases            : 23,481
    Unsolved cases            : 1,327
======================================================================
Resolved portfolio LGD        : 57.84%
    Resolution type 0         : 0.45%
    Resolution type 202       : 55.68%
    Resolution type 204       : 62.41%
    Resolution type 205       : 90.48%
======================================================================
Unsolved portfolio LGD        : 65.13%
    Resolution type 0         : 66.34%
    Resolution type 202       : 28.91%
    Resolution type 204       : 70.20%
    Resolution type 205       : 66.79%
======================================================================
Unbias portfolio LGD          : 58.18%
    Default status 100        Resolution type 0          : 12.90%
    Default status 100        Resolution type 202        : 45.18%
    Default status 100        Resolution type 204        : 62.77%
    Default status 100        Resolution type 205        : 60.13%
    Default status 202        Resolution type 202        : 57.84%
    Default status 204        Resolution type 204        : 75.08%
    Default status 205        Resolution type 205        : 98.52%
```

### 3. Residual LGD
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 LGD Model แบบ Workout period ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/a7a46938-cdca-4a97-b66c-6c1225ac9d81" />
</p>


## License
MIT · Built for learning purposes













