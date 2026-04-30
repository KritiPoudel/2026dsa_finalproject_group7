# 2026dsa_finalproject_group7 - Corn Yield Prediction

**Team Members:** 
- Kriti Poudel. 
- Mmaduabuchi Okeh.

------------------------------------------------------------------------

## A. Introduction.

This repository contains all code, data processing steps, and machine learning models developed for the Data Science Applied to Agriculture final project on corn yield prediction.

We were provided with a **corn variety trial** data set.
The data contains over **164,000 rows** spanning **10 years** (2014-2023), **45 sites** across the USA, for a total of **270** site-years, where over **5,000** corn hybrids were evaluated.

**Training data** provided for years 2014-2023:

-   Trait information (including site, year, hybrid, yield, and grain moisture)
-   Meta information (including site, year, previous crop, longitude, and latitude)
-   Soil information (including site, year, soil pH, soil organic matter, soil P, soil K).

**Testing data** for year 2024:
- Submission information (site, year, hybrid, **no yield**)
- Meta information (same as training)
- Soil information (same as training).

> We are tasked with training machine learning models using the training data to predict yield on the test data. We are not provided with the test data yield — only the predictor variables.

------------------------------------------------------------------------

## B. Open-Source Data

We used the longitude and latitude provided in the meta data to pull external open-source weather data.

-   Weather data was retrieved using the **Daymet API** via the `daymetr` package in R
-   Variables retrieved: maximum temperature, minimum temperature, precipitation, solar radiation, vapor pressure, daylight hours
-   Data retrieved for all training site-years (2014–2023) and test site-years (2024)

------------------------------------------------------------------------

# C. Feature Engineering

-   Weather data was summarized on a **monthly basis**:
    -   **Mean** for temperature, solar radiation, vapor pressure, and daylight hours
    -   **Sum** for precipitation
-   Only **growing season months (May–September)** were kept as predictors
-   Non-growing season months (Jan, Feb, Mar, Apr, Oct, Nov, Dec) were removed

------------------------------------------------------------------------

## D. Modeling Strategies

We trained two machine learning models:

-   XGBoost

-   Support Vector Machine (SVM)

------------------------------------------------------------------------

## E. Training Strategies

### XGBoost

| Parameter             | Value                                              |
|--------------------------------------------|----------------------------|
| Data engineering      | Monthly mean/sum weather summaries                 |
| Number of predictors  | 30                                                 |
| Data split            | 70% training / 30% test                            |
| Split type            | Stratified by yield (Mg/ha)                        |
| Pre-processing        | Removed non-growing season months                  |
| Hyperparameters tuned | trees, tree_depth, min_n, learn_rate               |
| Search algorithm      | ANOVA Racing                                       |
| V value               | 10                                                 |
| Resampling strategies | 10-fold CV, Leave-One-Year-Out, Leave-One-Site-Out |
| Best model metric     | RMSE (overall best)                                |

------------------------------------------------------------------------

## F. Repository Structure

-   `code/` — all R scripts
-   `data/` — training and testing datasets
-   `output/` — model figures and results
