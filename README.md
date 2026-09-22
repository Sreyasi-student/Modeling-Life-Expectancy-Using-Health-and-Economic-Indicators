# Modeling Life Expectancy Using Health and Economic Indicators

Multiple linear regression (OLS), Ridge, Lasso, and Principal Component Regression (PCR) models built on the **Life Expectancy WHO Updated** dataset to explain and predict life expectancy from health and economic indicators.

## Contents

- `Life_Expectancy_Regression_Final.ipynb` — the analysis notebook (data loading, EDA, preprocessing, modeling, diagnostics, model comparison)
- `Life_Expectancy_Data_Updated.csv` — dataset sourced from Kaggle

## Objectives

1. Load and inspect the data.
2. Perform a train/test split.
3. Fit multiple linear regression (OLS).
4. Check the major assumptions of linear regression.
5. Fit Ridge and Lasso regression.
6. Select the Ridge/Lasso regularization parameter (`alpha`) using cross-validation on the training set.
7. Compare OLS, Ridge, and Lasso (plus PCR) on the same untouched test set.

## Dataset

- Source file: `Life-Expectancy-Data-Updated.csv`
- Original shape: 2,864 rows × 21 columns, with no missing values
- The notebook restricts the analysis to a single year, **`Year == 2003`**, giving 179 countries across 9 regions
- Columns: `Country`, `Region`, `Year`, `Infant_deaths`, `Under_five_deaths`, `Adult_mortality`, `Alcohol_consumption`, `Hepatitis_B`, `Measles`, `BMI`, `Polio`, `Diphtheria`, `Incidents_HIV`, `GDP_per_capita`, `Population_mln`, `Thinness_ten_nineteen_years`, `Thinness_five_nine_years`, `Schooling`, `Economy_status_Developed`, `Economy_status_Developing`, `Life_expectancy`

### Target
`Life_expectancy`

### Final predictors (14 numeric + 1 categorical)
Numeric: `Infant_deaths`, `Alcohol_consumption`, `Hepatitis_B`, `Measles`, `BMI`, `Polio`, `Diphtheria`, `Incidents_HIV`, `GDP_per_capita`, `Population_mln`, `Thinness_ten_nineteen_years`, `Thinness_five_nine_years`, `Schooling`, `Economy_status`
Categorical: `Region` (one-hot encoded, first category dropped)

## Methodology

1. **Train/test split** — 80/20 (`test_size=0.20`, `random_state=42`), yielding 143 training and 36 test observations. The test set is untouched until final evaluation.
2. **Preprocessing pipeline** (`ColumnTransformer`) — `StandardScaler` on numeric features, `OneHotEncoder(drop="first")` on `Region`; fit on the training data only.
3. **OLS baseline** — `LinearRegression` on the preprocessed features.
4. **Assumption checks on the OLS model** — linearity (residuals vs. fitted), normality (histogram, Q-Q plot, Shapiro-Wilk), homoskedasticity (Breusch-Pagan test), independence (Durbin-Watson), multicollinearity (VIF), and influential points (Cook's distance).
5. **Principal Component Regression (PCR)** — PCA fit on the training design matrix only, retaining components that explain 95% of training variance, followed by `LinearRegression` on the retained components.
6. **Ridge and Lasso** — `alpha` tuned via `GridSearchCV` over `np.logspace(-4, 4, 100)` using `KFold(n_splits=3, shuffle=True, random_state=42)` and `neg_root_mean_squared_error` scoring.
7. **Model comparison** — OLS, Ridge, Lasso, and PCR evaluated on the same held-out test set using R², RMSE, and MAE.


## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
statsmodels
```

## How to run
Run the notebook cells in order — the pipeline fits all preprocessing, PCA, and regularization hyperparameters on the training split only, so cells must execute sequentially.

## Notes / Limitations

- No missing-value imputation is performed; the source file has no missing values.
- The analysis in this notebook uses only the 2003 cross-section of the panel dataset (179 countries), not the full multi-year panel.
- Several features (`Polio`/`Diphtheria`, and the two thinness measures) show high VIF, indicating multicollinearity in the OLS coefficients; Ridge and Lasso are included in part to address this.
