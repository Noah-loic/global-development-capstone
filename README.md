
# Capstone Project: Global Development Trends Analysis (1960–2021)

This project analyzes economic and human development indicators using World Bank and UN HDI data.

##  Tools
- Python (Data Cleaning, EDA, Modeling, Clustering)
- Power BI (Interactive Dashboard)
- GitHub (Documentation, Collaboration)

##  Problem Statement
Which countries and regions have developed the fastest in terms of GDP, HDI, population, and life expectancy? Can we model or cluster development patterns?

This project investigates:
- Which countries experienced the highest **GDP and population growth**
- Which countries and regions saw the most **HDI growth** in the 21st century
- Which **factors correlate** strongly with **life expectancy**
- How **clustering** can reveal patterns among countries beyond income groups

---

##  Data Sources

- `WorldBank.xlsx`: Economic indicators (1960–2018)
- `HDI.csv`: Human Development Index (1990–2021)
- These datasets were merged by country and year to produce `data_cleaned.csv`.

---

## 1. Data Cleaning 

####  What the Code Does:
- Reads in both datasets
- Standardizes column names (e.g., converts spaces to underscores, lowercase)
- Converts year to integer
- Merges the two datasets on `country` and `year`
- Handles missing values using mean imputation
- Outputs a cleaned dataset called `data_cleaned.csv`

#### Result:
We obtained a consistent, combined dataset ready for analysis with ~180 countries over multiple decades.

---

### 2 Exploratory Data Analysis (EDA) 

#### EDA Question 1: **Top GDP & Population Growth Countries**
- The code calculates the difference between the **max** and **min** GDP per capita and population for each country.
- Visualized top 10 countries in bar charts.

**Observation:**
- **China**, **India**, and **Vietnam** led in GDP growth.
- **Nigeria**, **Pakistan**, and **Bangladesh** led in population growth.
- Some countries like India appeared in both, indicating fast population + economic expansion.

---

#### EDA Question 2: **HDI Growth in the 21st Century**
- Filtered records from year 2000 onward.
- Calculated HDI improvement for each country using max - min values.
- Visualized top 10 countries using horizontal bar plots.

**Observation:**
- **Rwanda**, **Cambodia**, and **Ethiopia** showed the greatest HDI improvement.
- These countries invested in health and education after years of instability.

---

#### EDA Question 3: **Correlation with Life Expectancy**
- Used `.corr()` to generate a correlation matrix.
- Plotted top positive/negative correlations with life expectancy.

**Observation:**
- Strong positive correlations:
  - **HDI**
  - **gdp_per_capita_usd**
  - **electric_power_consumption_kwh_per_capita**
- Slight negative correlation with **CO₂ emissions**, showing development may come with environmental cost.

---

### 3. Modeling 

#### What the Code Does:
- Selected features like HDI, GDP, population, electricity use, CO₂ emissions
- Trained two regression models:
  - **Linear Regression**
  - **Random Forest Regressor**
- Evaluated using **R²** and **RMSE**

#### Results:
| Model | R² Score | RMSE |
|-------|----------|------|
| Linear Regression | ~0.84 | Moderate |
| Random Forest     | ~0.91 | Low |

**Observation:**
- Random Forest captured non-linear relationships better.
- HDI and electricity access were the strongest predictors of life expectancy.

---

### 4. Clustering

#### What the Code Does:
- Selected the most recent year (e.g., 2018 or 2021)
- Standardized features using `StandardScaler`
- Applied **K-Means clustering (k=3)**
- Reduced dimensions with PCA for visualization

#### Result:
- Countries were grouped into 3 distinct clusters based on development indicators.
- Used PCA to plot clusters in 2D for interpretability.

**Observation:**
- Cluster 0: Underdeveloped nations (low HDI, low GDP)
- Cluster 1: Transitional economies (moderate metrics)
- Cluster 2: Developed countries (high HDI, high life expectancy, strong infrastructure)

---
