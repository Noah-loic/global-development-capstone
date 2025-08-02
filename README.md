
# Capstone Project: Global Development Trends Analysis (1960–2021)

This project analyzes economic and human development indicators using World Bank and UN HDI data.

**Course:** INSY 8413 Introduction to Big Data Analytics  
**Student:** Kizungu Shyaka Noah Loic   
**ID:** 25919

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

- `WorldBank.xlsx`: Economic indicators (1960–2018):The World Bank funds infrastructure projects in the developing world. As part of its mission, it releases an annual report on economic indicators. This dataset includes important indicators of economic performance and development from 1960-2018, including fields like electricity consumption, GDP per capita, life expectancy, and more.
  
- `HDI.csv`: Human Development Index (1990–2021):To augment the World Bank data, Human Development Index (HDI) data has been provided by the United Nations (UN). HDI is a composite measure of how developed a country is based on life expectancy, GDP per capita, and educational attainment. This supplementary dataset contains additional indicators used by the UN to track development, environmental impact, and inequality for each country from 1990-2021.
  
- These datasets were merged by country and year to produce `data_cleaned.csv`.

---
## PYTHON ANALYTICS TASKS

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
- Visualized top 10 countries with growing countries in GDP and population were identified, with some overlap in bar charts.

<img width="989" height="590" alt="top 10 countries" src="https://github.com/user-attachments/assets/c22d03bb-26ba-42f7-9e89-b0844f358568" />

**Observation:**
- **China**, **India**, and **Vietnam** led in GDP growth.
- **Nigeria**, **Pakistan**, and **Bangladesh** led in population growth.
- Some countries like singapore and India appeared in both, indicating fast population + economic expansion.

---

#### EDA Question 2: **HDI Growth in the 21st Century**
- Filtered records from year 2000 onward.
- Calculated HDI improvement for each country using max - min values.
- Visualized top 10 countries using horizontal bar plots.

<img width="989" height="590" alt="top 10 by HDI" src="https://github.com/user-attachments/assets/6b80b834-2513-43af-ac44-e2ae376b4f30" />

**Observation:**
- **Rwanda**, **Myanmar**, and **Ethiopia** showed the greatest HDI improvement.
- HDI improvements are strongest in specific countries post-2000
- These countries invested in health and education after years of instability.

---

#### EDA Question 3: **Correlation with Life Expectancy**
- Used `.corr()` to generate a correlation matrix.
- Plotted top positive/negative correlations with life expectancy.

<img width="1239" height="547" alt="top correlation" src="https://github.com/user-attachments/assets/e41f5233-de16-49aa-a9ab-4367bc97c574" />

**Top Positive Correlations with Life Expectancy:**

 life_expectancy_at_birth_years                   1.000000
 
hdi                                              0.826773

gdp_per_capita_usd                               0.549062

individuals_using_the_internet__of_population    0.513566

electric_power_consumption_kwh_per_capita        0.365979

year                                             0.236142
Name: life_expectancy_at_birth_years, dtype: float64

**Top Negative Correlations with Life Expectancy:**

 unemployment__of_total_labor_force_modeled_ilo_estimate    0.035974
 
death_rate_crude_per_1000_people                          -0.664155

hdi_rank_2021                                             -0.835685

birth_rate_crude_per_1000_people                          -0.864584

infant_mortality_rate_per_1000_live_births                -0.920102

Name: life_expectancy_at_birth_years, dtype: float64

**Observation:**
- Strong positive correlations:
  - **HDI**
  - **gdp_per_capita_usd**
  - **electric_power_consumption_kwh_per_capita**
- Slight negative correlation with **CO₂ emissions**, showing development may come with environmental cost.

---

### 3. Modeling 
#### Build a regression model to predict Life Expectancy 

it was done following the following steps:

**1. Feature Selection**:	Identify predictors and target variable	Focus model on meaningful inputs.

**2. Data Splitting**:	Divide data into training and test sets	and ensures fair and generalizable evaluation.

**3. Model Training**:	Fit models to training data	and Learn relationships between inputs and output

**4. Model Evaluation**:	Test model on unseen data using R² and RMSE	Measure accuracy and robustness

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

**Linear Regression**

 - Purpose: Establish a simple, interpretable relationship between features and life expectancy.

 - How it works: Fits a straight line (linear equation) to model the relationship.

 - Strengths: Easy to understand, explain, and benchmark.

 - Limitations: Struggles with non-linear relationships or complex interactions between variables.

 Used as a baseline to measure how well simple linear assumptions perform.

**Random Forest Regressor**

 - Purpose: Capture non-linear, complex relationships that a linear model may miss.

 - How it works: Uses an ensemble of decision trees to predict the output — averaging many trees reduces overfitting.

Strengths:

 = Handles missing values and outliers better

 - Captures feature interactions

 - Can compute feature importance

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

<img width="989" height="590" alt="scatterplot" src="https://github.com/user-attachments/assets/dc98dfbe-c75d-4c16-b2bc-eabd49621836" />


**Observation:**
- Cluster 0: Underdeveloped nations (low HDI, low GDP)
- Cluster 1: Transitional economies (moderate metrics)
- Cluster 2: Developed countries (high HDI, high life expectancy, strong infrastructure)

---
