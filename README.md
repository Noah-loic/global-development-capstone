
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

```python
# Group by country and calculate growth over time
gdp_growth = merged_df.groupby('country')['gdp_per_capita_usd'].agg(['min', 'max']).reset_index()
gdp_growth['gdp_growth'] = gdp_growth['max'] - gdp_growth['min']

pop_growth = merged_df.groupby('country')['population_density_people_per_sq_km_of_land_area'].agg(['min', 'max']).reset_index()
pop_growth['population_growth'] = pop_growth['max'] - pop_growth['min']

# Top 10 countries by GDP growth
top_gdp = gdp_growth.sort_values(by='gdp_growth', ascending=False).head(10)

# Top 10 countries by population growth
top_pop = pop_growth.sort_values(by='population_growth', ascending=False).head(10)

print("Top 10 GDP Growth Countries:\n", top_gdp[['country', 'gdp_growth']])
print("\nTop 10 Population Growth Countries:\n", top_pop[['country', 'population_growth']])

# Find overlapping countries
overlap = set(top_gdp['country']).intersection(set(top_pop['country']))
print("\nCountries in both GDP and Population Growth Top 10:\n", list(overlap))
```

- The code calculates the difference between the **max** and **min** GDP per capita and population for each country.
- Visualized top 10 countries with growing countries in GDP and population were identified, with some overlap in bar charts.

<img width="988" height="590" alt="horizontal bar plot" src="https://github.com/user-attachments/assets/45265782-9e42-4d4b-81f0-edd3627519eb" />

<img width="989" height="590" alt="top 10 by gdp" src="https://github.com/user-attachments/assets/a5dc8f29-a4c5-45f3-88a1-b8a86f0ab477" />

<img width="1189" height="790" alt="GDP Growth vs Population Growth for Top Countries" src="https://github.com/user-attachments/assets/b0814824-b2f6-463b-8d14-e581549aede5" />


**Observation:**
- **Luxembourg** leads with the highest GDP growth, followed closely by **Norway** and **Qatar**.This suggests that smaller or resource-rich countries like Luxembourg, Norway, and Qatar have experienced substantial economic growth compared to larger economies like the United States over the specified period.
- The bar chart indicates **Singapore** has the highest population growth, followed by **Bahrain** and **Maldives**. The growth decreases significantly for countries like Burundi and Belgium, showing diverse population trends.
- Some countries like **singapore** appeared in both, indicating fast population + economic expansion.
- **Singapore** is the only country with significant overlap between GDP growth and population growth, both showing substantial values, with GDP growth slightly higher and highlighted in yellow.
- Most countries, such as Luxembourg, Norway, Qatar, and Ireland, exhibit high GDP growth but minimal population growth.
- Countries like Malta, Bangladesh, and Maldives show low growth in both GDP and population

---

#### EDA Question 2: **HDI Growth in the 21st Century**

```python
# Filter 2000 onwards
df_21st = merged_df[merged_df['year'] >= 2000]

# Calculate HDI change by country
hdi_change = df_21st.groupby('country')['hdi'].agg(['min', 'max']).reset_index()
hdi_change['hdi_growth'] = hdi_change['max'] - hdi_change['min']

# Top 10 countries with highest HDI improvement
top_hdi = hdi_change.sort_values(by='hdi_growth', ascending=False).head(10)
print("Top 10 HDI Growth Countries since 2000:\n", top_hdi[['country', 'hdi_growth']])
# Plot horizontal bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=top_hdi, y='country', x='hdi_growth', palette='coolwarm')
plt.title('Top 10 Countries by HDI Growth (2000–2021)')
plt.xlabel('HDI Growth')
plt.ylabel('Country')
plt.tight_layout()
plt.show()
```

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

```python
# Correlation matrix focused on life expectancy
correlations = merged_df.corr(numeric_only=True)
life_corr = correlations['life_expectancy_at_birth_years'].sort_values(ascending=False)

print("Top Positive Correlations with Life Expectancy:\n", life_corr.head(6))
print("\nTop Negative Correlations with Life Expectancy:\n", life_corr.tail(5))
# Visualize top correlations
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.barplot(x=life_corr.head(10).values, y=life_corr.head(10).index)
plt.title("Top Correlations with Life Expectancy")
plt.xlabel("Correlation")
plt.show()
```

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
```python
# Select features for modeling
features = [
    'gdp_per_capita_usd',
    'population_density_people_per_sq_km_of_land_area',
    'electric_power_consumption_kwh_per_capita',
    'hdi'
]

# Drop NA rows for selected features + target
model_df = merged_df[features + ['life_expectancy_at_birth_years']].dropna()

# Separate features and target
X = model_df[features]
y = model_df['life_expectancy_at_birth_years']

```

**2. Data Splitting**:	Divide data into training and test sets	and ensures fair and generalizable evaluation.
```python
from sklearn.model_selection import train_test_split

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

```

**3. Model Training**:	Fit models to training data	and Learn relationships between inputs and output
```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

```

**4. Model Evaluation**:	Test model on unseen data using R² and RMSE	Measure accuracy and robustness
```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.4f}")
    print()

# Ensure rf and y_pred_rf are defined
if 'rf' not in locals():
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
if 'y_pred_rf' not in locals():
    y_pred_rf = rf.predict(X_test)

# Evaluate both models
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest Regressor", y_test, y_pred_rf)


```

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

 - Handles missing values and outliers better

 - Captures feature interactions

 - Can compute feature importance

**Observation:**
- Random Forest captured non-linear relationships better.
- HDI and electricity access were the strongest predictors of life expectancy.

---

### 4. Clustering
**Step-by-Step Clustering Plan:**

1.Select relevant features
  ```python
# Select one row per country for latest available year (e.g., 2018 or max year)
latest_year = merged_df['year'].max()
df_latest = merged_df[merged_df['year'] == latest_year]

# Drop countries with missing values
cluster_features = [
    'country', 'gdp_per_capita_usd', 'population_density_people_per_sq_km_of_land_area', 'life_expectancy_at_birth_years', 'hdi', 'electric_power_consumption_kwh_per_capita'
]
df_cluster = df_latest[cluster_features].dropna()
df_cluster.set_index('country', inplace=True)

# Data to feed into the model
X_cluster = df_cluster.copy()


```
   
2.Normalize the data
  ```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

```
3.Apply K-means Clustering

  ```python
from sklearn.cluster import KMeans

# Try 3 clusters (adjust based on elbow curve later)
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

```
4.Visualize clusters
 ```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_cluster['pca1'] = X_pca[:, 0]
df_cluster['pca2'] = X_pca[:, 1]

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_cluster, x='pca1', y='pca2', hue='cluster', palette='Set2', s=100)
plt.title("Country Clusters Based on Development Indicators")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

```

#### What the Code Does:
- Selected the most recent year (e.g., 2018 or 2021)
- Standardized features using `StandardScaler`
- Applied **K-Means clustering (k=3)**
- Reduced dimensions with PCA for visualization

#### Result:
- Countries were grouped into 3 distinct clusters based on development indicators.
- Used PCA(principle component analysis ) to plot clusters in 2D for interpretability.

<img width="989" height="590" alt="scatterplot" src="https://github.com/user-attachments/assets/dc98dfbe-c75d-4c16-b2bc-eabd49621836" />


**Observation:**
- Cluster 0: Underdeveloped nations (low HDI, low GDP)
- Cluster 1: Transitional economies (moderate metrics)
- Cluster 2: Developed countries (high HDI, high life expectancy, strong infrastructure)

---
