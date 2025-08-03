
# Capstone Project: Global Development Trends Analysis (1960–2021)

This project analyzes economic and human development indicators using World Bank and UN HDI data.

**Course:** INSY 8413 Introduction to Big Data Analytics  
**Student:** Kizungu Shyaka Noah Loic   
**ID:** 25919

##  Tools
- Python (Data Cleaning, EDA, Modeling, Clustering)
- Power BI (Interactive Dashboard)
- GitHub (Documentation, Collaboration)

## PART 1: Problem Statement
Analyzing Global Development Trends using World Bank and UN HDI Data

**sector**: Government

This project investigates:
- Which countries experienced the highest **GDP and population growth**
- Which countries and regions saw the most **HDI growth** in the 21st century
- Which **factors correlate** strongly with **life expectancy**
- How **clustering** can reveal patterns among countries beyond income groups

---

##  Data Sources:
`https://mavenanalytics.io/data-playground?page=4&pageSize=5;`

#### Datasets:
- `WorldBank.xlsx`: Economic indicators (1960–2018):The World Bank funds infrastructure projects in the developing world. As part of its mission, it releases an annual report on economic indicators. This dataset includes important indicators of economic performance and development from 1960-2018, including fields like electricity consumption, GDP per capita, life expectancy, and more.
  
- `HDI.csv`: Human Development Index (1990–2021):To augment the World Bank data, Human Development Index (HDI) data has been provided by the United Nations (UN). HDI is a composite measure of how developed a country is based on life expectancy, GDP per capita, and educational attainment. This supplementary dataset contains additional indicators used by the UN to track development, environmental impact, and inequality for each country from 1990-2021.
  
- These datasets were merged by country and year to produce `data_cleaned.csv`.

---
## PART 2: PYTHON ANALYTICS TASKS

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

**What was done:**

Chose independent variables that are known to influence life expectancy:

 - hdi (Human Development Index)

 - gdp_per_capita

 - population

 - electricity_consumption_per_capita

 - co2_emissions_per_capita

Defined the target variable:
 - life_expectancy

**Why it was done:**
 - To ensure that only relevant, numerical, and complete data was used to build an accurate model.
 - These features are backed by economic and social theory as drivers of health outcomes.
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

**Divided the dataset into:**

 - Training set (80%) – to train the model.

 - Test set (20%) – to evaluate model performance on unseen data.

**Why it was done:**
 - To prevent overfitting (i.e., model memorizing data).
 - Ensures the model is evaluated fairly on data it has never seen before — mimicking real-world deployment.

```python
from sklearn.model_selection import train_test_split

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

```


**3. Model Training**:	Fit models to training data	and Learn relationships between inputs and output

**What was done:**
 - Trained two different models:
    1. Linear Regression: Fits a straight-line relationship between features and life expectancy.
    2. Random Forest Regressor: Builds multiple decision trees and averages their results for better predictions.

**Why it was done:**
 - To learn the patterns in the training data that relate predictors (X) to the target (y).
 - Comparing a simple model (linear) and a complex one (random forest) helps evaluate trade-offs between interpretability and accuracy.
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

**What was done:**

 - Used metrics like:
   1. R² Score – How much variance is explained by the model (closer to 1 is better).
   2. RMSE (Root Mean Squared Error) – Measures prediction error in the original unit (lower is better).

**Why it was done:**
 - To measure how well each model performs.
 - Linear Regression gives insights into directional relationships.
 - Random Forest shows how accurately complex, non-linear relationships can be modeled.

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

 - How it works: Uses an ensemble of decision trees to predict the output, averaging many trees reduces overfitting.

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

## PART 3: POWER BI DASHBOARD TASKS
The interactive dashboard includes:
### 1 .Slicers: Country, Year, Cluster
### 2 .KPIs: Global HDI, GDP per capita, life expectancy
<img width="580" height="329" alt="average of hdi,life expectecny,Gdp using kpi" src="https://github.com/user-attachments/assets/cd01a002-6f59-4e08-b733-0bfc7d1e297c" />

**Average Global HDI, GDP per Capital, and Life Expectancy (1990):**
 - It the image i took an example of country of Rwanda in 1990
 - The average for Rwanda in 1990 HDI is 0.67.
 - The average global GDP per capita is 349.87 USD.
 - The average life expectancy is 33.41 years. This shows that it was a time of high mortality rate.


### 3. Charts: Top growth countries, HDI trends, correlations
<img width="946" height="470" alt="image3" src="https://github.com/user-attachments/assets/38427349-97d0-4488-9f0f-266ee7d9e0d7" />

<img width="687" height="407" alt="example" src="https://github.com/user-attachments/assets/f7b6db4b-5f09-4b86-bdad-b939dd6a1d85" />

**Sum of GDP per Capita, Life Expectancy, and Death Rate by Country and Year (1990-2011):**
 - **Using a slicer, i set an example of the year of 1994 as you can see below:**
 - **GDP per Capital:** Luxembourg and Norway show the highest growth, with values exceeding 1.5M USD, while other countries like Iceland and Denmark are lower but still significant.
 - **Life Expectancy:** Japan and Switzerland lead with values around 2,000 years (likely aggregated or mislabeled data), while Rwanda,Sierra Leone and Uganda are lower, indicating health disparities.
 - **Death Rate:** Sierra Leone and Rwanda show the highest sums (around 500-600), while countries like Japan and Iceland are lower, reflecting differences in mortality which also confirms the life expectancy values.

<img width="643" height="378" alt="image2" src="https://github.com/user-attachments/assets/85ae6090-5fe1-4c1f-9ba8-f716ac6a9a0f" />

**HDI Trend Over Years (2008):**

The HDI trend for various countries in 2008 shows a relatively stable range between 0.3 and 0.9, with a slight decline over the years. Countries like Norway and Switzerland are at the higher end, while others like Afghanistan and Angola are lower, indicating varying development levels


<img width="644" height="379" alt="images 5" src="https://github.com/user-attachments/assets/63a447e3-4c00-404d-8616-6c1e1c7d2980" />

**Sum of Unemployment and Population Density by Country:**
  - The line chart shows a high initial sum of unemployment (of total labor force, modeled ILO estimate) and population density (people per sq. km of land area) that decreases sharply and then stabilizes. Countries like North Macedonia and Bosnia and Herzegovina start with high values, while Syria Arab Republic shows a notable peak, suggesting regional variations in these metrics.


### 4. Country clusters and development patterns
  <img width="640" height="375" alt="image4" src="https://github.com/user-attachments/assets/86c4c12f-c6ef-4c4a-870f-49b12cac5044" />
  
**Pie Charts by Cluster:**

 - **Birth Rate:** Cluster 0.0 dominates with 41.7%, followed by 2.0 (29.19%) and 1.0 (20.61%), indicating varied birth rates across clusters which could mean that one of the cause of the disparities between the cluster 2 and cluster 0 is the huge population of the underdeveloped countries.
 - **Population Density:** Cluster 2.0 has the highest sum (350.28K, 42.51%), followed by 0.0 (156.11K, 18.94%) and 1.0 (46.2K, 5.61%), showing significant density differences.
 - **Life Expectancy:** Cluster 1.0 leads with 152.9K (26.07%), followed by 0.0 (59.24K, 19.76%) and 2.0 (51.04K, 8.49%), reflecting diverse life expectancy distributions.
 - **Death Rate:** Cluster 0.0 has the highest sum (15.66K, 39.56%), followed by 1.0 (5.62K, 14.17%) and 2.0 (16.3K, 4.25%), indicating varied mortality rates.


# Overall Key Observations & Predictions

### 1. GDP and Population Growth Insights

- **Luxembourg, Norway, and Qatar** experienced the highest GDP per capita growth, reflecting strong economic performance likely driven by finance, oil, and innovation.
- **Singapore** showed simultaneous growth in both GDP and population, indicating robust and balanced development.
- **Countries like Bangladesh, Malta, and Maldives** exhibited modest growth, suggesting room for improvement in economic expansion and population health.
- **Insight**: GDP growth does not always translate to population growth ,wealth accumulation can be independent of demographic trends.

### 2. HDI Improvements in the 21st Century

- Countries like **Rwanda, Myanmar, and Ethiopia** had the most significant HDI improvements post-2000.
- These gains are likely due to post-conflict recovery, public health investments, and education reform.
- **Insight**: Even countries with challenging pasts can achieve rapid human development with focused policy interventions.

### 3. Life Expectancy Correlation Analysis

- **Top factors positively correlated with life expectancy**:
  - **HDI**: A comprehensive assessment of progress
  - **GDP per capital**: Wealth enables better healthcare
  - **Internet Access & Electricity Consumption**: Indicators of infrastructure and modern living
- **Top negative correlations**:
  - Infant mortality
  - Birth rate
  - HDI rank
  - Death rate
- **Insight**: Life expectancy is a reflection of not just income, but also healthcare, education, and infrastructure access.

### 4. Predictive Modeling of Life Expectancy

- Two models were developed:
  - **Linear Regression** (R² ≈ 0.84): Good baseline for simple relationships.
  - **Random Forest** (R² ≈ 0.91): Captured more complex interactions with lower RMSE.
- **Most Influential Predictors**:
  - HDI
  - Electricity consumption
  - GDP per capita
- **Prediction**: A country’s life expectancy can be predicted with high confidence using a few economic and development indicators.

### 5. Country Clustering Analysis

- Using **K-Means (k=3)**, countries were grouped into:
  - **Cluster 0 (Underdeveloped)**: Low HDI, high death/birth rates, low electricity and GDP.
  - **Cluster 1 (Developing)**: Moderate progress, transitional economies.
  - **Cluster 2 (Developed)**: High HDI, life expectancy, and access to electricity.
- **Insight**: Clustering reveals nuanced development profiles beyond just income categories. Birth rate and population density were key separators.

### 6. Power BI Dashboard Insights

- **KPI Panels** showed low average life expectancy and GDP in countries like Rwanda in 1990, emphasizing historical underdevelopment.
- **Line charts** revealed HDI progress post-2000 in many regions.
- **Maps and Pie Charts** visualized clear disparities across clusters in:
  - Life expectancy
  - Death rate
  - Birth rate
- Countries in **Cluster 2** dominated in infrastructure and health, while **Cluster 0** bore the burden of high death/birth rates and lower economic outputs.
- **Insight**: Interactive dashboards empower policymakers to isolate regional gaps and monitor progress toward SDGs.

### Final Summary: What This Project Shows

This project reveals that:

- **Development is multi-dimensional**, not just about wealth, but health, education, and infrastructure.
- Countries can rapidly improve HDI through targeted interventions.
- Predictive models and cluster analysis are valuable tools for guiding government strategies.
- Visual analytics (Power BI) helps communicate complex insights in an accessible way.

### Predictions Based on Trends

- Countries like **Rwanda, Ethiopia, and Bangladesh** may continue their upward HDI trends if investment continues.
- Regions in **Cluster 1** could shift to **Cluster 2** in the next 1 to 2 decades with policy support.
- **Electricity access and internet usage** will be crucial for future life expectancy improvements.
