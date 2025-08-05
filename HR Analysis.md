# 📊 Capstone Project: Data-Driven Insights for HR Decision-Making

## 🧩 Project Overview

This project is part of the [Google Advanced Data Analytics Professional Certificate on Coursera](https://www.coursera.org/professional-certificates/google-advanced-data-analytics).  
It focuses on analyzing a real-world HR dataset to identify key factors affecting employee attrition and to help the Human Resources (HR) department of a large consulting firm make informed, data-driven decisions.

## 🎯 Objectives

- Explore and clean the HR dataset
- Identify patterns and relationships associated with employee attrition
- Build predictive models to estimate the probability of employee turnover
- Create insightful visualizations to support strategic HR planning
- Discuss ethical considerations and data limitations

## 📦 Deliverables

- Cleaned and well-documented dataset
- Exploratory Data Analysis (EDA) and visualizations
- Predictive modeling and model evaluation
- Key findings and recommendations for the HR department
- Ethical reflections and resource documentation

# **PACE stages**

![pace](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/1ecb8a7a-4c24-4eb6-83d2-801e3588269b)

## 🧭 Pace: Plan

### Understanding the Business Scenario and Problem

The HR department at *Salifort Motors* seeks to improve employee retention by understanding the key factors contributing to employee turnover. Rather than predicting which specific employees might leave, the focus of this project is to explore the dataset, uncover patterns, and identify the main drivers behind employee attrition.

By analyzing trends in the HR data—such as job satisfaction, department, years at company, salary level, and performance metrics—the company aims to take proactive, data-driven actions that address the root causes of employee dissatisfaction. These insights can help reduce turnover, improve morale, and lower recruitment costs in the long term.

## 🗂️ Dataset Overview

The dataset used in this project contains **15,000 rows** and **10 variables** related to employee behavior, performance, and organizational context at *Salifort Motors*. It serves as the foundation for understanding **why employees may leave the company**, and what patterns distinguish those who leave from those who stay.

> 📌 Source: [Kaggle - HR Analytics and Job Prediction Dataset](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv)

| Variable | Description |
|----------|-------------|
| `satisfaction_level` | Employee-reported job satisfaction level *(scale: 0 to 1)* |
| `last_evaluation` | Score from the most recent performance evaluation *(scale: 0 to 1)* |
| `number_project` | Number of active projects the employee worked on |
| `average_monthly_hours` | Average number of hours worked per month |
| `time_spend_company` | Tenure with the company (in years) |
| `Work_accident` | Whether the employee had a work-related accident *(0 = No, 1 = Yes)* |
| `left` | Whether the employee left the company *(1 = Yes, 0 = No)* |
| `promotion_last_5years` | Promotion received in the last 5 years *(0 = No, 1 = Yes)* |
| `Department` | Department or job role of the employee |
| `salary` | Salary category *(low, medium, high)* |

This dataset allows for a range of analytical tasks including:

- Comparing characteristics of employees who left vs those who stayed
- Exploring trends in satisfaction, workload, and promotions
- Identifying high-risk groups or roles with elevated turnover

## Step 1: Importing Required Libraries

We start by importing the necessary Python packages for data manipulation, visualization, and basic machine learning analysis.

### Importing packages


```python
# Import packages

# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle
```

## Step 2: Loading the Dataset

We load the HR dataset into a Pandas dataframe for exploration and analysis.

```python


# Load dataset into a dataframe
df0 = pd.read_csv("HR_capstone_dataset.csv")


# Display first few rows of the dataframe
df0.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>Department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Data Exploration and Cleaning

In this step, we will perform an initial exploration of the dataset to understand its structure, detect any missing values or anomalies, and prepare it for analysis.

```python
# Gather basic information about the data
df0.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14999 entries, 0 to 14998
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   satisfaction_level     14999 non-null  float64
     1   last_evaluation        14999 non-null  float64
     2   number_project         14999 non-null  int64  
     3   average_montly_hours   14999 non-null  int64  
     4   time_spend_company     14999 non-null  int64  
     5   Work_accident          14999 non-null  int64  
     6   left                   14999 non-null  int64  
     7   promotion_last_5years  14999 non-null  int64  
     8   Department             14999 non-null  object 
     9   salary                 14999 non-null  object 
    dtypes: float64(2), int64(6), object(2)
    memory usage: 1.1+ MB


The info indicates that `Department` and `salary` are objects, which might be categorical variables. We will figure that out later. Additionally, it shows that there are no missing values in the dataset.

### Descriptive Statistics

Next, we will generate descriptive statistics to understand the distribution of numerical features in the dataset.

This helps identify:
- the central tendencies (mean, median),
- data dispersion (standard deviation, range),
- potential outliers (via min/max),
- and skewness in certain variables.

We will also take a closer look at how these features relate to the attrition variable (`left`).


```python
# Gather descriptive statistics about the data
df0.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.612834</td>
      <td>0.716102</td>
      <td>3.803054</td>
      <td>201.050337</td>
      <td>3.498233</td>
      <td>0.144610</td>
      <td>0.238083</td>
      <td>0.021268</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.248631</td>
      <td>0.171169</td>
      <td>1.232592</td>
      <td>49.943099</td>
      <td>1.460136</td>
      <td>0.351719</td>
      <td>0.425924</td>
      <td>0.144281</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.090000</td>
      <td>0.360000</td>
      <td>2.000000</td>
      <td>96.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.440000</td>
      <td>0.560000</td>
      <td>3.000000</td>
      <td>156.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.640000</td>
      <td>0.720000</td>
      <td>4.000000</td>
      <td>200.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.820000</td>
      <td>0.870000</td>
      <td>5.000000</td>
      <td>245.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>310.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The descriptive statistics reveal several insights:

1. All variables have 14,999 entries, which means there are no missing values.
2. Some employees are completely dissatisfied with their job (`satisfaction_level` = 0). However, the average satisfaction (~0.61) shows moderate overall satisfaction.
3. The average of `last_evaluation` (~0.72) suggests decrease in average of employees satisfaction.
4. Employees work about 210 hours per month on average, which is higher than the standard 175 hours.
5. No employee has less than 2 years of experience at the company, suggesting that interns or short-term workers are not included.
6. Around 14% of employees experienced a work accident. This is a concern that the HSSE department should address.
7. About 24% of employees have left the company. This attrition rate is high enough to be investigated.
8. The promotion rate is low, although the satisfaction level of employees is high, they did not get enough promotions to encourage them which may contribute to employee dissatisfaction and turnover.

---

### Renaming Columns

To clean and standardize the dataset, we'll rename the columns using `snake_case` for consistency.






```python
# Display all column names
df0.columns

```




    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',
           'promotion_last_5years', 'Department', 'salary'],
          dtype='object')




```python
# Rename columns as needed

df0 = df0.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})

# Display all column names after the update

df0.columns
```




    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_monthly_hours', 'tenure', 'work_accident', 'left',
           'promotion_last_5years', 'department', 'salary'],
          dtype='object')



### Checking for Missing Values

Although the dataset appears to be complete, it's a good practice to verify that there are no missing values in any of the columns.


```python
# Check for missing values
df0.isna().sum()

```




    satisfaction_level       0
    last_evaluation          0
    number_project           0
    average_monthly_hours    0
    tenure                   0
    work_accident            0
    left                     0
    promotion_last_5years    0
    department               0
    salary                   0
    dtype: int64



### Checking for Duplicates

To ensure data quality, let's check if there are any duplicate rows in the dataset.


```python
# Check for duplicates
df0.duplicated().sum()

```




    3008



The output shows that there are 3,008 duplicate rows, which makes up 20% of the dataset. This is a significant portion and can affect the quality of our analysis if not addressed.


```python
# Inspect some containing duplicates as needed
df0_duplicated = df0[df0.duplicated() == True]
df0_duplicated

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>396</th>
      <td>0.46</td>
      <td>0.57</td>
      <td>2</td>
      <td>139</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>866</th>
      <td>0.41</td>
      <td>0.46</td>
      <td>2</td>
      <td>128</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1317</th>
      <td>0.37</td>
      <td>0.51</td>
      <td>2</td>
      <td>127</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>1368</th>
      <td>0.41</td>
      <td>0.52</td>
      <td>2</td>
      <td>132</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>RandD</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1461</th>
      <td>0.42</td>
      <td>0.53</td>
      <td>2</td>
      <td>142</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14994</th>
      <td>0.40</td>
      <td>0.57</td>
      <td>2</td>
      <td>151</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>0.37</td>
      <td>0.48</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14996</th>
      <td>0.37</td>
      <td>0.53</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>0.11</td>
      <td>0.96</td>
      <td>6</td>
      <td>280</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14998</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
<p>3008 rows × 10 columns</p>
</div>



The output displays the first five instances of rows that are duplicated elsewhere in the dataset. How likely is it that these entries are legitimate? In other words, is it plausible that two employees provided identical responses across all 10 variables, including multiple continuous features? Given this level of similarity, such duplicates appear highly improbable. Therefore, it is reasonable to assume they are unintentional and should be removed. However, before dropping them, we should verify whether doing so would significantly affect the class balance of the target variable.


```python
# Checking duplicates for each category in left column
df0_duplicated['left'].value_counts()
```




    1    1580
    0    1428
    Name: left, dtype: int64



This indicates that both classes in the target variable are similarly represented among the duplicate rows. As a result, removing them is unlikely to introduce class imbalance in the dataset.


```python
# Drop duplicates and save resulting dataframe in a new variable as needed
df1 = df0.drop_duplicates(keep='first')

# Display first few rows of new dataframe as needed
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>department</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>



Even though there are chances to have completely the same information with different employees, since we don't have identification info and we don't know if they are truly the same person, we still dropped the duplicates for the following reasons:

1- We will still have 11,991 data after we dropped the duplicates.
2- Both of the classes have the duplicates, dropping the rows won't have a serious impact on the balance of the dataset.
3- In order to keep our model more fair, dropping the duplicates is going to be better than not dropping them.

### Checking for outliers



```python
# Create a boxplot to visualize distribution of `tenure` and detect any outliers

plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1['tenure'])
plt.show()
```

![output_38_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/05a4214c-0fc5-4a42-a68b-1f76274ded5d)   


The boxplot above reveals the presence of outliers in the `tenure` variable. To better understand the impact of these outliers, it is useful to investigate how many rows in the dataset contain unusually high `tenure` values.

As part of this capstone project, I will analyze an HR dataset to uncover insights that can help the Human Resources (HR) department of a large consulting firm.

My deliverables will include model evaluation (along with interpretation where applicable), data visualizations directly related to the core analytical questions, ethical considerations, and a list of resources I referenced to solve problems or explore concepts during the project.


```python
# Determine the number of rows containing outliers

# Compute the 25th percentile value in `tenure`
percentile25 = df1['tenure'].quantile(0.25)

# Compute the 75th percentile value in `tenure`
percentile75 = df1['tenure'].quantile(0.75)

# Compute the interquartile range in `tenure`
iqr = percentile75 - percentile25

# Define the upper limit and lower limit for non-outlier values in `tenure`
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Identify subset of data containing outliers in `tenure`
outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]

# Count how many rows in the data contain outliers in `tenure`
print("Number of rows in the data containing outliers in `tenure`:", len(outliers))


```

    Lower limit: 1.5
    Upper limit: 5.5
    Number of rows in the data containing outliers in `tenure`: 824


Certain types of models are more sensitive to outliers than others. When we begin building predictive models, we should carefully consider whether to remove outliers based on the assumptions and robustness of the chosen algorithm.

## 📊 pAce: Analyze Stage
- Performing Exploratory Data Analysis (EDA) to examine relationships between variables

## Step 2. Continued Data Exploration

To begin the analysis, let’s first understand how many employees left the company and what percentage they represent out of the total workforce.


```python
# Get numbers of people who left vs. stayed
print(df1['left'].value_counts())
print()

# Get percentages of people who left vs. stayed
print(df1['left'].value_counts(normalize=True))
```

    0    10000
    1     1991
    Name: left, dtype: int64
    
    0    0.833959
    1    0.166041
    Name: left, dtype: float64


### Data Visualizations

Now, let's explore the variables by creating plots that help visualize relationships between key features in the dataset.

We will begin by creating a **stacked boxplot** to examine the distribution of `average_monthly_hours` across different `number_project` categories. This plot will allow us to compare the working hours of employees who stayed versus those who left the company.

Boxplots are a powerful tool for visualizing the distribution and spread of continuous variables. However, they may not always convey the underlying sample size in each category. To supplement this, we will also generate a **stacked histogram** showing the distribution of `number_project` for employees who stayed versus those who left. This will pro_



```python


# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing `average_monthly_hours` distributions for `number_project`, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')

# Create histogram showing distribution of `number_project`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')

# Display the plots
plt.show()

```

![output_48_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/b99422bf-0e66-487c-8b66-1fb54241fb11)   




It may seem intuitive that employees assigned to more projects also tend to work longer hours. This trend is reflected in the data, where the **average monthly hours increase** with the number of projects for both employees who stayed and those who left. However, several important patterns emerge from the boxplot:

1. **Two distinct groups among those who left**:
   - **Group A**: Employees who worked **significantly fewer hours** than peers with the same number of projects. It's possible that some of these individuals were let go, or they had already resigned and were given lighter workloads before departure.
   - **Group B**: Employees who worked **significantly more hours** than their peers. These individuals may have quit voluntarily due to burnout or feeling overburdened, especially if they were top contributors.

2. **Extreme attrition among employees with high project counts**:
   - Every employee with **seven projects** left the company.
   - For those with **six or seven projects**, the interquartile range of monthly hours was approximately **255–295 hours**, which is well above normal work expectations.

3. **Project load of 3–4 appears optimal**:
   - Employees in these groups had the **lowest attrition rate**, suggesting a more sustainable workload.

4. **Overwork across all groups**:
   - Assuming a standard work schedule of **40 hours per week** with **two weeks vacation annually**, the average monthly working hours would be:
     
     \[
     (50 \text{ weeks} \times 40 \text{ hours/week}) / 12 = 166.67 \text{ hours/month}
     \]
     
   - With the exception of employees working on two projects, **every group**—even among those who stayed—**exceeded this threshold**, indicating a culture of overwork.

---


To confirm our observations, let's specifically verify whether **all employees with seven projects** indeed left the company.



```python
# Get value counts of stayed/left for employees with 7 projects
df1[df1['number_project']==7]['left'].value_counts()

```




    1    145
    Name: left, dtype: int64





This confirms that **all employees with 7 projects** indeed left the company.


###  Exploring Satisfaction vs. Average Monthly Hours

As the next step, let’s examine the relationship between **average monthly working hours** and **employee satisfaction levels**.

This analysis will help us understand:

- Whether employees who work significantly more or fewer hours tend to report lower satisfaction.
- If there is a “sweet spot” in terms of working hours that aligns with higher satisfaction.
- Whether extreme workloads (both under- and over-engagement) contribute to dissatisfaction and attrition.

A scatter plot or density plot could help reveal potential clusters or trends in this relationship.



```python

# Create scatterplot of `average_monthly_hours` versus `satisfaction_level`, comparing employees who stayed versus those who left
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');

```

![output_52_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/e742a8b4-d9ed-4ac4-881d-ecb383921741)   


The scatter plot above reveals several important patterns:

1. **Overworked Employees with Low Satisfaction**  
   There is a sizeable group of employees who worked around **240–315 hours per month**.  
   > _315 hours per month_ translates to **over 75 hours per week**, sustained across a full year.  
   It's not surprising that many employees in this group had **satisfaction levels close to zero**—a clear indicator of burnout or overwork.

2. **Moderate Hours, Moderate Dissatisfaction**  
   Another group of employees who left worked **more typical hours** but still had relatively **low satisfaction (~0.4)**.  
   While it's harder to pinpoint the exact reason, it's possible that they felt **pressured** to work more, especially if surrounded by peers with significantly higher workloads. This perceived pressure may have contributed to their dissatisfaction and eventual departure.

3. **High Satisfaction Amongst Reasonably Worked Employees**  
   A third group of employees worked **210–280 hours per month** and reported **higher satisfaction levels (~0.7–0.9)**. This suggests that maintaining a healthy workload is positively associated with job satisfaction.

4. **Odd Distribution Patterns**  
   The strange shapes and sharp boundaries observed in the data may indicate **synthetic data generation** or possible **data manipulation**. This is something to keep in mind when interpreting results.

   
###  Visualizing Satisfaction by Tenure

As a next step, it would be insightful to explore how **satisfaction levels change with tenure**—i.e., the number of years an employee has been at the company.

This can help answer questions like:

- Do longer-serving employees tend to be more or less satisfied?
- Is there a specific point in time (e.g., year 3 or year 5) when satisfaction dips?


```python

# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing distributions of `satisfaction_level` by tenure, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

# Create histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')

plt.show();

```

![output_54_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/549e8780-2593-4cc3-ab2d-53df2e5bbbe3)   


This visualization offers several interesting insights:

- **Two Types of Departures**  
  Employees who left the company generally fall into two distinct groups:  
  1. **Dissatisfied employees with short tenures**, likely those who left early due to unmet expectations or poor fit.  
  2. **Highly satisfied employees with medium-length tenures**, who may have left for promotions, better opportunities, or personal reasons unrelated to dissatisfaction.

- **Unusual Dip at Four Years**  
  Employees with exactly **four years of tenure** who left the company show **anomalously low satisfaction**.  
  This could signal an internal policy change or cultural shift that specifically affected employees at this stage. If possible, it would be valuable to investigate HR or management decisions made around this point.

- **Long-Tenured Employees Stayed**  
  The longest-serving employees all **remained at the company**, and their satisfaction levels appear consistent with those of newer employees who also stayed. This may suggest a strong sense of loyalty or better alignment with company culture among senior staff.

- **Skewed Tenure Distribution**  
  The histogram indicates a **small number of long-tenured employees**. It's plausible these are **higher-level or better-compensated roles**, which could correlate with higher retention and satisfaction.

###  Satisfaction Statistics by Attrition

To further understand the relationship between satisfaction and attrition, we will calculate the **mean and median satisfaction scores** for employees who left and those who stayed. This will help quantify any differences in overall satisfaction levels between the two groups.


```python
# Calculate mean and median satisfaction scores of employees who left and those who stayed
df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>left</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.667365</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.440271</td>
      <td>0.41</td>
    </tr>
  </tbody>
</table>
</div>





As expected, the **mean and median satisfaction scores** of employees who left are **lower** than those of employees who stayed.

Interestingly, for employees who **stayed**, the **mean satisfaction** is slightly **lower than the median**, which suggests a **left-skewed distribution**—that is, a subset of these employees have particularly low satisfaction levels, pulling the mean down. This could imply that some employees remained at the company despite being relatively dissatisfied.

---

###  Salary Levels by Tenure

Next, we’ll explore how **salary levels vary across different tenure groups**. This may help identify whether compensation is aligned with employee experience and whether it potentially influences retention.



```python
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Define short-tenured employees
tenure_short = df1[df1['tenure'] < 7]

# Define long-tenured employees
tenure_long = df1[df1['tenure'] > 6]

# Plot short-tenured histogram
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14')

# Plot long-tenured histogram
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14');

```

![output_58_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/d455d2ed-47a3-4842-a8ab-b19171b80629)   




The plots above show that **long-tenured employees were not disproportionately represented among the higher-paid salary groups**. This suggests that simply staying at the company for a longer time does not necessarily lead to higher pay.

---

### Exploring the Relationship Between Workload and Evaluation

Next, let's explore whether there's a relationship between the **number of hours worked** and **performance evaluations**.

We’ll create a **scatterplot of `average_monthly_hours` versus `last_evaluation`** to see if employees who work longer hours also tend to receive higher performance scores.



```python

# Create scatterplot of `average_monthly_hours` versus `last_evaluation`
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');

```

 ![output_60_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/09950596-0a3b-4f52-aa32-bb033cc9256f)   



The following observations can be made from the scatterplot above:

- There appear to be **two distinct groups** of employees who left:
  - **Overworked high performers**, who worked significantly more than average and received high evaluation scores.
  - Employees who worked **below the nominal monthly average** (~166.67 hours) and had **lower evaluation scores**.
  
- There seems to be a **positive correlation** between hours worked and evaluation score — in general, employees who work more tend to be evaluated more highly.

- However, the **upper-left quadrant** (high hours, low evaluation) has few data points, suggesting that **long hours do not guarantee high evaluations**.

- It's also noteworthy that **most employees work well above 167 hours per month**, which may point to a culture of overwork.

---

###  Looking at Promotions

As the next step, let's examine whether employees who worked very long hours were **promoted in the last five years**. This can help assess whether high workload is rewarded with career advancement.



```python
# Create plot to examine relationship between `average_monthly_hours` and `promotion_last_5years`
plt.figure(figsize=(16, 3))
sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by promotion last 5 years', fontsize='14');

```


![output_62_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/8a881c17-8254-4939-b7ab-257ceaf3a522)   




The plot above reveals several important insights:

- **Very few employees who were promoted in the last five years ended up leaving** the company. This suggests that promotions may be a strong retention factor.

- Among those who **worked the most hours**, **very few were promoted**, indicating that working long hours does not necessarily lead to promotion.

- Interestingly, **all of the employees who left were among those working the longest hours**, reinforcing the idea that overwork may be linked to attrition.

---

###  Department-Level Attrition Analysis

Next, let's inspect how employees who left are distributed across **different departments**. This can help identify specific areas with higher attrition rates and may reveal organizational patterns or management issues.



```python
# Display counts for each department
df1["department"].value_counts()
```




    sales          3239
    technical      2244
    support        1821
    IT              976
    RandD           694
    product_mng     686
    marketing       673
    accounting      621
    hr              601
    management      436
    Name: department, dtype: int64




```python

# Create stacked histogram to compare department distribution of employees who left to that of employees who didn't
plt.figure(figsize=(11,8))
sns.histplot(data=df1, x='department', hue='left', discrete=1, 
             hue_order=[0, 1], multiple='dodge', shrink=.5)
plt.xticks(rotation='45')
plt.title('Counts of stayed/left by department', fontsize=14);
```


![output_65_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/5eed937a-611f-43b7-a708-5141967c29e3)   


The bar chart suggests that **no single department stands out** as having a significantly higher or lower proportion of employees who left versus those who stayed. This implies that attrition may be **driven more by individual factors** (such as workload, satisfaction, or promotion) than by departmental culture or structure.

---

###  Correlation Analysis

Lastly, let’s check for **strong correlations** between variables in the dataset. This will help us identify which features might be most predictive of employee attrition and guide future modeling efforts.



```python
# Plot a correlation heatmap
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df0.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);
```


![output_67_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/11dc2cd1-a7de-46a3-8110-64a2fec762aa)   



The correlation heatmap confirmed several key relationships:

- 📈 **Positive Correlation**: `number_project`, `average_monthly_hours`, and `last_evaluation` are positively correlated, suggesting employees who work on more projects tend to work longer hours and may receive higher evaluations.
- 📉 **Negative Correlation**: The variable `satisfaction_level` shows a clear negative correlation with the target variable `left`, confirming that less satisfied employees are more likely to leave.

---

### 💡 Insights

The exploratory data analysis reveals several likely causes of employee attrition:

- 🚫 **Poor Management & Burnout**: High number of projects and long work hours without corresponding promotions or recognition correlate with attrition.
- ⚠️ **Burned-Out High Performers**: Many high-evaluation employees working excessive hours still chose to leave, possibly due to burnout or lack of advancement.
- 📉 **Low Satisfaction**: The strongest predictor of employee exit appears to be low satisfaction.
- 🧓 **Tenure Effect**: Employees who have stayed beyond six years are significantly more likely to remain at the company, possibly due to reaching senior or less stressful roles.

---

# 🧱 paCe: Construct Stage

## Step 3 & 4. Model Building and Evaluation

### Predictive Task: Binary Classification

Our goal is to predict **employee attrition** (i.e., whether an employee left the company). This is a **binary classification** problem where:

- `1` = Employee left
- `0` = Employee stayed

---

### Model Selection

Since our target variable is binary, appropriate models include:

- **Logistic Regression** – a simple, interpretable baseline model.
- **Decision Tree** – a tree-based model that allows interpretability.
- **Random Forest** – an ensemble of trees that improves accuracy and generalizability.

We will begin by isolating the **target variable** and preparing the data for modeling.




```python
# Copy the dataframe
df_enc = df1.copy()

# Encode the `salary` column as an ordinal numeric category
df_enc['salary'] = (
    df_enc['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False)

# Display the new dataframe
df_enc.head()
# Isolate the outcome variable
y = df_enc['left']

# Display the first few rows of `y`
y.head()
```




    0    1
    1    1
    2    1
    3    1
    4    1
    Name: left, dtype: int64



Next, selecting the features. 


```python
# Select the features
X = df_enc.drop('left', axis=1)

# Display the first few rows of `X`
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Lastly, the dataset is split into training, validation, and testing sets to prepare for model training and evaluation.


```python
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
```

####  Decision Tree - Round 1

We start by constructing a Decision Tree model and setting up a cross-validated grid search to exhaustively find the best model parameters.


```python
# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')
```

Fitting the decision tree model to the training data.


```python
%%time
tree1.fit(X_train, y_train)
```

    CPU times: user 2.96 s, sys: 0 ns, total: 2.96 s
    Wall time: 2.96 s





    GridSearchCV(cv=4, error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=0, splitter='best'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'max_depth': [4, 6, 8, None],
                             'min_samples_leaf': [2, 5, 1],
                             'min_samples_split': [2, 4, 6]},
                 pre_dispatch='2*n_jobs', refit='roc_auc', return_train_score=False,
                 scoring={'f1', 'roc_auc', 'precision', 'recall', 'accuracy'},
                 verbose=0)



Identifing the optimal values for the decision tree parameters.


```python
# Check best parameters
tree1.best_params_
```




    {'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 2}



Identifing the best AUC score achieved by the decision tree model on the training set.


```python
# Check best AUC score on CV
tree1.best_score_
```




    0.969819392792457



This is a strong AUC score, indicating that the model performs well in predicting which employees are likely to leave.

Next, we’ll write a function to extract all the scores from the grid search results.



```python
def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): 
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                        })
  
    return table

# Get all CV scores
tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
tree1_cv_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>F1</th>
      <th>accuracy</th>
      <th>auc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>decision tree cv</td>
      <td>0.914552</td>
      <td>0.916949</td>
      <td>0.915707</td>
      <td>0.971978</td>
      <td>0.969819</td>
    </tr>
  </tbody>
</table>
</div>



All of the scores from the decision tree model indicate strong overall performance.

Next, we’ll construct a random forest model.

####  Random Forest - Round 1  
Building a random forest model and setting up a cross-validated grid search to exhaustively find the best model parameters.



```python
# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }  

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')
```

Fitting the random forest model to the training data.


```python
%%time
rf1.fit(X_train, y_train) 
```

    CPU times: user 9min 10s, sys: 923 ms, total: 9min 11s
    Wall time: 9min 11s





    GridSearchCV(cv=4, error_score=nan,
                 estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                                  class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features='auto',
                                                  max_leaf_nodes=None,
                                                  max_samples=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  n_estimators=100, n_jobs=None,...
                                                  verbose=0, warm_start=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'max_depth': [3, 5, None], 'max_features': [1.0],
                             'max_samples': [0.7, 1.0],
                             'min_samples_leaf': [1, 2, 3],
                             'min_samples_split': [2, 3, 4],
                             'n_estimators': [300, 500]},
                 pre_dispatch='2*n_jobs', refit='roc_auc', return_train_score=False,
                 scoring={'f1', 'roc_auc', 'precision', 'recall', 'accuracy'},
                 verbose=0)



Specifing path to save the model.


```python
# Define a path to save the model
path = '/home/project/hr_analysis/'
```

Defineing functions to pickle the model and read in the model.


```python
def write_pickle(path, model_object, save_as:str):
    '''
    In: 
        path:         path of folder to save the pickle
        model_object: a model  to pickle
        save_as:      filename for how to save the model

    Out: A call to pickle the model in the folder indicated
    '''    

    with open(path + save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)
        
def read_pickle(path, saved_model_name:str):
    '''
    In: 
        path:             path to folder where  to read from
        saved_model_name: filename of pickled model to read in

    Out: 
        model: the pickled model 
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model
```

Useing the functions defined above to save the model in a pickle file and then read it in.


```python
# Write pickle
write_pickle(path, rf1, 'hr_rf1')

# Read pickle
rf1 = read_pickle(path, 'hr_rf1')
```

Identify the best AUC score achieved by the random forest model on the training set.


```python
# Check best AUC score on CV
rf1.best_score_
```




    0.9804250949807172



Identifing the optimal values for the parameters of the random forest model.


```python
# Check best params
rf1.best_params_
```




    {'max_depth': 5,
     'max_features': 1.0,
     'max_samples': 0.7,
     'min_samples_leaf': 1,
     'min_samples_split': 4,
     'n_estimators': 500}



Collecting the evaluation scores on the training set for the decision tree and random forest models.


```python
# Get all CV scores
rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(tree1_cv_results)
print(rf1_cv_results)
```

                  model  precision    recall        F1  accuracy       auc
    0  decision tree cv   0.914552  0.916949  0.915707  0.971978  0.969819
                  model  precision    recall        F1  accuracy       auc
    0  random forest cv   0.950023  0.915614  0.932467  0.977983  0.980425


The evaluation scores of the random forest model are better than those of the decision tree model — with the exception of recall, which is only about 0.001 lower (a negligible difference). This suggests that the random forest model generally outperforms the decision tree.

Next, we’ll define a function to extract all the evaluation scores from a model’s predictions.



```python
def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
  
    return table
```

Now we use the best performing model to predict on the test set.


```python
# Get predictions on test data
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
rf1_test_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>random forest1 test</td>
      <td>0.964211</td>
      <td>0.919679</td>
      <td>0.941418</td>
      <td>0.980987</td>
      <td>0.956439</td>
    </tr>
  </tbody>
</table>
</div>



The test scores closely match the validation scores, which is a positive sign. This suggests that the model generalizes well, and since the test set was used exclusively for this evaluation, we can be more confident that the model’s performance on unseen data will be similar.

#### Feature Engineering

I’m somewhat skeptical of the high evaluation scores — there may be some degree of data leakage. Data leakage occurs when the model is trained using information that it wouldn’t realistically have access to at prediction time. This often results in overly optimistic performance that won’t hold in production.

In this case, the company might not consistently have up-to-date satisfaction scores for all employees. Additionally, the `average_monthly_hours` variable could also be problematic. For example, employees planning to quit or identified for termination may start working fewer hours, introducing a bias that inflates model performance.

The initial decision tree and random forest models used all available variables as features. In this next round, I’ll introduce some feature engineering to help the model generalize better.

Specifically, I’ll drop the `satisfaction_level` variable and create a new binary feature called `overworked`, which captures whether an employee is putting in significantly more hours than average.



```python
# Drop `satisfaction_level` and save resulting dataframe in new variable
df2 = df_enc.drop('satisfaction_level', axis=1)

# Display first few rows of new dataframe
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create `overworked` column. For now, it's identical to average monthly hours.
df2['overworked'] = df2['average_monthly_hours']

# Inspect max and min average monthly hours values
print('Max hours:', df2['overworked'].max())
print('Min hours:', df2['overworked'].min())
```

    Max hours: 310
    Min hours: 96


An average monthly workload of **166.67 hours** corresponds to someone working 8 hours a day, 5 days a week, for 50 weeks in a year.

To define overwork, we'll set a threshold: employees working more than **175 hours per month** on average will be considered **overworked**.

We will create a new binary column, `overworked`, using a boolean mask:

- `df3['average_monthly_hours'] > 175` returns a series of boolean values (`True` for hours over 175, `False` otherwise).
- Applying `.astype(int)` converts these boolean values into `1` (overworked) and `0` (not overworked).



```python
# Define `overworked` as working > 175 hrs/week
df2['overworked'] = (df2['overworked'] > 175).astype(int)

# Display first few rows of new column
df2['overworked'].head()
```




    0    0
    1    1
    2    1
    3    1
    4    0
    Name: overworked, dtype: int64



Droping the `average_monthly_hours` column.



```python
# Drop the `average_monthly_hours` column
df2 = df2.drop('average_monthly_hours', axis=1)

# Display first few rows of resulting dataframe
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>tenure</th>
      <th>work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>salary</th>
      <th>department_IT</th>
      <th>department_RandD</th>
      <th>department_accounting</th>
      <th>department_hr</th>
      <th>department_management</th>
      <th>department_marketing</th>
      <th>department_product_mng</th>
      <th>department_sales</th>
      <th>department_support</th>
      <th>department_technical</th>
      <th>overworked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.53</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.86</td>
      <td>5</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.88</td>
      <td>7</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.87</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.52</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Again, isolating the features and target variables


```python
# Isolate the outcome variable
y = df2['left']

# Select the features
X = df2.drop('left', axis=1)
```

Spliting the data into training and testing sets.


```python
# Create test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
```

#### Decision tree - Round 2


```python
# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


```


```python
%%time
tree2.fit(X_train, y_train)
```

    CPU times: user 2.4 s, sys: 3.99 ms, total: 2.41 s
    Wall time: 2.41 s





    GridSearchCV(cv=4, error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=0, splitter='best'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'max_depth': [4, 6, 8, None],
                             'min_samples_leaf': [2, 5, 1],
                             'min_samples_split': [2, 4, 6]},
                 pre_dispatch='2*n_jobs', refit='roc_auc', return_train_score=False,
                 scoring={'f1', 'roc_auc', 'precision', 'recall', 'accuracy'},
                 verbose=0)




```python
# Check best params
tree2.best_params_
```




    {'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 6}




```python
# Check best AUC score on CV
tree2.best_score_

```




    0.9586752505340426



This model performs very well, even without satisfaction levels and detailed hours worked data.   
   
Next, checking the other scores.


```python
# Get all CV scores
tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
print(tree1_cv_results)
print(tree2_cv_results)
```

                  model  precision    recall        F1  accuracy       auc
    0  decision tree cv   0.914552  0.916949  0.915707  0.971978  0.969819
                   model  precision    recall        F1  accuracy       auc
    0  decision tree2 cv   0.856693  0.903553  0.878882  0.958523  0.958675


Some of the other scores fell. That's to be expected given fewer features were taken into account in this round of the model. Still, the scores are very good.

#### Random forest - Round 2


```python
# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }  

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')
```


```python
%%time
rf2.fit(X_train, y_train) 
```

    CPU times: user 7min 15s, sys: 1.11 s, total: 7min 16s
    Wall time: 7min 16s





    GridSearchCV(cv=4, error_score=nan,
                 estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                                  class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features='auto',
                                                  max_leaf_nodes=None,
                                                  max_samples=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  n_estimators=100, n_jobs=None,...
                                                  verbose=0, warm_start=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'max_depth': [3, 5, None], 'max_features': [1.0],
                             'max_samples': [0.7, 1.0],
                             'min_samples_leaf': [1, 2, 3],
                             'min_samples_split': [2, 3, 4],
                             'n_estimators': [300, 500]},
                 pre_dispatch='2*n_jobs', refit='roc_auc', return_train_score=False,
                 scoring={'f1', 'roc_auc', 'precision', 'recall', 'accuracy'},
                 verbose=0)




```python
# Write pickle
write_pickle(path, rf2, 'hr_rf2')

# Read in pickle
rf2 = read_pickle(path, 'hr_rf2')

# Check best params
rf2.best_params_

# Check best AUC score on CV
rf2.best_score_

# Get all CV scores
rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
print(tree2_cv_results)
print(rf2_cv_results)
```

                   model  precision    recall        F1  accuracy       auc
    0  decision tree2 cv   0.856693  0.903553  0.878882  0.958523  0.958675
                   model  precision    recall        F1  accuracy      auc
    0  random forest2 cv   0.866758  0.878754  0.872407  0.957411  0.96481


Again, the scores dropped slightly, but the random forest performs better than the decision tree if using AUC as the deciding metric. 



```python
# Get predictions on test data
rf2_test_scores = get_scores('random forest2 test', rf2, X_test, y_test)
rf2_test_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>AUC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>random forest2 test</td>
      <td>0.870406</td>
      <td>0.903614</td>
      <td>0.8867</td>
      <td>0.961641</td>
      <td>0.938407</td>
    </tr>
  </tbody>
</table>
</div>



This seems to be a stable, well-performing final model. 

Let's plot a confusion matrix to visualize how well it predicts on the test set.


```python
# Generate array of values for confusion matrix
preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf2.classes_)
disp.plot(values_format='');
```


![output_145_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/1062ba99-dcad-43e6-b29d-6661fe082047)    


The model predicts more false positives than false negatives, which means that some employees may be identified as at risk of quitting or getting fired, when that's actually not the case. But this is still a strong model.

For exploratory purpose, I want to inspect the splits of the decision tree model and the most important features in the random forest model. 

#### Decision tree splits


```python
# Plot the tree
plt.figure(figsize=(85,20))
plot_tree(tree2.best_estimator_, max_depth=6, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()
```

  
![output_148_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/a603584e-a8fc-4cd1-b676-fcd28ed58d3f)   



```python
#tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X.columns)
tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, 
                                 columns=['gini_importance'], 
                                 index=X.columns
                                )
tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending=False)

# Only extract the features with importances > 0
tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]
tree2_importances
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gini_importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>last_evaluation</th>
      <td>0.343958</td>
    </tr>
    <tr>
      <th>number_project</th>
      <td>0.343385</td>
    </tr>
    <tr>
      <th>tenure</th>
      <td>0.215681</td>
    </tr>
    <tr>
      <th>overworked</th>
      <td>0.093498</td>
    </tr>
    <tr>
      <th>department_support</th>
      <td>0.001142</td>
    </tr>
    <tr>
      <th>salary</th>
      <td>0.000910</td>
    </tr>
    <tr>
      <th>department_sales</th>
      <td>0.000607</td>
    </tr>
    <tr>
      <th>department_technical</th>
      <td>0.000418</td>
    </tr>
    <tr>
      <th>work_accident</th>
      <td>0.000183</td>
    </tr>
    <tr>
      <th>department_IT</th>
      <td>0.000139</td>
    </tr>
    <tr>
      <th>department_marketing</th>
      <td>0.000078</td>
    </tr>
  </tbody>
</table>
</div>



Then create a barplot to visualize the decision tree feature importances.


```python
sns.barplot(data=tree2_importances, x="gini_importance", y=tree2_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()
```


![output_151_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/f8eb864a-1dc6-45bb-b689-7be9ccd02ec5)   




The barplot above shows that in this decision tree model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`.

#### Random forest feature importance

Now, let's plot the feature importances for the random forest model.


```python
# Get feature importances
feat_impt = rf2.best_estimator_.feature_importances_

# Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]

# Get column labels of top 10 features 
feat = X.columns[ind]

# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]

y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)

y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")

plt.show()
```


![output_154_0](https://github.com/ImanBrjn/python_Salifort_Motors/assets/140934258/78b3b1b7-8fd4-40e9-92e3-c62c2ba40d9c)   



The plot above shows that in this random forest model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`, and they are the same as the ones used by the decision tree model.

# pacE: Execute Stage

## Recall evaluation metrics

- **AUC** is the area under the ROC curve; it's also considered the probability that the model ranks a random positive example more highly than a random negative example.
- **Precision** measures the proportion of data points predicted as True that are actually True, in other words, the proportion of positive predictions that are true positives.
- **Recall** measures the proportion of data points that are predicted as True, out of all the data points that are actually True. In other words, it measures the proportion of positives that are correctly classified.
- **Accuracy** measures the proportion of data points that are correctly classified.
- **F1-score** is an aggregation of precision and recall.


## Step 4. Results and Evaluation

### Summary of model results
After conducting feature engineering, the decision tree model achieved AUC of 93.8%, precision of 87.0%, recall of 90.4%, f1-score of 88.7%, and accuracy of 96.2%, on the test set. The random forest modestly outperformed the decision tree model. 

### Conclusion and Recommendations

The models and the feature importances extracted from the models confirm that employees at the company are overworked. 

To retain employees, the following recommendations could be presented to the stakeholders:

* Cap the number of projects that employees can work on.
* Consider promoting employees who have been with the company for atleast four years, or conduct further investigation about why four-year tenured employees are so dissatisfied. 
* Either reward employees for working longer hours, or don't require them to do so. 
* If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear. 
* Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts. 
* High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort. 



**Thank you for taking the time to read my Capstone Project!**


## 🔗 Resources and Troubleshooting
- Stack Overflow
- pandas documentation
- seaborn official site
- Google search for syntax/debugging

