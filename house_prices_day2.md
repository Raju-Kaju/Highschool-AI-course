# House Prices Competition - Day 2
## Data Exploration & Visualization
### Detailed Teacher Guide with Script & Complete Code

---

## Lesson Overview
**Duration:** 50 minutes  
**Objective:** Students explore the house prices dataset and create regression visualizations  
**Key Outcome:** Students understand key relationships and create scatter plots showing price correlations

---

## Materials Checklist
- [ ] Projector for live coding
- [ ] Student computers with Python/Jupyter OR Google Colab
- [ ] House Prices dataset (train.csv)
- [ ] Printed Code Reference Sheet
- [ ] Completed homework from Day 1
- [ ] Example scatter plot printouts

---

## Key Regression Visualization Concepts

**Unlike Titanic (Classification):**
- We used BAR CHARTS to show survival rates by category
- We counted survivors vs non-survivors

**For Regression:**
- We use SCATTER PLOTS to show relationships between features and price
- We calculate CORRELATIONS to measure strength of relationships
- We look for LINEAR patterns (does price go up as feature increases?)

---

## MINUTE-BY-MINUTE LESSON PLAN

### Minutes 0-5: Warm-Up & Homework Review

**TEACHER SCRIPT:**

"Good morning! Before we code, let's talk about your homework predictions.

You predicted which 5 features would be most important. Let's see what you thought:

Raise your hand if you predicted:
- Square footage / size features [count]
- Location / neighborhood [count]
- Quality ratings [count]
- Number of bedrooms/bathrooms [count]
- Age of house [count]

[WRITE TALLIES ON BOARD]

Interesting! Most of you focused on size, location, and quality. Makes sense!

Today we're going to test your predictions using REAL DATA. We'll create visualizations that show which features actually correlate most strongly with price.

And let me ask - who researched R²? Can someone explain it in their own words?

[CALL ON 2-3 STUDENTS]

Good! R² tells us 'how much of the price variation does our model explain?' - closer to 1 is better.

Let's dive into the data!"

---

### Minutes 5-10: Setup & Loading Data

**TEACHER SCRIPT:**

"Open your laptops and start your Python environment. We're using the same setup as Titanic.

[IF USING GOOGLE COLAB:]
Click the link I shared. It's a fresh notebook for House Prices.

[IF USING JUPYTER:]
Open Jupyter and create a new notebook called 'HousePrices_Day2.ipynb'

Watch my screen - I'll type first, then you follow."

[SWITCH TO SCREEN SHARE]

---

### Minutes 10-25: GUIDED CODING - Loading and Initial Exploration

**TEACHER SCRIPT:**

"First, let's import our tools and load the data."

[TYPE SLOWLY, EXPLAINING:]

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Set visualization style
sns.set_style('whitegrid')

print("✓ Libraries imported!")
```

**TEACHER SCRIPT:**

"Same libraries as Titanic, but now we're working with regression data. Let's load it:"

```python
# Load the training data
train = pd.read_csv('train.csv')

print("Data loaded successfully!")
print(f"Shape: {train.shape}")
print(f"({train.shape[0]} houses × {train.shape[1]} features)")
```

**TEACHER SCRIPT:**

"See that? 1,460 houses and 81 columns (79 features + Id + SalePrice).

Now let's peek at the first few houses:"

```python
# View first few rows
print("First 5 houses:")
print(train.head())
```

**TEACHER SCRIPT:**

"Wow! Look at all those columns scrolling across! That's a LOT of information per house.

Let's see specifically the price column and a few key features:"

```python
# Look at specific columns
print("\nKey features for first 5 houses:")
key_cols = ['Id', 'MSSubClass', 'LotArea', 'OverallQual', 'YearBuilt', 
            'GrLivArea', 'BedroomAbvGr', 'SalePrice']
print(train[key_cols].head(10))
```

**TEACHER SCRIPT:**

"Perfect! Now we can actually read it. Look at house ID 1:
- LotArea: 8,450 sq ft lot
- OverallQual: 7 (pretty good!)
- YearBuilt: 2003 (fairly new)
- GrLivArea: 1,710 sq ft
- BedroomAbvGr: 3 bedrooms
- **SalePrice: $208,500**

Now type all this code. Take your time!"

[GIVE 3-4 MINUTES]

---

**TEACHER SCRIPT (Basic Statistics):**

"Now let's get basic information about our dataset:"

```python
# Dataset information
print("="*60)
print("DATASET INFORMATION")
print("="*60)
print(train.info())
```

**TEACHER SCRIPT:**

"Notice some columns have fewer than 1460 non-null entries? That means missing data! We'll handle that tomorrow.

Now let's look at statistics for our target variable - SalePrice:"

```python
# SalePrice statistics
print("\n" + "="*60)
print("SALEPRICE STATISTICS")
print("="*60)
print(train['SalePrice'].describe())

print(f"\nMinimum price: ${train['SalePrice'].min():,.0f}")
print(f"Maximum price: ${train['SalePrice'].max():,.0f}")
print(f"Average price: ${train['SalePrice'].mean():,.0f}")
print(f"Median price: ${train['SalePrice'].median():,.0f}")
```

**TEACHER SCRIPT:**

"Look at this range!
- Cheapest house: $34,900 (maybe tiny and run-down?)
- Most expensive: $755,000 (luxury home!)
- Average: $180,921
- Median: $163,000

The median is lower than the mean - this tells us there are some very expensive houses pulling the average up.

Everyone run this code!"

[GIVE 2-3 MINUTES]

---

### Minutes 25-40: CREATING REGRESSION VISUALIZATIONS

**TEACHER SCRIPT:**

"Now for the exciting part - VISUALIZATIONS!

In Titanic, we made bar charts. For regression, we make SCATTER PLOTS.

**Why scatter plots?**
- X-axis: A feature (like square footage)
- Y-axis: Price
- Each dot: One house
- Pattern shows: Does price increase as feature increases?

Let's create our first scatter plot - GrLivArea (living area) vs Price:"

```python
# Visualization 1: Living Area vs Price
plt.figure(figsize=(10, 6))
plt.scatter(train['GrLivArea'], train['SalePrice'], alpha=0.5)
plt.title('Living Area vs Sale Price', fontsize=16, fontweight='bold')
plt.xlabel('Above Ground Living Area (sq ft)', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(train['GrLivArea'], train['SalePrice'], 1)
p = np.poly1d(z)
plt.plot(train['GrLivArea'], p(train['GrLivArea']), "r--", linewidth=2, label='Trend')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate correlation
correlation = train['GrLivArea'].corr(train['SalePrice'])
print(f"\nCorrelation between GrLivArea and SalePrice: {correlation:.3f}")
```

**TEACHER SCRIPT:**

"WOW! Look at that pattern!

See how the dots go UP and to the RIGHT? That's a POSITIVE CORRELATION.

As living area increases, price increases!

The red dashed line shows the trend. And look at the correlation number - probably around 0.70 or 0.71.

**Correlation ranges from -1 to +1:**
- +1 = perfect positive relationship
- 0 = no relationship
- -1 = perfect negative relationship

0.70 is STRONG! Living area is highly predictive of price!

Type this code carefully - especially the trend line part."

[GIVE 4-5 MINUTES - CIRCULATE HEAVILY]

---

**TEACHER SCRIPT:**

"Let's do another - OverallQual (quality rating 1-10) vs Price:"

```python
# Visualization 2: Overall Quality vs Price
plt.figure(figsize=(10, 6))

# Create box plot for each quality level
train.boxplot(column='SalePrice', by='OverallQual', figsize=(12, 6))
plt.title('Sale Price by Overall Quality Rating', fontsize=16, fontweight='bold')
plt.suptitle('')  # Remove default title
plt.xlabel('Overall Quality (1=Very Poor, 10=Very Excellent)', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.tight_layout()
plt.show()

# Calculate correlation
correlation_qual = train['OverallQual'].corr(train['SalePrice'])
print(f"\nCorrelation between OverallQual and SalePrice: {correlation_qual:.3f}")
```

**TEACHER SCRIPT:**

"This is a BOX PLOT - perfect for categorical features!

Each box shows the price distribution for that quality level.

See the pattern? As quality increases (1→10), the median price (line in middle) goes UP!

Quality 10 houses? Way more expensive than quality 1!

Correlation is probably around 0.79 or 0.80 - even STRONGER than living area!

This means quality rating is super important for price!"

---

**TEACHER SCRIPT:**

"Now let's look at Year Built - does newer = more expensive?"

```python
# Visualization 3: Year Built vs Price
plt.figure(figsize=(10, 6))
plt.scatter(train['YearBuilt'], train['SalePrice'], alpha=0.5, c='green')
plt.title('Year Built vs Sale Price', fontsize=16, fontweight='bold')
plt.xlabel('Year Built', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(train['YearBuilt'], train['SalePrice'], 1)
p = np.poly1d(z)
plt.plot(train['YearBuilt'], p(train['YearBuilt']), "r--", linewidth=2, label='Trend')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate correlation
correlation_year = train['YearBuilt'].corr(train['SalePrice'])
print(f"\nCorrelation between YearBuilt and SalePrice: {correlation_year:.3f}")
```

**TEACHER SCRIPT:**

"See? Newer houses (to the right) generally cost more!

Though there's more scatter here - some old houses are expensive (maybe historic/well-maintained), and some new houses are cheap (maybe small).

Correlation probably around 0.52 - moderate relationship.

Everyone create these three visualizations!"

[GIVE 3-4 MINUTES]

---

### Minutes 40-47: CORRELATION HEATMAP - The Big Picture

**TEACHER SCRIPT:**

"Now I'm going to show you something REALLY cool - a correlation heatmap!

This shows correlations between ALL numeric features and SalePrice at once!"

```python
# Find top correlated features with SalePrice
print("="*60)
print("TOP 10 FEATURES CORRELATED WITH SALEPRICE")
print("="*60)

# Calculate correlations
correlations = train.corr()['SalePrice'].sort_values(ascending=False)

# Show top 10
print(correlations.head(10))

# Visualize as heatmap
plt.figure(figsize=(8, 10))
top_features = correlations.head(10).index
sns.heatmap(train[top_features].corr(), annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Top 10 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**TEACHER SCRIPT:**

"This is GOLD! Look at the numbers showing correlation with SalePrice:

Typically you'll see something like:
1. SalePrice: 1.000 (correlates perfectly with itself!)
2. OverallQual: ~0.79 (quality is #1 predictor!)
3. GrLivArea: ~0.71 (living area is #2!)
4. GarageCars: ~0.64 (garage size matters!)
5. GarageArea: ~0.62 (similar to garage cars)
6. TotalBsmtSF: ~0.61 (basement size)
7. 1stFlrSF: ~0.60 (first floor size)
8. FullBath: ~0.56 (number of bathrooms)
9. TotRmsAbvGrd: ~0.53 (total rooms)
10. YearBuilt: ~0.52 (newer homes)

The heatmap colors show this:
- Dark red = strong positive correlation
- White = no correlation
- Dark blue = negative correlation

Were your homework predictions correct? Did you guess OverallQual and GrLivArea?

Type this code - this heatmap is super useful!"

[GIVE 3-4 MINUTES]

---

### Minutes 47-50: CLOSURE & SUMMARY

**TEACHER SCRIPT:**

"Amazing work today! Let's summarize what we discovered:

[WRITE ON BOARD:]

**Key Findings:**
1. **OverallQual (0.79)** - Quality rating is THE strongest predictor!
2. **GrLivArea (0.71)** - Living space size is #2
3. **GarageCars (0.64)** - Garage capacity matters
4. **YearBuilt (0.52)** - Newer houses generally cost more
5. **Price range**: $34,900 to $755,000 (huge variation!)

**Technical Skills Learned:**
- Loading regression data with pandas
- Creating scatter plots for continuous relationships
- Box plots for categorical vs continuous
- Calculating correlations
- Creating correlation heatmaps
- Interpreting strength of relationships

**Key Difference from Titanic:**
- Titanic: Bar charts showing survival RATES (percentages)
- House Prices: Scatter plots showing RELATIONSHIPS (correlations)

**Tomorrow's Preview:**
Tomorrow we'll clean this messy data:
- Fill missing values (lots of them!)
- Handle categorical variables (neighborhoods, quality ratings)
- Create new features (total square footage, house age)
- Prepare for our regression model

**Homework:**
1. Complete any visualizations you didn't finish
2. Answer the reflection questions on the handout
3. Based on today's correlation analysis, revise your prediction: Which feature do you NOW think is most important? Did your Day 1 prediction change?

Save your notebook - we need it tomorrow!

Any questions?"

[ANSWER QUESTIONS]

"Excellent exploring today! See you tomorrow for data preprocessing!"

---

## COMPLETE WORKING CODE - DAY 2

```python
# ============================================
# HOUSE PRICES DATA EXPLORATION - DAY 2
# Student Name: _______________
# Date: _______________
# ============================================

# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("✓ Libraries imported successfully!")

# ============================================

# Cell 2: Load Data
train = pd.read_csv('train.csv')

print("="*60)
print("DATA LOADED")
print("="*60)
print(f"Dataset shape: {train.shape}")
print(f"({train.shape[0]} houses × {train.shape[1]} columns)")
print("\n✓ Data loaded successfully!")

# ============================================

# Cell 3: Initial Exploration
print("\n" + "="*60)
print("FIRST 5 HOUSES")
print("="*60)
print(train.head())

print("\n" + "="*60)
print("KEY FEATURES FOR FIRST 10 HOUSES")
print("="*60)
key_cols = ['Id', 'MSSubClass', 'LotArea', 'OverallQual', 'YearBuilt', 
            'GrLivArea', 'BedroomAbvGr', 'SalePrice']
print(train[key_cols].head(10))

# ============================================

# Cell 4: Dataset Information
print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
print(train.info())

print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(train.describe())

# ============================================

# Cell 5: SalePrice Analysis
print("\n" + "="*60)
print("SALEPRICE STATISTICS")
print("="*60)
print(train['SalePrice'].describe())

print(f"\n💰 Price Range:")
print(f"   Minimum: ${train['SalePrice'].min():,.0f}")
print(f"   Maximum: ${train['SalePrice'].max():,.0f}")
print(f"   Average: ${train['SalePrice'].mean():,.0f}")
print(f"   Median:  ${train['SalePrice'].median():,.0f}")

# Distribution
plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(train['SalePrice'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribution of Sale Prices', fontsize=14, fontweight='bold')
plt.xlabel('Sale Price ($)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.axvline(train['SalePrice'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f"Mean: ${train['SalePrice'].mean():,.0f}")
plt.axvline(train['SalePrice'].median(), color='green', linestyle='--', 
            linewidth=2, label=f"Median: ${train['SalePrice'].median():,.0f}")
plt.legend()

# Box plot
plt.subplot(1, 2, 2)
plt.boxplot(train['SalePrice'])
plt.title('Sale Price Box Plot', fontsize=14, fontweight='bold')
plt.ylabel('Sale Price ($)', fontsize=11)

plt.tight_layout()
plt.show()

# ============================================

# Cell 6: Visualization 1 - Living Area vs Price
print("\n" + "="*60)
print("VISUALIZATION 1: LIVING AREA vs PRICE")
print("="*60)

plt.figure(figsize=(10, 6))
plt.scatter(train['GrLivArea'], train['SalePrice'], alpha=0.5, s=30)
plt.title('Living Area vs Sale Price', fontsize=16, fontweight='bold')
plt.xlabel('Above Ground Living Area (sq ft)', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(train['GrLivArea'], train['SalePrice'], 1)
p = np.poly1d(z)
plt.plot(train['GrLivArea'], p(train['GrLivArea']), "r--", 
         linewidth=2, label='Trend Line')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate correlation
correlation_area = train['GrLivArea'].corr(train['SalePrice'])
print(f"\n📊 Correlation: {correlation_area:.3f}")
print(f"   Interpretation: Strong positive relationship!")
print(f"   As living area increases, price increases.")

# ============================================

# Cell 7: Visualization 2 - Overall Quality vs Price
print("\n" + "="*60)
print("VISUALIZATION 2: OVERALL QUALITY vs PRICE")
print("="*60)

# Box plot by quality
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Box plot
train.boxplot(column='SalePrice', by='OverallQual', ax=axes[0])
axes[0].set_title('Sale Price by Overall Quality', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Overall Quality (1=Poor, 10=Excellent)', fontsize=11)
axes[0].set_ylabel('Sale Price ($)', fontsize=11)
plt.suptitle('')  # Remove default title

# Bar plot of average prices
avg_by_quality = train.groupby('OverallQual')['SalePrice'].mean()
avg_by_quality.plot(kind='bar', ax=axes[1], color='skyblue', edgecolor='black')
axes[1].set_title('Average Price by Quality', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Overall Quality', fontsize=11)
axes[1].set_ylabel('Average Sale Price ($)', fontsize=11)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Calculate correlation
correlation_qual = train['OverallQual'].corr(train['SalePrice'])
print(f"\n📊 Correlation: {correlation_qual:.3f}")
print(f"   Interpretation: Very strong positive relationship!")
print(f"   Quality is a TOP predictor of price!")

# ============================================

# Cell 8: Visualization 3 - Year Built vs Price
print("\n" + "="*60)
print("VISUALIZATION 3: YEAR BUILT vs PRICE")
print("="*60)

plt.figure(figsize=(10, 6))
plt.scatter(train['YearBuilt'], train['SalePrice'], alpha=0.5, c='green', s=30)
plt.title('Year Built vs Sale Price', fontsize=16, fontweight='bold')
plt.xlabel('Year Built', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(train['YearBuilt'], train['SalePrice'], 1)
p = np.poly1d(z)
plt.plot(train['YearBuilt'], p(train['YearBuilt']), "r--", 
         linewidth=2, label='Trend Line')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate correlation
correlation_year = train['YearBuilt'].corr(train['SalePrice'])
print(f"\n📊 Correlation: {correlation_year:.3f}")
print(f"   Interpretation: Moderate positive relationship")
print(f"   Newer houses tend to be more expensive")

# ============================================

# Cell 9: Correlation Analysis - Top Features
print("\n" + "="*60)
print("TOP 15 FEATURES CORRELATED WITH SALEPRICE")
print("="*60)

# Calculate all correlations with SalePrice
correlations = train.corr()['SalePrice'].sort_values(ascending=False)

# Display top 15
print(correlations.head(15))

# Visualize
plt.figure(figsize=(10, 8))
correlations.head(15).plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 15 Features Correlated with Sale Price', 
          fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ============================================

# Cell 10: Correlation Heatmap
print("\n" + "="*60)
print("CORRELATION HEATMAP - TOP 10 FEATURES")
print("="*60)

# Get top 10 features most correlated with SalePrice
top_features = correlations.head(10).index

# Create correlation matrix for these features
corr_matrix = train[top_features].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, 
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Top 10 Features vs SalePrice', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 How to read this heatmap:")
print("   - Dark RED = Strong positive correlation")
print("   - White = No correlation")
print("   - Dark BLUE = Strong negative correlation")
print("   - Look at the SalePrice row/column to see relationships")

# ============================================

# Cell 11: Additional Visualizations
print("\n" + "="*60)
print("ADDITIONAL INSIGHTS")
print("="*60)

# Create multiple scatter plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Total Basement SF
axes[0, 0].scatter(train['TotalBsmtSF'], train['SalePrice'], alpha=0.5)
axes[0, 0].set_title('Total Basement SF vs Price', fontweight='bold')
axes[0, 0].set_xlabel('Total Basement SF')
axes[0, 0].set_ylabel('Sale Price ($)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Garage Cars
axes[0, 1].boxplot([train[train['GarageCars']==i]['SalePrice'].dropna() 
                     for i in range(5)], labels=range(5))
axes[0, 1].set_title('Garage Capacity vs Price', fontweight='bold')
axes[0, 1].set_xlabel('Number of Cars')
axes[0, 1].set_ylabel('Sale Price ($)')

# Plot 3: Full Bathrooms
axes[1, 0].boxplot([train[train['FullBath']==i]['SalePrice'].dropna() 
                     for i in range(4)], labels=range(4))
axes[1, 0].set_title('Full Bathrooms vs Price', fontweight='bold')
axes[1, 0].set_xlabel('Number of Full Baths')
axes[1, 0].set_ylabel('Sale Price ($)')

# Plot 4: LotArea
axes[1, 1].scatter(train['LotArea'], train['SalePrice'], alpha=0.5, c='purple')
axes[1, 1].set_title('Lot Area vs Price', fontweight='bold')
axes[1, 1].set_xlabel('Lot Area (sq ft)')
axes[1, 1].set_ylabel('Sale Price ($)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================

# Cell 12: Summary Report
print("\n" + "="*70)
print("📊 EXPLORATION SUMMARY")
print("="*70)

print("\n🏆 TOP 5 PREDICTIVE FEATURES:")
print("="*70)
top_5 = correlations.head(6)[1:]  # Skip SalePrice itself
for i, (feature, corr) in enumerate(top_5.items(), 1):
    print(f"   {i}. {feature:20s} → Correlation: {corr:.3f}")

print("\n💰 PRICE STATISTICS:")
print("="*70)
print(f"   Range: ${train['SalePrice'].min():,} to ${train['SalePrice'].max():,}")
print(f"   Average: ${train['SalePrice'].mean():,.0f}")
print(f"   Median: ${train['SalePrice'].median():,.0f}")
print(f"   Std Dev: ${train['SalePrice'].std():,.0f}")

print("\n📈 KEY INSIGHTS:")
print("="*70)
print(f"   • OverallQual is the STRONGEST predictor (r={correlations['OverallQual']:.3f})")
print(f"   • Size features (GrLivArea, GarageCars) are very important")
print(f"   • Quality matters more than size!")
print(f"   • Age (YearBuilt) has moderate impact")

print("\n✓ Ready for preprocessing tomorrow!")
print("="*70)
```

---

## HOMEWORK ASSIGNMENT - DAY 2

**Name:** _________________ **Date:** _____________

### Part 1: Correlation Understanding

**1. What does a correlation of +0.79 mean?**

_____________________________________________________________________

_____________________________________________________________________

**2. Which is a stronger relationship?**
   - Correlation of 0.45
   - Correlation of 0.78
   
Circle one and explain why:

_____________________________________________________________________

**3. If a feature has correlation of -0.30 with price, what does that mean?**

_____________________________________________________________________

_____________________________________________________________________

---

### Part 2: Feature Analysis

**Based on today's visualizations, answer:**

**1. What was the #1 most correlated feature with SalePrice?**

Feature: _______________________ Correlation: _______

**2. What was the #2 most correlated feature?**

Feature: _______________________ Correlation: _______

**3. Did your Day 1 prediction match the actual top features? Explain:**

_____________________________________________________________________

_____________________________________________________________________

_____________________________________________________________________

---

### Part 3: Visualization Interpretation

**Look at the GrLivArea vs SalePrice scatter plot you created:**

**1. Describe the pattern you see:**

_____________________________________________________________________

_____________________________________________________________________

**2. Are there any outliers (dots far from the trend)? Describe them:**

_____________________________________________________________________

_____________________________________________________________________

**3. If a house has 3,000 sq ft of living area, approximately what price would you predict? (Estimate from your scatter plot)**

Predicted price: $______________

---

### Part 4: Reflection

**1. What surprised you most about today's data exploration?**

_____________________________________________________________________

_____________________________________________________________________

**2. Which visualization did you find most helpful? Why?**

_____________________________________________________________________

_____________________________________________________________________

**3. Based on the correlations, which feature do you NOW think is most important?**

Feature: _______________________

Has your opinion changed from Day 1? _______ Why or why not?

_____________________________________________________________________

_____________________________________________________________________

---

### Bonus Challenge (Extra Credit)

**Create your own scatter plot for a feature we didn't explore in class:**

**Feature chosen:** _______________________

**Code used:**
```python
# Write your code here


```

**Correlation found:** _______

**What pattern did you discover?**

_____________________________________________________________________

_____________________________________________________________________

---

## STUDENT CODE REFERENCE SHEET - DAY 2

### Regression Visualization Commands

**Scatter Plot (Continuous vs Continuous):**
```python
plt.scatter(train['Feature'], train['SalePrice'], alpha=0.5)
plt.xlabel('Feature Name')
plt.ylabel('Sale Price')
plt.show()
```

**Box Plot (Categorical vs Continuous):**
```python
train.boxplot(column='SalePrice', by='CategoryFeature')
plt.show()
```

**Calculate Correlation:**
```python
correlation = train['Feature'].corr(train['SalePrice'])
print(f"Correlation: {correlation:.3f}")
```

**Find Top Correlations:**
```python