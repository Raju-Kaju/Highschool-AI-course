# Day 3: Data Preprocessing & Feature Engineering
## Detailed Teacher Guide with Script & Complete Code

---

## Lesson Overview
**Duration:** 50 minutes  
**Objective:** Students clean messy data and prepare features for machine learning  
**Key Outcome:** Students transform raw data into model-ready format

---

## Materials Checklist
- [ ] Projector for live coding demonstration
- [ ] Student computers with notebooks from Day 2
- [ ] Printed "Data Cleaning Cheat Sheet" (included below)
- [ ] Completed Day 2 homework
- [ ] Visual aids: "Missing Data Decision Tree" poster
- [ ] Backup: Pre-cleaned dataset if students fall behind

---

## Big Ideas for Day 3
1. Real data is messy - missing values, inconsistent formats
2. Computers need numbers, not text
3. Feature engineering = creating new useful information from existing data
4. Garbage in = garbage out (clean data = better predictions)

---

## MINUTE-BY-MINUTE LESSON PLAN

### Minutes 0-5: Warm-Up & Homework Review

**TEACHER SCRIPT:**

"Good morning! Quick show of hands:
- Who found that SEX was most important? [count hands]
- Who found CLASS was most important? [count hands]
- Who was surprised by something in the data? [pick 2-3 to share]

[CALL ON STUDENT WHO RAISED HAND]

Great! Today we're going to solve a problem. Remember all those missing ages from yesterday? And how our computer can't understand the word 'male' - it only understands numbers?

Today we become DATA JANITORS. Not the most glamorous job, but absolutely essential. About 80% of a data scientist's time is spent cleaning data, not building fancy models.

By the end of today, our messy data will be squeaky clean and ready for machine learning!"

[WRITE ON BOARD:]
**Today's Goal: Transform messy data → clean, model-ready numbers**

---

### Minutes 5-12: The Missing Data Problem

**TEACHER SCRIPT:**

"Open your notebook from yesterday. Let's look at our missing data again."

[PROJECT YOUR SCREEN - RUN THIS CODE:]

```python
# Reload our data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv('train.csv')

# Check missing data
print("Missing Values Count:")
print(train.isnull().sum())
print("\n")

# Percentage missing
print("Percentage Missing:")
missing_pct = (train.isnull().sum() / len(train)) * 100
print(missing_pct[missing_pct > 0])
```

**TEACHER SCRIPT:**

"Look at this:
- **Age**: 177 missing (20% of our data!)
- **Cabin**: 687 missing (77% of our data!)
- **Embarked**: 2 missing (basically nothing)

Now, let me ask you: What should we do about missing ages?

Remember your homework - you had options:
A) Delete all passengers with missing ages
B) Fill with average age
C) Fill with zero
D) Other idea

Who chose A - delete them? [hands] 
Who chose B - fill with average? [hands]

Let me show you why each matters..."

[DRAW ON BOARD - Simple visualization:]

```
OPTION A (Delete):
891 passengers → 714 passengers
PROBLEM: We lose 20% of our data! Less data = worse learning

OPTION B (Fill with average):
Missing age = 29.7 (average)
PROBLEM: Not perfect, but we keep all data

OPTION C (Fill with zero):
Missing age = 0 (baby)
PROBLEM: Totally wrong! Makes no sense

OPTION D (Smart fill):
Fill based on other features (advanced - maybe later!)
```

**TEACHER SCRIPT:**

"In real data science, there's no perfect answer. But here's the rule:

**If more than 50% is missing → Drop the column (like Cabin)**
**If less than 50% is missing → Fill it intelligently**

For Age, we'll fill with the **median** (middle value) because it's less affected by extreme ages than the average.

Let's do it! Everyone type this code:"

```python
# Before filling
print(f"Missing ages before: {train['Age'].isnull().sum()}")

# Fill missing ages with median
train['Age'].fillna(train['Age'].median(), inplace=True)

# After filling
print(f"Missing ages after: {train['Age'].isnull().sum()}")

print(f"\nMedian age we used: {train['Age'].median():.1f} years")
```

**TEACHER SCRIPT:**

"See? 177 missing ages → 0 missing ages! We filled them all with 28 (the median).

Is this perfect? No. But it's reasonable and lets us keep all our passengers.

Now you try - type and run this code!"

[GIVE 2 MINUTES - CIRCULATE]

---

### Minutes 12-20: Handling Embarked & Dropping Cabin

**TEACHER SCRIPT:**

"Now let's handle Embarked - remember, only 2 passengers are missing this.

For just 2 missing values, we'll use the **mode** - the most common value. It's like saying 'most people got on at Southampton, so these 2 probably did too.'"

```python
# Check Embarked values
print("Embarkation ports:")
print(train['Embarked'].value_counts())

print(f"\nMost common port: {train['Embarked'].mode()[0]}")

# Fill missing Embarked with most common
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Verify
print(f"\nMissing Embarked after: {train['Embarked'].isnull().sum()}")
```

**TEACHER SCRIPT:**

"Perfect! Now for Cabin - this is 77% missing. We can't fill 77% - that would be mostly made-up data!

So we're going to DROP this column entirely. Watch:"

```python
# Drop Cabin column
train = train.drop('Cabin', axis=1)

# Verify it's gone
print("Remaining columns:")
print(train.columns.tolist())
```

**TEACHER SCRIPT:**

"See? Cabin is gone! That's okay - we still have plenty of good features.

**Quick check:** Let's verify we have NO missing data in our important columns:"

```python
# Final missing data check
print("Missing data in features we'll use:")
important_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
print(train[important_cols].isnull().sum())
```

**TEACHER SCRIPT:**

"All zeros! Our data is clean!

Everyone run this code - make sure you have no missing values in these columns."

[GIVE 2-3 MINUTES]

---

### Minutes 20-32: Converting Text to Numbers

**TEACHER SCRIPT:**

"Alright, here's a fundamental truth about machine learning:

**COMPUTERS ONLY UNDERSTAND NUMBERS.**

Right now, we have text:
- Sex: 'male' or 'female'
- Embarked: 'S', 'C', or 'Q'

The computer looks at 'male' and goes: 🤷 'I don't know what this means!'

We need to convert text → numbers. This is called **encoding**.

Let me show you the simplest way - for Sex, we'll map:
- female → 0
- male → 1

Watch this magic:"

```python
# BEFORE conversion
print("Sex column BEFORE:")
print(train['Sex'].head(10))

# Convert Sex to numbers
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})

# AFTER conversion
print("\nSex column AFTER:")
print(train['Sex'].head(10))

# Verify
print(f"\nUnique values in Sex: {train['Sex'].unique()}")
```

**TEACHER SCRIPT:**

"See what happened? 
- 'male' became 1
- 'female' became 0

Now the computer can work with it! The numbers don't mean 'better' or 'worse' - they're just labels.

Everyone type and run this!"

[GIVE 2 MINUTES]

---

**TEACHER SCRIPT:**

"Now for Embarked - this is trickier because we have 3 values: S, C, and Q.

We CAN'T just do: S=1, C=2, Q=3 because the computer would think Q is '3 times more' than S. That's wrong!

Instead, we use **one-hot encoding**. We create separate columns:
- Embarked_S → 1 if Southampton, 0 if not
- Embarked_C → 1 if Cherbourg, 0 if not  
- Embarked_Q → 1 if Queenstown, 0 if not

Let me show you:"

```python
# BEFORE one-hot encoding
print("Embarked BEFORE:")
print(train['Embarked'].head())

# Create dummy variables (one-hot encoding)
embarked_dummies = pd.get_dummies(train['Embarked'], prefix='Embarked')

print("\nNew columns created:")
print(embarked_dummies.head())

# Add these new columns to our data
train = pd.concat([train, embarked_dummies], axis=1)

# Drop original Embarked column
train = train.drop('Embarked', axis=1)

print("\nColumns now include:")
print([col for col in train.columns if 'Embarked' in col])
```

**TEACHER SCRIPT:**

"Watch what happened:
- If someone embarked at Southampton → Embarked_S = 1, others = 0
- If someone embarked at Cherbourg → Embarked_C = 1, others = 0

This way, the computer doesn't think one port is 'bigger' than another - they're just different!

This is called 'one-hot encoding' - like turning on ONE light switch.

Type this code carefully - make sure you get the exact spelling!"

[GIVE 3-4 MINUTES - THIS IS COMPLEX, CIRCULATE HEAVILY]

---

### Minutes 32-40: Feature Engineering - Creating New Features

**TEACHER SCRIPT:**

"Alright, now for the cool part - FEATURE ENGINEERING!

This is where we become creative. We create NEW information from existing data.

Think about it: If I told you someone traveled with 2 siblings and 3 parents, what could we learn?

[PAUSE FOR STUDENT RESPONSES]

Right! Total family size = 2 + 3 + 1 (themselves) = 6 people!

Maybe family size matters for survival:
- Traveling alone → might have disadvantage
- Big family → might help each other
- Very big family → might be hard to keep together

Let's create this feature:"

```python
# Create FamilySize feature
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

# Look at results
print("FamilySize examples:")
print(train[['SibSp', 'Parch', 'FamilySize']].head(10))

# Check survival by family size
print("\nSurvival rate by family size:")
print(train.groupby('FamilySize')['Survived'].mean())
```

**TEACHER SCRIPT:**

"Interesting! Look at the survival rates:
- FamilySize 1 (alone): about 30% survived
- FamilySize 2-4: about 50-70% survived (best!)
- FamilySize 5+: drops down again

So having some family helped, but huge families had trouble!

We just created a NEW piece of information that might help our model!

Everyone create this feature now."

[GIVE 2 MINUTES]

---

**TEACHER SCRIPT:**

"Let's create one more - IsAlone. This is a simple yes/no:
- If FamilySize = 1 → IsAlone = 1 (yes)
- If FamilySize > 1 → IsAlone = 0 (no)

Watch:"

```python
# Create IsAlone feature
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)

# Check it
print("IsAlone examples:")
print(train[['FamilySize', 'IsAlone']].head(10))

# Survival rate
print(f"\nSurvival rate when alone: {train[train['IsAlone']==1]['Survived'].mean():.1%}")
print(f"Survival rate with family: {train[train['IsAlone']==0]['Survived'].mean():.1%}")
```

**TEACHER SCRIPT:**

"See? People traveling alone had only 30% survival, while people with family had 50%!

That's feature engineering - we created useful information from what we already had.

Type this code!"

[GIVE 2 MINUTES]

---

### Minutes 40-45: Selecting Final Features for the Model

**TEACHER SCRIPT:**

"Okay, we've done a LOT of work. Let's see what we have now:"

```python
# Show all columns
print("All columns we have:")
print(train.columns.tolist())

print("\n" + "="*50)
print("FINAL FEATURE SELECTION")
print("="*50)

# Select features for our model
feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 
                   'Fare', 'FamilySize', 'IsAlone',
                   'Embarked_C', 'Embarked_Q', 'Embarked_S']

# Create X (features) and y (target)
X = train[feature_columns]
y = train['Survived']

print(f"\nFeatures we're using: {len(feature_columns)}")
print(feature_columns)

print(f"\nDataset shape: {X.shape}")
print(f"That's {X.shape[0]} passengers and {X.shape[1]} features")

# Check for any missing data
print(f"\nAny missing data? {X.isnull().sum().sum()} (should be 0!)")

# Look at first few rows
print("\nFirst 5 passengers (model-ready data):")
print(X.head())
```

**TEACHER SCRIPT:**

"Perfect! We now have:
- **891 passengers** (rows)
- **11 features** (columns)
- **0 missing values**
- **All numbers** (no text!)

This data is READY for machine learning!

Compare this to Day 2:
- We HAD: messy data with missing values and text
- We NOW HAVE: clean, numerical data

This is what data preprocessing means - preparing raw data for the model.

Everyone run this code and verify you have 0 missing values!"

[GIVE 2-3 MINUTES]

---

### Minutes 45-48: Visualizing Our Clean Data

**TEACHER SCRIPT:**

"Let's make ONE visualization with our new features to see if they're useful:"

```python
# Visualize survival by our new features
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Family Size
train.groupby('FamilySize')['Survived'].mean().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Survival Rate by Family Size', fontweight='bold')
axes[0].set_ylabel('Survival Rate')
axes[0].set_xlabel('Family Size')
axes[0].set_ylim(0, 1)

# Plot 2: Alone vs Not Alone
train.groupby('IsAlone')['Survived'].mean().plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
axes[1].set_title('Survival Rate: Alone vs With Family', fontweight='bold')
axes[1].set_ylabel('Survival Rate')
axes[1].set_xlabel('Is Alone (0=No, 1=Yes)')
axes[1].set_xticklabels(['With Family', 'Alone'], rotation=0)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()
```

**TEACHER SCRIPT:**

"Beautiful! Our new features show clear patterns - this is good news! It means they'll help our model make better predictions.

Run this code to see your visualizations!"

[GIVE 2 MINUTES]

---

### Minutes 48-50: CLOSURE & Summary

**TEACHER SCRIPT:**

"Excellent work today! Let's review what we accomplished.

[WRITE ON BOARD:]

**DATA PREPROCESSING CHECKLIST ✓**
1. ✓ Filled missing Ages with median (28 years)
2. ✓ Filled missing Embarked with mode (S)
3. ✓ Dropped Cabin (too much missing)
4. ✓ Converted Sex to numbers (0 and 1)
5. ✓ One-hot encoded Embarked (3 new columns)
6. ✓ Created FamilySize feature
7. ✓ Created IsAlone feature
8. ✓ Selected final 11 features

**RESULT:** Clean, numerical, model-ready data!

[TURN TO CLASS]

This seems like a lot of work, and it is! But here's the thing - garbage in, garbage out. If we feed messy data to our model, we get bad predictions.

Clean data = Good predictions.

Tomorrow is THE BIG DAY - we build our machine learning model! We'll take this clean data and teach the computer to predict survival.

**Homework:**
1. Save your notebook - we NEED this for tomorrow!
2. Answer the reflection questions on the handout
3. Think: What other features could we create? (Extra credit!)

**Tomorrow preview:** Decision Trees, Training, and our FIRST predictions!

Any questions?"

[ANSWER QUESTIONS]

"Great work today, data scientists! See you tomorrow!"

---

## STUDENT CODE REFERENCE SHEET - DAY 3

### Data Cleaning Quick Reference

**Fill Missing Values:**
```python
# Fill with median (for numbers)
df['Column'].fillna(df['Column'].median(), inplace=True)

# Fill with mode (for categories)
df['Column'].fillna(df['Column'].mode()[0], inplace=True)

# Fill with specific value
df['Column'].fillna(0, inplace=True)
```

**Drop Columns:**
```python
df = df.drop('ColumnName', axis=1)
```

**Convert Text to Numbers:**
```python
# Simple mapping
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# One-hot encoding
dummies = pd.get_dummies(df['Column'], prefix='Column')
df = pd.concat([df, dummies], axis=1)
```

**Create New Features:**
```python
# Mathematical operations
df['NewFeature'] = df['Col1'] + df['Col2']

# Conditional features
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
```

**Check for Issues:**
```python
df.isnull().sum()          # Missing values
df.dtypes                  # Data types
df['Column'].unique()      # Unique values
```

---

## COMPLETE WORKING CODE - DAY 3

```python
# ============================================
# TITANIC DATA PREPROCESSING - DAY 3
# Student Name: _______________
# Date: _______________
# ============================================

# Cell 1: Import and Load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline

# Load data
train = pd.read_csv('train.csv')

print("Data loaded successfully!")
print(f"Shape: {train.shape}")

# ============================================

# Cell 2: Examine Missing Data
print("MISSING DATA ANALYSIS")
print("="*50)

print("\nMissing value counts:")
print(train.isnull().sum())

print("\nPercentage missing:")
missing_pct = (train.isnull().sum() / len(train)) * 100
print(missing_pct[missing_pct > 0])

# ============================================

# Cell 3: Fill Missing Ages
print("FILLING MISSING AGES")
print("="*50)

print(f"Missing ages before: {train['Age'].isnull().sum()}")
print(f"Median age: {train['Age'].median():.1f}")

# Fill missing ages with median
train['Age'].fillna(train['Age'].median(), inplace=True)

print(f"Missing ages after: {train['Age'].isnull().sum()}")
print("✓ Ages cleaned!")

# ============================================

# Cell 4: Fill Missing Embarked
print("\nFILLING MISSING EMBARKED")
print("="*50)

print("Embarkation port counts:")
print(train['Embarked'].value_counts())

print(f"\nMost common: {train['Embarked'].mode()[0]}")

# Fill missing Embarked
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

print(f"Missing Embarked after: {train['Embarked'].isnull().sum()}")
print("✓ Embarked cleaned!")

# ============================================

# Cell 5: Drop Cabin
print("\nDROPPING CABIN COLUMN")
print("="*50)

cabin_missing = train['Cabin'].isnull().sum()
cabin_pct = (cabin_missing / len(train)) * 100

print(f"Cabin missing: {cabin_missing} ({cabin_pct:.1f}%)")
print("Too much missing - dropping this column")

# Drop Cabin
train = train.drop('Cabin', axis=1)

print("✓ Cabin dropped!")

# ============================================

# Cell 6: Convert Sex to Numbers
print("\nCONVERTING SEX TO NUMBERS")
print("="*50)

print("BEFORE conversion:")
print(train['Sex'].head())

# Map female=0, male=1
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})

print("\nAFTER conversion:")
print(train['Sex'].head())
print(f"\nUnique values: {train['Sex'].unique()}")
print("✓ Sex converted!")

# ============================================

# Cell 7: One-Hot Encode Embarked
print("\nONE-HOT ENCODING EMBARKED")
print("="*50)

print("Creating dummy variables...")

# Create dummy variables
embarked_dummies = pd.get_dummies(train['Embarked'], prefix='Embarked')

print("\nNew columns created:")
print(embarked_dummies.head())

# Add to dataframe
train = pd.concat([train, embarked_dummies], axis=1)

# Drop original
train = train.drop('Embarked', axis=1)

print("\n✓ Embarked one-hot encoded!")

# ============================================

# Cell 8: Feature Engineering - FamilySize
print("\nCREATING FAMILYSIZE FEATURE")
print("="*50)

# Create FamilySize
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

print("Examples:")
print(train[['SibSp', 'Parch', 'FamilySize']].head(10))

print("\nSurvival rate by FamilySize:")
survival_by_family = train.groupby('FamilySize')['Survived'].mean()
print(survival_by_family)

print("\n✓ FamilySize created!")

# ============================================

# Cell 9: Feature Engineering - IsAlone
print("\nCREATING ISALONE FEATURE")
print("="*50)

# Create IsAlone
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)

print("Examples:")
print(train[['FamilySize', 'IsAlone']].head(10))

print("\nSurvival rates:")
print(f"Alone: {train[train['IsAlone']==1]['Survived'].mean():.1%}")
print(f"With family: {train[train['IsAlone']==0]['Survived'].mean():.1%}")

print("\n✓ IsAlone created!")

# ============================================

# Cell 10: Select Final Features
print("\nFINAL FEATURE SELECTION")
print("="*50)

# Define feature columns
feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 
                   'Fare', 'FamilySize', 'IsAlone',
                   'Embarked_C', 'Embarked_Q', 'Embarked_S']

# Create X (features) and y (target)
X = train[feature_columns]
y = train['Survived']

print(f"Features selected: {len(feature_columns)}")
print(feature_columns)

print(f"\nDataset shape: {X.shape}")
print(f"({X.shape[0]} passengers × {X.shape[1]} features)")

# Check for missing data
total_missing = X.isnull().sum().sum()
print(f"\nTotal missing values: {total_missing}")

if total_missing == 0:
    print("✓ NO MISSING DATA - READY FOR ML!")
else:
    print("⚠ WARNING: Still have missing data!")
    print(X.isnull().sum())

# ============================================

# Cell 11: Preview Clean Data
print("\nCLEAN DATA PREVIEW")
print("="*50)

print("\nFirst 5 passengers (model-ready):")
print(X.head())

print("\nData types:")
print(X.dtypes)

print("\nStatistical summary:")
print(X.describe())

# ============================================

# Cell 12: Visualize New Features
print("\nVISUALIZING NEW FEATURES")
print("="*50)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Survival by Family Size
survival_family = train.groupby('FamilySize')['Survived'].mean()
survival_family.plot(kind='bar', ax=axes[0,0], color='skyblue', edgecolor='black')
axes[0,0].set_title('Survival Rate by Family Size', fontweight='bold', fontsize=12)
axes[0,0].set_ylabel('Survival Rate')
axes[0,0].set_xlabel('Family Size')
axes[0,0].set_ylim(0, 1)
axes[0,0].axhline(y=train['Survived'].mean(), color='red', linestyle='--', label='Overall avg')
axes[0,0].legend()

# Plot 2: Alone vs Not Alone
survival_alone = train.groupby('IsAlone')['Survived'].mean()
survival_alone.plot(kind='bar', ax=axes[0,1], color='coral', edgecolor='black')
axes[0,1].set_title('Survival Rate: Alone vs With Family', fontweight='bold', fontsize=12)
axes[0,1].set_ylabel('Survival Rate')
axes[0,1].set_xlabel('Traveling Status')
axes[0,1].set_xticklabels(['With Family', 'Alone'], rotation=0)
axes[0,1].set_ylim(0, 1)

# Plot 3: Age Distribution (cleaned)
axes[1,0].hist(train['Age'], bins=30, edgecolor='black', color='lightgreen')
axes[1,0].set_title('Age Distribution (After Cleaning)', fontweight='bold', fontsize=12)
axes[1,0].set_xlabel('Age (years)')
axes[1,0].set_ylabel('Count')
axes[1,0].axvline(train['Age'].median(), color='red', linestyle='--', 
                  linewidth=2, label=f"Median: {train['Age'].median():.1f}")
axes[1,0].legend()

# Plot 4: Feature Correlation with Survival
correlations = X.corrwith(y).sort_values(ascending=False)
correlations.plot(kind='barh', ax=axes[1,1], color='purple', edgecolor='black')
axes[1,1].set_title('Feature Correlation with Survival', fontweight='bold', fontsize=12)
axes[1,1].set_xlabel('Correlation Coefficient')
axes[1,1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

print("\n✓ Visualizations complete!")

# ============================================

# Cell 13: Summary Report
print("\n" + "="*50)
print("DATA PREPROCESSING SUMMARY")
print("="*50)

print("\n✓ CLEANING STEPS COMPLETED:")
print("  1. Filled 177 missing Ages with median (28.0)")
print("  2. Filled 2 missing Embarked with mode ('S')")
print("  3. Dropped Cabin column (77% missing)")
print("  4. Converted Sex to numeric (0/1)")
print("  5. One-hot encoded Embarked (3 columns)")

print("\n✓ FEATURE ENGINEERING:")
print("  6. Created FamilySize (SibSp + Parch + 1)")
print("  7. Created IsAlone (binary: 1=alone, 0=family)")

print("\n✓ FINAL DATASET:")
print(f"  - Passengers: {len(X)}")
print(f"  - Features: {len(feature_columns)}")
print(f"  - Missing values: {X.isnull().sum().sum()}")
print(f"  - All numeric: {X.dtypes.apply(lambda x: x in ['int64', 'float64']).all()}")

print("\n✓ READY FOR MACHINE LEARNING!")

print("\n" + "="*50)
print("Save this notebook - we'll use it tomorrow!")
print("="*50)
```

---

## HOMEWORK ASSIGNMENT - DAY 3

**Name:** _________________ **Date:** _____________

### Part 1: Reflection Questions

**1. Why did we fill missing Ages with the MEDIAN instead of the MEAN (average)?**

_____________________________________________________________________

_____________________________________________________________________

**Hint:** Think about what happens if one person is 100 years old - how does that affect mean vs median?

---

**2. Explain in your own words: What is one-hot encoding and why do we need it?**

_____________________________________________________________________

_____________________________________________________________________

_____________________________________________________________________

**Example:** Embarked had values S, C, Q. We created...

---

**3. We created a new feature called "FamilySize". Why might this be more useful than just using SibSp and Parch separately?**

_____________________________________________________________________

_____________________________________________________________________

---

**4. TRUE or FALSE (circle and explain):**

**T / F** - Deleting all rows with missing data is always the best solution.

**Explanation:**

_____________________________________________________________________

_____________________________________________________________________

---

**5. Look at this passenger's data. What would our preprocessing do?**

```
PassengerId: 900
Sex: 'female'
Age: NaN (missing)
Embarked: 'C'
SibSp: 1
Parch: 2
```

**After preprocessing:**
- Sex = _______
- Age = _______ (approximately)
- Embarked_C = _______
- Embarked_Q = _______
- Embarked_S = _______
- FamilySize = _______
- IsAlone = _______

---

### Part 2: Challenge - Create Your Own Feature! (Extra Credit)

**Can you think of a NEW feature we could create from the existing data?**

Some ideas to spark creativity:
- Title from Name (Mr., Mrs., Miss., Master)
- Fare per person (Fare / FamilySize)
- Age groups (child, adult, elderly)
- Ticket price categories (cheap, medium, expensive)

**My feature idea:**

Feature name: _______________________

What it represents: 

_____________________________________________________________________

How to create it (write code or description):

_____________________________________________________________________

_____________________________________________________________________

_____________________________________________________________________

Why it might help predict survival:

_____