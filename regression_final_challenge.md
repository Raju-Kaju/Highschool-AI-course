# Programming Challenge: Student Exam Score Prediction
## Regression Machine Learning Challenge

---

## Challenge Overview

**Scenario:** You are a data scientist hired by a high school to build a model that predicts final exam scores based on student study habits, attendance, and other factors. This will help identify students who might need extra support.

**Your Task:** Train a regression model to predict exam scores (0-100) and evaluate its performance.

**Difficulty Level:** Beginner to Intermediate  
**Estimated Time:** 45-60 minutes  
**Skills Tested:** Data exploration, preprocessing, feature engineering, model training, evaluation

---

##  Learning Objectives

By completing this challenge, you will demonstrate:
1. Loading and exploring regression data
2. Handling missing values appropriately
3. Creating visualizations to understand relationships
4. Engineering useful features
5. Training a Linear Regression model
6. Evaluating model performance using RMSE and R²
7. Making predictions on new data

---

##  Dataset Description

### Training Data (200 students)

**Features:**
- `student_id`: Unique identifier (1-200)
- `study_hours_per_week`: Hours spent studying per week (0-40)
- `attendance_rate`: Percentage of classes attended (0-100)
- `previous_exam_score`: Score on previous exam (0-100)
- `homework_completion_rate`: Percentage of homework completed (0-100)
- `hours_of_sleep`: Average hours of sleep per night (4-10)
- `extracurricular_hours`: Hours per week in extracurricular activities (0-20)
- `tutoring_sessions`: Number of tutoring sessions attended (0-10)
- `parent_involvement`: Parent involvement level (Low=0, Medium=1, High=2)
- `has_study_group`: Whether student is in a study group (0=No, 1=Yes)

**Target Variable:**
- `final_exam_score`: Final exam score to predict (0-100)

### Test Data (50 students)
Same features as training data, but **WITHOUT** `final_exam_score` - you need to predict these!

---

##  Sample Dataset

### Create the Dataset (Run this code first)

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate training data (200 students)
n_train = 200

train_data = {
    'student_id': range(1, n_train + 1),
    'study_hours_per_week': np.random.uniform(5, 35, n_train),
    'attendance_rate': np.random.uniform(60, 100, n_train),
    'previous_exam_score': np.random.uniform(40, 95, n_train),
    'homework_completion_rate': np.random.uniform(50, 100, n_train),
    'hours_of_sleep': np.random.uniform(5, 9, n_train),
    'extracurricular_hours': np.random.uniform(0, 15, n_train),
    'tutoring_sessions': np.random.randint(0, 11, n_train),
    'parent_involvement': np.random.choice([0, 1, 2], n_train, p=[0.3, 0.4, 0.3]),
    'has_study_group': np.random.choice([0, 1], n_train, p=[0.4, 0.6])
}

# Create realistic exam scores based on features
# Formula: weighted combination of features + some randomness
train_data['final_exam_score'] = (
    0.3 * train_data['study_hours_per_week'] +
    0.2 * train_data['attendance_rate'] / 100 * 100 +
    0.25 * train_data['previous_exam_score'] +
    0.15 * train_data['homework_completion_rate'] / 100 * 100 +
    2 * train_data['hours_of_sleep'] +
    0.5 * train_data['tutoring_sessions'] +
    3 * train_data['parent_involvement'] +
    4 * train_data['has_study_group'] +
    np.random.normal(0, 5, n_train)  # Add noise
)

# Clip scores to 0-100 range
train_data['final_exam_score'] = np.clip(train_data['final_exam_score'], 0, 100)

# Add some missing values randomly (realistic scenario)
train_df = pd.DataFrame(train_data)
missing_indices_sleep = np.random.choice(train_df.index, 15, replace=False)
missing_indices_tutoring = np.random.choice(train_df.index, 10, replace=False)
train_df.loc[missing_indices_sleep, 'hours_of_sleep'] = np.nan
train_df.loc[missing_indices_tutoring, 'tutoring_sessions'] = np.nan

# Generate test data (50 students) - same process but no final_exam_score
n_test = 50

test_data = {
    'student_id': range(201, 201 + n_test),
    'study_hours_per_week': np.random.uniform(5, 35, n_test),
    'attendance_rate': np.random.uniform(60, 100, n_test),
    'previous_exam_score': np.random.uniform(40, 95, n_test),
    'homework_completion_rate': np.random.uniform(50, 100, n_test),
    'hours_of_sleep': np.random.uniform(5, 9, n_test),
    'extracurricular_hours': np.random.uniform(0, 15, n_test),
    'tutoring_sessions': np.random.randint(0, 11, n_test),
    'parent_involvement': np.random.choice([0, 1, 2], n_test, p=[0.3, 0.4, 0.3]),
    'has_study_group': np.random.choice([0, 1], n_test, p=[0.4, 0.6])
}

test_df = pd.DataFrame(test_data)

# Add missing values to test set too
missing_test_sleep = np.random.choice(test_df.index, 3, replace=False)
test_df.loc[missing_test_sleep, 'hours_of_sleep'] = np.nan

print("✓ Datasets created!")
print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")

# Save to CSV (optional)
train_df.to_csv('student_scores_train.csv', index=False)
test_df.to_csv('student_scores_test.csv', index=False)

print("\n✓ CSV files saved!")
```

---

##  Challenge Tasks

### **Part 1: Data Exploration (15 points)**

1. Load the training data and display the first 10 rows
2. Check the shape of the dataset
3. Display summary statistics using `.describe()`
4. Check for missing values and report how many in each column
5. Calculate the correlation between each feature and `final_exam_score`
6. Create a scatter plot showing `study_hours_per_week` vs `final_exam_score`

**Deliverable:** Print statements showing your findings

---

### **Part 2: Data Preprocessing (20 points)**

1. Handle missing values:
   - Fill missing `hours_of_sleep` with the median
   - Fill missing `tutoring_sessions` with 0 (assume they didn't attend)

2. Create at least 2 new features through feature engineering:
   - **Suggestion 1:** `total_effort` = study_hours + homework_completion_rate/10
   - **Suggestion 2:** `academic_preparation` = previous_exam_score * attendance_rate/100
   - **Be creative!** Create your own features

3. Verify that you have no missing values remaining

**Deliverable:** Clean dataset with new features, verification printout

---

### **Part 3: Visualization (15 points)**

Create the following visualizations:

1. **Correlation heatmap** showing top 8 features most correlated with final_exam_score
2. **Box plot** comparing exam scores by parent_involvement level (0, 1, 2)
3. **Scatter plot** with trend line for your strongest correlated feature

**Deliverable:** Three clear, labeled visualizations

---

### **Part 4: Model Training (25 points)**

1. Split your data into training (80%) and validation (20%) sets using train_test_split
2. Create a Linear Regression model
3. Train the model on your training set
4. Make predictions on your validation set
5. Calculate and print:
   - **RMSE (Root Mean Squared Error)**
   - **MAE (Mean Absolute Error)**
   - **R² Score**

**Success Criteria:**
- R² score > 0.70 (Good)
- R² score > 0.80 (Excellent)
- RMSE < 10 points (Good)
- RMSE < 7 points (Excellent)

**Deliverable:** Trained model with performance metrics printed

---

### **Part 5: Model Evaluation & Interpretation (15 points)**

1. Create a scatter plot: Actual vs Predicted scores
2. Calculate and display feature importance (model coefficients)
3. Answer these questions in comments:
   - Which feature is most important for predicting exam scores?
   - Is your model overfitting or underfitting? How can you tell?
   - What could you do to improve the model?

**Deliverable:** Visualization and written analysis in code comments

---

### **Part 6: Make Predictions on Test Data (10 points)**

1. Load the test data (student_scores_test.csv)
2. Apply the SAME preprocessing steps you used on training data
3. Make predictions for all 50 test students
4. Create a submission dataframe with columns: `student_id`, `predicted_score`
5. Save to CSV: `predictions.csv`

**Deliverable:** CSV file with predictions for 50 students

---

##  Scoring Rubric

| Task | Points | Criteria |
|------|--------|----------|
| Part 1: Exploration | 15 | All 6 tasks completed correctly |
| Part 2: Preprocessing | 20 | Missing values handled, 2+ features created |
| Part 3: Visualization | 15 | 3 clear, labeled visualizations |
| Part 4: Model Training | 25 | Model trained correctly, metrics calculated |
| Part 5: Evaluation | 15 | Scatter plot + feature importance + analysis |
| Part 6: Predictions | 10 | Test predictions made and saved |
| **Total** | **100** | |

**Bonus Points (+10):** Achieve R² > 0.85 OR create 3+ meaningful engineered features

---

##  Starter Code Template

```python
# ============================================
# STUDENT EXAM SCORE PREDICTION CHALLENGE
# Name: _________________
# Date: _________________
# ============================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# PART 1: DATA EXPLORATION
# ============================================

# TODO: Load training data
train = pd.read_csv('student_scores_train.csv')

# TODO: Display first 10 rows
print("First 10 students:")
# YOUR CODE HERE

# TODO: Check shape
print(f"\nDataset shape: ...")
# YOUR CODE HERE

# TODO: Summary statistics
print("\nSummary statistics:")
# YOUR CODE HERE

# TODO: Check missing values
print("\nMissing values:")
# YOUR CODE HERE

# TODO: Calculate correlations with final_exam_score
print("\nCorrelations with final exam score:")
# YOUR CODE HERE

# TODO: Create scatter plot
# YOUR CODE HERE

# ============================================
# PART 2: DATA PREPROCESSING
# ============================================

# TODO: Handle missing values
# YOUR CODE HERE

# TODO: Create new features
# Example: train['total_effort'] = ...
# YOUR CODE HERE

# TODO: Verify no missing values
# YOUR CODE HERE

# ============================================
# PART 3: VISUALIZATION
# ============================================

# TODO: Correlation heatmap
# YOUR CODE HERE

# TODO: Box plot by parent involvement
# YOUR CODE HERE

# TODO: Scatter plot with trend line
# YOUR CODE HERE

# ============================================
# PART 4: MODEL TRAINING
# ============================================

# TODO: Prepare features and target
# X = train[feature_columns]
# y = train['final_exam_score']
# YOUR CODE HERE

# TODO: Train/test split
# YOUR CODE HERE

# TODO: Create and train model
# YOUR CODE HERE

# TODO: Make predictions
# YOUR CODE HERE

# TODO: Calculate metrics
# YOUR CODE HERE

print("\nModel Performance:")
print(f"RMSE: ...")
print(f"MAE: ...")
print(f"R² Score: ...")

# ============================================
# PART 5: MODEL EVALUATION
# ============================================

# TODO: Actual vs Predicted plot
# YOUR CODE HERE

# TODO: Feature importance
# YOUR CODE HERE

# TODO: Answer analysis questions in comments
"""
Q1: Which feature is most important?
A: [YOUR ANSWER]

Q2: Is the model overfitting or underfitting?
A: [YOUR ANSWER]

Q3: How could you improve the model?
A: [YOUR ANSWER]
"""

# ============================================
# PART 6: TEST PREDICTIONS
# ============================================

# TODO: Load test data
# YOUR CODE HERE

# TODO: Apply same preprocessing
# YOUR CODE HERE

# TODO: Make predictions
# YOUR CODE HERE

# TODO: Create submission file
# YOUR CODE HERE

print("\n Challenge complete!")
```

---

## 💡 Hints & Tips

### For Data Preprocessing:
- Use `.fillna()` for missing values
- Use `.median()` for numeric features
- Remember: test data needs SAME preprocessing as training data!

### For Feature Engineering Ideas:
- Combine related features (study + homework)
- Create ratios (attendance / homework completion)
- Create interactions (study_hours * parent_involvement)
- Create binary flags (is_high_achiever: previous_score > 80)

### For Evaluation:
- **RMSE** tells you average error in same units as target (points)
- **R²** tells you % of variance explained (0 to 1, higher is better)
- **MAE** is easier to interpret than RMSE (average absolute error)

### Common Mistakes to Avoid:
- ❌ Not applying same preprocessing to test data
- ❌ Including student_id in features
- ❌ Forgetting to handle missing values
- ❌ Not splitting data before training
- ❌ Using final_exam_score as a feature (data leakage!)

---

##  Solution Validation

Run this code to check if your predictions are reasonable:

```python
# Validate your predictions
predictions = pd.read_csv('predictions.csv')

print("Prediction Validation:")
print(f"Number of predictions: {len(predictions)}")
print(f"Min predicted score: {predictions['predicted_score'].min():.1f}")
print(f"Max predicted score: {predictions['predicted_score'].max():.1f}")
print(f"Mean predicted score: {predictions['predicted_score'].mean():.1f}")

# Check if predictions are in valid range
if predictions['predicted_score'].min() >= 0 and predictions['predicted_score'].max() <= 100:
    print(" Predictions are in valid range (0-100)")
else:
    print("  Warning: Some predictions are outside valid range!")

# Check if you have all 50 predictions
if len(predictions) == 50:
    print(" Correct number of predictions (50)")
else:
    print(f" Warning: Expected 50 predictions, got {len(predictions)}")
```

---
