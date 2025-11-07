# Titanic Survival Prediction - Complete Lesson Plan

## Problem Overview
**Kaggle Competition:** Titanic - Machine Learning from Disaster  
**Difficulty:** Beginner  
**Topic:** Binary Classification (Survived or Not)  
**Duration:** 4-6 class periods (45-60 minutes each)

---

## Learning Objectives

By the end of this lesson, students will be able to:
1. Understand what machine learning is and how it's applied to real problems
2. Load and explore datasets using Python and pandas
3. Perform basic data cleaning and preprocessing
4. Create visualizations to understand patterns in data
5. Build a simple machine learning model (Decision Tree)
6. Evaluate model performance and make predictions
7. Submit predictions to Kaggle

---

## Prerequisites

- Basic Python knowledge (variables, loops, functions)
- Understanding of basic statistics (mean, median, percentages)
- Computer with internet access
- Kaggle account (free)

---

## Materials Needed

- Computers with Python installed (or Google Colab access)
- Kaggle accounts for all students
- Titanic dataset from Kaggle
- Jupyter Notebook or Google Colab
- Projector for demonstrations

---

## Day 1: Introduction to Machine Learning & The Titanic Problem

### Hook Activity (10 minutes)
"If you were on the Titanic, what factors would increase your chances of survival?"
- Have students brainstorm in pairs
- Create a class list on the board
- Introduce: This is exactly what ML does - finds patterns!

### Direct Instruction (20 minutes)
**What is Machine Learning?**
- Definition: Teaching computers to learn from data
- Types: Supervised vs Unsupervised
- Real-world applications (recommendations, face recognition, medical diagnosis)

**The Titanic Problem**
- Historical context: April 15, 1912
- 2,224 passengers, 1,502 died
- Goal: Predict who survived based on passenger characteristics

**Dataset Features**
- Pclass: Ticket class (1st, 2nd, 3rd)
- Sex: Male or Female
- Age: Passenger age
- SibSp: Number of siblings/spouses aboard
- Parch: Number of parents/children aboard
- Fare: Ticket price
- Embarked: Port of embarkation

### Guided Practice (15 minutes)
**Setting Up Kaggle**
1. Create Kaggle accounts
2. Join the Titanic competition
3. Download the dataset
4. Review competition rules and leaderboard

### Homework
- Read about the Titanic disaster
- Think about which features might be most important for survival

---

## Day 2: Data Exploration & Visualization

### Warm-Up (5 minutes)
Quick poll: "Which feature do you think is MOST predictive of survival?"

### Direct Instruction (15 minutes)
**Introduction to pandas and data exploration**
- What is pandas?
- Loading data with `pd.read_csv()`
- Basic commands: `.head()`, `.info()`, `.describe()`
- Understanding missing data

### Guided Practice (30 minutes)
**Exploring the Titanic Dataset**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv('train.csv')

# Basic exploration
print(train.head())
print(train.info())
print(train.describe())

# Check for missing values
print(train.isnull().sum())

# Survival rate
print(f"Survival Rate: {train['Survived'].mean():.2%}")
```

**Creating Visualizations**
```python
# Survival by Sex
sns.barplot(x='Sex', y='Survived', data=train)
plt.title('Survival Rate by Sex')
plt.show()

# Survival by Class
sns.barplot(x='Pclass', y='Survived', data=train)
plt.title('Survival Rate by Class')
plt.show()

# Age distribution
train['Age'].hist(bins=30)
plt.title('Age Distribution')
plt.show()
```

### Independent Practice (10 minutes)
Students create their own visualizations:
- Survival by embarkation port
- Fare distribution
- Age by survival status

### Wrap-Up (5 minutes)
Discussion: What patterns did we discover?

---

## Day 3: Data Preprocessing & Feature Engineering

### Review (5 minutes)
Quick recap of yesterday's discoveries

### Direct Instruction (20 minutes)
**Why Preprocessing Matters**
- Missing data problems
- Categorical vs numerical data
- Feature scaling

**Handling Missing Data**
- Age: Fill with median
- Embarked: Fill with most common
- Cabin: Too many missing, drop it

**Converting Categorical to Numerical**
- Sex: Male=1, Female=0
- Embarked: One-hot encoding

### Guided Practice (25 minutes)
```python
# Fill missing Age values with median
train['Age'].fillna(train['Age'].median(), inplace=True)

# Fill missing Embarked with most common
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Convert Sex to numerical
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})

# Convert Embarked to numerical (one-hot encoding)
embarked_dummies = pd.get_dummies(train['Embarked'], prefix='Embarked')
train = pd.concat([train, embarked_dummies], axis=1)

# Create family size feature
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

# Select features for model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
X = train[features]
y = train['Survived']

print(X.head())
```

### Independent Practice (10 minutes)
Students experiment with creating new features:
- IsAlone: 1 if FamilySize = 1
- AgeGroup: Categorize ages into groups

---

## Day 4: Building the Machine Learning Model

### Warm-Up (5 minutes)
"How does a computer make decisions? Let's think of a decision tree game..."

### Direct Instruction (15 minutes)
**Introduction to Decision Trees**
- Visual explanation using flowchart
- How trees make decisions
- Training vs Testing data
- Why we split data (avoid overfitting)

### Guided Practice (30 minutes)
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2%}")

# Feature importance
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance)
```

### Discussion (10 minutes)
- What accuracy did we achieve?
- Which features were most important?
- Does this match our intuition?

---

## Day 5: Making Predictions & Kaggle Submission

### Review (10 minutes)
Recap the entire process from data to model

### Guided Practice (25 minutes)
**Preparing Test Data and Making Predictions**

```python
# Load test data
test = pd.read_csv('test.csv')

# Apply same preprocessing
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})

embarked_dummies = pd.get_dummies(test['Embarked'], prefix='Embarked')
test = pd.concat([test, embarked_dummies], axis=1)

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

# Select same features
X_test_final = test[features]

# Make predictions
final_predictions = model.predict(X_test_final)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': final_predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file created!")
```

### Independent Practice (15 minutes)
**Submit to Kaggle**
1. Navigate to Titanic competition
2. Click "Submit Predictions"
3. Upload submission.csv
4. View score on leaderboard

### Wrap-Up (10 minutes)
- Class discussion: Compare scores
- What could we improve?
- Celebrate first ML project!

---

## Day 6: Extensions & Improvements (Optional)

### Challenges for Advanced Students

**Challenge 1: Try Different Models**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
```

**Challenge 2: Feature Engineering**
- Extract titles from names (Mr., Mrs., Miss.)
- Create age groups
- Fare bins
- Cabin deck extraction

**Challenge 3: Hyperparameter Tuning**
- Adjust max_depth
- Try different random_states
- Cross-validation

---

## Assessment Rubric

### Knowledge (40%)
- [ ] Can explain what machine learning is
- [ ] Understands the difference between training and testing
- [ ] Can interpret basic visualizations
- [ ] Knows what features are

### Skills (40%)
- [ ] Successfully loads and explores data
- [ ] Creates at least 2 visualizations
- [ ] Preprocesses data correctly
- [ ] Builds and trains a model
- [ ] Makes predictions

### Application (20%)
- [ ] Successfully submits to Kaggle
- [ ] Achieves accuracy >75%
- [ ] Can suggest improvements
- [ ] Completes at least one extension challenge

---

## Common Pitfalls & Solutions

**Problem 1:** "My accuracy is only 50%!"
- Solution: Check that Sex was encoded correctly (female should have higher survival)

**Problem 2:** "I get errors about missing columns"
- Solution: Ensure test data has same preprocessing as training data

**Problem 3:** "My submission file is rejected"
- Solution: Check PassengerId matches test.csv exactly

**Problem 4:** "I don't understand why we split the data"
- Solution: Use the "teaching to the test" analogy - we need fresh data to evaluate

---

## Extensions & Real-World Connections

### Career Connections
- Data Scientist
- Machine Learning Engineer
- Data Analyst
- AI Researcher

### Related Projects
- Predict house prices
- Classify flowers (Iris dataset)
- Handwritten digit recognition (MNIST)
- Movie recommendation systems

### Discussion Questions
1. What are the ethical considerations in ML?
2. Could this model be biased? How?
3. Where else could we apply these techniques?
4. What are the limitations of our model?

---

## Resources for Students

**Learning Resources:**
- Kaggle Learn (free courses)
- Google Colab (free Python environment)
- Scikit-learn documentation
- YouTube: StatQuest Machine Learning

**Dataset Documentation:**
- Kaggle Titanic Competition page
- Encyclopedia Titanica

**Next Steps:**
- Try other Kaggle competitions
- Build a personal ML project
- Join a data science club
- Explore AI ethics

---

## Teacher Notes

**Preparation:**
- Test all code before class
- Have backup plans for technical issues
- Prepare Google Colab notebooks as backup
- Create helper function library for struggling students

**Differentiation:**
- Advanced: Challenges and additional models
- Struggling: Provide completed code sections to modify
- Visual learners: Emphasize visualizations
- Kinesthetic: Use physical decision tree activities

**Time Management:**
- Can be compressed to 3 days if needed
- Can be extended with competitions between students
- Works well as a week-long intensive

**Success Metrics:**
- 90%+ students successfully submit to Kaggle
- Average class accuracy >75%
- Students can explain their process
- Increased interest in AI/ML careers