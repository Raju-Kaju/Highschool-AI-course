# Day 4: Building Your First Machine Learning Model
## Detailed Teacher Guide with Script & Complete Code

---

## Lesson Overview
**Duration:** 50 minutes  
**Objective:** Students build, train, and evaluate their first ML model  
**Key Outcome:** Students create a Decision Tree model and understand accuracy

---

## Materials Checklist
- [ ] Projector for live coding demonstration
- [ ] Student computers with Day 3 notebooks (preprocessed data)
- [ ] Printed "ML Vocabulary Cheat Sheet" (included below)
- [ ] Physical decision tree props (optional - paper, markers)
- [ ] Celebration plan for successful model training!
- [ ] Backup: Pre-trained model file if students fall behind

---

## Big Ideas for Day 4
1. Machine learning is teaching computers to find patterns
2. We split data to test fairly (like not studying from the test!)
3. Decision trees make decisions like a flowchart
4. Accuracy tells us how well our model works
5. Some features matter more than others (feature importance)

---

## Pre-Class Setup (10 minutes before students arrive)

**Prepare:**
1. Test that scikit-learn works on all computers
2. Have your own completed model ready to show
3. Prepare celebration (music, certificate template, etc.)
4. Write success criteria on board

**Success Criteria for Today:**
- [ ] Train/test split complete
- [ ] Model trained successfully
- [ ] Predictions made
- [ ] Accuracy calculated (goal: >75%)
- [ ] Feature importance visualized

---

## MINUTE-BY-MINUTE LESSON PLAN

### Minutes 0-3: Warm-Up & The Big Moment

**TEACHER SCRIPT:**

"Good morning, data scientists! Today is THE DAY. We've spent three days preparing - cleaning data, handling missing values, converting text to numbers. All of that was building up to THIS moment.

Today, you're going to train your first machine learning model!

Quick poll - how are we feeling?
- Excited? [hands]
- Nervous? [hands]
- Ready? [hands]

All of those feelings are normal! Even professional data scientists get excited about training new models.

Let me tell you what's going to happen today..."

[WRITE ON BOARD:]

**Today's Journey:**
1. Split our data (train vs test)
2. Create a Decision Tree model
3. Train it on our data
4. Make predictions
5. Calculate accuracy
6. See which features matter most

"By the end of class, you'll have a working AI model. Let's go!"

---

### Minutes 3-12: Understanding Train/Test Split

**TEACHER SCRIPT:**

"Before we code, I need to explain something crucial. 

Imagine I'm your teacher and I give you a practice test. You study the practice test really hard, memorize all the answers. Then on test day... I give you THE EXACT SAME TEST. You'd get 100%, right?

Does that mean you actually learned the material? [pause for responses] 

NO! You just memorized those specific questions!

This is the same problem in machine learning. If we train our model on ALL the data, then test it on the SAME data, it would cheat! It memorized the answers.

Instead, we do this..."

[DRAW ON BOARD:]

```
ALL DATA (891 passengers)
        |
        |---> SPLIT
        |
    ___/_\___
   /         \
TRAINING     TEST
(80%)       (20%)
712         179
passengers  passengers

TRAINING: Model learns from these
TEST: Model proves it learned (never seen before!)
```

**TEACHER SCRIPT:**

"We hide 20% of our data - the model NEVER sees it during training. Then we test on that hidden data to see if it truly learned patterns, or just memorized.

This is like:
- Studying from practice problems (training)
- Taking a real test with NEW problems (testing)

Make sense? Let's do it in code!"

[OPEN YOUR NOTEBOOK]

---

### Minutes 12-22: GUIDED CODING - Train/Test Split & Model Creation

**TEACHER SCRIPT:**

"Open your notebook from yesterday. We should have our clean data ready - X (features) and y (survived/died).

Let me show you on my screen first."

[TYPE AND NARRATE:]

```python
# Import the tools we need
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Machine Learning libraries imported! ✓")
```

**TEACHER SCRIPT:**

"These are our ML tools:
- **train_test_split**: Splits our data
- **DecisionTreeClassifier**: Our ML model
- **accuracy_score**: Tells us how well we did

Now let's split our data. Watch carefully:"

```python
# Assuming we have X and y from Day 3
# If not, recreate them:
train = pd.read_csv('train.csv')

# Quick preprocessing (from Day 3)
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})

embarked_dummies = pd.get_dummies(train['Embarked'], prefix='Embarked')
train = pd.concat([train, embarked_dummies], axis=1)

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)

# Select features
feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 
                   'Fare', 'FamilySize', 'IsAlone',
                   'Embarked_C', 'Embarked_Q', 'Embarked_S']

X = train[feature_columns]
y = train['Survived']

print("Data prepared! ✓")
print(f"Total passengers: {len(X)}")
```

**TEACHER SCRIPT:**

"Good! Now the magic split:"

```python
# Split into training and testing sets
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 makes it reproducible (everyone gets same split)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print("="*50)
print("DATA SPLIT COMPLETE!")
print("="*50)
print(f"Training set: {len(X_train)} passengers")
print(f"Test set: {len(X_test)} passengers")
print(f"\nTraining percentage: {len(X_train)/len(X)*100:.1f}%")
print(f"Test percentage: {len(X_test)/len(X)*100:.1f}%")
```

**TEACHER SCRIPT:**

"Look! We now have FOUR pieces:
- **X_train**: Training features (712 passengers)
- **X_test**: Test features (179 passengers)
- **y_train**: Training answers (who survived)
- **y_test**: Test answers (we'll compare our predictions to this)

The model will ONLY see X_train and y_train. The test set is hidden!

Everyone type this code now - take your time with the split!"

[GIVE 3-4 MINUTES - CIRCULATE HEAVILY]

**COMMON ISSUES:**
- NameError for X or y → need to recreate from preprocessing
- Import errors → run import cell first
- Different split sizes → check random_state=42

---

**TEACHER SCRIPT (after students complete):**

"Perfect! Now comes the exciting part - creating our model!

In scikit-learn, creating a model is surprisingly simple. Watch:"

```python
# Create the Decision Tree model
model = DecisionTreeClassifier(
    max_depth=5,           # Don't grow tree too deep
    random_state=42        # Reproducible results
)

print("Decision Tree model created! ✓")
print(f"Model type: {type(model)}")
print("\nModel is ready to learn, but hasn't learned anything yet!")
```

**TEACHER SCRIPT:**

"That's it! We created a Decision Tree. Think of it like buying a new brain - it exists, but it's empty. No knowledge yet.

Parameters we set:
- **max_depth=5**: Tree can only go 5 levels deep (prevents overfitting)
- **random_state=42**: Makes results consistent

Now we TRAIN it - we teach it patterns!"

---

### Minutes 22-30: TRAINING THE MODEL

**TEACHER SCRIPT:**

"Here's the moment we've been waiting for. Three days of preparation come down to TWO LINES of code.

Ready? Watch:"

```python
# Train the model
print("Training the model...")
print("="*50)

model.fit(X_train, y_train)

print("✓ TRAINING COMPLETE!")
print("="*50)
print("\nYour model has learned from 712 passengers!")
print("It found patterns connecting features to survival.")
```

**TEACHER SCRIPT:**

"THAT'S IT! The `.fit()` method is where the magic happens!

In that one line, the computer:
1. Looked at all 712 training passengers
2. Examined their features (age, sex, class, etc.)
3. Saw who survived and who didn't
4. Found patterns: 'When Sex=0 AND Pclass=1, survival is likely'
5. Built a decision tree of rules

All of that happened in a fraction of a second!

Now type these two lines - create your model and train it!"

[GIVE 2 MINUTES]

**TEACHER SCRIPT (when most finish):**

"Congratulations! You just trained your first machine learning model! 

[PAUSE FOR EFFECT - maybe clap or play brief celebratory sound]

But... how do we know if it's any good? That's where testing comes in!"

---

### Minutes 30-38: MAKING PREDICTIONS & EVALUATING ACCURACY

**TEACHER SCRIPT:**

"Now we're going to test our model on data it's NEVER seen - the test set.

Remember, it learned from 712 passengers. Now we'll show it 179 NEW passengers and ask: 'Who do you think survived?'

Watch:"

```python
# Make predictions on the test set
predictions = model.predict(X_test)

print("Predictions made! ✓")
print(f"\nFirst 10 predictions: {predictions[:10]}")
print("0 = died, 1 = survived")

# Let's see some examples
print("\n" + "="*50)
print("SAMPLE PREDICTIONS vs REALITY")
print("="*50)

comparison = pd.DataFrame({
    'Actual': y_test[:10].values,
    'Predicted': predictions[:10],
    'Correct?': (y_test[:10].values == predictions[:10])
})
print(comparison)
```

**TEACHER SCRIPT:**

"See? For each passenger in our test set, the model made a guess: 0 or 1.

Some it got right ✓, some it got wrong ✗. That's normal - no model is perfect!

But what's our overall accuracy? Let's calculate:"

```python
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Accuracy: {accuracy:.2%}")
print(f"\nOut of {len(y_test)} test passengers:")
print(f"  Correct: {(predictions == y_test).sum()}")
print(f"  Wrong: {(predictions != y_test).sum()}")
```

**TEACHER SCRIPT:**

"This is THE moment! What's your accuracy?

[PAUSE - let students check their results]

Most of you should be getting around 80-85% accuracy. That means your model correctly predicted survival for 8 out of 10 passengers it had NEVER seen before!

Think about that - based on just age, sex, class, and a few other features, we can predict with 80% accuracy who survived a disaster from 1912.

Is 80% good? Let me put it in perspective..."

[WRITE ON BOARD:]

```
Random guessing: ~50% (flip a coin)
Always guess "died": ~62% (since 62% died)
Our model: ~80-85% ✓✓✓

That's pretty good for our first model!
```

**TEACHER SCRIPT:**

"Type this code and see your accuracy!"

[GIVE 3 MINUTES]

---

### Minutes 38-45: CONFUSION MATRIX & FEATURE IMPORTANCE

**TEACHER SCRIPT:**

"Let's dig deeper. WHERE is our model making mistakes?

We'll use something called a CONFUSION MATRIX - sounds scary, but it's actually simple. It shows four numbers:
- True Positives: Correctly predicted survival
- True Negatives: Correctly predicted death
- False Positives: Predicted survival but they died (sad)
- False Negatives: Predicted death but they survived (missed opportunity)

Watch:"

```python
# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

# Visualize it
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.show()

print("\nConfusion Matrix Breakdown:")
print(f"True Negatives (correctly predicted died): {cm[0,0]}")
print(f"False Positives (predicted survived but died): {cm[0,1]}")
print(f"False Negatives (predicted died but survived): {cm[1,0]}")
print(f"True Positives (correctly predicted survived): {cm[1,1]}")
```

**TEACHER SCRIPT:**

"The darker the blue, the more predictions. Ideally, we want dark blues in the top-left and bottom-right (correct predictions) and light blues elsewhere (mistakes).

Now, the coolest part - which features did the model think were most important?"

```python
# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE RANKING")
print("="*50)
print(feature_importance)

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance - What Matters Most?', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()
```

**TEACHER SCRIPT:**

"Look at this! The model figured out what matters most!

Usually, Sex is #1 - remember, 74% of females survived vs 19% of males. The model learned that pattern!

Pclass (ticket class) is often #2 - wealth mattered.

This tells us the model is learning the RIGHT patterns - the same ones we discovered in our exploratory analysis!

Run this code and see what your model thinks is important!"

[GIVE 3-4 MINUTES]

---

### Minutes 45-48: REFLECTION & DISCUSSION

**TEACHER SCRIPT:**

"Alright everyone, pause your coding. Let's reflect on what just happened.

[TURN AWAY FROM SCREEN - FACE CLASS]

You just:
1. Split data like a real data scientist
2. Created a machine learning model
3. Trained it on historical data
4. Made predictions on new data
5. Evaluated its performance

This is EXACTLY what data scientists at Google, Netflix, hospitals, and research labs do every day. The process is the same - only the data changes.

Discussion questions:

**1. Why do you think our model isn't 100% accurate?**

[TAKE 2-3 RESPONSES]

Expected answers:
- Some things are random/unpredictable
- Missing information (we don't know everything about passengers)
- Features overlap (rich young males vs poor old females)
- Survival had some luck involved

**2. If you could add ONE more feature to improve accuracy, what would it be?**

[TAKE 2-3 RESPONSES]

Good answers:
- Exact location on ship
- Who they were with
- Which lifeboat they could access
- Time they reached deck

**3. Could this model be biased? How?**

[TAKE 2-3 RESPONSES]

Important discussion:
- Yes! It learned that being female increased survival
- That was TRUE for Titanic, but reflects 1912 social norms
- If we used this model today, it would be unfair
- ML models learn from history - including historical biases

Great thinking! This is why data scientists need to think about ethics."

---

### Minutes 48-50: CLOSURE & TOMORROW'S PREVIEW

**TEACHER SCRIPT:**

"Incredible work today! Let me summarize what you accomplished:

[WRITE ON BOARD:]

**Today's Achievements:**
✓ Understood train/test split concept
✓ Created a Decision Tree Classifier
✓ Trained model on 712 passengers
✓ Made predictions on 179 new passengers
✓ Achieved ~80% accuracy!
✓ Analyzed feature importance
✓ Discussed ML ethics

**Tomorrow - THE FINALE:**
Tomorrow we're going to:
1. Load the REAL test data (418 passengers we know nothing about)
2. Preprocess it the same way
3. Make predictions
4. Submit to Kaggle!
5. See our names on the leaderboard!

This is it - the culmination of everything we've learned!

**Homework:**
1. Make sure your model is saved and working
2. Answer the reflection questions on the handout
3. Think: How could you improve your model? (We'll discuss tomorrow)

**Before you leave:**
Give yourself a round of applause - you're officially machine learning engineers!

[LEAD APPLAUSE]

See you tomorrow for our Kaggle submission!"

---

## COMPLETE WORKING CODE - DAY 4

```python
# ============================================
# TITANIC ML MODEL - DAY 4
# Building Your First Machine Learning Model
# Student Name: _______________
# Date: _______________
# ============================================

# Cell 1: Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

print("✓ All libraries imported successfully!")

# ============================================

# Cell 2: Load and Preprocess Data (From Day 3)
print("LOADING AND PREPROCESSING DATA")
print("="*50)

# Load data
train = pd.read_csv('train.csv')
print(f"✓ Data loaded: {train.shape}")

# Preprocessing steps from Day 3
# Fill missing values
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
print("✓ Missing values filled")

# Convert Sex to numeric
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
print("✓ Sex converted to numeric")

# One-hot encode Embarked
embarked_dummies = pd.get_dummies(train['Embarked'], prefix='Embarked')
train = pd.concat([train, embarked_dummies], axis=1)
train = train.drop('Embarked', axis=1)
print("✓ Embarked one-hot encoded")

# Feature engineering
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
print("✓ New features created")

# Select features
feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 
                   'Fare', 'FamilySize', 'IsAlone',
                   'Embarked_C', 'Embarked_Q', 'Embarked_S']

X = train[feature_columns]
y = train['Survived']

print(f"\n✓ Feature matrix X: {X.shape}")
print(f"✓ Target vector y: {y.shape}")
print("\nData preprocessing complete!")

# ============================================

# Cell 3: Train/Test Split
print("\n" + "="*50)
print("SPLITTING DATA")
print("="*50)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # Reproducible results
)

print(f"\n✓ Training set: {len(X_train)} passengers ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Test set: {len(X_test)} passengers ({len(X_test)/len(X)*100:.1f}%)")

print("\nTrain/Test split complete!")
print("The model will learn from the training set,")
print("and we'll evaluate it on the test set (data it's never seen).")

# ============================================

# Cell 4: Create the Model
print("\n" + "="*50)
print("CREATING THE MODEL")
print("="*50)

# Create Decision Tree Classifier
model = DecisionTreeClassifier(
    max_depth=5,         # Limit tree depth to prevent overfitting
    min_samples_split=20, # Minimum samples required to split a node
    min_samples_leaf=10,  # Minimum samples required in a leaf
    random_state=42
)

print("✓ Decision Tree Classifier created!")
print(f"\nModel parameters:")
print(f"  - Max depth: {model.max_depth}")
print(f"  - Min samples split: {model.min_samples_split}")
print(f"  - Min samples leaf: {model.min_samples_leaf}")

print("\nModel created but not trained yet...")

# ============================================

# Cell 5: Train the Model
print("\n" + "="*50)
print("TRAINING THE MODEL")
print("="*50)

print("Teaching the model patterns from 712 passengers...")

# Train the model
model.fit(X_train, y_train)

print("\n✓✓✓ TRAINING COMPLETE! ✓✓✓")
print("\nYour model has learned patterns!")
print("It can now make predictions on new passengers.")

# ============================================

# Cell 6: Make Predictions
print("\n" + "="*50)
print("MAKING PREDICTIONS")
print("="*50)

# Predict on test set
predictions = model.predict(X_test)

print(f"✓ Predictions made for {len(predictions)} test passengers")

# Show some examples
print("\nFirst 15 predictions:")
print(predictions[:15])
print("(0 = died, 1 = survived)")

# Compare predictions to reality
print("\n" + "="*50)
print("SAMPLE PREDICTIONS vs ACTUAL")
print("="*50)

comparison = pd.DataFrame({
    'Actual': y_test[:15].values,
    'Predicted': predictions[:15],
    'Correct?': (y_test[:15].values == predictions[:15])
})
comparison.index = range(1, len(comparison) + 1)
print(comparison)

# ============================================

# Cell 7: Calculate Accuracy
print("\n" + "="*60)
print("MODEL PERFORMANCE - ACCURACY")
print("="*60)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\n🎯 ACCURACY: {accuracy:.2%}")
print(f"\nThis means the model correctly predicted {accuracy:.1%}")
print(f"of the {len(y_test)} test passengers.")

print(f"\nBreakdown:")
print(f"  ✓ Correct predictions: {(predictions == y_test).sum()}")
print(f"  ✗ Incorrect predictions: {(predictions != y_test).sum()}")

# Context
print("\n" + "="*60)
print("CONTEXT: Is this good?")
print("="*60)
print(f"  Random guessing: ~50%")
print(f"  Always guess 'died': ~{(y_test == 0).sum()/len(y_test)*100:.1f}%")
print(f"  Our model: {accuracy:.1%} ✓✓✓")

# ============================================

# Cell 8: Confusion Matrix
print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)

# Calculate confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)
print("\nBreakdown:")
print(f"  True Negatives (correctly predicted died): {cm[0,0]}")
print(f"  False Positives (predicted survived but died): {cm[0,1]}")
print(f"  False Negatives (predicted died but survived): {cm[1,0]}")
print(f"  True Positives (correctly predicted survived): {cm[1,1]}")

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Predictions vs Reality', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)

# Add text explanation
plt.text(0.5, -0.15, 'Darker blue = more predictions', 
         ha='center', transform=plt.gca().transAxes, fontsize=10, style='italic')

plt.tight_layout()
plt.show()

# ============================================

# Cell 9: Classification Report
print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT")
print("="*50)

report = classification_report(y_test, predictions, 
                               target_names=['Died', 'Survived'])
print(report)

print("\nWhat this means:")
print("  - Precision: When model predicts 'survived', how often is it right?")
print("  - Recall: Of all who actually survived, how many did we catch?")
print("  - F1-score: Balance between precision and recall")

# ============================================

# Cell 10: Feature Importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRanking of most important features:")
print(feature_importance)

print("\nInterpretation:")
top_feature = feature_importance.iloc[0]['Feature']
print(f"  🥇 Most important: {top_feature}")
print(f"     The model relied on this feature the most!")

# Visualize
plt.figure(figsize=(10, 6))
colors = sns.color_palette('viridis', len(feature_importance))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette=colors)
plt.title('Feature Importance - What Matters Most for Survival?', 
          fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)

# Add value labels
for i, (idx, row) in enumerate(feature_importance.iterrows()):
    plt.text(row['Importance'] + 0.01, i, f"{row['Importance']:.3f}", 
             va='center', fontsize=10)

plt.tight_layout()
plt.show()

# ============================================

# Cell 11: Visualize the Decision Tree (Optional)
print("\n" + "="*50)
print("VISUALIZING THE DECISION TREE")
print("="*50)

from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=feature_columns,
          class_names=['Died', 'Survived'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ This shows how the model makes decisions!")
print("  - Each box is a decision point")
print("  - Color indicates prediction (orange=died, blue=survived)")
print("  - Follows path from top to bottom")

# ============================================

# Cell 12: Model Summary
print("\n" + "="*60)
print("FINAL MODEL SUMMARY")
print("="*60)

print(f"\n📊 Dataset:")
print(f"   Total passengers: {len(X)}")
print(f"   Training set: {len(X_train)}")
print(f"   Test set: {len(X_test)}")

print(f"\n🎯 Performance:")
print(f"   Accuracy: {accuracy:.2%}")
print(f"   Correct predictions: {(predictions == y_test).sum()}/{len(y_test)}")

print(f"\n⭐ Top 3 Most Important Features:")
for i, row in feature_importance.head(3).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.3f}")

print(f"\n✓ Model Status: TRAINED AND READY")
print(f"✓ Next step: Use this model on real Kaggle test data!")

print("\n" + "="*60)
```

---

## ML VOCABULARY CHEAT SHEET

### Key Terms

**Training Set**
- Data the model learns from
- Like practice problems before a test
- Usually 70-80% of your data

**Test Set**
- Data the model has NEVER seen
- Used to check if model truly learned
- Usually 20-30% of your data

**Features (X)**
- Input variables (age, sex, class, etc.)
- Information we use to make predictions
- Also called "independent variables"

**Target (y)**
- What we're trying to predict (survived yes/no)
- Also called "dependent variable" or "label"

**Training**
- Teaching the model patterns from data
- Done with `.fit()` method
- Model learns relationships between X and y

**Prediction**
- Model's guess for new data
- Done with `.predict()` method
- Returns predicted y values

**Accuracy**
- Percentage of correct predictions
- (Correct predictions / Total predictions) × 100
- Higher is better, but 100% is rare!

**Decision Tree**
- ML model that makes decisions like a flowchart
- Asks yes/no questions about features
- Splits data until it finds patterns

**Feature Importance**
- Shows which features matter most
- Higher number = more important
- Helps us understand the model

**Overfitting**
- Model memorizes training data too well
- Performs great on training, bad on test
- Like memorizing test answers vs learning

**Confusion Matrix**
- Shows where model makes mistakes
- 4 numbers: True Pos, True Neg, False Pos, False Neg
- Helps identify specific error patterns

---

## HOMEWORK ASSIGNMENT - DAY 4

**Name:** _________________ **Date:** _____________

### Part 1: Understanding Train/Test Split

**1. In your own words, why do we split data into training and test sets?**

_____________________________________________________________________

_____________________________________________________________________

_____________________________________________________________________

**2. What would happen if we trained on ALL the data and tested on the