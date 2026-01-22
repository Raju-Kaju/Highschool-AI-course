# House Prices Kaggle Competition
## Teacher Quick Reference Guide

---

## 📅 LESSON STRUCTURE AT A GLANCE

### Day 1 - 50 minutes
**Topic:** Regression vs Classification  
**Hook:** "Price This House" activity  
**Key Script Points:** Classification=categories, Regression=numbers  
**Homework:** Predict important features  

### Day 2 - 50 minutes
**Topic:** Data Exploration & Visualization  
**Activities:** Scatter plots, correlations, heatmaps  
**Code:** ~60 lines of pandas/matplotlib  
**Key Finding:** OverallQual is strongest predictor (r≈0.79)  

### Day 3 - 50 minutes ⭐
**Topic:** Data Cleaning & Missing Values  
**Problem:** 5-99% missing in various features  
**Solutions:** Drop columns >70%, fill with 0/median/"Unknown"  
**Code:** ~80 lines - missing data analysis + cleaning  
**Output:** train_cleaned.csv (complete data)

### Day 4 - 50 minutes ⭐
**Topic:** Feature Engineering  
**Create:** 10 new features from existing ones  
**Examples:** HouseAge, TotalSqFt, PricePer_SqFt, ratios  
**Encoding:** Ordinal mapping + one-hot encoding  
**Code:** ~100 lines - feature transformations + validation  
**Output:** train_engineered.csv

### Day 5 - 50 minutes ⭐
**Topic:** Model Training & Evaluation  
**Models:** Linear, Ridge, Decision Tree, Random Forest  
**Metrics:** RMSE, MAE, R², Cross-validation  
**Code:** ~150 lines - training 4 models + comparison  
**Typical Winner:** Random Forest (RMSE ~$18,000)

### Day 6 - 50 minutes ⭐
**Topic:** Optimization & Kaggle Submission  
**Method:** GridSearchCV for hyperparameters  
**Output:** Final predictions + submission.csv  
**Code:** ~170 lines - tuning, final training, analysis  
**Result:** See Kaggle leaderboard ranking

---

## 🎯 KEY TEACHING POINTS BY DAY

### DAY 1: "Think Like a House Inspector"
- Classification: "Is this house in good condition? Yes/No"
- Regression: "What is the exact selling price? $180,000"
- RMSE: "How far off are you on average? ±$18,000"

### DAY 2: "Find the Treasure Map"
- Scatter plot = visualize relationships
- Correlation = measure strength of relationship
- Heatmap = see all relationships at once
- Students should notice: Size features + Quality features = top predictors

### DAY 3: "Be a Data Detective" ⭐
- Missing data is INFORMATION (not a problem!)
- PoolQC missing 99% → almost nobody has pool data
- GarageYrBlt missing 5% → a few people didn't report
- Different treatment for different missingness patterns
- Key phrase: "Cleaning ≠ throwing away data"

### DAY 4: "Build Better Clues" ⭐
- Domain knowledge matters: "Realtors talk about price/sqft!"
- HouseAge might matter more than YearBuilt (interactions)
- One big feature (TotalSqFt) better than many small ones
- Encoding: "Computers need numbers, not 'Excellent'"
- Test new features: "Does it correlate with price?"

### DAY 5: "Let the Machine Learn" ⭐
- Different models, different strengths
- Linear: "Simple line through data"
- Tree: "Series of yes/no questions"
- Forest: "100 trees voting together"
- CV: "Test on different data folds = more honest evaluation"

### DAY 6: "Tune the Engine" ⭐
- GridSearch: "Try hundreds of combinations"
- More training data (full dataset): "Better learning"
- Residuals: "Where did we mess up?"
- Leaderboard: "How do we rank globally?"

---

## ⏱️ MINUTE-BY-MINUTE TIMING

### DAY 1 (50 min)
```
0-2:   Welcome & context                          [2 min]
2-12:  Hook: "Price This House"                  [10 min]
12-25: Classification vs Regression lesson        [13 min]
25-35: House Prices problem intro                 [10 min]
35-47: Kaggle setup walkthrough                  [12 min]
47-50: Homework explanation                      [3 min]
```

### DAY 2 (50 min)
```
0-5:   Homework review + recap                    [5 min]
5-10:  Setup & environment                        [5 min]
10-25: Load data + initial exploration            [15 min]
25-40: Create visualizations (code along)         [15 min]
40-48: Feature analysis + discussion              [8 min]
48-50: Homework assignment                        [2 min]
```

### DAY 3 (50 min) ⭐
```
0-5:   Context + problem intro                    [5 min]
5-15:  Missing data analysis (code along)         [10 min]
15-35: Data cleaning strategies (code along)      [20 min]
35-45: Validation & before/after comparison       [10 min]
45-50: Summary & homework                         [5 min]
```

### DAY 4 (50 min) ⭐
```
0-5:   Hook: "What else can you calculate?"       [5 min]
5-20:  Feature engineering fundamentals           [15 min]
20-40: Numeric features (code along: 10 features) [20 min]
40-48: Validation + correlation check             [8 min]
48-50: Summary & homework                         [2 min]
```

### DAY 5 (50 min) ⭐
```
0-5:   Big picture - model training               [5 min]
5-15:  Data preparation (code along)              [10 min]
15-40: Train 4 models (code along)               [25 min]
40-48: Model comparison + analysis                [8 min]
48-50: Summary & homework                         [2 min]
```

### DAY 6 (50 min) ⭐
```
0-5:   Final push context                         [5 min]
5-25:  Hyperparameter tuning (code along)         [20 min]
25-40: Model analysis & interpretations           [15 min]
40-48: Kaggle submission & ranking                [8 min]
48-50: Reflection & next steps                    [2 min]
```

---

## 📊 CRITICAL CODE SECTIONS

### Day 2: Core Visualization
```python
# Scatter plot with trend line
plt.scatter(train['GrLivArea'], train['SalePrice'], alpha=0.5)
z = np.polyfit(train['GrLivArea'], train['SalePrice'], 1)
p = np.poly1d(z)
plt.plot(train['GrLivArea'], p(train['GrLivArea']), "r--", linewidth=2)

# Calculate correlation
correlation = train['GrLivArea'].corr(train['SalePrice'])
print(f"Correlation: {correlation:.3f}")
```

### Day 3: Missing Data Handling
```python
# Identify missing
missing_pct = (train.isnull().sum() / len(train)) * 100

# Drop high-missing columns
cols_to_drop = missing_pct[missing_pct > 70].index
train = train.drop(columns=cols_to_drop)

# Fill categorical with "Unknown"
train['PoolQC'].fillna('Unknown', inplace=True)

# Fill numeric with 0 or median
train['GarageYrBlt'].fillna(0, inplace=True)
train['LotFrontage'].fillna(train['LotFrontage'].median(), inplace=True)
```

### Day 4: Feature Creation
```python
# Age features
train_eng['HouseAge'] = 2024 - train_eng['YearBuilt']

# Ratio features
train_eng['PricePer_SqFt'] = train_eng['SalePrice'] / train_eng['TotalSqFt']

# Encoding
quality_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
train_eng['Quality_encoded'] = train_eng['Quality'].map(quality_map)

# One-hot encoding
dummies = pd.get_dummies(train_eng['Neighborhood'], prefix='Neighborhood', drop_first=True)
train_eng = pd.concat([train_eng, dummies], axis=1)
```

### Day 5: Model Training
```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=20)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
```

### Day 6: Hyperparameter Tuning
```python
# GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X, y)

# Best parameters
print(grid_search.best_params_)
print(grid_search.best_score_)
```

---

## 🎓 COMMON STUDENT QUESTIONS & ANSWERS

### "Why do we need to clean data?"
**Answer:** Real data is messy! Missing values, outliers, errors. We can't train models on incomplete data. Cleaning is how real data scientists spend 60% of their time.

### "What if I fill missing data wrong?"
**Answer:** Good question! That's why we validate. If missing means "no pool," filling with 0 makes sense. If it's a measurement error, median is better. Wrong filling introduces bias.

### "Why not use the feature with highest correlation?"
**Answer:** Good features work TOGETHER. Also, correlation can be misleading. A feature might correlate only because of another feature (collinearity).

### "Why Random Forest over Linear Regression?"
**Answer:** RF captures non-linear patterns. Houses don't follow a straight line relationship with price—there are complex interactions (size, location, condition all matter together).

### "What's overfitting?"
**Answer:** Memorizing training data instead of learning patterns. If CV R² (fair test) much lower than training R² (on seen data), you're overfitting. Good CV score = good generalization.

### "How many features should I engineer?"
**Answer:** Quality over quantity! 10 good features beat 100 random features. Create features based on domain knowledge, then test their correlation.

### "Will my RMSE be better than the leaderboard winner?"
**Answer:** Unlikely! Top Kaggle submissions use advanced techniques and lots of experiments. This course is about learning the PROCESS. $18K RMSE is excellent for a week's work!

### "Can I use deep learning?"
**Answer:** Yes! After this course, try neural networks. But tree-based methods (Random Forest, XGBoost) usually win house price competitions. Start simple, add complexity if needed.

---

## 🔴 COMMON MISTAKES TO PREVENT

### Day 1
❌ Confusing regression with continuous data
✓ Emphasize: "Regression = predicting a NUMBER, classification = predicting a CATEGORY"

### Day 2
❌ Not interpreting correlations correctly (0.3 vs 0.8)
✓ Use visual: "0.8 = tight cluster, 0.3 = scattered cloud"

### Day 3
❌ Filling all missing with mean (loses information)
✓ Explain: "Different missing patterns need different strategies"

❌ Not checking remaining missing after cleaning
✓ Always run: `print(train.isnull().sum().sum())`

### Day 4
❌ Creating too many features randomly
✓ Emphasize: "Why would this feature matter? Connect to domain."

❌ Forgetting to encode categorical variables
✓ Checkpoint: "Does X have any non-numeric columns?"

### Day 5
❌ Evaluating only on training data (apparent R² = 0.95)
✓ Always use: train_test_split + CV

❌ Not comparing multiple models
✓ Make it competitive: "Which model wins?"

### Day 6
❌ Overfitting through excessive hyperparameter tuning
✓ Use CV, watch for: training R² >> CV R²

❌ Submitting predictions without validation
✓ Sanity check: "Are predictions in reasonable price range?"

---

## 💻 GOOGLE COLAB SETUP (If not using Jupyter)

**Advantages:**
- Free GPU (though not needed here)
- Pre-installed libraries
- No local setup needed
- Easy sharing with students

**Day 1 - 2 Setup:**
```python
# Mount Google Drive (for saving files)
from google.colab import drive
drive.mount('/content/drive')

# Download data from Kaggle (or upload manually)
# Then read: train = pd.read_csv('/path/to/train.csv')
```

**For each day:**
- New notebook or cell section
- Copy-paste provided code
- Students can run cells incrementally

---

## 🏆 GRADING RUBRIC (Optional)

### Homework (40%)
- Day 1: Feature predictions + research
- Day 2: Correlation analysis accuracy
- Day 3: Cleaning strategy justification
- Day 4: Feature engineering reasoning
- Day 5: Model interpretation
- Day 6: Reflection essay

### Engagement (20%)
- Class participation
- Asks good questions
- Helps classmates
- Completes bonus challenges

### Code Quality (20%)
- Runs without errors
- Adds comments
- Uses proper variable names
- Produces sensible results

### Final RMSE Score (20%)
- < $15,000: A (Top 5%)
- < $20,000: A- (Top 10%)
- < $25,000: B+ (Top 25%)
- < $30,000: B (Solid)
- > $30,000: Needs improvement

---

## 📱 STUDENT SURVEY (End of Course)

Ask students to rate 1-5:

1. How clear was the classification vs regression concept?
2. Did the visualizations help you understand correlations?
3. Did you understand the data cleaning decisions?
4. Did feature engineering make sense?
5. Could you follow the model training?
6. Did you understand why Random Forest won?
7. Overall, how confident are you in machine learning now?
8. Would you recommend this course to others?

---

## 🔗 RESOURCES TO SHARE WITH STUDENTS

**Free Courses:**
- Kaggle Learn (30-minute micro-courses)
- Fast.ai (practical deep learning)
- Andrew Ng ML course (fundamentals)

**Competitions:**
- Kaggle Titanic (classification)
- Kaggle House Prices (this one!)
- Kaggle competitions with cash prizes

**Books:**
- "Hands-On Machine Learning" by Aurelien Geron
- "Introduction to Statistical Learning" (ISLR)
- "The Hundred-Page ML Book"

**Tools:**
- Google Colab (free notebooks + GPU)
- Kaggle kernels (community code sharing)
- GitHub (version control + portfolio)

---

## ✅ PRE-LESSON CHECKLIST

**Before Day 1:**
- [ ] Create Kaggle account (test drive it)
- [ ] Download train.csv & test.csv
- [ ] Prepare printed handouts
- [ ] Test all code on your machine
- [ ] Create presentation slides (optional)

**Before Days 2-6:**
- [ ] Run code through to completion
- [ ] Check execution time
- [ ] Note any error messages
- [ ] Prepare example outputs
- [ ] Have backup code files

**Day-of:**
- [ ] Arrive 10 minutes early
- [ ] Test projector/screen sharing
- [ ] Open IDE (Jupyter/Colab)
- [ ] Have code files ready
- [ ] Print handouts/homework

---

## 📞 TROUBLESHOOTING QUICK FIXES

| Problem | Solution |
|---------|----------|
| Code runs slow | Use `.head(100)` for testing, reduce GridSearch grid |
| Data not found | Check working directory, use full file path |
| Library not installed | `pip install pandas numpy sklearn seaborn` |
| Different results than provided | Different random seed? Use `random_state=42` |
| Student can't follow code | Use pre-written templates, go slower, pair programming |
| GridSearchCV takes forever | Reduce `cv=3` or reduce param grid size |
| Low RMSE suddenly | Check for data leakage (using test data in training) |

---

## 🎬 FINAL PRESENTATION TIPS

**Day 1:** 
- Show actual Kaggle competition page
- Display price range visually (bar: $34K to $755K)

**Days 2-4:**
- Celebrate discoveries ("Look! OverallQual is strongest!")
- Make it interactive ("What do YOU think will matter?")

**Day 5:**
- Show model comparison side-by-side
- Emphasize: "Random Forest beats Linear! Here's why."

**Day 6:**
- Pull up real Kaggle leaderboard
- Show student's rank: "You're in top X%!"
- Celebrate: "You did what data scientists do!"

---

**Remember:** This structure is PROVEN. Trust it, teach it, refine it based on student feedback. You're doing amazing work! 🚀

