# Day 2: Data Exploration & Visualization
## Detailed Teacher Guide with Script & Complete Code

---

## Lesson Overview
**Duration:** 50 minutes  
**Objective:** Students load, explore, and visualize the Titanic dataset  
**Key Outcome:** Students create 3+ visualizations and discover survival patterns

---

## Materials Checklist
- [ ] Projector for live coding demonstration
- [ ] Student computers with Python/Jupyter OR Google Colab access
- [ ] Titanic dataset downloaded (train.csv, test.csv)
- [ ] Printed Code Reference Sheet (included below)
- [ ] Completed homework from Day 1
- [ ] Backup: Pre-loaded Google Colab notebook

---

## Pre-Class Setup (15 minutes before students arrive)

### Option A: Using Google Colab (RECOMMENDED for ease)
1. Create a Google Colab notebook
2. Upload train.csv to Colab OR use code to load from Kaggle
3. Test all code runs without errors
4. Share link with students via Google Classroom/email

### Option B: Using Local Jupyter Notebooks
1. Ensure Python, pandas, matplotlib, seaborn installed on all computers
2. Download train.csv and test.csv to a shared folder
3. Test notebook on student computers

**Teacher's Demo Notebook:** Have your own notebook ready with completed code

---

## MINUTE-BY-MINUTE LESSON PLAN

### Minutes 0-5: Warm-Up & Homework Review

**TEACHER SCRIPT:**

"Good morning! Before we dive into coding today, let's do a quick poll. Yesterday I asked you to predict which feature would be MOST important for survival.

Stand up if you chose:
- Sex (women and children first) - [count]
- Pclass (wealth/ticket class) - [count]
- Age (children saved first) - [count]
- Other - [count]

Great! Today we're going to find out who was right using REAL DATA. By the end of class, we'll have actual evidence.

Quick homework check - hold up your completed reading if you have it. [SCAN ROOM] Good.

One question from the reading: Why did ticket class matter for survival? [CALL ON 2-3 STUDENTS]

Expected answers:
- First class was higher up on the ship
- Better access to lifeboats
- Wealthier people got priority
- Closer to the boat deck

Perfect! Let's see if the data agrees with you."

---

### Minutes 5-10: Setup & Loading Data

**TEACHER SCRIPT:**

"Alright, everyone open your laptops. Today you're going to write your first data science code!

[IF USING GOOGLE COLAB:]
Click on the link I shared - it should open Google Colab. You should see a notebook with empty cells. This is where we'll write code.

[IF USING JUPYTER:]
Open Jupyter Notebook from your applications. Navigate to the folder where train.csv is saved.

[PAUSE FOR SETUP]

Okay, everyone should be looking at a blank notebook or the one I shared. 

**What is a Jupyter Notebook?**
Think of it like a mix between a Word document and a calculator:
- You write code in 'cells'
- You run the code and see results immediately
- You can add notes and explanations
- It saves everything

See these rectangular boxes? Those are cells. We'll type code in them and run them one at a time.

Let me show you on my screen first, then you'll do it."

[SWITCH TO SCREEN SHARE OF YOUR NOTEBOOK]

---

### Minutes 10-25: GUIDED CODING - Loading and Exploring Data

**TEACHER SCRIPT:**

"The first thing we need to do is load our data. Watch my screen.

I'm going to click in the first cell and type this code. Don't type yet - just watch."

[TYPE SLOWLY ON SCREEN, NARRATING EACH LINE]

```python
# Import libraries (these are tools we need)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This makes our plots show up in the notebook
%matplotlib inline
```

**TEACHER SCRIPT:**

"Let me explain what this means:

- **import pandas as pd** - Pandas is our main tool for working with data. 'pd' is just a nickname so we don't have to type 'pandas' every time.
- **import matplotlib** - This helps us make charts
- **import seaborn** - This makes our charts look prettier
- **%matplotlib inline** - This magic command makes our plots appear right here in the notebook

Now, to run this cell, I'll press Shift+Enter. Watch what happens."

[PRESS SHIFT+ENTER]

"See? No error messages! The number [1] appeared on the left - that means it ran successfully.

Now it's your turn. In your first cell, type this exact code. Take your time. Raise your hand if you need help.

[GIVE 2-3 MINUTES - CIRCULATE TO HELP]

Once you have it typed, press Shift+Enter to run it. You should see nothing happen - that's good! It means the libraries loaded successfully."

[WAIT FOR MOST STUDENTS TO COMPLETE]

---

**TEACHER SCRIPT (LOADING DATA):**

"Great! Now let's actually load our Titanic data. Watch me first."

[TYPE IN NEXT CELL:]

```python
# Load the training data
train = pd.read_csv('train.csv')

# Show the first 5 rows
print("First 5 passengers:")
print(train.head())

print("\n" + "="*50 + "\n")

# How big is our dataset?
print(f"Total passengers in dataset: {len(train)}")
print(f"Total features: {len(train.columns)}")
```

**TEACHER SCRIPT:**

"Let me break this down:

- **train = pd.read_csv('train.csv')** - This reads our CSV file and stores it in a variable called 'train'. Think of 'train' as a giant spreadsheet.
  
- **train.head()** - Shows us the first 5 rows. Like peeking at the top of a spreadsheet.

- **len(train)** - Tells us how many passengers we have

- **len(train.columns)** - Tells us how many columns (features) we have

Watch what happens when I run this..."

[RUN CELL]

"Whoa! Look at all this data! See the table? Each row is one passenger. Each column is a feature.

Look at the first passenger - Passenger ID 1:
- Survived = 0 (sadly, they died)
- Pclass = 3 (third class)
- Name = Braund, Mr. Owen Harris
- Sex = male
- Age = 22

Now you try! Type this code in your next cell."

[GIVE 3-4 MINUTES - CIRCULATE HEAVILY]

**COMMON ISSUES TO WATCH FOR:**
- File not found error → Check file path
- Typos in commands
- Forgetting quotation marks around 'train.csv'

---

**TEACHER SCRIPT (MORE EXPLORATION):**

"Excellent! Now let's learn more about this data. Type this in your next cell:"

```python
# Get information about each column
print("Dataset Information:")
print(train.info())

print("\n" + "="*50 + "\n")

# Get statistical summary
print("Statistical Summary:")
print(train.describe())

print("\n" + "="*50 + "\n")

# Check for missing data
print("Missing Values:")
print(train.isnull().sum())
```

[RUN AND EXPLAIN:]

**TEACHER SCRIPT:**

"This gives us three important things:

**1. Info()** - Shows data types and missing values
- See 'Age' has only 714 non-null values? That means 177 ages are missing!
- 'Cabin' is mostly missing - only 204 values
- This is common in real data

**2. Describe()** - Shows statistics for numeric columns
- Mean age: about 29.7 years
- Mean fare: about £32
- Notice it only shows numeric columns

**3. Isnull().sum()** - Counts missing values per column
- Age: 177 missing
- Cabin: 687 missing (most are missing!)
- Embarked: 2 missing

This is important - we'll need to handle these missing values later.

Now the exciting part - let's calculate survival rate!"

```python
# Calculate overall survival rate
survival_rate = train['Survived'].mean()
print(f"Overall survival rate: {survival_rate:.1%}")

# Count survivors vs non-survivors
print("\nSurvival counts:")
print(train['Survived'].value_counts())
```

**TEACHER SCRIPT:**

"Look at this! Only about 38% survived. That means 62% died. 
- 549 people died (Survived = 0)
- 342 people survived (Survived = 1)

This is what we're trying to predict for the test data passengers.

Everyone type and run this code!"

[GIVE 2-3 MINUTES]

---

### Minutes 25-40: CREATING VISUALIZATIONS

**TEACHER SCRIPT:**

"Okay, now for the cool part - VISUALIZATIONS! A picture is worth a thousand numbers. Let's see if our predictions from yesterday were right.

**First visualization: Survival by Sex**

Remember yesterday many of you predicted sex would be important because of 'women and children first'? Let's check!"

```python
# Survival rate by sex
print("Survival rate by sex:")
print(train.groupby('Sex')['Survived'].mean())

print("\n")

# Create a bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=train)
plt.title('Survival Rate by Sex', fontsize=16, fontweight='bold')
plt.ylabel('Survival Rate', fontsize=12)
plt.xlabel('Sex', fontsize=12)
plt.ylim(0, 1)

# Add value labels on bars
for i, p in enumerate(plt.gca().patches):
    height = p.get_height()
    plt.text(p.get_x() + p.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', fontsize=12, fontweight='bold')

plt.show()
```

**TEACHER SCRIPT:**

[AFTER RUNNING]

"WOW! Look at this difference!
- Females: about 74% survived
- Males: only about 19% survived

'Women and children first' was REAL! Being female nearly quadrupled your chances of survival.

Who predicted sex would be most important? You were onto something!

Now type this code and run it. You should see the same chart."

[GIVE 3-4 MINUTES - HELP STUDENTS]

**TEACHER SCRIPT:**

"Now let's check ticket class - Pclass. Many of you thought wealth would matter."

```python
# Survival rate by passenger class
print("Survival rate by class:")
print(train.groupby('Pclass')['Survived'].mean())

print("\n")

# Create visualization
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=train)
plt.title('Survival Rate by Passenger Class', fontsize=16, fontweight='bold')
plt.ylabel('Survival Rate', fontsize=12)
plt.xlabel('Passenger Class (1=1st, 2=2nd, 3=3rd)', fontsize=12)
plt.ylim(0, 1)

# Add value labels
for i, p in enumerate(plt.gca().patches):
    height = p.get_height()
    plt.text(p.get_x() + p.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', fontsize=12, fontweight='bold')

plt.show()
```

**TEACHER SCRIPT:**

"Look at this pattern!
- 1st class: 63% survived
- 2nd class: 47% survived  
- 3rd class: only 24% survived

The richer you were, the better your chances. Why? Because:
- First class cabins were higher up (closer to lifeboats)
- Wealthier passengers got priority
- Third class was deep in the ship - harder to escape

The data confirms what you predicted!"

---

**TEACHER SCRIPT:**

"Let's do one more - age distribution. This shows us how old passengers were."

```python
# Age distribution
plt.figure(figsize=(10, 5))
train['Age'].hist(bins=30, edgecolor='black')
plt.title('Age Distribution of Titanic Passengers', fontsize=16, fontweight='bold')
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.axvline(train['Age'].mean(), color='red', linestyle='--', linewidth=2, label='Mean Age')
plt.legend()
plt.show()

print(f"Average age: {train['Age'].mean():.1f} years")
print(f"Youngest passenger: {train['Age'].min():.1f} years")
print(f"Oldest passenger: {train['Age'].max():.1f} years")
```

**TEACHER SCRIPT:**

"This shows us:
- Most passengers were in their 20s and 30s
- Average age was about 30 years
- Youngest passenger: less than 1 year old
- Oldest passenger: 80 years old

There were entire families on the ship - babies to grandparents."

---

### Minutes 40-47: INDEPENDENT PRACTICE

**TEACHER SCRIPT:**

"Great work! Now YOU'RE going to create your own visualization. I want you to pick ONE of these options and create a chart:

**Option 1:** Survival rate by where people embarked (Embarked: S, C, or Q)

**Option 2:** Survival rate by age group (make groups: children under 12, adults 12-60, elderly over 60)

**Option 3:** Create a chart showing the distribution of ticket fares

I'll give you the starter code for each option on the handout. You pick one, customize it, and run it.

You have 5 minutes - GO!"

[DISTRIBUTE CODE REFERENCE SHEET]

**Option 1 Starter Code:**
```python
# Survival by embarkation port
sns.barplot(x='Embarked', y='Survived', data=train)
plt.title('Survival Rate by Embarkation Port')
plt.show()
```

**Option 2 Starter Code:**
```python
# Create age groups
train['AgeGroup'] = pd.cut(train['Age'], bins=[0, 12, 60, 100], labels=['Child', 'Adult', 'Elderly'])

# Plot survival by age group
sns.barplot(x='AgeGroup', y='Survived', data=train)
plt.title('Survival Rate by Age Group')
plt.show()
```

**Option 3 Starter Code:**
```python
# Fare distribution
train['Fare'].hist(bins=30, edgecolor='black')
plt.title('Distribution of Ticket Fares')
plt.xlabel('Fare (British Pounds)')
plt.ylabel('Number of Passengers')
plt.show()
```

[CIRCULATE AND HELP - CELEBRATE SUCCESSES]

---

### Minutes 47-50: CLOSURE & SUMMARY

**TEACHER SCRIPT:**

"Time! Let's share what we found. Who did Option 1 - embarkation port? What did you discover?"

[CALL ON 2-3 STUDENTS]

"Who did age groups? What pattern did you see?"

[CALL ON 2-3 STUDENTS]

"Excellent work! Let me summarize what we learned today:

[WRITE ON BOARD AS YOU SAY IT:]

**Key Findings:**
1. **Sex matters MOST** - 74% of females survived vs 19% of males
2. **Class matters** - 1st class had 3x survival rate of 3rd class
3. **Age had some effect** - children had better chances
4. **We have missing data** - especially in Age and Cabin columns

**Technical Skills Learned:**
- Loading data with pandas
- Using .head(), .info(), .describe()
- Creating bar plots with seaborn
- Calculating survival rates

**Tomorrow's Preview:**
Tomorrow we'll clean up this messy data (fill those missing ages!) and prepare it for our machine learning model.

**Homework:**
1. Complete your visualization if you didn't finish
2. Answer these questions on the handout:
   - Which feature do you NOW think is most important? (Based on today's data)
   - Name one surprise you found in the data
   - What do you think we should do about missing ages?

Save your notebook - we'll use it tomorrow!

Great work today - you're officially data scientists! See you tomorrow!"

---

## STUDENT CODE REFERENCE SHEET

### Day 2: Quick Reference Guide

**Running Code:**
- Type code in a cell
- Press **Shift + Enter** to run
- [Number] appears = success!
- Error message = check for typos

---

### Essential Commands

**Load data:**
```python
train = pd.read_csv('train.csv')
```

**View data:**
```python
train.head()          # First 5 rows
train.tail()          # Last 5 rows
train.info()          # Column information
train.describe()      # Statistics
```

**Missing data:**
```python
train.isnull().sum()  # Count missing per column
```

**Survival rate:**
```python
train['Survived'].mean()           # Overall rate
train.groupby('Sex')['Survived'].mean()  # By sex
```

---

### Visualization Templates

**Bar plot:**
```python
sns.barplot(x='COLUMN_NAME', y='Survived', data=train)
plt.title('YOUR TITLE HERE')
plt.show()
```

**Histogram:**
```python
train['COLUMN_NAME'].hist(bins=30)
plt.title('YOUR TITLE HERE')
plt.xlabel('X LABEL')
plt.ylabel('Y LABEL')
plt.show()
```

---

### Common Errors & Fixes

**Error: NameError: name 'pd' is not defined**
- Fix: Run the import cell first!
```python
import pandas as pd
```

**Error: FileNotFoundError: train.csv**
- Fix: Check file is in same folder as notebook
- Or use full path: 'C:/Users/YourName/Desktop/train.csv'

**Error: KeyError: 'Survived'**
- Fix: Check spelling - it's case-sensitive!

**Plot doesn't show:**
- Fix: Add `plt.show()` at the end
- Or run `%matplotlib inline` at the start

---

## HOMEWORK ASSIGNMENT - DAY 2

**Name:** _________________ **Date:** _____________

### Part 1: Complete Your Visualization
If you didn't finish your chosen visualization in class, complete it now. Paste a screenshot or sketch it below:

[SPACE FOR VISUALIZATION]

**What pattern did you find?**

_____________________________________________________________________

_____________________________________________________________________

---

### Part 2: Reflection Questions

**1. Based on TODAY'S DATA, which feature do you now think is most important for predicting survival?**

My choice: ______________________

Why (use evidence from today's data):

_____________________________________________________________________

_____________________________________________________________________

_____________________________________________________________________

**2. What was ONE surprise you found in the data today?**

_____________________________________________________________________

_____________________________________________________________________

**3. We have 177 missing ages in our dataset. What do you think we should do about this? (Circle one and explain)**

- A) Delete all passengers with missing ages
- B) Fill missing ages with the average age (about 30)
- C) Fill missing ages with 0
- D) Other idea: _______________________

**Why?**

_____________________________________________________________________

_____________________________________________________________________

**4. If you had to predict survival for a new passenger with ONLY this information, would they survive?**

- **Sex:** Female
- **Pclass:** 3 (third class)
- **Age:** 25

**Your prediction:** SURVIVED / DIED (circle one)

**Reasoning:**

_____________________________________________________________________

_____________________________________________________________________

---

### Part 3: Challenge (Optional - Extra Credit)

Create one NEW visualization we didn't do in class. Some ideas:
- Survival rate by SibSp (number of siblings/spouses)
- Survival rate by Parch (number of parents/children)
- Compare fare prices between survivors and non-survivors

Paste screenshot or sketch below:

[SPACE FOR CHALLENGE]

---

## COMPLETE STARTER CODE (For Google Colab)

*This is the complete notebook students can reference*

```python
# ============================================
# TITANIC DATA EXPLORATION - DAY 2
# Student Name: _______________
# Date: _______________
# ============================================

# Cell 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

print("Libraries loaded successfully!")

# ============================================

# Cell 2: Load Data
# Load the training data
train = pd.read_csv('train.csv')

# Show first 5 passengers
print("First 5 passengers:")
print(train.head())

print("\n" + "="*50 + "\n")

# Dataset size
print(f"Total passengers: {len(train)}")
print(f"Total features: {len(train.columns)}")

# ============================================

# Cell 3: Explore Data
# Get info about columns
print("Dataset Information:")
print(train.info())

print("\n" + "="*50 + "\n")

# Statistical summary
print("Statistical Summary:")
print(train.describe())

print("\n" + "="*50 + "\n")

# Missing values
print("Missing Values:")
print(train.isnull().sum())

# ============================================

# Cell 4: Calculate Survival Rate
# Overall survival rate
survival_rate = train['Survived'].mean()
print(f"Overall survival rate: {survival_rate:.1%}")

print("\nSurvival counts:")
print(train['Survived'].value_counts())

# ============================================

# Cell 5: Survival by Sex
print("Survival rate by sex:")
print(train.groupby('Sex')['Survived'].mean())

# Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=train)
plt.title('Survival Rate by Sex', fontsize=16, fontweight='bold')
plt.ylabel('Survival Rate', fontsize=12)
plt.xlabel('Sex', fontsize=12)
plt.ylim(0, 1)

# Add percentage labels
for i, p in enumerate(plt.gca().patches):
    height = p.get_height()
    plt.text(p.get_x() + p.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', fontsize=12, fontweight='bold')

plt.show()

# ============================================

# Cell 6: Survival by Class
print("Survival rate by passenger class:")
print(train.groupby('Pclass')['Survived'].mean())

# Visualization
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=train)
plt.title('Survival Rate by Passenger Class', fontsize=16, fontweight='bold')
plt.ylabel('Survival Rate', fontsize=12)
plt.xlabel('Passenger Class (1=1st, 2=2nd, 3=3rd)', fontsize=12)
plt.ylim(0, 1)

# Add percentage labels
for i, p in enumerate(plt.gca().patches):
    height = p.get_height()
    plt.text(p.get_x() + p.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', fontsize=12, fontweight='bold')

plt.show()

# ============================================

# Cell 7: Age Distribution
plt.figure(figsize=(10, 5))
train['Age'].hist(bins=30, edgecolor='black')
plt.title('Age Distribution of Titanic Passengers', fontsize=16, fontweight='bold')
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.axvline(train['Age'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f"Mean: {train['Age'].mean():.1f}")
plt.legend()
plt.show()

print(f"Average age: {train['Age'].mean():.1f} years")
print(f"Youngest passenger: {train['Age'].min():.1f} years")
print(f"Oldest passenger: {train['Age'].max():.1f} years")

# ============================================

# Cell 8: YOUR INDEPENDENT VISUALIZATION
# Choose one option and uncomment it:

# OPTION 1: Survival by Embarkation Port
# sns.barplot(x='Embarked', y='Survived', data=train)
# plt.title('Survival Rate by Embarkation Port')
# plt.show()

# OPTION 2: Survival by Age Group
# train['AgeGroup'] = pd.cut(train['Age'], bins=[0, 12, 60, 100], 
#                             labels=['Child', 'Adult', 'Elderly'])
# sns.barplot(x='AgeGroup', y='Survived', data=train)
# plt.title('Survival Rate by Age Group')
# plt.show()

# OPTION 3: Fare Distribution
# train['Fare'].hist(bins=30, edgecolor='black')
# plt.title('Distribution of Ticket Fares')
# plt.xlabel('Fare (British Pounds)')
# plt.ylabel('Number of Passengers')
# plt.show()

# ============================================

# Cell 9: Summary Statistics
print("="*50)
print("SUMMARY OF KEY FINDINGS")
print("="*50)

print("\n1. SURVIVAL BY SEX:")
print(train.groupby('Sex')['Survived'].agg(['mean', 'count']))

print("\n2. SURVIVAL BY CLASS:")
print(train.groupby('Pclass')['Survived'].agg(['mean', 'count']))

print("\n3. AGE STATISTICS:")
print(f"   Average age: {train['Age'].mean():.1f} years")
print(f"   Age range: {train['Age'].min():.0f} to {train['Age'].max():.0f} years")
print(f"   Missing ages: {train['Age'].isnull().sum()} passengers")

print("\n4. OVERALL:")
print(f"   Survival rate: {train['Survived'].mean():.1%}")
print(f"   Total passengers: {len(train)}")

print("="*50)
```

---

## TEACHER NOTES & TIPS

### Time Management
- **If running short:** Skip independent practice, assign as homework
- **If extra time:** Add a combined visualization (sex AND class together)
- **Pacing:** Spend most time on first visualizations - students catch on quickly

### Common Student Questions

**Q: "Why do we use Shift+Enter instead of just Enter?"**
A: Enter creates a new line in the cell. Shift+Enter runs the code.

**Q: "What's the difference between print() and just typing the variable?"**
A: Both work! print() gives us more control over formatting.

**Q: "Can we change the colors of the charts?"**
A: Yes! Add parameter: `sns.barplot(x='Sex', y='Survived', data=train, palette='Set2')`

**Q: "Why is the survival rate a decimal like 0.74?"**
A: That's how computers represent percentages. 0.74 = 74%. We use `:.1%` to format it.

### Differentiation Strategies

**For Advanced Students:**
- Challenge: Create a 2-way visualization (sex AND class together)
- Use `sns.catplot()` for more complex plots
- Calculate correlation between numeric features
```python
train[['Age', 'Fare', 'Survived']].corr()
```

**For Struggling Students:**
- Provide pre-written code to copy/paste
- Pair with stronger partner
- Focus on running code and interpreting results, not writing from scratch
- Use analogy: "Code is like a recipe - follow steps exactly"

**For Visual Learners:**
- Draw flowcharts of what each line does
- Use color coding for different types of commands
- Show before/after of data transformations

### Technical Troubleshooting

**Issue: Students can't see plots**
- Check `%matplotlib inline` was run
- Try `plt.show()` at end
- Restart kernel and run all cells from top

**Issue: "File not found" errors**
- Check file location matches code
- Use Google Colab to avoid file path issues
- Or provide full path: `/Users/studentname/Desktop/train.csv`

**Issue: Slow computers**
- Pre-load data before class
- Use smaller subset: `train.head(100)` for practice
- Close other applications

### Assessment Checkpoints

By end of Day 2, students should be able to:
- [ ] Load a CSV file with pandas
- [ ] View first/last rows of data
- [ ] Check for missing values
- [ ] Calculate basic statistics (mean, count)
- [ ] Create a bar plot
- [ ] Interpret survival rates from visualizations
- [ ] Explain why sex and class affected survival

### Tomorrow's Preview

"Tomorrow we'll clean this messy data - fill in missing ages, convert text to numbers, and prepare everything for our machine learning model. Bring your questions!"

---

## BACKUP PLAN (If Technology Fails)

### Low-Tech Version
1. Print out sample data (first 20 rows)
2. Calculate survival rates by hand with calculators
3. Create bar charts on graph paper
4. Still teaches the concepts!

### Pre-made Google Colab
- Create a shareable Colab with all code
- Students just press "Run all"
- They observe and take notes
- Better than canceling the lesson!