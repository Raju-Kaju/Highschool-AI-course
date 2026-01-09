# House Prices Competition - Day 1
## Introduction to Regression & Problem Setup
### Detailed Teacher Guide with Script & Student Materials

---

## Lesson Overview
**Duration:** 50 minutes  
**Objective:** Students understand regression vs classification and set up for House Prices competition  
**Key Outcome:** All students have Kaggle accounts and understand predicting continuous values

---

## Materials Checklist
- [ ] Projector/screen for demonstrations
- [ ] Student computers with internet access
- [ ] Printed handout: "Student Activity Guide - Day 1"
- [ ] Whiteboard/markers for key concepts
- [ ] Visual: Classification vs Regression poster
- [ ] House images (optional - various home types/prices)

---

## Pre-Class Setup

**Prepare Visual Aid on Board:**
```
CLASSIFICATION          vs          REGRESSION
(Categories)                        (Numbers)

Survived: Yes/No                    Price: $150,234
Email: Spam/Not Spam                Temperature: 72.5°F
Animal: Cat/Dog/Bird                Height: 5.8 feet
Grade: A/B/C/D/F                    Weight: 165.3 lbs
```

---

## MINUTE-BY-MINUTE LESSON PLAN

### Minutes 0-2: Welcome & Context

**TEACHER SCRIPT:**

"Welcome back, data scientists! Remember when we predicted survival on the Titanic? We were answering a yes/no question: Did they survive? 0 or 1. Two options.

Today, we're starting something different - and more complex. We're going to predict HOUSE PRICES.

Quick question: If I asked you to predict the price of a house, could you answer with just yes or no?"

[PAUSE FOR RESPONSES]

"Exactly! You'd need to give me an actual number - $150,000? $300,000? $500,000?

This is the difference between **classification** (categories) and **regression** (numbers). Today we enter the world of regression!"

---

### Minutes 2-12: HOOK ACTIVITY - "Price This House"

**TEACHER SCRIPT:**

"Let's do a quick exercise. I'm going to describe three houses. I want you to estimate the price of each one. Write down your guesses.

**House A:**
- 3 bedrooms, 2 bathrooms
- 1,500 square feet
- Built in 1950, needs work
- Bad neighborhood
- No garage

**House B:**
- 4 bedrooms, 3 bathrooms
- 2,500 square feet
- Built in 2010, excellent condition
- Great neighborhood with good schools
- 2-car garage

**House C:**
- 5 bedrooms, 4 bathrooms
- 4,000 square feet
- Built in 2020, luxury finishes
- Waterfront property
- 3-car garage with finished basement

Take 2 minutes - write down a price for each house."

[GIVE 2 MINUTES]

**TEACHER SCRIPT (after collecting estimates):**

"Let's see what you guessed. Raise your hand:

House A - Who said under $100,000? $100-150k? Over $150k?
House B - Who said under $200k? $200-300k? $300-400k? Over $400k?
House C - Who said under $400k? $400-600k? Over $600k?

[WRITE RANGES ON BOARD]

Interesting! You all used the FEATURES of the house to make your predictions:
- Size (square feet)
- Number of bedrooms/bathrooms
- Age and condition
- Location
- Amenities (garage, basement)

You just did what a machine learning regression model does! You looked at features and predicted a NUMBER - the price.

But here's the question: What if we had DATA from 1,460 houses with all these features AND their actual selling prices? Could a computer learn the patterns better than we can guess?

That's exactly what we're going to do!"

[WRITE ON BOARD:]
**Today's Goal: Understand regression & set up to predict house prices in Ames, Iowa**

---

### Minutes 12-25: DIRECT INSTRUCTION - Classification vs Regression

**TEACHER SCRIPT:**

"Let's make this crystal clear. There are two main types of supervised machine learning:

[POINT TO BOARD VISUAL]

**CLASSIFICATION: Predicting Categories**
- Question: 'Which category does this belong to?'
- Output: A label or class
- Examples: Spam/Not Spam, Survived/Died, Cat/Dog
- Titanic was classification: Survived (yes=1, no=0)

**REGRESSION: Predicting Numbers**
- Question: 'What is the exact value?'
- Output: A continuous number
- Examples: Price, temperature, age, distance
- House Prices is regression: Price ($34,900 to $755,000)

[DRAW ON BOARD:]

```
CLASSIFICATION               REGRESSION
Discrete buckets            Continuous line
[  A  ] [  B  ] [  C  ]    |————————————————>
                            0   $100k  $200k  $300k

Titanic Examples:           House Price Examples:
Survived: 1                 Price: $215,000
Survived: 0                 Price: $189,500
Survived: 1                 Price: $342,000
```

Let me give you more examples to understand this:

**CLASSIFICATION Questions:**
- Will it rain tomorrow? (Yes/No)
- Is this email spam? (Spam/Not Spam)
- What grade will I get? (A/B/C/D/F)
- Is this a picture of a cat or dog? (Cat/Dog)

**REGRESSION Questions:**
- What temperature will it be tomorrow? (73.4°F)
- How many inches of rain will fall? (2.3 inches)
- What score will I get on the test? (87.5)
- How tall is this person? (5.7 feet)

Notice the difference? 
- Classification → Pick from categories
- Regression → Predict exact number

[PAUSE FOR QUESTIONS]

**Why does this matter?**

1. **Different models**: Classification uses models like Logistic Regression, Decision Trees for categories. Regression uses Linear Regression, Ridge Regression for numbers.

2. **Different metrics**: 
   - Classification: Accuracy (% correct), Precision, Recall
   - Regression: RMSE (how far off are we?), MAE, R²

3. **Different interpretations**:
   - Classification: 'I was right or wrong'
   - Regression: 'I was off by $15,000' or 'I was close!'

**Quick Check:**
Let me ask you - which type of problem is this?

1. Predicting tomorrow's stock price → [REGRESSION!]
2. Deciding if a student will pass or fail → [CLASSIFICATION!]
3. Estimating how many hours you'll study → [REGRESSION!]
4. Identifying if a tumor is malignant or benign → [CLASSIFICATION!]

Great! Now let's talk about OUR specific problem."

---

### Minutes 25-35: THE HOUSE PRICES PROBLEM

**TEACHER SCRIPT:**

"We're going to work with real data from Ames, Iowa - a small college town. 

[SHOW MAP OR DESCRIBE:]
Ames is home to Iowa State University, population about 65,000. It's a typical American town with various neighborhoods, from student housing to upscale residential areas.

**Our Dataset:**
- **1,460 houses** in our training set (we know the prices)
- **1,459 houses** in our test set (we need to predict prices)
- **79 features** describing each house
- **Target variable:** SalePrice (what we're predicting)

**Price Range:**
- Minimum: $34,900 (probably a small, old house in bad condition)
- Maximum: $755,000 (probably a large, new house in great location)
- Average: around $180,000

Let me show you some of the 79 features:

[WRITE ON BOARD - ORGANIZE BY CATEGORY:]

**SIZE FEATURES:**
- LotArea: Lot size in square feet
- GrLivArea: Above ground living area
- TotalBsmtSF: Total basement area
- GarageArea: Size of garage

**QUALITY FEATURES:**
- OverallQual: Overall material and finish (1-10 scale)
- OverallCond: Overall condition (1-10 scale)
- KitchenQual: Kitchen quality (Excellent, Good, Average, Fair, Poor)
- ExterQual: Exterior material quality

**AGE FEATURES:**
- YearBuilt: Original construction year
- YearRemodAdd: Remodel date
- GarageYrBlt: Year garage was built

**LOCATION FEATURES:**
- Neighborhood: Physical location in Ames (25 neighborhoods!)
- MSZoning: General zoning classification

**COUNT FEATURES:**
- BedroomAbvGr: Number of bedrooms above ground
- FullBath: Number of full bathrooms
- HalfBath: Number of half bathrooms
- TotRmsAbvGrd: Total rooms above ground
- GarageCars: Size of garage in car capacity

**AMENITY FEATURES:**
- Fireplaces: Number of fireplaces
- PoolArea: Pool area in square feet
- Fence: Fence quality
- PavedDrive: Paved driveway

[PAUSE]

That's just some of them! 79 features total. That's a LOT of information.

**Our Goal:**
Use these 79 features to predict SalePrice as accurately as possible.

**How will we be evaluated?**
Kaggle will use **RMSE: Root Mean Squared Error**

Let me explain RMSE simply:

[WRITE ON BOARD:]
```
If actual price = $200,000
And you predict = $210,000
Your error = $10,000

RMSE basically measures: 'On average, how many dollars off are your predictions?'

Lower RMSE = Better predictions!

Good RMSE for this competition: around $25,000-$30,000
Great RMSE: under $20,000
Top scores: around $12,000
```

So if you have RMSE of $20,000, you're predicting house prices within about $20,000 on average. Not bad!

Think about it - house prices range from $34,900 to $755,000. Being off by $20,000 is pretty good!

**Quick Discussion:**
Which features do YOU think will be most important for predicting price?

[TAKE 3-4 STUDENT RESPONSES]

Common good answers:
- Size (bigger = more expensive)
- Location (neighborhood matters!)
- Quality (better condition = higher price)
- Age (newer often = more expensive)

We'll find out if you're right when we explore the data!"

---

### Minutes 35-47: GUIDED PRACTICE - Setting Up Kaggle

**TEACHER SCRIPT:**

"Alright, everyone open your laptops. Time to get set up!

If you already have a Kaggle account from Titanic, great! If not, we'll create one now.

I'm going to do this step-by-step. Follow along with me."

**STEP 1: Navigate to the Competition**

"Go to Kaggle.com and search for 'House Prices Advanced Regression'

Or go directly to: kaggle.com/c/house-prices-advanced-regression-techniques

You should see the competition page. Click on it."

[SHOW ON SCREEN]

**STEP 2: Join the Competition**

"Click the blue 'Join Competition' button.

If you don't have an account, you'll need to create one first - just like we did for Titanic.

Accept the rules and click 'I Understand and Accept'

[WAIT FOR STUDENTS]

Raise your hand when you see 'You are now competing!'"

[WAIT FOR MAJORITY]

**STEP 3: Explore the Overview**

"Great! Now let's explore what's on this page. Click through these tabs:

**Overview Tab:** Read the competition goal. It says:
'Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.'

**Data Tab:** This is where you'll download the data. You should see:
- train.csv (your training data - 1,460 houses WITH prices)
- test.csv (your test data - 1,459 houses WITHOUT prices)
- data_description.txt (explains all 79 features)
- sample_submission.csv (shows submission format)

**Code Tab:** Where people share their solutions (we'll look at this later)

**Discussion Tab:** Where you can ask questions and get help

**Leaderboard Tab:** Where rankings appear (we'll be on here soon!)"

[GIVE 3-4 MINUTES TO EXPLORE]

**STEP 4: Download Data Description**

"Click on the Data tab. Download 'data_description.txt' - this is like a dictionary that explains every feature.

For example, it tells you:
- MSSubClass: The building class (20 = 1-story, 60 = 2-story, etc.)
- LotShape: Shape of property (Reg=Regular, IR1=Slightly irregular, etc.)
- Neighborhood: Physical location (25 different neighborhoods in Ames)

You'll reference this file constantly when working with the data.

Don't download the actual data files yet - we'll do that tomorrow in class together."

---

### Minutes 47-50: CLOSURE & HOMEWORK

**TEACHER SCRIPT:**

"Excellent work today! Let's recap what we learned:

[POINT TO BOARD]

**Key Concepts:**
1. ✓ **Classification** = predicting categories (Titanic)
2. ✓ **Regression** = predicting numbers (House Prices)
3. ✓ **RMSE** = our error metric (lower is better)
4. ✓ **79 features** = lots of information about each house
5. ✓ **Goal** = predict SalePrice as accurately as possible

**Tomorrow's Preview:**
Tomorrow we'll:
- Download the actual data
- Explore the 1,460 houses
- Create visualizations showing which features affect price
- Calculate correlations
- See which features matter most

This is going to be exciting! With 79 features, we'll discover lots of patterns.

**Homework:**
Take out your handout. There are three tasks:

1. **Read the Competition Overview** on Kaggle (take notes)

2. **Think about features**: Which 5 features do you think will be MOST important for predicting house price? Write them down and explain why.

3. **Research question**: What is R² (R-squared)? This is another metric for regression we'll use. Write 2-3 sentences explaining what it measures.

**Bonus:** Browse the data_description.txt file and find 3 features that surprise you or that you didn't know existed in house data.

Bring your completed handout tomorrow!

Any questions before we finish?"

[ANSWER QUESTIONS]

"Great! Tomorrow we dive into the data. See you then!"

---

## STUDENT HANDOUT - DAY 1

### Student Name: _________________ Date: _____________

## Part 1: Classification vs Regression

**Fill in the blanks:**

**Classification** is when we predict _________________ like yes/no or cat/dog.

**Regression** is when we predict _________________ like prices or temperatures.

**Examples - Write CL for Classification or RG for Regression:**

_____ Predicting whether it will rain tomorrow (yes/no)
_____ Predicting how much rain will fall (in inches)
_____ Determining if an email is spam
_____ Predicting a student's test score (0-100)
_____ Classifying an image as cat, dog, or bird
_____ Estimating a person's age from their photo
_____ Predicting if a customer will buy (yes/no)
_____ Predicting how much a customer will spend ($)

---

## Part 2: The House Prices Problem

**Answer these questions:**

1. How many houses are in our training dataset? __________

2. How many features describe each house? __________

3. What is the target variable we're trying to predict? __________

4. What does RMSE stand for?

   R: __________
   M: __________
   S: __________
   E: __________

5. If a house actually sells for $200,000 and you predict $180,000, what is your error?
   
   Error = $__________

6. True or False: Lower RMSE means better predictions. __________

---

## Part 3: Feature Categories

**Match the feature to its category:**

| Feature | Category |
|---------|----------|
| GrLivArea | A. Location |
| Neighborhood | B. Size |
| OverallQual | C. Quality |
| YearBuilt | D. Age |
| BedroomAbvGr | E. Count |

---

## Part 4: Kaggle Setup Checklist

- [ ] Joined House Prices competition
- [ ] Explored Overview tab
- [ ] Found Data tab
- [ ] Downloaded data_description.txt
- [ ] Can see train.csv and test.csv listed

**My Kaggle username:** _____________________________

---

## HOMEWORK - Due Tomorrow

### Task 1: Competition Overview Notes

Read the Overview tab on Kaggle. Write 3 key facts you learned:

1. _________________________________________________________________

2. _________________________________________________________________

3. _________________________________________________________________

---

### Task 2: Feature Importance Predictions

**Which 5 features do you think will be MOST important for predicting house price?**

List them and explain WHY for each:

**Feature 1:** _______________________

Why: _______________________________________________________________

**Feature 2:** _______________________

Why: _______________________________________________________________

**Feature 3:** _______________________

Why: _______________________________________________________________

**Feature 4:** _______________________

Why: _______________________________________________________________

**Feature 5:** _______________________

Why: _______________________________________________________________

---

### Task 3: Research - What is R²?

**Research R² (R-squared) and explain what it measures (2-3 sentences):**

_____________________________________________________________________

_____________________________________________________________________

_____________________________________________________________________

**What does R² = 1.0 mean?**

_____________________________________________________________________

**What does R² = 0.0 mean?**

_____________________________________________________________________

---

### Bonus Challenge (Extra Credit)

**Browse data_description.txt and find 3 surprising features:**

1. Feature: ___________________
   
   Why surprising: ______________________________________________

2. Feature: ___________________
   
   Why surprising: ______________________________________________

3. Feature: ___________________
   
   Why surprising: ______________________________________________

---

## TEACHER ANSWER KEY

### Part 1: Classification vs Regression
- Classification predicts **categories**
- Regression predicts **numbers**

Examples:
- CL: Rain yes/no
- RG: Rain amount
- CL: Spam detection
- RG: Test score
- CL: Image classification
- RG: Age estimation
- CL: Will buy yes/no
- RG: Amount spent

### Part 2: House Prices Problem
1. 1,460 houses
2. 79 features
3. SalePrice
4. Root Mean Squared Error
5. $20,000
6. True

### Part 3: Matching
- GrLivArea = B (Size)
- Neighborhood = A (Location)
- OverallQual = C (Quality)
- YearBuilt = D (Age)
- BedroomAbvGr = E (Count)

### Part 4: R² Research
R² measures how well the model fits the data, ranging from 0 to 1.
- R² = 1.0 means perfect predictions (model explains 100% of variance)
- R² = 0.0 means model is no better than guessing the average

### Bonus: Surprising Features (Examples)
Students might find surprising:
- LotShape (irregular lot shapes)
- RoofMatl (different roof materials)
- BsmtExposure (walkout basement)
- Functional (home functionality rating)
- MiscFeature (elevator, tennis court)

---

## TEACHER REFLECTION NOTES

### What Went Well:
- [ ] Students understood classification vs regression difference
- [ ] "Price This House" activity engaged students
- [ ] All students successfully joined competition
- [ ] Good discussion about important features

### Challenges:
- [ ] Students confused about: ______________________
- [ ] Technical issues: ____________________________
- [ ] Pacing: too fast / too slow
- [ ] Students needing extra support: _______________

### Adjustments for Tomorrow:
- [ ] Review concepts: ____________________________
- [ ] Prepare: ___________________________________
- [ ] Download datasets in advance as backup
- [ ] Print correlation heatmap examples

### Notes:
_____________________________________________________________

_____________________________________________________________

_____________________________________________________________