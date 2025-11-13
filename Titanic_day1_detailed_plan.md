# Day 1: Introduction to Machine Learning & The Titanic Problem
## Detailed Teacher Guide with Script & Student Materials

---

## Lesson Overview
**Duration:** 50 minutes  
**Objective:** Students understand what ML is and set up for the Titanic competition  
**Key Outcome:** All students have Kaggle accounts and understand the problem

---

## Materials Checklist
- [ ] Projector/screen for demonstrations
- [ ] Student computers with internet access
- [ ] Printed handout: "Student Activity Guide - Day 1" (included below)
- [ ] Whiteboard/markers
- [ ] Titanic image for hook (optional)
- [ ] Post-it notes for brainstorming activity

---

## MINUTE-BY-MINUTE LESSON PLAN

### Minutes 0-2: Settling In & Attendance
**What Students See:** Slide with Titanic image and question: *"Could AI have predicted who would survive the Titanic?"*

**Teacher Actions:**
- Take attendance
- Have students sit at computers (but don't open anything yet)
- Display opening slide

---

### Minutes 2-12: HOOK ACTIVITY - "Who Would Survive?"

**TEACHER SCRIPT:**

"Good morning! Today we're starting something completely different - we're going to become machine learning engineers. But before I tell you what that means, I want you to do a quick activity.

[SHOW TITANIC IMAGE]

You all probably know about the Titanic - the 'unsinkable' ship that struck an iceberg on April 15, 1912 and sank. Out of 2,224 passengers and crew, only 722 survived. That's about 32% - roughly 1 in 3 people.

Here's the question: **If you were trying to predict who would survive, what information about a passenger would help you make that prediction?**

I'm going to give you 3 minutes to work with your partner. Write down as many factors as you can think of. Think like a detective - what clues would matter?"

**Teacher Actions:**
- Set 3-minute timer
- Circulate while students brainstorm
- Listen for good ideas to call on later

**After 3 Minutes:**

"Alright, let's hear what you came up with. Who wants to share one factor?"

[CALL ON STUDENTS - write responses on board]

**Expected student responses:**
- Gender (women and children first!)
- Age
- Ticket class / wealth
- Location on ship
- Cabin number
- Where they got on the ship
- Job/occupation
- Who they were with

**TEACHER SCRIPT (after collecting ~8-10 ideas):**

"Excellent! You all just did what data scientists do - you identified **features** that might predict an **outcome**. In data science language:
- **Features** = the information we know about someone (age, gender, ticket class)
- **Outcome** = what we're trying to predict (survived or died)

And guess what? You're about to teach a computer to do exactly what you just did - but with actual data from the real Titanic. Welcome to machine learning!"

---

### Minutes 12-32: DIRECT INSTRUCTION - What is Machine Learning?

**TEACHER SCRIPT:**

"So what exactly IS machine learning? Let me show you.

[ADVANCE TO 'WHAT IS ML?' SLIDE]

**Traditional Programming:**
You tell the computer exact rules: 
- IF it's raining THEN bring umbrella
- IF temperature > 80 THEN turn on AC

You write every single rule.

**Machine Learning:**
You give the computer examples, and IT figures out the rules:
- Show it 1,000 pictures of cats and dogs
- It learns what makes a cat a cat
- Now it can recognize cats it's never seen before

[DRAW SIMPLE DIAGRAM ON BOARD:]
```
TRADITIONAL:          MACHINE LEARNING:
Rules → Program       Data + Answers → Computer learns rules
```

Real-world examples you use every day:
- **Netflix recommendations** - learns what you like
- **Spotify playlists** - learns your music taste  
- **Face unlock** on your phone - learned your face
- **Spam filters** - learned what spam looks like
- **Google Translate** - learned from millions of translations

[PAUSE FOR QUESTIONS]

Now, there are two main types of machine learning:

**1. Supervised Learning** (what we're doing):
- We give the computer data WITH the answers
- Like flash cards - question on front, answer on back
- Computer learns the pattern
- Example: We tell it 'this person survived, this person died'

**2. Unsupervised Learning** (more advanced):
- We give the computer data WITHOUT answers
- It finds patterns on its own
- Example: Grouping customers by shopping behavior

We're doing **supervised learning** - we have historical data that tells us who survived and who didn't. The computer will learn from this.

[ADVANCE TO TITANIC PROBLEM SLIDE]

**Our Specific Problem: The Titanic Dataset**

Here's what we know about each passenger:

[WRITE ON BOARD AS YOU EXPLAIN:]

**PassengerId** - Just a number (passenger 1, 2, 3...)  
**Survived** - 0 = died, 1 = survived (THIS IS WHAT WE'RE PREDICTING!)  
**Pclass** - Ticket class: 1st, 2nd, or 3rd class  
**Name** - Passenger's name  
**Sex** - Male or Female  
**Age** - Age in years  
**SibSp** - Number of siblings/spouses on board  
**Parch** - Number of parents/children on board  
**Ticket** - Ticket number  
**Fare** - How much they paid (in 1912 British pounds)  
**Cabin** - Cabin number  
**Embarked** - Where they got on: C = Cherbourg, Q = Queenstown, S = Southampton

[POINT TO FEATURES ON BOARD]

Look familiar? These are exactly the kinds of things you brainstormed! The only difference is we have REAL data - 891 passengers with all this information.

**Our Goal:**
- Learn patterns from passengers we know about
- Predict survival for passengers we DON'T know about
- Submit our predictions to Kaggle (a real data science competition website)
- See how accurate we are!

**Quick Check for Understanding:**
Let me ask you three questions - just shout out answers:

1. Is this supervised or unsupervised learning? [supervised!]
2. What's our outcome variable - what are we predicting? [survived!]
3. Name one feature we'll use. [any feature they mention]

Great! Now let's get you set up so you can actually DO this."

---

### Minutes 32-47: GUIDED PRACTICE - Setting Up Kaggle

**TEACHER SCRIPT:**

"Alright, everyone open your laptops and go to your web browser. We're going to create accounts on Kaggle - this is where real data scientists compete and share work.

I'm going to do this step-by-step on my screen. Don't go ahead - follow along with me. Raise your hand if you get stuck."

**STEP-BY-STEP WALKTHROUGH:**

[PROJECT YOUR SCREEN]

**Step 1: Go to Kaggle.com**

"Type in K-A-G-G-L-E dot com. You should see a homepage with lots of data science content. 

Click the 'Register' button in the top right corner."

[PAUSE - wait for students to catch up]

**Step 2: Create Account**

"You can sign up with:
- Your Google account (easiest!)
- Email address

I recommend using your school email if you have one. Choose whichever method you prefer.

[Give 2 minutes for account creation]

If you get stuck, raise your hand and I'll come help."

[CIRCULATE AND HELP STUDENTS]

**Step 3: Join the Competition**

"Once you're logged in, look at the top of the page. There's a search bar. Type 'Titanic' and hit enter.

You should see 'Titanic - Machine Learning from Disaster' as one of the first results. Click on it.

[PAUSE]

You should now see a page with lots of information. There are several tabs:
- **Overview** - explains the competition
- **Data** - where you download the dataset
- **Code** - where people share their solutions
- **Discussion** - where people help each other
- **Leaderboard** - shows rankings

Look for a button that says 'Join Competition' - it might be on the right side or at the top. Click it.

[PAUSE]

You might need to accept some rules. That's fine - just click 'I Understand and Accept.'

[GIVE 1-2 MINUTES]

Raise your hand when you see the message 'You are now competing!'"

[WAIT UNTIL MOST STUDENTS HAVE JOINED]

**Step 4: Explore the Overview Page**

"Great! Now everyone scroll down on the Overview page. Let's look at what's here:

- **Goal of Competition**: Predict survival
- **Metric**: Accuracy (what percentage you get right)
- **Dataset**: Two files - train.csv and test.csv

Read this section: [READ ALOUD THE GOAL]

'The sinking of the Titanic is one of the most infamous shipwrecks in history... your job is to predict if a passenger survived the sinking...'

Now click on the 'Data' tab at the top."

**Step 5: Understand the Data**

"This page shows you all the columns in the dataset. We talked about these already!

See 'train.csv' and 'test.csv'?

- **train.csv** = 891 passengers where we KNOW if they survived (this is how our computer will learn!)
- **test.csv** = 418 passengers where we DON'T know (we have to predict!)

We'll download these next class, but for now, just know where they are.

Scroll down and read about what each column means. This is your data dictionary - like a glossary."

---

### Minutes 47-50: CLOSURE & HOMEWORK

**TEACHER SCRIPT:**

"Excellent work today! Let's recap what we learned:

[POINT TO BOARD WITH KEY TERMS]

1. **Machine Learning** - teaching computers to learn from data
2. **Supervised Learning** - learning from labeled examples
3. **Features** - information we have (age, class, sex...)
4. **Outcome** - what we're predicting (survived yes/no)
5. **Kaggle** - where we'll compete!

Tomorrow, we're going to actually download this data and start exploring it. You'll see what those 891 passengers really look like as data, and we'll create some cool visualizations.

**For homework:** 
Take out the handout I'm giving you. There are just three quick things:

1. Read the short article about the Titanic (2 pages)
2. Answer the 5 questions on the back
3. Make a prediction: Which feature do you think will be MOST important for predicting survival? Write 2-3 sentences explaining why.

Bring this with you tomorrow. Any questions?"

[DISTRIBUTE HOMEWORK HANDOUT]

"Great first day! See you tomorrow!"

---

## STUDENT HANDOUT - DAY 1 ACTIVITY GUIDE

### Student Name: _________________ Date: _____________

## Part 1: Brainstorming Activity (in class)

**Question:** What factors might help predict who survived the Titanic?

Work with your partner to list as many factors as you can:

1. _____________________________
2. _____________________________
3. _____________________________
4. _____________________________
5. _____________________________
6. _____________________________
7. _____________________________
8. _____________________________

---

## Part 2: Key Terms (fill in during lesson)

**Machine Learning:** _________________________________________________

_____________________________________________________________________

**Supervised Learning:** ______________________________________________

_____________________________________________________________________

**Features:** ________________________________________________________

**Outcome:** ________________________________________________________

**Our specific outcome for Titanic:** __________________________________

---

## Part 3: Titanic Dataset Features

Match the feature to its description:

| Feature | Description |
|---------|-------------|
| Pclass | A. How much the ticket cost |
| Sex | B. Where they boarded the ship |
| Age | C. 1st, 2nd, or 3rd class ticket |
| Fare | D. Male or Female |
| Embarked | E. Age in years |

---

## Part 4: Kaggle Setup Checklist

- [ ] Created Kaggle account
- [ ] Joined Titanic competition  
- [ ] Found the Data tab
- [ ] Can see train.csv and test.csv listed

**My Kaggle username:** _____________________________

---

## HOMEWORK - Due Tomorrow

### Reading: The Titanic Disaster (Read the article provided)

### Questions:

1. How many passengers were on the Titanic? _________________________

2. What was the "women and children first" policy? 
   
   _________________________________________________________________
   
   _________________________________________________________________

3. Which class (1st, 2nd, or 3rd) do you think had the highest survival rate? Why?
   
   _________________________________________________________________
   
   _________________________________________________________________

4. The Titanic sank at 2:20 AM. Why might the time of day matter for survival?
   
   _________________________________________________________________
   
   _________________________________________________________________

5. Based on what you learned, name THREE factors that likely affected survival:
   
   a) _______________________________________________________________
   
   b) _______________________________________________________________
   
   c) _______________________________________________________________

### Prediction Task:

Which ONE feature do you think will be MOST important for predicting survival? (Choose from: Pclass, Sex, Age, Fare, Embarked, SibSp, Parch)

**My prediction:** _____________________

**Why I think this (2-3 sentences):**

_____________________________________________________________________

_____________________________________________________________________

_____________________________________________________________________

---

## TEACHER ANSWER KEY

### Part 3: Matching
- Pclass = C (1st, 2nd, or 3rd class)
- Sex = D (Male or Female)
- Age = E (Age in years)
- Fare = A (How much ticket cost)
- Embarked = B (Where they boarded)

### Homework Questions:
1. 2,224 passengers (or ~2,200)
2. During evacuations, women and children were given priority for lifeboats
3. 1st class (had better access to lifeboats, higher up on ship)
4. People were sleeping, harder to react quickly, less time to reach deck
5. Accept: Class, Gender, Age, Location on ship, Lifeboat access, etc.

### Common Student Predictions:
- **Sex** (most common - women and children first policy)
- **Pclass** (wealth = better access)
- **Age** (children saved first)
All are valid with reasoning!

---

## STARTER CODE FOR STUDENTS (Preview for Day 2)

*Note: Share this at the end of class or via email so students can preview*

```python
# Day 1 Preview: Our First Look at Data
# We'll run this code tomorrow in class!

import pandas as pd

# This loads our Titanic data
train = pd.read_csv('train.csv')

# See the first few passengers
print(train.head())

# How many passengers do we have?
print(f"Total passengers: {len(train)}")

# What percentage survived?
survival_rate = train['Survived'].mean() * 100
print(f"Survival rate: {survival_rate:.1f}%")
```

**What this code does:**
1. Imports pandas (a tool for working with data)
2. Loads the training data
3. Shows us the first 5 passengers
4. Tells us how many total passengers we have
5. Calculates what % survived

**Don't worry if this looks confusing!** We'll go through it line by line tomorrow. This is just a preview!

---

## TEACHER REFLECTION NOTES

### What Went Well:
- [ ] Students engaged with hook activity
- [ ] All students successfully created Kaggle accounts
- [ ] Students understood supervised vs unsupervised learning
- [ ] Good questions asked during instruction

### Challenges:
- [ ] Technical issues (internet, login problems)
- [ ] Students confused about: ______________________
- [ ] Pacing issues: too fast / too slow
- [ ] Students who need extra support: _______________

### Adjustments for Tomorrow:
- [ ] Review concepts: _____________________________
- [ ] Bring: ______________________________________
- [ ] Pre-download datasets in case of internet issues
- [ ] Pair struggling students with tech-savvy partners

### Notes:
_____________________________________________________________

_____________________________________________________________

_____________________________________________________________