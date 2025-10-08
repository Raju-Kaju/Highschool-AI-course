# Module 1 Detailed Lesson Plan: "Is This Song a Hit?"
## Introduction to Classification

**Duration:** 60 minutes  
**Students:** High School (Ages 14-18)  
**Tools:** Google Colab (or Jupyter Notebook)

---

## Pre-Class Setup (5 minutes before class)
```
- Ensure all students have Google accounts
- Share Google Colab notebook link
- Have backup dataset files ready
- Test internet connection and projector
```

---

## Opening Hook (5 minutes)

### Teacher Script:
"Good morning everyone! Before we start, I want everyone to take out their phones - yes, you heard that right! Open your music app and look at your most recent playlist. 

*[Wait for students to do this]*

Now, here's a million-dollar question: How does Spotify know which songs to recommend to you? How does it predict which new songs will become hits before they even go viral on TikTok?

*[Pause for responses - let students share ideas]*

Today, you're going to build your own AI system that can predict whether a song will be a hit or a flop - just like the algorithms that power Spotify, Apple Music, and TikTok's music discovery. And the best part? You'll do it with just a few lines of code.

By the end of this hour, you'll have a working AI model that can analyze any song and tell you if it has what it takes to be the next big hit. Ready to become music industry AI consultants?"

---

## Section 1: Data Exploration (15 minutes)

### Teacher Script & Code Walkthrough

"First, let's see what data we're working with. In AI, we call this 'Exploratory Data Analysis' - it's like being a detective looking for clues in the data."

```python
# Let's import our tools - think of these as our AI toolkit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load our song data
# This dataset contains real songs from Spotify with their audio features
songs_data = pd.read_csv('spotify_songs_dataset.csv')

# Let's see what we're working with
print("Our dataset has", len(songs_data), "songs!")
songs_data.head()
```

### Teacher Explanation:
"Each row represents one song, and each column is what we call a 'feature' - these are the characteristics that describe each song. Let's understand what these features mean."

```python
# Let's explore our features
print("Columns in our dataset:")
for col in songs_data.columns:
    print(f"- {col}")
    
# Display basic statistics
songs_data.describe()
```

### Interactive Discussion (5 minutes):
**Teacher:** "Looking at these features, let's decode what they mean for music:

- **Danceability (0.0-1.0)**: How suitable is the track for dancing? 0 = funeral march, 1 = ultimate dance floor banger
- **Energy (0.0-1.0)**: How intense and powerful does the track feel? 0 = meditation music, 1 = death metal
- **Loudness (-60 to 0 dB)**: How loud is the track? -60 = whisper quiet, 0 = maximum volume
- **Tempo (BPM)**: Beats per minute - slow ballad vs fast electronic
- **Popularity (0-100)**: This is our TARGET - what we want to predict! 0 = nobody listens, 100 = global phenomenon

Can anyone guess which songs in your playlist might have high danceability? High energy?"

### Visual Exploration:
```python
# Let's visualize our data to find patterns
plt.figure(figsize=(12, 4))

# Plot 1: Distribution of popularity scores
plt.subplot(1, 3, 1)
plt.hist(songs_data['popularity'], bins=20, alpha=0.7, color='skyblue')
plt.title('Song Popularity Distribution')
plt.xlabel('Popularity Score')
plt.ylabel('Number of Songs')

# Plot 2: Danceability vs Popularity
plt.subplot(1, 3, 2)
plt.scatter(songs_data['danceability'], songs_data['popularity'], alpha=0.5, color='green')
plt.title('Danceability vs Popularity')
plt.xlabel('Danceability')
plt.ylabel('Popularity')

# Plot 3: Energy vs Popularity
plt.subplot(1, 3, 3)
plt.scatter(songs_data['energy'], songs_data['popularity'], alpha=0.5, color='red')
plt.title('Energy vs Popularity')
plt.xlabel('Energy')
plt.ylabel('Popularity')

plt.tight_layout()
plt.show()
```

**Teacher:** "What patterns do you notice? Do more danceable songs tend to be more popular? This is exactly how AI learns - by finding patterns in data!"

---

## Section 2: Creating Our Classification Problem (10 minutes)

### Teacher Script:
"Now here's where we transform this into a classification problem. Instead of predicting exact popularity scores (which is hard), we're going to predict whether a song will be a 'HIT' or 'FLOP'. This is called binary classification - we have two categories to choose from."

```python
# Let's define what makes a song a "hit"
# We'll say songs with popularity > 70 are hits, others are flops
def classify_song(popularity_score):
    if popularity_score > 70:
        return 'HIT'
    else:
        return 'FLOP'

# Apply this to our entire dataset
songs_data['hit_or_flop'] = songs_data['popularity'].apply(classify_song)

# Let's see how many hits vs flops we have
hit_flop_counts = songs_data['hit_or_flop'].value_counts()
print("Distribution of Hits vs Flops:")
print(hit_flop_counts)

# Visualize this
plt.figure(figsize=(6, 4))
hit_flop_counts.plot(kind='bar', color=['red', 'green'])
plt.title('Number of Hits vs Flops in Our Dataset')
plt.ylabel('Number of Songs')
plt.xticks(rotation=0)
plt.show()
```

### Concept Explanation:
**Teacher:** "This is a crucial concept in AI - we just converted a regression problem (predicting numbers) into a classification problem (predicting categories). 

Think about it: It's easier to say 'this song will be popular' than to say 'this song will have exactly 73.2 popularity points.' That's why classification is often more practical for real-world decisions.

In the music industry, executives don't need exact popularity scores - they need to know: Should we promote this song? Should we sign this artist? Hit or Flop answers those questions."

---

## Section 3: Preparing Our Data for AI (15 minutes)

### Teacher Script:
"Now we need to prepare our data for the AI algorithm. This involves two critical concepts that every AI engineer must understand."

```python
# Step 1: Select our features (inputs) and target (output)
# Features: what the AI uses to make predictions
# Target: what we want the AI to predict

feature_columns = ['danceability', 'energy', 'loudness', 'tempo']
X = songs_data[feature_columns]  # X is traditionally used for features
y = songs_data['hit_or_flop']    # y is traditionally used for targets

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFirst 5 feature rows:")
print(X.head())
print("\nFirst 5 target values:")
print(y.head())
```

### The Big Concept: Train vs Test Split

**Teacher:** "Now I'm going to teach you one of the most important concepts in AI. Raise your hand if you've ever studied for a test using practice problems."

*[Most students raise hands]*

"Exactly! You practice on homework problems, then take the test on NEW problems you haven't seen before. That's exactly how we train AI systems.

Here's why this matters: Imagine I gave you a math test, but the test had the exact same problems you practiced on. You might get 100%, but did you really learn math, or did you just memorize answers?

AI has the same problem. If we test our model on the same songs it learned from, it might just memorize the answers instead of learning the real patterns. So we always split our data into:

1. **Training Set**: Songs the AI learns from (like homework problems)
2. **Test Set**: Songs we use to see how well it really learned (like the actual test)

This prevents 'overfitting' - when AI just memorizes instead of learning patterns."

```python
from sklearn.model_selection import train_test_split

# Split our data: 80% for training, 20% for testing
# random_state=42 ensures we get the same split every time (for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% for testing
    random_state=42,  # Makes results reproducible
    stratify=y        # Ensures equal proportion of hits/flops in both sets
)

print("Training set size:", len(X_train), "songs")
print("Test set size:", len(X_test), "songs")

# Let's verify our split maintained the hit/flop ratio
print("\nTraining set distribution:")
print(y_train.value_counts())
print("\nTest set distribution:")
print(y_test.value_counts())
```

### Interactive Check:
**Teacher:** "Let's make sure everyone understands this crucial concept. Turn to the person next to you and explain in your own words why we split the data. Use the test analogy if it helps."

*[Give students 2 minutes to discuss]*

---

## Section 4: Training Our First AI Model (10 minutes)

### Teacher Script:
"Now for the exciting part - we're going to train our AI! We'll use something called a Decision Tree. Think of it as a series of yes/no questions that help us make a decision."

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create our AI model
# Think of this as creating a "blank brain" that's ready to learn
model = DecisionTreeClassifier(
    random_state=42,    # For consistent results
    max_depth=5         # Prevents the tree from getting too complex
)

print("Created our Decision Tree model!")

# Step 2: Train the model (this is where the AI learns!)
model.fit(X_train, y_train)
print("Model training complete! 🎉")

# Step 3: Make predictions on our test set
predictions = model.predict(X_test)

print("Predictions made for", len(predictions), "songs!")
print("First 10 predictions:", predictions[:10])
print("First 10 actual results:", y_test.head(10).values)
```

### Understanding What Just Happened:

**Teacher:** "In those three lines of code, something amazing just happened. Our AI:

1. **Analyzed** thousands of songs and their features
2. **Found patterns** like 'If danceability > 0.7 AND energy > 0.8, then it's probably a HIT'
3. **Built a decision tree** of these rules
4. **Applied these rules** to new songs it had never seen before

Let's see how well it did!"

```python
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Our AI got {accuracy:.2%} of predictions correct!")

# Detailed performance report
print("\nDetailed Performance Report:")
print(classification_report(y_test, predictions))

# Let's see some specific examples
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predictions,
    'Danceability': X_test['danceability'].values,
    'Energy': X_test['energy'].values,
    'Tempo': X_test['tempo'].values
})

print("\nSample predictions:")
print(results_df.head(10))
```

---

## Section 5: Testing Our Model & Real-World Application (10 minutes)

### Teacher Script:
"Now let's have some fun and test our model on songs you might actually know!"

```python
# Let's create some test songs manually
# Students can suggest songs they know!

test_songs = pd.DataFrame({
    'danceability': [0.9, 0.3, 0.8, 0.6],  # High, Low, High, Medium
    'energy': [0.8, 0.2, 0.9, 0.7],        # High, Low, High, Medium  
    'loudness': [-5, -15, -4, -8],          # Loud, Quiet, Loud, Medium
    'tempo': [128, 70, 140, 100]            # Fast, Slow, Very Fast, Medium
})

# Make predictions
song_predictions = model.predict(test_songs)

print("Predictions for our test songs:")
for i, prediction in enumerate(song_predictions):
    print(f"Song {i+1}: {prediction}")
    print(f"  - Danceability: {test_songs.iloc[i]['danceability']}")
    print(f"  - Energy: {test_songs.iloc[i]['energy']}")
    print(f"  - Tempo: {test_songs.iloc[i]['tempo']}")
    print()
```

### Interactive Activity:
**Teacher:** "Now let's test this on some real current songs! I want volunteers to look up the audio features of their favorite songs on Spotify and we'll see what our AI predicts!"

*[Have students volunteer their favorite songs, look up features on Spotify for Artists or similar tools, and test them]*

### Understanding Decision Trees:
```python
# Let's visualize how our AI makes decisions
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
tree.plot_tree(model, 
               feature_names=['danceability', 'energy', 'loudness', 'tempo'],
               class_names=['FLOP', 'HIT'],
               filled=True,
               max_depth=3)  # Show only top 3 levels for clarity
plt.title("Our AI's Decision Tree (Simplified View)")
plt.show()
```

**Teacher:** "This tree shows exactly how our AI makes decisions! Each box represents a question:

- The top box might ask: 'Is energy > 0.65?'
- If YES, go right; if NO, go left
- Keep following the branches until you reach a final decision: HIT or FLOP

This is why Decision Trees are so powerful - we can actually see HOW the AI thinks, unlike some 'black box' algorithms."

---

## Section 6: Reflection & Real-World Applications (5 minutes)

### Teacher Script:
"Let's step back and think about what we just accomplished. You've built a real AI system that music industry professionals use variations of every single day.

**Real-world applications of what you just learned:**

1. **Spotify's Discover Weekly**: Uses similar algorithms to recommend new music
2. **Record Labels**: Use AI to decide which artists to sign and which songs to promote
3. **Radio Stations**: Use AI to decide playlist ordering and song rotation
4. **TikTok**: Uses audio analysis to predict which sounds will go viral
5. **Music Producers**: Use AI to analyze trends and create songs with 'hit potential'

**Key Concepts You Mastered Today:**

✅ **Classification**: Predicting categories (Hit/Flop) instead of exact numbers  
✅ **Features vs Targets**: Understanding inputs vs outputs  
✅ **Train/Test Split**: Why we never test on data we trained on  
✅ **Decision Trees**: AI that makes human-interpretable decisions  
✅ **Model Evaluation**: Measuring how good our AI really is  

**Questions for Next Time:**
- What other music features might predict hits? (Lyrics sentiment? Artist popularity?)  
- How might this system be biased? (Toward certain genres? Demographics?)  
- What happens if music tastes change over time?

Your homework: Find one song you think should be a hit but isn't popular yet, and one song that's popular but you think shouldn't be. Bring the Spotify audio features next class and we'll test our model!"

---

## Complete Code Notebook

```python
# COMPLETE MODULE 1 CODE - SPOTIFY HIT PREDICTOR
# Copy this entire code into a new Google Colab notebook

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

# Step 2: Load and explore data
# Note: In a real class, provide the CSV file or use Spotify's API
songs_data = pd.read_csv('spotify_songs_dataset.csv')

print("Dataset Info:")
print(f"Number of songs: {len(songs_data)}")
print("\nColumns:", songs_data.columns.tolist())
print("\nFirst 5 rows:")
print(songs_data.head())

# Step 3: Data visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(songs_data['popularity'], bins=20, alpha=0.7, color='skyblue')
plt.title('Song Popularity Distribution')
plt.xlabel('Popularity Score')
plt.ylabel('Number of Songs')

plt.subplot(1, 3, 2)
plt.scatter(songs_data['danceability'], songs_data['popularity'], alpha=0.5)
plt.title('Danceability vs Popularity')
plt.xlabel('Danceability')
plt.ylabel('Popularity')

plt.subplot(1, 3, 3)
plt.scatter(songs_data['energy'], songs_data['popularity'], alpha=0.5)
plt.title('Energy vs Popularity')
plt.xlabel('Energy')
plt.ylabel('Popularity')

plt.tight_layout()
plt.show()

# Step 4: Create classification target
def classify_song(popularity_score):
    return 'HIT' if popularity_score > 70 else 'FLOP'

songs_data['hit_or_flop'] = songs_data['popularity'].apply(classify_song)
print("\nHit vs Flop distribution:")
print(songs_data['hit_or_flop'].value_counts())

# Step 5: Prepare features and target
feature_columns = ['danceability', 'energy', 'loudness', 'tempo']
X = songs_data[feature_columns]
y = songs_data['hit_or_flop']

# Step 6: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} songs")
print(f"Test set: {len(X_test)} songs")

# Step 7: Train the model
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)
print("\n✅ Model training complete!")

# Step 8: Make predictions and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n🎯 Model Accuracy: {accuracy:.2%}")
print("\nDetailed Performance:")
print(classification_report(y_test, predictions))

# Step 9: Test on custom songs
print("\n🎵 Testing on Custom Songs:")
test_songs = pd.DataFrame({
    'danceability': [0.9, 0.3, 0.8],
    'energy': [0.8, 0.2, 0.9],
    'loudness': [-5, -15, -4],
    'tempo': [128, 70, 140]
})

custom_predictions = model.predict(test_songs)
for i, pred in enumerate(custom_predictions):
    print(f"Song {i+1}: {pred}")

# Step 10: Visualize the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(model, 
               feature_names=feature_columns,
               class_names=['FLOP', 'HIT'],
               filled=True,
               max_depth=3)
plt.title("AI Decision Tree - How It Decides Hit vs Flop")
plt.show()

print("\n🎉 Congratulations! You've built your first AI music predictor!")
```

---

## Assessment Rubric

### Participation & Understanding (40 points)
- **Excellent (36-40)**: Active participation, asks thoughtful questions, helps classmates
- **Good (32-35)**: Regular participation, shows understanding of concepts
- **Satisfactory (28-31)**: Basic participation, follows along with activities
- **Needs Improvement (0-27)**: Limited participation, struggles with basic concepts

### Technical Execution (30 points)
- **Excellent (27-30)**: Successfully runs all code, makes modifications independently
- **Good (24-26)**: Runs most code successfully with minimal help
- **Satisfactory (21-23)**: Follows along but needs guidance
- **Needs Improvement (0-20)**: Struggles to execute basic code

### Conceptual Grasp (30 points)
- **Excellent (27-30)**: Can explain train/test split, classification, and decision trees clearly
- **Good (24-26)**: Understands most concepts with minor gaps
- **Satisfactory (21-23)**: Basic understanding of key concepts
- **Needs Improvement (0-20)**: Limited understanding of fundamental concepts

---

## Materials Needed
- Computers/tablets with internet access
- Google Colab accounts for all students
- Spotify song dataset (CSV file)
- Projector for code demonstrations
- Handout with key terms and code snippets

## Next Module Preview
"Next week, we'll take this to the next level with Netflix! We'll build a movie recommendation system that can predict what genre you'll love based on plot descriptions. We'll also learn why some AI algorithms work better than others - and you'll see your accuracy scores improve dramatically!"