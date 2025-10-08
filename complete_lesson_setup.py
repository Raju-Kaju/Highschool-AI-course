# ================================================
# COMPLETE MODULE 1: "Is This Song a Hit?"
# AI Classification for High School Students
# ================================================

# PART 1: SETUP AND DATA GENERATION
# Run this first to create the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import random

print("🎵 Welcome to AI Music Prediction!")
print("=" * 50)

# Generate the dataset (only run this once)
def create_spotify_dataset():
    """Create a realistic Spotify-style dataset for the lesson"""
    
    np.random.seed(42)  # For consistent results
    random.seed(42)
    
    # Sample artists (mix of real and fictional)
    artists = [
        "Taylor Swift", "Ed Sheeran", "The Weeknd", "Billie Eilish", "Drake", 
        "Ariana Grande", "Post Malone", "Olivia Rodrigo", "Harry Styles", "Dua Lipa",
        "Electric Sunrise", "Velvet Thunder", "Crystal Waters", "Phoenix Rising", 
        "Midnight Velocity", "Golden Hour", "Urban Legends", "Digital Hearts"
    ]
    
    # Generate song titles
    adjectives = ["Lost", "Broken", "Golden", "Electric", "Midnight", "Summer", 
                 "Dancing", "Flying", "Burning", "Falling", "Rising", "Shining"]
    nouns = ["Hearts", "Dreams", "Lights", "Stars", "Fire", "Rain", "Thunder", 
            "Roads", "Nights", "Days", "Love", "Time", "World", "Sky"]
    
    songs = []
    
    for i in range(800):  # Generate 800 songs
        # Create song title
        song_title = f"{random.choice(adjectives)} {random.choice(nouns)}"
        artist = random.choice(artists)
        
        # Generate realistic audio features with correlations
        # Create different "genre types" with typical characteristics
        genre = random.choice(['pop', 'hip-hop', 'rock', 'electronic', 'ballad'])
        
        if genre == 'pop':
            danceability = np.random.beta(3, 2) * 0.7 + 0.3
            energy = np.random.beta(2, 2) * 0.6 + 0.4
            tempo = np.random.normal(120, 15)
        elif genre == 'hip-hop':
            danceability = np.random.beta(4, 2) * 0.8 + 0.2
            energy = np.random.beta(3, 2) * 0.7 + 0.3
            tempo = np.random.normal(95, 20)
        elif genre == 'electronic':
            danceability = np.random.beta(4, 1) * 0.9 + 0.1
            energy = np.random.beta(3, 2) * 0.8 + 0.2
            tempo = np.random.normal(128, 20)
        elif genre == 'rock':
            danceability = np.random.beta(2, 3) * 0.6 + 0.2
            energy = np.random.beta(4, 1) * 0.8 + 0.2
            tempo = np.random.normal(140, 25)
        else:  # ballad
            danceability = np.random.beta(1, 4) * 0.4 + 0.1
            energy = np.random.beta(1, 3) * 0.3 + 0.1
            tempo = np.random.normal(70, 15)
        
        # Clip values to valid ranges
        danceability = np.clip(danceability, 0.0, 1.0)
        energy = np.clip(energy, 0.0, 1.0)
        tempo = np.clip(tempo, 60, 180)
        
        # Generate other features
        loudness = np.random.normal(-8, 4)
        loudness = np.clip(loudness, -20, 0)
        
        valence = np.random.beta(2, 2)  # Positivity/happiness
        
        # Generate popularity with some correlation to features
        popularity_score = (
            danceability * 30 +
            energy * 25 + 
            valence * 20 +
            np.random.normal(0, 15)  # Random factor
        )
        
        # Popular artists get boost
        if artist in ["Taylor Swift", "Drake", "The Weeknd", "Billie Eilish"]:
            popularity_score += 20
        
        popularity = np.clip(popularity_score, 0, 100)
        
        songs.append({
            'track_name': song_title,
            'artist_name': artist,
            'danceability': round(danceability, 3),
            'energy': round(energy, 3),
            'loudness': round(loudness, 1),
            'tempo': round(tempo, 1),
            'valence': round(valence, 3),
            'popularity': round(popularity, 1)
        })
    
    return pd.DataFrame(songs)

# Create the dataset
print("📊 Creating song dataset...")
songs_data = create_spotify_dataset()
print(f"✅ Created dataset with {len(songs_data)} songs!")

# ================================================
# PART 2: MAIN LESSON CODE
# ================================================

print("\n" + "="*50)
print("🔍 STEP 1: EXPLORING OUR MUSIC DATA")
print("="*50)

# Display basic information
print(f"Our dataset contains {len(songs_data)} songs")
print(f"Features we can analyze: {list(songs_data.columns)}")

print("\n📊 First few songs in our dataset:")
print(songs_data.head())

print("\n📈 Statistical summary:")
print(songs_data[['danceability', 'energy', 'loudness', 'tempo', 'popularity']].describe())

# ================================================
# STEP 2: VISUALIZING THE DATA
# ================================================

print("\n" + "="*50)
print("📊 STEP 2: VISUALIZING PATTERNS IN MUSIC")
print("="*50)

plt.figure(figsize=(15, 10))

# Plot 1: Popularity distribution
plt.subplot(2, 3, 1)
plt.hist(songs_data['popularity'], bins=25, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('How Popular Are Songs?', fontsize=14, fontweight='bold')
plt.xlabel('Popularity Score (0-100)')
plt.ylabel('Number of Songs')
plt.grid(True, alpha=0.3)

# Plot 2: Danceability vs Popularity
plt.subplot(2, 3, 2)
plt.scatter(songs_data['danceability'], songs_data['popularity'], alpha=0.6, color='green')
plt.title('Danceability vs Popularity', fontsize=14, fontweight='bold')
plt.xlabel('Danceability (0=not danceable, 1=very danceable)')
plt.ylabel('Popularity Score')
plt.grid(True, alpha=0.3)

# Plot 3: Energy vs Popularity  
plt.subplot(2, 3, 3)
plt.scatter(songs_data['energy'], songs_data['popularity'], alpha=0.6, color='red')
plt.title('Energy vs Popularity', fontsize=14, fontweight='bold')
plt.xlabel('Energy (0=low energy, 1=high energy)')
plt.ylabel('Popularity Score')
plt.grid(True, alpha=0.3)

# Plot 4: Tempo vs Popularity
plt.subplot(2, 3, 4)
plt.scatter(songs_data['tempo'], songs_data['popularity'], alpha=0.6, color='purple')
plt.title('Tempo vs Popularity', fontsize=14, fontweight='bold')
plt.xlabel('Tempo (beats per minute)')
plt.ylabel('Popularity Score')
plt.grid(True, alpha=0.3)

# Plot 5: Valence vs Popularity
plt.subplot(2, 3, 5)
plt.scatter(songs_data['valence'], songs_data['popularity'], alpha=0.6, color='orange')
plt.title('Happiness vs Popularity', fontsize=14, fontweight='bold')
plt.xlabel('Valence (0=sad, 1=happy)')
plt.ylabel('Popularity Score')
plt.grid(True, alpha=0.3)

# Plot 6: Top artists
plt.subplot(2, 3, 6)
top_artists = songs_data['artist_name'].value_counts().head(8)
plt.bar(range(len(top_artists)), top_artists.values, color='gold')
plt.title('Most Frequent Artists', fontsize=14, fontweight='bold')
plt.xlabel('Artists')
plt.ylabel('Number of Songs')
plt.xticks(range(len(top_artists)), top_artists.index, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================================
# STEP 3: CREATING CLASSIFICATION PROBLEM
# ================================================

print("\n" + "="*50)
print("🎯 STEP 3: TURNING THIS INTO HIT vs FLOP PREDICTION")
print("="*50)

# Define what makes a song a "hit"
def classify_hit_or_flop(popularity_score):
    """Convert popularity score to Hit or Flop classification"""
    if popularity_score >= 70:
        return 'HIT'
    else:
        return 'FLOP'

# Apply classification
songs_data['hit_or_flop'] = songs_data['popularity'].apply(classify_hit_or_flop)

# Show distribution
hit_flop_counts = songs_data['hit_or_flop'].value_counts()
print("📊 Distribution of Hits vs Flops:")
print(hit_flop_counts)
print(f"Percentage of Hits: {hit_flop_counts['HIT'] / len(songs_data) * 100:.1f}%")

# Visualize hit vs flop
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
hit_flop_counts.plot(kind='bar', color=['red', 'green'], alpha=0.8)
plt.title('Number of Hits vs Flops', fontsize=14, fontweight='bold')
plt.ylabel('Number of Songs')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)

# Compare features between hits and flops
plt.subplot(1, 2, 2)
hits_avg = songs_data[songs_data['hit_or_flop'] == 'HIT'][['danceability', 'energy', 'valence']].mean()
flops_avg = songs_data[songs_data['hit_or_flop'] == 'FLOP'][['danceability', 'energy', 'valence']].mean()

x = range(len(hits_avg))
width = 0.35
plt.bar([i - width/2 for i in x], hits_avg.values, width, label='Hits', color='green', alpha=0.8)
plt.bar([i + width/2 for i in x], flops_avg.values, width, label='Flops', color='red', alpha=0.8)
plt.title('Average Features: Hits vs Flops', fontsize=14, fontweight='bold')
plt.xlabel('Audio Features')
plt.ylabel('Average Score')
plt.xticks(x, ['Danceability', 'Energy', 'Happiness'])
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================================
# STEP 4: PREPARING DATA FOR AI
# ================================================

print("\n" + "="*50)
print("🤖 STEP 4: PREPARING DATA FOR AI TRAINING")
print("="*50)

# Select features for prediction
feature_columns = ['danceability', 'energy', 'loudness', 'tempo']
X = songs_data[feature_columns]  # Features (inputs)
y = songs_data['hit_or_flop']    # Target (output)

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

print("\n🔍 Sample of our features:")
print(X.head())
print("\n🎯 Sample of our targets:")
print(y.head().values)

# ================================================
# STEP 5: THE CRUCIAL TRAIN/TEST SPLIT
# ================================================

print("\n" + "="*50)
print("📚 STEP 5: SPLITTING DATA - TRAINING vs TESTING")
print("="*50)

print("🧠 Why do we split data?")
print("- Training set: AI learns patterns from these songs")
print("- Test set: We test how well AI learned on NEW songs it's never seen")
print("- This prevents 'cheating' - like testing students on homework problems!")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducible results
    stratify=y          # Keep same ratio of hits/flops in both sets
)

print(f"\n📊 Data split results:")
print(f"Training songs: {len(X_train)} ({len(X_train)/len(songs_data)*100:.1f}%)")
print(f"Testing songs: {len(X_test)} ({len(X_test)/len(songs_data)*100:.1f}%)")

print(f"\n✅ Training set distribution:")
print(y_train.value_counts())
print(f"\n✅ Test set distribution:")
print(y_test.value_counts())

# ================================================
# STEP 6: TRAINING THE AI MODEL
# ================================================

print("\n" + "="*50)
print("🚀 STEP 6: TRAINING OUR AI MUSIC PREDICTOR")
print("="*50)

# Create the AI model
print("🧠 Creating Decision Tree AI model...")
model = DecisionTreeClassifier(
    random_state=42,    # Consistent results
    max_depth=6,        # Prevent overly complex tree
    min_samples_split=10  # Require at least 10 songs to make a split
)

print("✅ Model created! Now training...")

# Train the model
model.fit(X_train, y_train)
print("🎉 Training complete! Our AI has learned from", len(X_train), "songs!")

# ================================================
# STEP 7: TESTING THE AI
# ================================================

print("\n" + "="*50)
print("🎯 STEP 7: TESTING OUR AI'S PREDICTIONS")
print("="*50)

# Make predictions
predictions = model.predict(X_test)
print(f"📊 Made predictions for {len(predictions)} songs!")

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\n🏆 Our AI's accuracy: {accuracy:.1%}")
print(f"That means it got {accuracy*len(y_test):.0f} out of {len(y_test)} songs correct!")

# Detailed performance report
print(f"\n📋 Detailed Performance Report:")
print(classification_report(y_test, predictions))

# Show some example predictions
print(f"\n🔍 Example predictions vs actual results:")
results_comparison = pd.DataFrame({
    'Predicted': predictions[:10],
    'Actual': y_test.values[:10],
    'Danceability': X_test['danceability'].values[:10],
    'Energy': X_test['energy'].values[:10],
    'Tempo': X_test['tempo'].values[:10]
})
print(results_comparison)

# ================================================
# STEP 8: TESTING ON NEW SONGS
# ================================================

print("\n" + "="*50)
print("🎵 STEP 8: TESTING ON BRAND NEW SONGS")
print("="*50)

# Create some test songs manually
print("🧪 Let's test some hypothetical songs:")

test_songs = pd.DataFrame({
    'danceability': [0.9, 0.2, 0.8, 0.5],
    'energy': [0.8, 0.1, 0.9, 0.6],
    'loudness': [-4, -18, -3, -10],
    'tempo': [128, 65, 140, 95]
})

song_descriptions = [
    "High-energy dance track (like electronic/pop)",
    "Slow, quiet ballad (like sad piano song)", 
    "Party anthem (like upbeat pop/hip-hop)",
    "Medium tempo song (like radio-friendly pop)"
]

predictions_new = model.predict(test_songs)

print("🎯 AI Predictions for new songs:")
for i, (pred, desc) in enumerate(zip(predictions_new, song_descriptions)):
    print(f"\nSong {i+1}: {desc}")
    print(f"  Features: Dance={test_songs.iloc[i]['danceability']}, "
          f"Energy={test_songs.iloc[i]['energy']}, "
          f"Tempo={test_songs.iloc[i]['tempo']}")
    print(f"  🔮 AI Prediction: {pred}")

# ================================================
# STEP 9: UNDERSTANDING HOW AI THINKS
# ================================================

print("\n" + "="*50)
print("🧠 STEP 9: HOW DOES OUR AI MAKE DECISIONS?")
print("="*50)

# Show feature importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("🎯 Which features matter most for predicting hits?")
print(importance_df)

# Visualize feature importance
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(importance_df['Feature'], importance_df['Importance'], 
        color=['gold', 'silver', 'orange', 'lightblue'])
plt.title('Feature Importance for Hit Prediction', fontsize=14, fontweight='bold')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Visualize decision tree (simplified)
plt.subplot(1, 2, 2)
tree.plot_tree(model, 
               feature_names=feature_columns,
               class_names=['FLOP', 'HIT'],
               filled=True,
               max_depth=3,  # Show only top 3 levels
               fontsize=8)
plt.title('AI Decision Tree (Simplified)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ================================================
# STEP 10: FINAL INSIGHTS & WRAP-UP
# ================================================

print("\n" + "="*50)
print("🎉 CONGRATULATIONS! YOU BUILT AN AI MUSIC PREDICTOR!")
print("="*50)

print("🏆 What you accomplished today:")
print("✅ Loaded and explored real music data")
print("✅ Created visualizations to find patterns")  
print("✅ Converted a complex problem into classification")
print("✅ Split data properly for training and testing")
print("✅ Trained a Decision Tree AI model")
print("✅ Achieved {:.1%} accuracy on unseen songs!".format(accuracy))
print("✅ Understood how AI makes decisions")

print("\n🌟 Real-world applications:")
print("🎵 Spotify uses similar AI for Discover Weekly")
print("📻 Radio stations use AI for playlist curation") 
print("🎤 Record labels use AI to decide which artists to sign")
print("📱 TikTok uses audio analysis to predict viral sounds")

print("\n🤔 Questions to think about:")
print("• What other song features might predict hits? (lyrics sentiment? artist popularity?)")
print("• Could this AI be biased toward certain genres or demographics?")
print("• How might music tastes change over time?")

print("\n📝 Your homework challenge:")
print("Find one song you think should be a hit but isn't popular yet.")
print("Also find one popular song you think shouldn't be a hit.")
print("Next class: we'll test our AI on your examples!")

print(f"\n🎵 Final fun fact: Our AI thinks the most important factor")
print(f"for predicting hits is: {importance_df.iloc[0]['Feature'].upper()}!")

print("\n" + "="*50)
print("Thanks for learning AI with music! 🎵🤖")
print("="*50)