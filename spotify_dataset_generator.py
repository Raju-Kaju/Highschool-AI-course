import pandas as pd
import numpy as np
import random

# Set random seed for reproducible results
np.random.seed(42)
random.seed(42)

# Create realistic song data based on actual Spotify audio features
def generate_spotify_dataset(n_songs=1000):
    """
    Generate a realistic dataset of songs with audio features similar to Spotify's API
    """
    
    # Artist names (mix of real and fictional to avoid copyright issues)
    artists = [
        "Taylor Swift", "Ed Sheeran", "The Weeknd", "Billie Eilish", "Drake", 
        "Ariana Grande", "Post Malone", "Olivia Rodrigo", "Harry Styles", "Dua Lipa",
        "BTS", "Bad Bunny", "Kendrick Lamar", "Adele", "Bruno Mars",
        "The Midnight Echoes", "Neon Dreams", "Electric Sunrise", "Velvet Thunder", 
        "Crystal Waters", "Phoenix Rising", "Midnight Velocity", "Golden Hour",
        "Urban Legends", "Digital Hearts", "Cosmic Drift", "Retro Vibes",
        "Shadow Dancers", "Ocean Waves", "Mountain High", "City Lights"
    ]
    
    # Song name components for generating realistic titles
    adjectives = ["Lost", "Broken", "Golden", "Electric", "Midnight", "Summer", "Winter", 
                 "Dancing", "Flying", "Burning", "Falling", "Rising", "Shining", "Wild"]
    nouns = ["Hearts", "Dreams", "Lights", "Stars", "Fire", "Rain", "Thunder", "Waves",
            "Roads", "Nights", "Days", "Love", "Time", "World", "Sky", "Moon"]
    
    songs = []
    
    for i in range(n_songs):
        # Generate song title
        if random.random() < 0.3:  # 30% chance of single word title
            song_title = random.choice(adjectives + nouns)
        else:  # 70% chance of two word title
            song_title = f"{random.choice(adjectives)} {random.choice(nouns)}"
        
        # Select random artist
        artist = random.choice(artists)
        
        # Generate correlated audio features for more realistic data
        # Some genres/styles tend to cluster together
        
        # Create genre bias
        genre_type = random.choice(['pop', 'hip-hop', 'rock', 'electronic', 'ballad'])
        
        if genre_type == 'pop':
            danceability = np.random.beta(3, 2) * 0.8 + 0.2  # Tend to be danceable
            energy = np.random.beta(2, 2) * 0.7 + 0.3
            valence = np.random.beta(3, 2) * 0.7 + 0.3  # Tend to be positive
            tempo = np.random.normal(120, 15)
            
        elif genre_type == 'hip-hop':
            danceability = np.random.beta(4, 2) * 0.8 + 0.2  # Very danceable
            energy = np.random.beta(3, 2) * 0.8 + 0.2
            valence = np.random.beta(2, 2) * 0.8 + 0.2
            tempo = np.random.normal(95, 20)
            
        elif genre_type == 'rock':
            danceability = np.random.beta(2, 3) * 0.7 + 0.1  # Less danceable
            energy = np.random.beta(4, 1) * 0.9 + 0.1  # Very energetic
            valence = np.random.beta(2, 2) * 0.8 + 0.2
            tempo = np.random.normal(130, 25)
            
        elif genre_type == 'electronic':
            danceability = np.random.beta(4, 1) * 0.9 + 0.1  # Very danceable
            energy = np.random.beta(3, 2) * 0.8 + 0.2
            valence = np.random.beta(2, 2) * 0.8 + 0.2
            tempo = np.random.normal(128, 20)
            
        else:  # ballad
            danceability = np.random.beta(1, 4) * 0.5 + 0.1  # Not danceable
            energy = np.random.beta(1, 3) * 0.4 + 0.1  # Low energy
            valence = np.random.beta(2, 3) * 0.6 + 0.1  # Often sad
            tempo = np.random.normal(70, 15)
        
        # Ensure values stay in valid ranges
        danceability = np.clip(danceability, 0.0, 1.0)
        energy = np.clip(energy, 0.0, 1.0)
        valence = np.clip(valence, 0.0, 1.0)
        tempo = np.clip(tempo, 50, 200)
        
        # Other audio features
        acousticness = np.random.beta(2, 3)  # Most songs are not very acoustic
        instrumentalness = np.random.beta(1, 9)  # Most songs have vocals
        liveness = np.random.beta(1, 4)  # Most songs are studio recordings
        speechiness = np.random.beta(1, 4)  # Most songs are not very speech-like
        loudness = np.random.normal(-8, 4)  # Typical range for modern music
        loudness = np.clip(loudness, -25, 0)
        
        # Duration in milliseconds (3-6 minutes typical)
        duration_ms = np.random.normal(210000, 45000)  # ~3.5 minutes average
        duration_ms = np.clip(duration_ms, 120000, 400000)  # 2-6.7 minutes range
        
        # Generate popularity score with some correlation to features
        # More danceable, energetic, positive songs tend to be more popular
        # But add significant randomness to reflect real-world complexity
        
        popularity_base = (
            danceability * 25 +           # Danceable songs more popular
            energy * 20 +                # Energetic songs more popular  
            valence * 15 +               # Happy songs more popular
            (1 - acousticness) * 10 +    # Produced songs more popular
            np.random.normal(0, 20)      # Random factors
        )
        
        # Add artist popularity bias (some artists are just more popular)
        if artist in ["Taylor Swift", "Drake", "The Weeknd", "Billie Eilish", "Ariana Grande"]:
            popularity_base += 20
        elif artist in ["Ed Sheeran", "Post Malone", "Harry Styles", "Dua Lipa"]:
            popularity_base += 15
        
        popularity = np.clip(popularity_base, 0, 100)
        
        # Create song dictionary
        song = {
            'track_name': song_title,
            'artist_name': artist,
            'danceability': round(danceability, 3),
            'energy': round(energy, 3),
            'loudness': round(loudness, 1),
            'tempo': round(tempo, 1),
            'valence': round(valence, 3),
            'acousticness': round(acousticness, 3),
            'instrumentalness': round(instrumentalness, 3),
            'liveness': round(liveness, 3),
            'speechiness': round(speechiness, 3),
            'duration_ms': int(duration_ms),
            'popularity': round(popularity, 1)
        }
        
        songs.append(song)
    
    return pd.DataFrame(songs)

# Generate the dataset
print("Generating Spotify-style dataset...")
df = generate_spotify_dataset(1000)

# Display basic info about the dataset
print(f"\nDataset created with {len(df)} songs")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset statistics:")
print(df.describe())

# Check the hit/flop distribution
df['hit_or_flop'] = df['popularity'].apply(lambda x: 'HIT' if x > 70 else 'FLOP')
print(f"\nHit vs Flop distribution:")
print(df['hit_or_flop'].value_counts())

# Save to CSV
df.to_csv('spotify_songs_dataset.csv', index=False)
print(f"\n✅ Dataset saved as 'spotify_songs_dataset.csv'")

print("\n🎵 Sample songs from the dataset:")
sample_songs = df.sample(5)
for _, song in sample_songs.iterrows():
    print(f"'{song['track_name']}' by {song['artist_name']}")
    print(f"  Danceability: {song['danceability']}, Energy: {song['energy']}, Popularity: {song['popularity']}")
    print()