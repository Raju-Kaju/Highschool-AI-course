# High School AI Course: Classification & Regression
## Course Overview
**Target Audience:** High School Students (Ages 14-18)  
**Duration:** 8 modules × 1 hour each  
**Focus:** Practical applications with simple Python coding  
**Goal:** Build excitement about AI while developing real skills

---

## Module 1: "Is This Song a Hit?" - Introduction to Classification (1 hour)

### Learning Objectives
- Understand what classification means in AI
- Learn the difference between features and labels
- Build first classification model

### Project: Spotify Hit Predictor
Students will predict if a song will be popular based on audio features like:
- Danceability, Energy, Loudness, Tempo
- Use a pre-made dataset of popular vs. unpopular songs

### Activities
1. **Explore the Data** (15 min)
   - Load Spotify dataset with pandas
   - Visualize popular vs unpopular songs
   
2. **Understand Features** (15 min)
   - What makes a song "danceable"?
   - How does energy relate to popularity?
   
3. **Train Your First Model** (25 min)
   ```python
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.model_selection import train_test_split
   
   # Simple 5-line model training
   ```

4. **Test and Reflect** (5 min)
   - Predict if current trending songs would be hits
   - Discuss results

### Takeaway
Students leave with a working model that can predict song popularity!

---

## Module 2: "Netflix Recommendation Engine" - Advanced Classification (1 hour)

### Learning Objectives
- Understand different types of classification algorithms
- Learn about training vs testing data
- Improve model accuracy

### Project: Movie Genre Classifier
Predict movie genres based on plot summaries using movie descriptions.

### Activities
1. **Data Detective** (20 min)
   - Explore movie dataset (genre, plot, rating, year)
   - Find patterns in different genres

2. **Feature Engineering** (20 min)
   - Convert text to numbers (simple bag-of-words)
   - Select best features for prediction

3. **Model Comparison** (15 min)
   ```python
   # Compare Decision Tree vs Random Forest
   # Students see accuracy improvements
   ```

4. **Build Personal Recommender** (5 min)
   - Input favorite movie descriptions
   - Get genre predictions

### Fun Element
Students can test their model on movies they haven't seen yet!

---

## Module 3: "House Price Prophet" - Introduction to Regression (1 hour)

### Learning Objectives
- Understand regression vs classification
- Learn to predict continuous values
- Interpret model predictions

### Project: Local Housing Market Predictor
Predict house prices based on features like:
- Square footage, bedrooms, bathrooms, location score

### Activities
1. **Market Research** (15 min)
   - Explore real housing data from their area
   - Identify what affects house prices

2. **Visualization Magic** (20 min)
   ```python
   import matplotlib.pyplot as plt
   # Create scatter plots showing price relationships
   ```

3. **Linear Regression Training** (20 min)
   ```python
   from sklearn.linear_model import LinearRegression
   # Simple, interpretable model
   ```

4. **Price Your Dream Home** (5 min)
   - Students input their dream home specs
   - Model predicts the price

### Real-world Connection
Connect to students' future (college costs, first home purchase)

---

## Module 4: "Grade Predictor" - Personal Regression Application (1 hour)

### Learning Objectives
- Apply regression to personal scenarios
- Understand feature importance
- Learn model limitations

### Project: Academic Performance Predictor
Predict final grades based on:
- Study hours, attendance, assignment scores, sleep hours

### Activities
1. **Self-Reflection Survey** (10 min)
   - Students fill out anonymous study habit survey
   - Create class dataset

2. **Pattern Discovery** (15 min)
   - Analyze which factors correlate with good grades
   - Surprising insights about study habits

3. **Personal Model Training** (25 min)
   ```python
   # Multiple regression with study factors
   # Feature importance visualization
   ```

4. **What-If Scenarios** (10 min)
   - "If I study 2 more hours per week..."
   - "If I get 8 hours of sleep instead of 6..."

### Ethical Discussion
Brief talk about correlation vs causation

---

## Module 5: "Social Media Sentiment Detective" - Text Classification (1 hour)

### Learning Objectives
- Apply classification to text data
- Understand natural language processing basics
- Explore AI bias and ethics

### Project: Tweet Mood Analyzer
Classify social media posts as positive, negative, or neutral sentiment.

### Activities
1. **Text Exploration** (15 min)
   - Examine tweet dataset
   - Manual sentiment labeling exercise

2. **Text to Numbers** (20 min)
   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   # Transform tweets into numerical features
   ```

3. **Sentiment Classification** (20 min)
   ```python
   from sklearn.naive_bayes import MultinomialNB
   # Train sentiment classifier
   ```

4. **Test on Real Data** (5 min)
   - Analyze sentiment of tweets about current events
   - Compare model predictions with human judgment

### Critical Thinking
Discuss AI bias in sentiment analysis and real-world implications

---

## Module 6: "Sports Analytics Champion" - Advanced Regression (1 hour)

### Learning Objectives
- Handle more complex datasets
- Feature selection and engineering
- Model evaluation metrics

### Project: Basketball Player Performance Predictor
Predict player points per game based on various statistics.

### Activities
1. **Sports Statistics Deep Dive** (15 min)
   - Explore NBA/WNBA player statistics
   - Identify meaningful performance metrics

2. **Advanced Feature Engineering** (20 min)
   - Create new features (shooting efficiency, usage rate)
   - Handle missing data

3. **Model Optimization** (20 min)
   ```python
   from sklearn.ensemble import RandomForestRegressor
   # Compare multiple algorithms
   # Cross-validation for better evaluation
   ```

4. **Fantasy Sports Application** (5 min)
   - Predict performance for upcoming games
   - Connect to fantasy sports leagues

### Engagement Factor
Many students follow sports, making this highly relatable

---

## Module 7: "Climate Change Tracker" - Time Series & Social Impact (1 hour)

### Learning Objectives
- Apply AI to important global issues
- Understand time series data
- Connect AI to social responsibility

### Project: Temperature Trend Analyzer
Predict future temperature trends based on historical climate data.

### Activities
1. **Climate Data Exploration** (20 min)
   - Load historical temperature data
   - Visualize climate trends over decades

2. **Trend Analysis** (15 min)
   ```python
   from sklearn.linear_model import LinearRegression
   import numpy as np
   # Simple time series regression
   ```

3. **Future Predictions** (20 min)
   - Predict temperatures for next decade
   - Discuss model limitations and uncertainty

4. **Impact Discussion** (5 min)
   - How AI helps climate scientists
   - Student ideas for AI solutions to environmental problems

### Social Connection
Addresses issues students care about deeply

---

## Module 8: "Build Your Own AI App" - Capstone Project (1 hour)

### Learning Objectives
- Integrate all learned concepts
- Create original AI application
- Present findings to peers

### Project Options (Students Choose One)
1. **Playlist Mood Matcher** - Classify songs by mood for workout/study playlists
2. **College Admission Predictor** - Predict admission chances based on grades/activities
3. **Video Game Success Predictor** - Predict game ratings based on features
4. **Food Recipe Classifier** - Classify recipes by cuisine type
5. **Custom Project** - Student's own idea with instructor approval

### Activities
1. **Project Planning** (15 min)
   - Choose project and define problem
   - Identify data sources and features

2. **Implementation** (30 min)
   - Apply learned techniques
   - Instructor provides individual guidance

3. **Testing and Refinement** (10 min)
   - Evaluate model performance
   - Iterate and improve

4. **Mini-Presentations** (5 min)
   - 1-minute project showcases
   - Peer feedback and celebration

---

## Course Materials & Tools

### Required Software
- Python 3.x with Jupyter Notebooks
- Libraries: pandas, scikit-learn, matplotlib, numpy
- Google Colab (browser-based, no installation needed)

### Datasets Provided
- Pre-cleaned, student-appropriate datasets for each module
- Real-world data from Spotify, movie databases, housing markets, etc.
- All datasets < 1MB for quick loading

### Assessment Method
- **Participation & Engagement** (40%)
- **Weekly Mini-Projects** (40%)
- **Final Capstone Project** (20%)
- No traditional tests - focus on practical application

### Extension Activities
For advanced students:
- Deploy models using Streamlit
- Create web interfaces for their models
- Explore deep learning with TensorFlow/PyTorch basics

---

## Why This Curriculum Works

### Age-Appropriate Projects
- Connects to student interests (music, movies, sports, social media)
- Addresses relevant life decisions (college, housing, career)
- Tackles issues they care about (climate, social justice)

### Practical Focus
- Every module produces a working model
- Students see immediate results
- Code examples are simple but powerful

### Progressive Complexity
- Starts with basic concepts, builds systematically
- Each module reinforces previous learning
- Scaffolded approach builds confidence

### Real-World Relevance
- Uses actual datasets and problems
- Discusses ethical implications
- Connects to current events and social issues

### Engagement Strategies
- Hands-on coding from day one
- Personal data and choices when appropriate
- Competitive elements (accuracy challenges)
- Creative final projects

This curriculum transforms abstract AI concepts into tangible, exciting experiences that will inspire students to continue learning about artificial intelligence and its applications!