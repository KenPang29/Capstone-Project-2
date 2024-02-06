# Import libraries
import spacy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Update the file path without double quotes
file_path = r"C:\Users\kenpa\Desktop\Hyperion\T21 - Capstone Project - NLP Applications\amazon_products_reviews.csv"
dataframe = pd.read_csv(file_path)

# Drop rows with missing review text
clean_data = dataframe.dropna(subset=['reviews.text'])

# Preprocess function to apply various preprocessing steps
def preprocess_text(text):
    # Lowercase the text, remove whitespaces
    text = text.lower().strip()
    
    # Process the text using spaCy
    doc = nlp(text)
    
    # Remove punctuation and stopwords, and lemmatize the tokens
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    
    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function for sentiment analysis
def predict_sentiment(review_text):
    # Preprocess the review text
    preprocessed_review = preprocess_text(review_text)
    
    # Analyze sentiment using VADER
    sentiment_scores = sid.polarity_scores(preprocessed_review)
    
    # Determine sentiment label based on the compound score
    compound_score = sentiment_scores['compound']
    if compound_score > 0.5:
        return 'positive'
    elif compound_score < -0.5:
        return 'negative'
    else:
        return 'neutral'

# Function to compare similarity between two reviews
def compare_similarity(review1, review2):
    # Process the reviews using spaCy
    doc1 = nlp(review1)
    doc2 = nlp(review2)
    
    # Calculate the similarity between the two reviews
    similarity_score = doc1.similarity(doc2)
    
    return similarity_score

# Test the sentiment analysis function on sample product reviews from the dataframe
sample_reviews_indices = random.sample(range(len(clean_data)), 3)  # Select 3 random reviews
sample_reviews = clean_data.loc[sample_reviews_indices, 'reviews.text'].tolist()

print("Sentiment Analysis Results:")
for review in sample_reviews:
    sentiment = predict_sentiment(review)
    print(f"Review: '{review}'")
    print(f"Predicted Sentiment: {sentiment}")
    print()

# Test the similarity comparison function on two sample reviews from the dataset
review1 = clean_data['reviews.text'].iloc[0]  # Select the first review
review2 = clean_data['reviews.text'].iloc[1]  # Select the second review
similarity_score = compare_similarity(review1, review2)
print("Similarity Comparison Results:")
print(f"Review 1: '{review1}'")
print(f"Review 2: '{review2}'")
print(f"Similarity Score: {similarity_score}")
