
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the NLTK data needed for sentiment analysis
nltk.download('vader_lexicon')

# Create a SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# Define a function to perform sentiment analysis
def analyze_sentiment(text):
    # Use the SentimentIntensityAnalyzer to analyze the text
    sentiment = sia.polarity_scores(text)
    return sentiment

# Test the function with a sample text
text = "I love this product! It's amazing!"
sentiment = analyze_sentiment(text)
print(sentiment)