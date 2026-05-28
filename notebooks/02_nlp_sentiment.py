import pandas as pd
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# Load cleaned data
df = pd.read_csv("data/cleaned_reviews.csv")
print("Data loaded! Shape:", df.shape)

# Function to get TextBlob sentiment score
def get_textblob_sentiment(text):
    analysis = TextBlob(str(text))
    score = analysis.sentiment.polarity
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Apply TextBlob sentiment
print("\nApplying TextBlob sentiment analysis...")
df["TextBlob_Sentiment"] = df["Review"].apply(get_textblob_sentiment)
df["TextBlob_Score"] = df["Review"].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)

print("\nTextBlob Sentiment Distribution:")
print(df["TextBlob_Sentiment"].value_counts())

# Compare with original rating sentiment
print("\nOriginal Rating Sentiment:")
print(df["Sentiment"].value_counts())

# Accuracy check
correct = (df["Sentiment"] == df["TextBlob_Sentiment"]).sum()
accuracy = correct / len(df) * 100
print(f"\nTextBlob Accuracy: {accuracy:.2f}%")

# Show sample reviews with scores
print("\nSample Positive Reviews:")
print(df[df["TextBlob_Sentiment"] == "Positive"]["Review"].head(3).to_string())

print("\nSample Negative Reviews:")
print(df[df["TextBlob_Sentiment"] == "Negative"]["Review"].head(3).to_string())

# Save results
df.to_csv("data/sentiment_results.csv", index=False)
print("\nSentiment results saved!")