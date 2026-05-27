import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("data/tripadvisor_hotel_reviews.csv")
print("Dataset loaded!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nRating Distribution:")
print(df["Rating"].value_counts())

# Adding Sentiment Label based on Rating
def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Rating"].apply(get_sentiment)

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

# Clean Reviews
df["Review"] = df["Review"].str.lower()
df["Review"] = df["Review"].str.replace("[^a-zA-Z\s]", "", regex=True)
df = df.dropna(subset=["Review"])
df = df.drop_duplicates()

print("\nCleaned dataset shape:", df.shape)

# Save cleaned data
df.to_csv("data/cleaned_reviews.csv", index=False)
print("Cleaned data saved!")