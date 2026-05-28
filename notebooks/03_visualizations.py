import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load sentiment results
df = pd.read_csv("data/sentiment_results.csv")
print("Data loaded!")

# Chart 1: Sentiment Distribution
plt.figure(figsize=(8, 5))
colors = ["green", "red", "gray"]
df["TextBlob_Sentiment"].value_counts().plot(
    kind="bar", color=colors)
plt.title("Hotel Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("notebooks/chart1_sentiment_distribution.png")
plt.show()
print("Chart 1 saved!")

# Chart 2: Rating Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="Rating", data=df, palette="viridis")
plt.title("Hotel Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("notebooks/chart2_rating_distribution.png")
plt.show()
print("Chart 2 saved!")

# Chart 3: Positive Word Cloud
positive_reviews = " ".join(
    df[df["TextBlob_Sentiment"] == "Positive"]["Review"].astype(str)
)
wordcloud_pos = WordCloud(
    width=800, height=400,
    background_color="white",
    colormap="Greens"
).generate(positive_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews - Word Cloud")
plt.tight_layout()
plt.savefig("notebooks/chart3_positive_wordcloud.png")
plt.show()
print("Chart 3 saved!")

# Chart 4: Negative Word Cloud
negative_reviews = " ".join(
    df[df["TextBlob_Sentiment"] == "Negative"]["Review"].astype(str)
)
wordcloud_neg = WordCloud(
    width=800, height=400,
    background_color="white",
    colormap="Reds"
).generate(negative_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews - Word Cloud")
plt.tight_layout()
plt.savefig("notebooks/chart4_negative_wordcloud.png")
plt.show()
print("Chart 4 saved!")

print("\nAll charts saved!")