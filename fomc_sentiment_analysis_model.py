#Library Installations:
%pip install datasets 
%pip install scikit-learn 
%pip install vaderSentiment
%pip install huggingface_hub --upgrade

# Import Libraries
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Hugging Face (Public Datasets; no authentification required
# Optional: Login only required for private datasets
# from huggingface_hub import login
# login(token=os.getenv("HF_TOKEN"))

# Load datasets
finben = load_dataset("TheFinAI/finben-fomc", split="test")
sorour = load_dataset("Sorour/fomc", split="train")
fomc_comm = load_dataset("gtfintechlab/fomc_communication", split="train")

# Normalize labels (Convert all labels to DOVISH/HAWKISH/NEUTRAL)
def normalize_label(label):
    if isinstance(label, int):
        return {0: "DOVISH", 1: "HAWKISH", 2: "NEUTRAL"}.get(label, "NEUTRAL")
    elif isinstance(label, str):
        l = label.strip().upper()
        if l in {"DOVISH", "HAWKISH", "NEUTRAL"}:
            return l
        elif l in {"DOVE", "DOVES"}:
            return "DOVISH"
        elif l in {"HAWK", "HAWKS"}:
            return "HAWKISH"
        return "NEUTRAL"
    else:
        return "NEUTRAL"

# Combine datasets into one table & Merge
df1 = pd.DataFrame({"text": finben["query"], "label": [normalize_label(lbl) for lbl in finben["answer"]]})
df2 = pd.DataFrame({"text": sorour["text"], "label": [normalize_label(lbl) for lbl in sorour["label"]]})
df3 = pd.DataFrame({"text": fomc_comm["sentence"], "label": [normalize_label(lbl) for lbl in fomc_comm["label"]]})
df = pd.concat([df1, df2, df3], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Split into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Convert text to numbers
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=30000, stop_words="english")
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Train classifier
clf = LogisticRegression(
    max_iter=500,
    solver="lbfgs",
    class_weight="balanced"
)
clf.fit(x_train_vec, y_train)

# Show Test Set Performance Metrics
y_pred = clf.predict(x_test_vec)

print("\nTest Set Metrics:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Setup VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to classify new FOMC statements
def classify_fomc_statement(text):
    # Clean text
    text_clean = re.sub(r"\s+", " ", text.strip())

    # VADER sentiment
    vader_scores = analyzer.polarity_scores(text_clean)

    # Predict stance using classifier
    vec = vectorizer.transform([text_clean])
    label = clf.predict(vec)[0]
    probs = clf.predict_proba(vec)[0]

    # Convert probabilities to plain float
    prob_map = {x: float(v) for x,v in zip(clf.classes_, probs)}
    vader_map = {x: float(v) for x,v in vader_scores.items()}

    return {
        "stance_label": label,
        "stance_probabilities": prob_map,
        "vader": vader_map
    }

# FOMC Statement
fomc_text = """
The Federal Reserve on Tuesday announced the establishment of 
a temporary repurchase agreement facility for foreign and international 
monetary authorities (FIMA Repo Facility) to help support the smooth 
functioning of financial markets, including the U.S. Treasury market, 
and thus maintain the supply of credit to U.S. households and businesses. 
The FIMA Repo Facility will allow FIMA account holders, which consist of 
central banks and other international monetary authorities with accounts at 
the Federal Reserve Bank of New York, to enter into repurchase agreements with 
the Federal Reserve. In these transactions, FIMA account holders temporarily 
exchange their U.S. Treasury securities held with the Federal Reserve for U.S. 
dollars, which can then be made available to institutions in their jurisdictions.
This facility should help support the smooth functioning of the U.S. Treasury 
market by providing an alternative temporary source of U.S. dollars other than
sales of securities in the open market. It should also serve, along with the U.S.
dollar liquidity swap lines the Federal Reserve has established with other central
banks, to help ease strains in global U.S. dollar funding markets.
"""

# Analyze statement
result = classify_fomc_statement(fomc_text)

print("\nFOMC Statement Analysis: ")
print("Predicted Stance:", result["stance_label"])
print("Stance Probabilities:", result["stance_probabilities"])
print("VADER Sentiment:", result["vader"])
