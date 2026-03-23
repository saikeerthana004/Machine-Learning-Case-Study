# Email Spam Classification 

# Step-1: Import Necessary Libraries 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Step-2: Create Email Spam Detection Dataset(DataFrame)
data = {
    "Email": [
        "Win money now",
        "Win a free iPhone today. Hurry up!",
        "Meeting scheduled at 10 AM tomorrow",
        "Project deadline tomorrow",
        "Congratulations you won a lottery",
        "Let's have lunch",
        "Submit your assignment by tonight",
        "Limited time offer! Buy now!",
        "Free vacation offer, claim now!",
        "Team meeting postponed to evening",
        "You have been selected for a prize",
        "Exam results are published",
        "Win a brand new car now!"
    ],
    "Label": [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam
}
df = pd.DataFrame(data)
print(df)

# Step-3: Separate Features and Labels
X_text = df["Email"]
y = df["Label"]

# Step-4: Convert Text to Numerical Data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

# Model-1: Logistic Regression 
lr_model = LogisticRegression()
lr_model.fit(X, y)

# Model-2: Naive Bayes 
nb_model = MultinomialNB()
nb_model.fit(X, y)

# Step 5: Test with New Emails
test_emails = [
    "Win cash prize now",
    "Meeting scheduled at 10 AM tomorrow",
    "Limited time offer! Buy now!",
    "Free vacation offer, claim now!",
    "Project deadline tomorrow"
]

# Convert test emails
test_vec = vectorizer.transform(test_emails)
# Predictions
lr_predictions = lr_model.predict(test_vec)
nb_predictions = nb_model.predict(test_vec)

# Step 6: Show Results
print(" Email Predictions:\n")
for i in range(len(test_emails)):
    print("Email:", test_emails[i])
    print("Logistic Regression:", "Spam" if lr_predictions[i] == 1 else "Not Spam")
    print("Naive Bayes       :", "Spam" if nb_predictions[i] == 1 else "Not Spam")
    print("-" * 50)

# Step 7: Visualization Graph

# 1. Dataset Distribution (Spam vs Not Spam)
plt.figure()
df["Label"].value_counts().plot(kind='bar')
plt.title("Spam vs Not Spam Distribution", fontsize=14)
plt.xlabel("Label (0 = Not Spam, 1 = Spam)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=0)
plt.show()

# 2. Model Prediction Comparison
# Count predictions
lr_counts = pd.Series(lr_predictions).value_counts()
nb_counts = pd.Series(nb_predictions).value_counts()
# Create DataFrame for plotting
comparison_df = pd.DataFrame({
    "Logistic Regression": lr_counts,
    "Naive Bayes": nb_counts
}).fillna(0)

# Plot comparison
comparison_df.plot(kind='bar')
plt.title("Model Prediction Comparison", fontsize=14)
plt.xlabel("Class (0 = Not Spam, 1 = Spam)", fontsize=12)
plt.ylabel("Number of Predictions", fontsize=12)
plt.xticks(rotation=0)
plt.show()
