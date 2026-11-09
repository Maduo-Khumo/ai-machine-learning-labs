from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. The Training Data (Emails and Labels)
# 1 = Phishing/Scam, 0 = Safe
emails = [
    "Click this link to reset your password immediately", # 1
    "Your account has been suspended, verify now", # 1
    "Hey, are we still meeting for lunch today?", # 0
    "Please find the attached report for Q3", # 0
    "Congratulations! You won a free iPhone, claim now", # 1
    "Can you send me the updated project timeline?" # 0
]
labels = [1, 1, 0, 0, 1, 0]

# 2. Data Preprocessing (Converting text to numbers)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# 3. Train the Machine Learning Model (Naive Bayes)
model = MultinomialNB()
model.fit(X, labels)

# 4. Test the Model on NEW, unseen emails
test_emails = [
    "Urgent: Your bank account is locked. Click here to verify.",
    "Hi team, attached is the agenda for tomorrow's meeting."
]

# 5. Predict whether they are phishing or safe
test_X = vectorizer.transform(test_emails)
predictions = model.predict(test_X)

# 6. Print the Results
print("--- AI Phishing Detection Results ---")
for email, prediction in zip(test_emails, predictions):
    if prediction == 1:
        print(f"🚨 PHISHING DETECTED: '{email}'")
    else:
        print(f"✅ SAFE: '{email}'")