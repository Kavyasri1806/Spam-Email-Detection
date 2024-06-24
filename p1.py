# Importing required Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# In this example, we create a small dataset of email text and Labels (0 for not spam, 1 for spam)
emails = [
    "Get rich Quick! Click here to win a million dollars!",
    "Hello, could you please review this document for me",
    "Discounts on luxury watches and handbags!",
    "Meeting scheduled for tomorrow, please confirm your attendance.",
    "Congratulations, you've won a free gift card!"
]

labels = [1, 0, 1, 0, 1]

# Convert text data into numerical features using Count Vectorization
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(emails)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Create a Multinomial Naive Bayes classifier
model = MultinomialNB()

# Train the model on training data
model.fit(X_train, y_train)

# Make prediction on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Classification Report:\n", report)

# Predict whether a new email is spam or not 
new_emails = [
    "You've won a free cruise vacation",  # Expected spam
    "Please find the attached report for your review",  # Expected non-spam
    "Exclusive offer! Get 50% off on your next purchase",  # Expected spam
    "Let's catch up tomorrow at the meeting",  # Expected non-spam
    "Limited time offer! Claim your free trial now",  # Expected spam
    "Can we reschedule our appointment to next week?"  # Expected non-spam
]

for email in new_emails:
    new_email_vectorized = vectorizer.transform([email])
    predicted_label = model.predict(new_email_vectorized)
    result = "spam" if predicted_label[0] == 1 else "not spam"
    print(f"Email: \"{email}\" - Predicted as {result}.")

# Running additional test cases
try:
    # Test Case 1
    test_email_1 = ["Get rich Quick! Click here to win a million dollars!"]
    test_email_1_vectorized = vectorizer.transform(test_email_1)
    predicted_label_1 = model.predict(test_email_1_vectorized)
    assert predicted_label_1[0] == 1, "Test Case 1 Failed: Should be classified as spam."
    
    # Test Case 2
    test_email_2 = ["Hello, could you please review this document for me"]
    test_email_2_vectorized = vectorizer.transform(test_email_2)
    predicted_label_2 = model.predict(test_email_2_vectorized)
    assert predicted_label_2[0] == 0, "Test Case 2 Failed: Should be classified as not spam."
    
    # Test Case 3
    test_email_3 = ["Congratulations, you've won a free cruise vacation"]
    test_email_3_vectorized = vectorizer.transform(test_email_3)
    predicted_label_3 = model.predict(test_email_3_vectorized)
    assert predicted_label_3[0] == 1, "Test Case 3 Failed: Should be classified as spam."
    
    # Test Case 4
    test_email_4 = ["Meeting reminder for tomorrow, please confirm your attendance"]
    test_email_4_vectorized = vectorizer.transform(test_email_4)
    predicted_label_4 = model.predict(test_email_4_vectorized)
    assert predicted_label_4[0] == 0, "Test Case 4 Failed: Should be classified as not spam."
    
    # Test Case 5
    test_email_5 = ["Exclusive offer! Get 50% off on your next purchase"]
    test_email_5_vectorized = vectorizer.transform(test_email_5)
    predicted_label_5 = model.predict(test_email_5_vectorized)
    assert predicted_label_5[0] == 1, "Test Case 5 Failed: Should be classified as spam."
    
    # Test Case 6
    test_email_6 = ["Let's catch up tomorrow at the meeting"]
    test_email_6_vectorized = vectorizer.transform(test_email_6)
    predicted_label_6 = model.predict(test_email_6_vectorized)
    assert predicted_label_6[0] == 0, "Test Case 6 Failed: Should be classified as not spam."
    
    print("All test cases passed!")
except AssertionError as e:
    print(e)