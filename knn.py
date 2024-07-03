import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load dataset (Assuming 'text' column contains news article text and 'topic' column contains labels)
df = pd.read_csv("news.csv", sep=";")
df.dropna(inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['title'], df['topic'], test_size=0.20, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize kNN classifier
k = 3  # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train kNN classifier
knn_classifier.fit(X_train_tfidf, y_train)

# Predict labels for test data
y_pred = knn_classifier.predict(X_test_tfidf)

for i in range(len(X_test)):
  print (X_test.iloc[i])
  print (y_pred[i])

# Evaluate classifier performance
print(classification_report(y_test, y_pred))

