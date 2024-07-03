import requests
import random
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("news.csv", sep=";")

# Initialize an empty dictionary to store the article titles and content
dictcontent = {"title": [], "content": []}

# Loop through the rows in the DataFrame
for index, row in df.iterrows():
    try:
        # Fetch the article content from the URL
        content = ""
        url = row['link']
        page = requests.get(url)

        # Check if the request was successful
        if page.status_code == 200:
            soup = BeautifulSoup(page.content, "html.parser")
            elements = soup.find_all("p")
            for element in elements:
                content += element.text + "\n"
            dictcontent["title"].append(row['title'])
            dictcontent["content"].append(content)
        else:
            continue
    except requests.exceptions.SSLError as e:
        continue
    except requests.exceptions.RequestException as e:
        continue

# Convert the dictionary to a pandas DataFrame
df1 = pd.DataFrame(dictcontent)

# Combine title and content for each article
corpus = df1['title'] + ' ' + df1['content']

# Create a TF-IDF vectorizer with English stopwords
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus)

# Train the LDA model
n_components = 4  # Number of topics
lda = LatentDirichletAllocation(n_components=n_components, random_state=42, learning_method='online', max_iter=100)
lda.fit(X)

# Transform all articles into topic distributions
topic_distribution = lda.transform(X)

# Initialize lists to store article titles for each topic
topic_lists = [[] for _ in range(n_components)]

# Assign each article to the topic it is most associated with
for i, topic_dist in enumerate(topic_distribution):
    max_index = topic_dist.argmax()
    topic_lists[max_index].append(df['title'].iloc[i])

# Function to suggest similar articles
def suggest_similar_articles(topic_index, selected_article_title, num_suggestions=3):
    # Get indices of articles in the selected topic
    topic_articles = topic_lists[topic_index]

    # Get index of the selected article in the DataFrame
    selected_article_df_index = df[df['title'] == selected_article_title].index[0]

    # Get the TF-IDF vector representation of the selected article
    selected_article_vector = X[selected_article_df_index]

    # Calculate cosine similarity between the selected article and all other articles in the same topic
    similarities = cosine_similarity(selected_article_vector, X)

    # Get the indices of articles sorted by cosine similarity (excluding the selected article itself)
    sorted_indices = similarities.argsort()[0][::-1][1:]  # Exclude the first element which is the selected article itself

    # Get the top similar articles
    similar_articles = [df['title'].iloc[index] for index in sorted_indices[:num_suggestions]]
    return similar_articles

# Choose a random topic (let's say Topic 1)
topic_index = 0  # Remember, Python is 0-indexed
topic_articles = topic_lists[topic_index]

# Choose a random article from the selected topic
selected_article_title = random.choice(topic_articles)

# Print the selected article
print("Selected Article:")
print(selected_article_title)
print()

# Get and print the top 3 similar articles
similar_articles = suggest_similar_articles(topic_index, selected_article_title)
print("Top 3 Similar Articles:")
for article in similar_articles:
    print(f"- {article}")
