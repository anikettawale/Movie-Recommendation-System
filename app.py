import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
movies = pd.read_csv("movies.csv")

# Convert Genre into Vectors
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(movies["Genre"])

# Calculate Similarity
similarity = cosine_similarity(matrix)

def recommend(movie_name):

    movie_name = movie_name.lower()
    movies["Title"] = movies["Title"].str.lower()

    if movie_name not in movies["Title"].values:
        print("\nMovie not found!")
        return

    index = movies[movies["Title"] == movie_name].index[0]

    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:\n")

    count = 0

    for movie in scores[1:]:

        print(movies.iloc[movie[0]].Title.title())

        count += 1

        if count == 5:
            break

print("=" * 50)
print("      MOVIE RECOMMENDATION SYSTEM")
print("=" * 50)

while True:

    movie = input("\nEnter Movie Name : ")

    recommend(movie)

    again = input("\nSearch Again? (y/n): ")

    if again.lower() != "y":
        print("\nThank You!")
        break
