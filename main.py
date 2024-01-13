import pandas as pd

df = pd.read_csv('movie_data.csv')
df = df.drop('Poster Path',axis=1)
df = df.drop(['Original Language','Video','Title','Release Date'], axis =1)
df.dropna(inplace=True)
C = df['Vote Average'].mean()
m = df['Vote Count'].quantile(0.9)
q_movies = df.copy().loc[df['Vote Count'] >= m]


def weighted_rating(x, m = m,C = C):
    v = x['Vote Count']
    R = x['Vote Average']
    #Calculation based on the IMDB formula
    return (v/(v+m) *R + (m/(m+v)) * C)

df['Weighted Rating'] = df.apply(weighted_rating, axis=1)

#So now we have added the Rating We can go through Buliding content based learning
from sklearn.feature_extraction.text import TfidfVectorizer

'''we create an instance of TfidfVectorizer. The stop_words='english' 
argument specifies that common English words like "the," "and," and 
"is" should be removed from the text data because they typically 
don't carry much meaning.'''

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Overview'])

from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)

def get_recommendations(movie_title, cosine_sim=cosine_sim, num_recommendations=10):
    idx = df[df['Original Title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(num_recommendations+1)]  # Top N similar movies (excluding itself)
    movie_indices = [i[0] for i in sim_scores]
    return df[['Original Title', 'Weighted Rating']].iloc[movie_indices]

# let try out

input_movie = "Harry Potter and the Philosopher's Stone"
recommendations = get_recommendations(input_movie)
print(recommendations)