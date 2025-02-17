#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


movies = pd.read_csv('movies.csv')


# In[3]:


print("Dataset Overview:")
print(movies.head())


# In[4]:


print("\nMissing Values:")
print(movies.isnull().sum())


# In[5]:


movies['genre'] = movies['genre'].fillna('') 
movies['genre'] = movies['genre'].str.replace('|', ' ')


# In[6]:


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genre'])


# In[7]:


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[8]:


def recommend_movie(input_movie):
     if input_movie not in movies['names'].values:
        print(f"Sorry, '{input_movie}' is not in the dataset.")
        return


# In[9]:


input_movie = input("Enter the name of a movie: ")
recommend_movie(input_movie)


# In[10]:


idx = movies.index[movies['names'] == input_movie].tolist()[0]


# In[11]:


sim_scores = list(enumerate(cosine_sim[idx]))


# In[12]:


sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)


# In[13]:


sim_scores = sim_scores[1:6]
movie_indices = [i[0] for i in sim_scores]


# In[14]:


recommended_movies = movies['names'].iloc[movie_indices]


# In[15]:


print(f"\nMovies recommended similar to '{input_movie}':")
for movie in recommended_movies:
    print(f"- {movie}")


# In[ ]:




