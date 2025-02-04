import streamlit as st
#Packages
import pandas as pd
import pickle
import numpy as np
import random
from tqdm import tqdm
from gensim.models import Word2Vec 
import spacy
import plotly.graph_objects as go
import itertools
import pandas as pd
import re
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string 
import time

from bs4 import BeautifulSoup
from collections import Counter
from itertools import chain
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords as sw, wordnet as wn
from pandas.io.json import json_normalize
from pprint import pprint
from recipe_scrapers import scrape_me

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm_notebook as tqdm


# with open('cosine_sim.pickle', 'rb') as file:
#    cosine_sim = pickle.load(file)

df = pd.read_csv(r'/Users/stephaniekendall/Desktop/Errthang/Flatiron/projects/Recipe_Recs/final.csv')

# with open('tfidf_matrix.pickle','rb') as file:
#     tfidf = pickle.load(file)

# with open('cosine_sim.pickle', 'rb') as file:
#    cosine_sim = pickle.load(file)


# cosine_sim = linear_kernel(tfidf, tfidf)
# indices = pd.Series(df.index, index=df['Name'])

# img = Image.open('macrofit-meals.jpg')
# st.image(img, width = 700)


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# def Pics(img_urls):
#     print(color.BOLD + df.Name.iloc[img_urls] + color.END)

#     urls = [url for url in df.img_urls[img_urls].split(',')]

#     images = []
#     for i in range(0, len(urls) - 1):
#         url = urls[i].strip()
#         response = requests.get(url)
#         img = Image.open(BytesIO(response.content))
#         images.append(img)

#         if img.size[0] > 125:
#             st.image(img)
#             plt.figure()


def Ingredients(ind):
    print(color.BOLD + df.name.iloc[ind] + color.END)
    ing = {key + 1: i.strip() for key, i in enumerate(df.Ingredients[ind].split(','))}
    ingredient = {}
    for key, value in ing.items():
        if value:
            ingredient[key] = value

    return ingredient


# def Directions(ind):
#     print(color.BOLD + df.Name.iloc[ind] + color.END)
#     direction = {key + 1: i.strip() for key, i in enumerate(df.Directions[ind].split('.'))}
#     directionss = {}
#     for key, value in direction.items():
#         if value:
#             directionss[key] = value

#     return directionss


def Stats(ind):
    print(color.BOLD + df.name.iloc[ind] + color.END)
    return df[['rating', 'prep_time', 'cook_time', 'total_time', 'avg_rating','calories_g','fat_g']].iloc[ind]



def get_recommendations(title, cosine_sim=cosine_sim):
   # Get the index of the movie that matches the title
   idx = indices[title]

   # Get the pairwsie similarity scores of all movies with that movie
   sim_scores = list(enumerate(cosine_sim[idx]))

   # Sort the movies based on the similarity scores
   sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

   # Get the scores of the 10 most similar movies
   sim_scores = sim_scores[1:11]

   # Return the top 10 most similar movies
   return df.name.iloc[movie_indices], movie_indices


# meal = df.Name
def main():
    st.title('Meal Recommendations')

    choice = st.selectbox('choose a meal', meal)
   # if st.button('Find Meal'):
    rec = st.text(get_recommendations((choice))[0])

    all_services = ['Ingredients', 'Stats']
    service_choice = st.selectbox('Please select the service', all_services)
    if service_choice == 'Ingredients':
        cho = st.selectbox('chose one', get_recommendations((choice))[1])
        st.success('Ingredients of Your Choice')
        st.text(df.Name.iloc[indices[choice]])
        st.text(Ingredients(indices[choice]))
        st.success('ingredients of The recommendation')
        st.text(df.Name.iloc[cho])
#         st.text(Ingredients(cho))

#    if service_choice == 'Directions':
#       cho = st.selectbox('chose one', get_recommendations((choice))[1])
#       st.success('Directions of Your Choice')
#       st.text(df.Name.iloc[indices[choice]])
#       st.text(Directions(indices[choice]))
#       st.success('Directions of The recommendation')
#       st.text(df.Name.iloc[cho])
#       st.text(Directions(cho))

#    if service_choice == 'Stats':
#       cho = st.selectbox('chose one', get_recommendations((choice))[1])
#       st.success('Stats of Your Choice')
#       st.text(df.Name.iloc[indices[choice]])
#       st.text(Stats(indices[choice]))
#       st.success('Stats of The recommendation')
#       st.text(df.Name.iloc[cho])
#       st.text(Stats(cho))

#    if service_choice == 'Photos':
#       cho = st.selectbox('chose one', get_recommendations((choice))[1])
#       st.success('Photos of Your Choice')
#       st.text(df.Name.iloc[indices[choice]])
#       st.text(Pics(indices[choice]))
#       st.success('Photos of The recommendation')
#       st.text(df.Name.iloc[cho])
#       st.text(Pics(cho))


# img = Image.open('food.jpg')
# st.image(img, width = 750)




if __name__== '__main__':
    main()