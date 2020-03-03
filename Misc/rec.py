import streamlit as st 

#Packages
import pandas as pd
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


nlp = spacy.load('en_core_web_lg')


# regex for separating ingredients list
SEPARATOR_RE = re.compile(r'^([\d\s*[\d\.,/]*)\s*(.+)')

# create a normalized string for ingredients
def normalize(st):
    """

    :param st:
    :return:
    """
    return re.sub(r'\s+', ' ', SEPARATOR_RE.sub('\g<1> \g<2>', st)).strip()


def escape_re_string(text):
    """

    :param text:
    :return:
    """
    text = text.replace('.', '\.')
    return re.sub(r'\s+', ' ', text)

# list of common units of measurements for ingredients
UNITS = {"cup": ["cups", "cup", "c.", "c"], "fluid_ounce": ["fl. oz.", "fl oz", "fluid ounce", "fluid ounces"],
         "gallon": ["gal", "gal.", "gallon", "gallons"], "ounce": ["oz", "oz.", "ounce", "ounces"],
         "pint": ["pt", "pt.", "pint", "pints"], "pound": ["lb", "lb.", "pound", "pounds"],
         "quart": ["qt", "qt.", "qts", "qts.", "quart", "quarts"],
         "tablespoon": ["tbsp.", "tbsp", "T", "T.", "tablespoon", "tablespoons", "tbs.", "tbs"],
         "teaspoon": ["tsp.", "tsp", "t", "t.", "teaspoon", "teaspoons"],
         "gram": ["g", "g.", "gr", "gr.", "gram", "grams"], "kilogram": ["kg", "kg.", "kilogram", "kilograms"],
         "liter": ["l", "l.", "liter", "liters"], "milligram": ["mg", "mg.", "milligram", "milligrams"],
         "milliliter": ["ml", "ml.", "milliliter", "milliliters"], "pinch": ["pinch", "pinches"],
         "dash": ["dash", "dashes"], "touch": ["touch", "touches"], "handful": ["handful", "handfuls"],
         "stick": ["stick", "sticks"], "clove": ["cloves", "clove"], "can": ["cans", "can"], "large": ["large"],
         "small": ["small"], "scoop": ["scoop", "scoops"], "filets": ["filet", "filets"], "sprig": ["sprigs", "sprig"],
        "fillets": ["fillet", "fillets"],"jar":["jar","jars"], "packet": ["packet","packets"], "package": ["package","packages"], 
         "bottle":["bottle","bottles"],"slice":["slice","slices"], "cube":["cube","cubes"], 
         "container":["container","containers"],"envelope":["envelope","envelopes"], "ground":["ground"], "quick":["quick"],
        "to taste":["to taste", "or to taste"],'minced':['minced']}


# numbers to separate quantities from ingredients
NUMBERS = ['seventeen', 'eighteen', 'thirteen', 'nineteen', 'fourteen', 'sixteen', 'fifteen', 'seventy', 'twelve',
           'eleven', 'eighty', 'thirty', 'ninety', 'twenty', 'seven', 'fifty', 'sixty', 'forty', 'three', 'eight',
           'four', 'zero', 'five', 'nine', 'ten', 'one', 'six', 'two', '½', '⅓','¼', '⅛', '¾','half',
          'halves','pieces','cubes','chunks','whole','cube', 'extra', 'pieces','piece','cube', 'long', 
           'jumbo', 'small','medium', 'large', 'slices', 'sliced', 'cubes','cubed','minced','divided','or to taste',
          'canned','crushed','lean', 'all-purpose', 'or as needed','needed','thinly','freshly','or more', 'to taste',
          'to cover','chopped','-inch','inch','smashed and cut into 1-inch pieces','cut','and','halved','quartered',
          'can','ground','breast','thighs','fillet','fillets','bunch','fresh','to cover','lean','crushed','finely',
          'packed','diced','boneless','skinless','shredded','beaten','light','in half']

prepositions = ["of"]

a = list(chain.from_iterable(UNITS.values()))
a.sort(key=lambda x: len(x), reverse=True)
a = map(escape_re_string, a)

PARSER_RE = re.compile(
    r'(?P<quantity>(?:[\d\.,][\d\.,\s/]*)?\s*(?:(?:%s)\s*)*)?(\s*(?P<unit>%s)\s+)?(\s*(?:%s)\s+)?(\s*(?P<name>.+))?' % (
        '|'.join(NUMBERS), '|'.join(a), '|'.join(prepositions)))


def parse(st):
    """

    :param st:
    :return:
    """
    st = normalize(st)
    res = PARSER_RE.match(st)
    
    return ((res.group('name') or '').strip())


df = pd.read_csv(r'/Users/stephaniekendall/Desktop/Errthang/Flatiron/projects/Recipe_Recs/CSV Files/final.csv')
df = df.iloc[:5201]

if st.checkbox('Show dataframe'):
    st.write(df)


all_ings = df['ingredients'].str.replace("]",'').str.replace("'",'').str.replace("[",'').str.replace("(",'').str.replace(')','').str.replace('"','').str.split(',') 

parsed_ingredients = []
for ings in all_ings:
    between = []
    for ing in ings:
        if len(parse(ing)) > 1:
            between.append(parse(ing))
    parsed_ingredients.append(between)
    
import spacy

nlp = spacy.load("en_core_web_lg")


lemmas = []
pos_text = []
for val in tqdm(parsed_ingredients):
    between = []
    for v in val:
        pp = nlp(v)
        for p in pp:
            if p.lemma_ != 'mix' and p.lemma_ != 'powder' and p.lemma_ != 'inch' and p.lemma_ !=  'uncooked' and p.lemma_ !=  'whole' and p.lemma_ !=  'boneless' and p.lemma_ !=  'heavy' and p.lemma_ !=  'such' and p.lemma_ !=  'style' and p.lemma_ !=  'thick' and p.lemma_ !=  'root' and p.lemma_ !=  'bite' and p.lemma_ != 'bottle' and p.tag_ != 'NNS' and p.lemma_ != 'package' and p.lemma_ != 'envelope' and p.lemma_ != 'cup' and p.lemma_ != 'ola' and p.lemma_ != 'cubed' and p.lemma_ != 'skinless' and p.lemma_ != 'slices' and p.lemma_ != 'rest' and p.lemma_ != 'undrained' and p.lemma_ != 'cooking' and p.lemma_ != 'cut' and p.lemma_ != 'purpose' and p.lemma_ != 'optional' and p.lemma_ != 'extra' and p.pos_ != 'X' and p.lemma_ != 'thigh' and p.lemma_ != 'aluminum' and p.lemma_ != 'foil' and p.lemma_ != 'spray' and p.lemma_ != 'fillet' and p.pos_ != 'SCONJ' and p.text != 'ground' and p.lemma_ != 'breast' and p.lemma_ != 'half' and p.dep_ != 'prep' and p.dep_ != 'mark' and p.pos_ != 'NUM' and p.dep_ != 'pobj' and p.dep_ != 'advmod' and p.pos_ != 'VERB' and p.pos_ != 'CCONJ' and p.pos_ != 'DET' and p.pos_ != 'PUNCT' and p.pos_ != 'PART' and p.pos_ != 'ADP' and p.dep_ != 'conj' and p.dep_ != 'nsubj' and p.dep_ != 'nmod':
                if p.lemma_ not in between:
                    between.append(p.lemma_)
    lemmas.append(between)
    

    
st.title('Recipe Recommender')
""" NLP and Recommendations"""

# Processing Keywords
# flatten list for count of values
flat_list = [' '.join(x) for x in lemmas]

from nltk.tokenize import word_tokenize
keywords = [word_tokenize(keyword.lower()) for keyword in flat_list]

def no_commas(doc):
    no_commas = [t for t in doc if t!=',']
    return(no_commas)



keywords = [no_commas(kw) for kw in keywords]
processed_keywords = keywords

from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(processed_keywords) # create a dictionary of words from our keywords

corpus = [dictionary.doc2bow(doc) for doc in processed_keywords] 

from gensim.models.tfidfmodel import TfidfModel
tfidf = TfidfModel(corpus) #create tfidf model of the corpus

from gensim.similarities import MatrixSimilarity
# Create the similarity data structure. This is the most important part where we get the similarities between the movies.
sims = MatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

def keywords_recommendation(keys):
    number_of_hits=8
    query_doc_bow = dictionary.doc2bow(keys) # get a bag of words from the query_doc
    query_doc_tfidf = tfidf[query_doc_bow] #convert the regular bag of words model to a tf-idf model where we have tuples
    # of the recipe name and it's tf-idf value for the recipe

    similarity_array = sims[query_doc_tfidf] # get the array of similarity values between our recipe and every other recipe. 
    #So the length is the number of recipe we have. To do this, we pass our list of tf-idf tuples to sims.

    similarity_series = pd.Series(similarity_array.tolist(), index=df.name.values) #Convert to a Series
    top_hits = similarity_series.sort_values(ascending=False)[:number_of_hits] #get the top matching results, 
    # i.e. most similar recipes

    # Print the top matching eecipes
    st.write("Our top %s most similar recipes for the keywords %s are:" %(number_of_hits, keys))
    for idx, (recipe) in enumerate(zip(top_hits.index, top_hits)):
        st.write("%d '%s'" %(idx, recipe))
            

#ask user for input
ingredients = st.text_input('Enter ingredient(s) with commas between')


test = st.button('Search for recommended recipes')

#run recommender
if test:
    if ingredients:
        x = ingredients
        st.write(x.split(','))
        st.write(keywords_recommendation(x.split(',')))
        st.success('Searching for recipes similar to your inputs' )

