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


df = pd.read_csv(r'/Users/stephaniekendall/Desktop/Errthang/Flatiron/projects/FP_Practice/my_info.csv')

if st.checkbox('Show dataframe'):
    st.write(df)


import pickle 

with open("lemmas.txt", "rb") as fp:   # Unpickling
    lemmas = pickle.load(fp)
    
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
        st.write("%d '%s' with a similarity score of %.3f" %(idx, recipe))
            

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


# def text_analyzer(my_text):
#     nlp = spacy.load('en_core_web_lg')
#     docx = nlp(my_text)
#     tokens = [token.text for token in docx]
#     allData = [('"Tokens":{},\n"Lemma":{},\n"Point-of-Speech":{}'.format(token.text, token.lemma_, token.pos_)) for token in docx]
#     return allData


# checkbox_1 = st.sidebar.checkbox('Get Your Rec On!')

# option = st.sidebar.text_area('Enter Ingredient(s)','Here')
# # if checkbox_1 == True:
    


# def main():

# Tokenization
# if st.sidebar.checkbox('Show Tokens and Lemma' == True):
#     st.subheader('Tokenize Your Text')
#     message = st.text_area('Enter Text', 'Type Here')
# #     if st.button('Analyze'):
# #         nlp_result = text_analyzer(message)
# #         st.json(nlp_result)
# if checkbox_1 == True:
#     st.subheader('Get Your Recommendations On!')
#     option = st.text_area('Enter Ingredient(s)', 'Type Here')
#     prediction = keywords_recommendation(option)
#     prediction2 = pd.DataFrame(keywords_recommendation((search_terms),5), columns = ["Product name", "% similarity"])
#     st.table(prediction2)

    

    

#     rand_topic_distr, summ, tags, most_sim, most_dif = get_rec_random(matrix, final_lda_dtm, talk_df, 5)
#     st.plotly_chart(rand_topic_distr)
#     st.subheader('SUMMARY:')
#     st.write(summ)
#     st.subheader('CURRENT TED TAGS:')
#     tag_str_ = ', '.join(tags)
#     st.write(tag_str_)

#     st.subheader('MOST SIMILAR TALKS:')
#     for talk in most_sim:
#         st.write(talk_df.iloc[talk]['title'])

#     st.subheader('MOST DIFFERENT TALKS:')
#     for talk in most_dif:
#         st.write(talk_df.iloc[talk]['title'])
            

# search_terms = st.sidebar.text_input('Enter ingredients',"HERE")
# prediction = keywords_recommendation((search_terms),5)
# prediction2 = pd.DataFrame(keywords_recommendation((search_terms),5), columns = ["Product name", "% similarity"])
# st.table(prediction2)

# option = st.sidebar.text_input("YOUR INGREDIENTS")

# if st.sidebar.button('Submit'):
#     keywords_recommendation([search_terms],5)
    
# st.write("YOUR INGREDIENTS")
    
# # Text Input
# firstname = st.text_input("Enter Your First Name","Type Here..")
# if st.button("Submit"):
#     result = firstname.title()
#     st.success(result)

# # Text Area
# message = st.text_area("Enter Your Message","Type Here..")
# if st.button("Enter"):
#     result = message.title()
#     st.success(result)

 # # SelectBox
# occupation = st.selectbox("Your Occupation",["Programmer","DataScientist","Chef","Doctor"])
# st.write("You selected this option ", occupation)

# # MultiSelect
# location = st.multiselect("Where do you work?",("London","New York","Accra","Kieve","Paris"))
# st.write("You selected",len(location),"locations")


# checkbox_1 = st.sidebar.checkbox('Recommendations based on keywords')

# option = st.sidebar.selectbox('Select a keyword', keywords)

# keywords_recommendation(st.multiselect,8)

# #Text/Title
# st.title('Streamlit Tutorials')

# # Header/Subheader
# st.header('This is a header')
# st.subheader('This is a subheader')

# # Text
# st.text('Hello Streamlit')



# # Error/Colorful Text
# st.success('Successful')
# st.info('Information')
# st.warning('This is a warning')
# st.error('This is an error')
# st.exception('NameError("name three not defined")')

# # # Get Help Info About Python
# # st.help(range)

# # Writing Text | Functions
# st.write('Text with write')
# st.write(range(10))

# # # Images
# from PIL import Image
# img = Image.open("pizza.jpeg")
# st.image(img,width=300,caption='Pizza')

# # Videos
# # vid_file = open("example.mp4","rb")
# # vid_bytes = vid_file.read()
# # or vid_file = open("example.mp4","rb").read()
# # st.video(vid_file)

# # Audio
# # audio_file = open("examplemusic.mp3","rb").read()
# # st.audio(audio_file,format='audio/mp3')

# # # Widget
# # Checkbox
# if st.checkbox("Show/Hide"):
#     st.text("Showing or Hiding Widget")
    
# # Radio Buttons
# status = st.radio("What is your status", ("Active","Inactive"))

# if status == "Active":
#     st.success("You are Active")
# else:
#     st.warning("Inactive, Activate!")
    
# # SelectBox
# occupation = st.selectbox("Your Occupation",["Programmer","DataScientist","Chef","Doctor"])
# st.write("You selected this option ", occupation)

# # MultiSelect
# location = st.multiselect("Where do you work?",("London","New York","Accra","Kieve","Paris"))
# st.write("You selected",len(location),"locations")

# # Slider
# age = st.slider("What is your age?",1,100)

# # Buttons
# st.button("Simple Button")

# if st.button("About"):
#     st.text("Streamlit is Cool")
    
    
# # Text Input
# firstname = st.text_input("Enter Your First Name","Type Here..")
# if st.button("Submit"):
#     result = firstname.title()
#     st.success(result)

# # Text Area
# message = st.text_area("Enter Your Message","Type Here..")
# if st.button("Enter"):
#     result = message.title()
#     st.success(result)

# # Date Input
# import datetime
# today = st.date_input("Today is", datetime.datetime.now())


# # Time
# the_time = st.time_input("The time is", datetime.time())


# # Displaying JSON
# st.text("Display JSON")
# st.json({'name':'Jesse','gender':'male'})

# # Display Raw Code
# st.text("Display Raw Code")
# st.code("import numpy as np")

# # Display Raw Code
# with st.echo():
#     # This will also show as a comment
#     import pandas as pd
#     df = pd.DataFrame()

# # # Progress Bar
# # import time
# # my_bar = st.progress(0)
# # for p in range(10):
# #     my_bar.progress(p + 1)

# # # Spinner
# # with st.spinner("Loading"):
# #     time.sleep(5)
# # st.success("Finished")


# # # Balloons
# # st.balloons()

# # SIDEBARS
# st.sidebar.header("About")
# st.sidebar.text("This is Streamlit")

# # Functions
# @st.cache
# def run_fxn():
#     return range(100)

# st.write(run_fxn())

# # Plot
# st.pyplot()

# # DataFrames
# st.dataframe(df)

# # Tables
# st.table(df)


