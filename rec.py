import streamlit as st


# import pandas as pd
# import numpy as np
# import random
# from tqdm import tqdm
# from gensim.models import Word2Vec 

# st.title('Recipe Recommender')

# data = (r'/Users/stephaniekendall/Desktop/Errthang/Flatiron/projects/FP_Practice/final.csv')

# @st.cache
# def load_data(nrows):
#     df = pd.read_csv(data,nrows=nrows)
#     return df
# df = load_data(5000)

# st.dataframe(df)

# def no_commas(doc):
#     no_commas = [t for t in doc if t!=',']
#     return(no_commas)

# from nltk.tokenize import word_tokenize

# keywords = df['ings_str'].tolist()
# keywords = [word_tokenize(keyword.lower()) for keyword in keywords]
# keywords = [no_commas(kw) for kw in keywords]


# from gensim.corpora.dictionary import Dictionary
# dictionary_keywords = Dictionary(keywords)
# corpus = [dictionary_keywords.doc2bow(keyword) for keyword in keywords]  


# from gensim.models.tfidfmodel import TfidfModel
# tfidf = TfidfModel(corpus)


# from gensim.similarities import MatrixSimilarity
# sims = MatrixSimilarity(tfidf[corpus], num_features=len(dictionary_keywords))

# def keywords_recommendation(keywords, number_of_hits):
#     dictionary = Dictionary()
#     tfidf = TfidfModel()
#     query_doc_bow = dictionary.doc2bow(keywords) # get a bag of words from the query_doc
#     query_doc_tfidf = tfidf[query_doc_bow] #convert the regular bag of words model to a tf-idf model where we have tuples
#     # of the recipe name and it's tf-idf value for the recipe

#     similarity_array = sims[query_doc_tfidf] # get the array of similarity values between our recipe and every other recipe. 
#     #So the length is the number of recipe we have. To do this, we pass our list of tf-idf tuples to sims.

#     similarity_series = pd.Series(similarity_array.tolist(), index=df.name.values) #Convert to a Series
#     top_hits = similarity_series.sort_values(ascending=False)[:number_of_hits] #get the top matching results, 
#     # i.e. most similar recipes

#     # Print the top matching eecipes
#     print("Our top %s most similar recipes for the keywords %s are:" %(number_of_hits, keywords))
#     for idx, (recipe,score) in enumerate(zip(top_hits.index, top_hits)):
#         return ("%d '%s' with a similarity score of %.3f" %(idx+1, recipe, score))
        
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

#Text/Title
st.title('Streamlit Tutorials')

# Header/Subheader
st.header('This is a header')
st.subheader('This is a subheader')

# Text
st.text('Hello Streamlit')



# Error/Colorful Text
st.success('Successful')
st.info('Information')
st.warning('This is a warning')
st.error('This is an error')
st.exception('NameError("name three not defined")')

# # Get Help Info About Python
# st.help(range)

# Writing Text | Functions
st.write('Text with write')
st.write(range(10))

# # Images
from PIL import Image
img = Image.open("pizza.jpeg")
st.image(img,width=300,caption='Pizza')

# # Videos
# # vid_file = open("example.mp4","rb")
# # vid_bytes = vid_file.read()
# # or vid_file = open("example.mp4","rb").read()
# # st.video(vid_file)

# # Audio
# # audio_file = open("examplemusic.mp3","rb").read()
# # st.audio(audio_file,format='audio/mp3')

# # Widget
# Checkbox
if st.checkbox("Show/Hide"):
    st.text("Showing or Hiding Widget")
    
# Radio Buttons
status = st.radio("What is your status", ("Active","Inactive"))

if status == "Active":
    st.success("You are Active")
else:
    st.warning("Inactive, Activate!")
    
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


