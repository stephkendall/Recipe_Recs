import streamlit as st
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import pickle
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt


with open('tfidf_matrix.pickle','rb') as file:
    tfidf = pickle.load(file)

with open('sims_matrix.pickle', 'rb') as file:
    sims = pickle.load(file)


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
#    print(color.BOLD + df.Name.iloc[img_urls] + color.END)

#    urls = [url for url in df.img_urls[img_urls].split(',')]

#    images = []
#    for i in range(0, len(urls) - 1):
#       url = urls[i].strip()
#       response = requests.get(url)
#       img = Image.open(BytesIO(response.content))
#       images.append(img)

#       if img.size[0] > 125:
#          st.image(img)
#          plt.figure()


# def Ingredients(ind):
#    print(color.BOLD + df.Name.iloc[ind] + color.END)
#    ing = {key + 1: i.strip() for key, i in
#           enumerate(df.Ingredients[ind].split(','))}
#    ingredient = {}
#    for key, value in ing.items():
#       if value:
#          ingredient[key] = value

#    return ingredient


# def Directions(ind):
#    print(color.BOLD + df.Name.iloc[ind] + color.END)
#    direction = {key + 1: i.strip() for key, i in enumerate(df.Directions[ind].split('.'))}
#    directionss = {}
#    for key, value in direction.items():
#       if value:
#          directionss[key] = value

#    return directionss


# def Stats(ind):
#    print(color.BOLD + df.Name.iloc[ind] + color.END)
#    return df[['Prep_time', 'Cook_time', 'Calorie', 'Rating', 'Review_count']].iloc[ind]



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
        
        
# def keywords_recommendation(keywords, number_of_hits):
#     query_doc_bow = dictionary.doc2bow(keywords) # get a bag of words from the query_doc
#     query_doc_tfidf = tfidf[query_doc_bow] #convert the regular bag of words model to a tf-idf model where we have tuples
#     # of the movie ID and it's tf-idf value for the recipe

#     similarity_array = sims[query_doc_tfidf] # get the array of similarity values between our ingredients and every other recipe. 
#     #So the length is the number of recipes we have. To do this, we pass our list of tf-idf tuples to sims.

#     similarity_series = pd.Series(similarity_array.tolist(), index=df_final.name.values) #Convert to a Series
#     top_hits = similarity_series.sort_values(ascending=False)[:number_of_hits] #get the top matching results, 
#     # i.e. most similar recipes
    
#     # Get the recipe indices
#     rec_indices = [i[0] for i in dfs]
    
#     # Print the top matching recipes
#     print("Our top %s most similar recipes for the keywords %s are:" %(number_of_hits, keywords))
#     for idx, (recipe) in enumerate(zip(top_hits.index, top_hits)):
#         print("%d '%s' with a similarity score of %.3f" %(idx+1, recipe, score))

# meal = df.Name


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


# def main():
#     st.title('Meal Recommendations')

#     choice = st.selectbox('choose a meal', meal)
#     # if st.button('Find Meal'):
#     rec = st.text(get_recommendations((choice))[0])

#     all_services = ['Ingredients', 'Directions', 'Stats', 'Photos']
#     service_choice = st.selectbox('Please select the service', all_services)
#     if service_choice == 'Ingredients':
#         cho = st.selectbox('chose one', get_recommendations((choice))[1])
#         st.success('Ingredients of Your Choice')
#         st.text(df.Name.iloc[indices[choice]])
#         st.text(Ingredients(indices[choice]))
#         st.success('ingredients of The recommendation')
#         st.text(df.Name.iloc[cho])
#         st.text(Ingredients(cho))





# if __name__== '__main__':
#     main()