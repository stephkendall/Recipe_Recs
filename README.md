# Recipe Recommendations

Creating a collaborative-based recommendation system that incorporates user input in the form of ingredient keywords to make accurate suggestions for dinner

### Data Sources

5k+ main dish recipes with 13k+ ingredients were scraped from AllRecipes.com, including the nutritional information, total cooking time, average user rating, instructions, and an image of the recipe. Natural Language Processing through NLTK and SpaCy was implemented to parse the units from ingredients and other stop words to lemmatize the ingredients into a uniform list.

### EDA

Correlative effects of the continuous variables were assessed, determining that carbohydrates, protein, sodium, and cholesterol were the only variables that had some form of negative correlation between them. Total fat and calories were positively related, as expected with the nature of these nutrients.

![Heat Map](Misc/heatmap.pdf)

### Results

Knowledge-based recommendations were made from user-inputs to determine how similar their ingredients are to the recipes in the dataframe. Cosine similarities were used to assess the similarity between ingredients on hand and the recipes indexed. 
