import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import texthero as hero
#from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
import warnings
warnings.filterwarnings("ignore")

def filter_reviews(df):
    #Creating column with clean text
    df['cleanText'] = hero.clean(df['Text'])
    
    #Creating Vader Object
    vaderobj = SentimentIntensityAnalyzer() 
    
    #Creating columns with positive and negative sentiment derived from Vader
    df['VaderPos'] = df.apply(lambda row : vaderobj.polarity_scores(row['cleanText'])['pos'], axis = 1)
    df['VaderNeg'] = df.apply(lambda row : vaderobj.polarity_scores(row['cleanText'])['neg'], axis = 1)
    #df['Blob Rating'] = df.apply(lambda row : TextBlob(row['cleanText']).sentiment[0], axis = 1)
    
    #Filtering dataset using
    # - >0.65 as cutoff for high positive sentiment
    # - <=2 as criteria for low rating
    filtered_df = df[(df['Star']<=2) 
                     & (df['VaderPos'] > 0.65) 
                     ][['Text','Star','VaderPos']]
    
    return filtered_df

def create_wordcloud(df):
    fig = WordCloud().generate(' '.join(df['Text']))
    plt.imshow(fig)
    plt.axis("off")
    