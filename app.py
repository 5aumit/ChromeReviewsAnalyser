import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from filter_reviews import filter_reviews
from wordcloud import WordCloud
import warnings

st.set_option('deprecation.showPyplotGlobalUse', False)

pd.set_option("display.max_rows", None, "display.max_columns", None,'display.max_colwidth', -1)

st.set_page_config(page_title = 'Review Analyser')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
if st.button("Analyze"):
    with st.spinner("Analyzing"):
        result = filter_reviews(df)
    
    st.write('These reviews had a positive sentiment but a low rating')
    
    st.dataframe(result)
    
    st.write("WordCloud of such reviews:")
    
    fig = WordCloud().generate(' '.join(result['Text']))
    plt.imshow(fig, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()