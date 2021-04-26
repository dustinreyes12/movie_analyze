import streamlit as st

import imdb
import pickle
import numpy as np
import pandas as pd
import warnings
import datetime as dt
from tqdm import tqdm
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact
from random import randint
from time import sleep
from pytrends.request import TrendReq
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

pytrend = TrendReq(hl='en-US', tz=360)

def get_top_15_actors(title):
    ia = imdb.IMDb()
    actors = []
    try:
        search_results = ia.search_movie(title)
        if search_results:
            movieID = search_results[0].movieID
            movie = ia.get_movie(movieID)
            if movie:
                cast = movie.get('cast')
                topActors = 15
                for actor in cast[:topActors]:
                    actors.append(actor['name'])
        return actors
    except:
        actors.append('')
        return actors

def get_trends_specific(list_names, year):
    df = pd.DataFrame()
#     df2 = pd.DataFrame()
    df['actors'] = list_names
    df['year'] = year
    df['year'] = df['year'].astype(str)
    df['Year1'] = df['year'] + '-' + '01' + '-' + '01'
    df['year'] = df['year'].astype(int)
    df['next_year'] = df['year'] + 1
    df['next_year'] = df['next_year'].astype(str)
    df['Year2'] = df['next_year'] + '-' + '01' + '-' + '01'

    dataset = []
    for x in range(0, len(list_names)):
        keywords = [list_names[x]]
        pytrend.build_payload(
            kw_list=keywords,
            cat=0,
            timeframe=df['Year1'].unique()[0] + " " + df['Year2'].unique()[0],
            geo='US')
        data = pytrend.interest_over_time()
        if not data.empty:
            data = data.drop(labels=['isPartial'], axis='columns')
            dataset.append(data)
    result = pd.concat(dataset, axis=1)
    result.reset_index(inplace=True)
    result['year'] = result['date'].dt.year
    name_cols = result.columns.tolist()[1:-1]
    result.drop('date', axis=1, inplace=True)
    df2 = pd.DataFrame(result[name_cols].sum())
    df2.columns = ['search_interest']
    df2['year'] = year
    return df2

@st.cache()
def movie_analyze(df, year):
    test_df = df[df['startYear'] == year].sort_values(by = 'opening', ascending = False).head()
    best_opening = test_df['title'].values.tolist()[0]
#     print(best_opening)
    top_actors = get_top_15_actors(best_opening)
    df_interest= get_trends_specific(top_actors, year)
    df_interest.sort_values(by = 'search_interest', ascending = False, inplace = True)
    return year, best_opening, df_interest
    # return best_opening

def top_search_visualizer(df, year_choice):
    # return movie_analyze(df, year=year_choice)
    year, movie_title, df_analysis = movie_analyze(df, year=year_choice)
    fig_dims = (15, 10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.barplot(y=df_analysis.index, x="search_interest",
                ax=ax, data=df_analysis)
    ax.set(xlabel='Search Interests', ylabel='Most Searched Actors',
           title=f"{year}'s Most Searched Actors for the Film with the Best Opening Performance: {movie_title}")
    return fig





def filter_data(df_movie_actorgross, number = 10,  year=2011):
    df_movie_actorgross = df_movie_actorgross.groupby(['startYear']).apply(lambda x: x.nlargest(number,['gross'])).reset_index(drop=True)
    test_df = df_movie_actorgross[df_movie_actorgross['startYear'] == year].sort_values(by = 'gross', ascending = True)
    # sns.barplot(x="gross", y="actors", data=test_df)
    # plt.show()
    fig = go.Figure(go.Bar(
        x=test_df['gross'],
        y=test_df['actors'],
        orientation='h'))

    fig.update_layout(title_text=f" Most Popular Actors for {year} based on Movie Gross Performance ",
                    autosize=True,
                    title_x=0.5,
                    xaxis_title='Gross',
                    yaxis_title='Actors')
    return fig

@st.cache(suppress_st_warning=True)
def load_data():
    df_movies2 = pd.read_csv('data_small/lead_actors.csv')
    df_movies2.drop(['release','startYear'], axis =1, inplace = True)
    df_movies3 = pd.read_csv('data_small/titles_complete_info.csv')
    df_movies3 = df_movies3.merge(df_movies2, on='title').sort_values(by='release').reset_index(drop=True)
    df_movies3.dropna(inplace = True)
    df_movies3['top3_actors'] = df_movies3['top3_actors'].apply(lambda x: literal_eval(x))
    col_names = ['title', 'top3_actors', 'gross', 'release', 'startYear']
    expanded_data = []
    for idx, row in df_movies3[col_names].iterrows():
        for name in row['top3_actors']:
            expanded_data.append(
                [row['title'], name.strip(), row['gross'], row['release'], row['startYear']])
    df_movie_expanded = pd.DataFrame(expanded_data, columns=[
                                'title', 'actors', 'gross', 'release', 'startYear'])
    df_movie_actorgross = df_movie_expanded.groupby(['startYear', 'actors'])['gross'].sum().reset_index()
    df_movie_actorgross = df_movie_actorgross[df_movie_actorgross['actors'] != '']
    df_movie_actorgross = df_movie_actorgross[df_movie_actorgross['actors'] != 'Cole Konis']

    df_lists = pd.read_csv('data_small/actor_searches.csv')

    return df_movie_actorgross, df_lists, df_movies3

df_movie_actorgross_orig, df_lists_orig, df_movies3_orig = load_data()


def app():
    
    st.title('Actor Analysis')
    st.markdown(f"<p><strong>Disclaimer: </strong><em> This web application was created by Dustin Reyes. </strong></em>", unsafe_allow_html=True)
    # st.write("This page incorporates analysis on movies from 2011 to 2021 and dashboards to visualize the insights that were observed.")
    st.markdown("""<p align="justify"><em>This page involves analyzing the data about the different actors that appear on films that have complete data for analysis.
                </em>""", unsafe_allow_html=True)

    st.markdown("""<p align="justify"> A commercially successful movie is not without the casts that were involved during production. Actor popularity
                    can be a factor when it comes to a movie's success as this can due to the cast's proven track records, awards and accolades. 
                    The main goal of this page is to analyze the popularity of film actors/actresses in terms of two criterias. The first criteria 
                    involves actor popularity based on movie gross performance. The second criteria then involves identifying the movie with the 
                    best opening performance for a specific year and then analyzing the top 15 actors' search interests.""", unsafe_allow_html=True)


    st.markdown(f"<h2> I. Most Popular Actors in terms of Movie Gross Performance", unsafe_allow_html=True)
    st.write('\n')
    st.write('\n')
    st.markdown("""<p align="justify">This section aims to identify the most popular actors in terms of the gross performances of the movies he/she was casted in. 
                        The idea of popularity in this method shows us that popularity can be tied with the overall gross performances of each actor's movies""", unsafe_allow_html=True)
    
    
    df_movie_actorgross = df_movie_actorgross_orig.copy()
    df_lists = df_lists_orig.copy()
    df_movies3 = df_movies3_orig.copy()

    years = []

    for i in df_movie_actorgross['startYear'].unique():
        years.append(i)

    option_1 = st.selectbox(
        'Pls select the year', years)
    option_2 = st.slider('Pls. choose the number of movies to consider?', 1, 15, 10)

    figure_1 = filter_data(df_movie_actorgross, number = option_2, year = option_1)
    st.plotly_chart(figure_1)


    st.markdown(f"<h2> II. Most Popular Actor of the Film with the Best Opening Performance", unsafe_allow_html=True)
    st.write('\n')
    st.write('\n')
    st.markdown("""<p align="justify">The aim of this section is to identify the most popular actor for the year by means of identifying the movie with
                    the best overall opening performance. The idea for this that movies with a good opening performance were already popular beforehand
                    through marketing efforts therefore, we can observe the top N actors for this film and analyze the search interests for the actors of this particular movie
                    to identify his/her popularity.""", unsafe_allow_html=True)
    
    years2 = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

    # for i in df_movie_actorgross['startYear'].unique():
    #     years2.append(int(i))


    option_4 = st.selectbox(
        'Pls select the year', years2)

    figure_2 = top_search_visualizer(df_movies3, year_choice=option_4)
    # figure_2  = movie_analyze(df_movies3, year=2018)
    st.write(figure_2)
    # st.plotly_chart(figure2)