import streamlit as st

import json
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
import ipywidgets as widgets
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

from bokeh.plotting import figure, output_file, show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker, LogTicker, ContinuousTicker, AdaptiveTicker,
    PrintfTickFormatter,
    ColorBar,
    FactorRange
)
import bokeh.palettes
from bokeh.transform import factor_cmap
from wordcloud import WordCloud, STOPWORDS
from math import floor
from bokeh.io import output_notebook
from bokeh.models import NumeralTickFormatter
from bokeh.resources import CDN
from bokeh.embed import file_html

# %config InlineBackend.figure_format = 'retina'

warnings.filterwarnings('ignore')

@st.cache(suppress_st_warning=True)
def load_data():
    cols = ['tconst', 'title', 'release', 'startYear', 'director', 'budget', 'opening', 'gross',
    'worldwide_gross', 'metacritic_score', 'mpaa_rating', 'budget_mil',
    'opening_mil', 'runtimeMinutes', 'genres', 'averageRating',
    'numVotes']
    df = pd.read_csv('data_small/titles_complete_info.csv', usecols = cols)
    title_basics_df = pd.read_csv('data_small/title_basics.zip')
    title_basics_df = title_basics_df[title_basics_df['startYear'] != '\\N']
    title_basics_df = title_basics_df[title_basics_df['runtimeMinutes'] != '\\N']
    title_basics_df['titleType'] = title_basics_df['titleType'].astype(str)
    title_basics_df['isAdult'] = title_basics_df['isAdult'].astype(int)
    title_basics_df['runtimeMinutes'] = title_basics_df['runtimeMinutes'].astype(
        int)
    title_basics_df['startYear'] = title_basics_df['startYear'].astype(int)
    title_basics_df.drop('endYear', axis=1, inplace=True)
    title_basics_df['startYear'] = title_basics_df['startYear'].apply(clean_year)
    title_basics_df['runtimeMinutes'] = title_basics_df['runtimeMinutes'].apply(clean_year)
    title_basics_df['genres'] = title_basics_df['genres'].apply(clean_genre)
    title_basics_df.dropna(inplace=True, how='any', subset=['startYear', 'runtimeMinutes'])
    title_basics_df = title_basics_df[title_basics_df['titleType'] == 'movie']
    title_basics_df = title_basics_df.replace(r'^\s*$', np.nan, regex=True)
    title_basics_df.dropna(inplace = True)
    title_basics_df.sort_values(by = 'startYear', inplace = True)
    title_basics_df.reset_index(drop = True, inplace = True)


    # Load scrapy json to my_data
    with open('data_small/results_final1.json', 'r') as f:
        my_data = json.load(f)
        
    with open('data_small/results_final2.json', 'r') as f:
        my_data2 = json.load(f)   

    imdb_info1 = pd.DataFrame(my_data)
    imdb_info2 = pd.DataFrame(my_data2)
    imdb_info = imdb_info1.append(imdb_info2, ignore_index=True)
    imdb_info = imdb_info.replace(r'^\s*$', np.nan, regex=True)
    imdb_info.rename({'title_id':'tconst'}, axis =1, inplace = True)
    imdb_info = imdb_info.merge(title_basics_df, on = 'tconst')
    imdb_info_withbudget = imdb_info.dropna(subset = ['budget'])
    imdb_info_withbudget['budget'] = imdb_info_withbudget['budget'].astype(int)
    return df, title_basics_df, imdb_info_withbudget


# Helper Functions
def clean_year(y):
    # Return year as an integer or 'NaN' if empty
    import numpy as np
    try:
        return int(y)
    except:
        return np.nan


def clean_genre(y):
    # Return only the first genre listed
    y = str(y)
    if y == '\\N':
        return ''
    return y.split(',')[0].strip()

# @st.cache(suppress_st_warning=True)
def movie_analyzer(df_movies, category='Metacritic Score', year=2011):
    df_movies_sorted = df_movies.groupby(['startYear']).apply(
        lambda x: x.nlargest(10, [category])).reset_index(drop=True)
    df_movies_sorted = df_movies_sorted[df_movies_sorted['startYear'] == year]
    df_movies_sorted = df_movies_sorted.sort_values(category, ascending=True)
    fig = go.Figure(go.Bar(
        x=df_movies_sorted[category],
        y=df_movies_sorted['title'],
        orientation='h'))

    fig.update_layout(title_text=f"Movie Rankings by: {category}",
                    title_x=0.5, xaxis={
                        'categoryorder': 'total ascending'},
                    xaxis_title=category,
                    yaxis_title='Movies')
    # fig.show()
    return fig

# @st.cache(suppress_st_warning=True)
def runtimemovie_analyzer(title_basics_df, number=10, genre='All'):
    if genre == 'All':
        df = title_basics_df.nlargest(number, 'runtimeMinutes')
        df.sort_values(by='runtimeMinutes', ascending=True, inplace=True)
        fig = go.Figure(go.Bar(
            x=df['runtimeMinutes'],
            y=df['primaryTitle'],
            orientation='h'))

        fig.update_layout(title_text=f"Top {number} Longest Movies with {genre} genres ",

                        title_x=0.5,
                        xaxis_title='Runtime (Minutes)',
                        yaxis_title='Movies')
        return fig
    else:
        df = title_basics_df[title_basics_df['genres'] == genre]
        df = df.nlargest(number, 'runtimeMinutes')
        df.sort_values(by='runtimeMinutes', ascending=True, inplace=True)
        fig = go.Figure(go.Bar(
            x=df['runtimeMinutes'],
            y=df['primaryTitle'],
            orientation='h'))

        fig.update_layout(title_text=f"Top {number} Longest Movies with {genre} genre ",
                        autosize=True,
                        title_x=0.5,
                        xaxis_title='Runtime (Minutes)',
                        yaxis_title='Movies')
        return fig

def genre_opening_analyzer(df_movies, category = 'Opening'):
    genre_list = ['Action', 'Biography', 'Comedy', 'Drama', 'Adventure', 'Crime', 'Horror', 'Animation']
    grouped_data = df_movies.groupby(['startYear', 'genres'])[category].mean().reset_index()
    fig = go.Figure()
    for i in genre_list:
        fig.add_trace(
            go.Scatter(
                y=grouped_data[grouped_data['genres'] == i][category].values,
                x=grouped_data[grouped_data['genres'] == i]['startYear'].values,
                name=i
            ))
    fig.update_layout(
        autosize=False,
        title= f'{category} Performance for each Genre Across the Years',
        title_x=0.5,
        xaxis_title='Years',
        yaxis_title=f'{category} Performance',
        width=800,
        height=650,
        margin=dict(
            l=0,
            r=0,
            b=100,
            t=100,
            pad=0,
        ),
            paper_bgcolor='rgba(0,0,0,0)',
        # showgrid = True,
        plot_bgcolor='Grey',
        legend_title_text='Genres'
        # paper_bgcolor="LightSteelBlue"
    )
    return fig



df_movies_orig, title_basics_df_orig, imdb_info_withbudget_orig = load_data()

# @st.cache(suppress_st_warning=True)
def app():
    st.title('Analysis on Movies from 2011 - 2021')
    st.markdown(f"<p><strong>Disclaimer: </strong><em> This web application was created by Dustin Reyes. </strong></em>", unsafe_allow_html=True)
    # st.write("This page incorporates analysis on movies from 2011 to 2021 and dashboards to visualize the insights that were observed.")
    st.markdown("""<p align="justify"><em>This page incorporates analysis on movies from 2011 to 2021 and dashboards to visualize the insights that were observed. 
                It must be noted that movies with complete information, released in theaters and with reliable sources are only 
                considered for this analysis.</em>""", unsafe_allow_html=True)



    # df_movies = pd.read_csv('data/titles_complete_info.csv', usecols = cols)
    df_movies = df_movies_orig.copy()
    title_basics_df = title_basics_df_orig.copy()
    imdb_info_withbudget = imdb_info_withbudget_orig.copy()

    df_movies.dropna(subset = ['worldwide_gross', 'metacritic_score'], inplace = True)
    df_movies.reset_index(drop = True, inplace = True)
    df_movies.sort_values(by = 'release', inplace = True)
    df_movies.reset_index(drop = True, inplace = True)
    df_movies.rename({'worldwide_gross': 'Worldwide Gross', 'metacritic_score': 'Metacritic Score',
                  'budget': 'Budget', 'opening': 'Opening', 
                  'gross': 'Gross', 
                  'runtimeMinutes': 'Runtime (Minutes)',
                  'averageRating': 'Average Rating', 'numVotes': 'Number of Votes'},
                axis = 1, inplace = True)

    st.markdown("""<p align="justify"> A commercially successful movie not only provides entertainment to the audience but also enables film producers to generate significant profits. 
    Several factors such as veteran actors, social media presence, popularity, and release time are important for profitability, 
    but they do not always guarantee how a movie will have a great reception to the audience. 
    In this page, we sought to understand temporal patterns affecting movie opening performance, 
    see how popular genres change over years, see movie rankings based on chosen metrics, observe movie runtimes across different genres and observe changes in movie ratings and vote averages over time""", unsafe_allow_html=True)
    # st.write("See `apps/home.py` to know how to use it.")
    st.markdown(f"<h2> I. Temporal Pattern of Movie Openings", unsafe_allow_html=True)
    st.markdown("""<p align="justify">This section aims to analyze the months wherein movies have the best opening performance. 
                        The analysis of temporal patterns across the years enables film makers to strategically release films on months wherein such movies are in demand""", unsafe_allow_html=True)

    df_movies['month'] = pd.DatetimeIndex(df_movies['release']).month
    opening_by_month_year = df_movies.groupby(["startYear","month"]).Opening.mean().reset_index()
    newdata = ColumnDataSource(opening_by_month_year)

    mapper = LinearColorMapper(palette=bokeh.palettes.RdBu[9],
                            low=opening_by_month_year["Opening"].min(), high=opening_by_month_year["Opening"].max())


    hover = HoverTool(
        tooltips=[
            ("Opening", "@Opening{$,}"),
        ]
    )

    TOOLS = [hover, "save,pan,box_zoom,reset,wheel_zoom"]


    p = figure(x_axis_label='Year',
            y_axis_label='Month',
            tools=TOOLS,
            plot_width=900)

    p.rect(x="startYear", y="month", width=1, height=1, source=newdata,
        fill_color={'field': 'Opening', 'transform': mapper})

    color_bar = ColorBar(color_mapper=mapper, location=(20, 0), label_standoff=18,
                        ticker=AdaptiveTicker(), formatter=NumeralTickFormatter(format="$,"))

    p.add_layout(color_bar, 'right')

    p.title.text = "Movie Opening Performance by Year and Month"
    p.title.align = "center"
    p.title.text_font_size = "20px"
    st.write(p)


    st.markdown(f"<h2> II. Movie Ranking Analysis", unsafe_allow_html=True)
    st.markdown("""<p align="justify">This section visualizes the rankings of movies per year based on the following criterias: 
    <strong>Budget, Opening, Gross, Worldwide Gross, Metacritic Score, Runtime (Minutes), Average Rating, and Number of Votes</strong>. This section enables
    analysts to know what are the qualities and characteristics that movies that have appeared on these rankings have. """, unsafe_allow_html=True)
    years = []
    categories = ['Budget', 'Opening', 'Gross',
                'Worldwide Gross', 'Metacritic Score', 'Runtime (Minutes)', 'Average Rating', 'Number of Votes']

    for i in df_movies['startYear'].unique():
        years.append(i)

    option1 = st.selectbox(
        'Pls select the category', categories)

    option2 = st.selectbox(
        'Pls select the year', years)

    figure1 = movie_analyzer(df_movies, category = option1, year = option2)
    st.plotly_chart(figure1)
    # st.write('You selected:', option)




    st.markdown(f"<h2> III. What are the Most Popular Movie Genres?", unsafe_allow_html=True)
    st.markdown("""<p align="justify">This section visualizes the most popular genres as a WordCloud. The larger the font, the more frequently appearing
                            the word is. From the WordCloud, we can observe that Action Movies were the most popular movie genres among
                            film makers during the last 10 years.""", unsafe_allow_html=True)
    # Join the different processed abstracts together.
    colors = ["#BF0A30", "#002868"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    long_string = ' '.join(df_movies['genres'].values.tolist())

    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", colormap=cmap, width=1000,
                            height=300, max_font_size=500,relative_scaling=0.3,
                            min_font_size=5)

    # Generate a word cloud
    wordcloud = wordcloud.generate(long_string)

    # Visualize the word cloud
    plt.figure(figsize=(100,100))
    fig_cld, axes_cld = plt.subplots(1,1) 
    axes_cld.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")
    st.pyplot(fig_cld)



    st.markdown(f"<h2> IV. Movie Runtimes per Genre Analysis", unsafe_allow_html=True)
    st.markdown("""<p align="justify">This section visualizes movie runtimes per genre. It is important that we identify the characterictics
                    of movies whose runtimes are not normal as these may or may not affect viewership of the said movie. It is also quite possible that
                    these films are experimental in nature and that the director mainly created the movie for test subjects.""", unsafe_allow_html=True)
    genres = title_basics_df['genres'].unique().tolist()
    genres.append('All')

    option3 = st.slider('Pls. choose the number of movies to consider?', 2, 20, 10)
    option4 = st.selectbox(
        'Pls select the genre', genres)

    figure2 = runtimemovie_analyzer(title_basics_df, number = option3, genre = option4)
    st.plotly_chart(figure2)






    st.markdown(f"<h2> V. Performance for each Genre Across the Years", unsafe_allow_html=True)
    st.markdown("""<p align="justify">This section aims to visualize the different performance of each genre based on metrics (opening, gross and worldwide gross) across the years 2011 to 2021.
                      """, unsafe_allow_html=True)
    categories = ['Opening', 'Gross', 'Worldwide Gross']
    option5 = st.selectbox(
        'Pls select the category', categories)
    figure3 = genre_opening_analyzer(df_movies, category = option5)
    st.plotly_chart(figure3)




    st.markdown(f"<h2> VI. Average Budget per Genre", unsafe_allow_html=True)
    st.markdown("""<p align="justify">This section visualizes the average budget per genre across the available data. From the visualization, we
                        can observe that the Action genre has average budgets that were considered as outliers through all the average budgets across genres.
                        Meanwhile, other genres usually have lower budget allocations when being made and such genres include horror, drama, documentaries, comedies.
                      """, unsafe_allow_html=True) 
    fig = plt.figure(figsize = (15, 10))

    # fliersize is the size of outlier markers
    g = sns.boxplot(x = 'genres', y = 'budget', data = imdb_info_withbudget, 
                    palette="Set2", linewidth = 1, fliersize= 1.5)

    g.set(title = 'Average Budget per Genre', 
        ylabel = "Average Budget ($M)", xlabel = "")

    # put a horizontal line on overall mean
    plt.axhline(imdb_info_withbudget.budget.mean(), ls='--', lw = 1, color = 'black')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    #fig.savefig("filename.png")
    st.pyplot(fig)
    