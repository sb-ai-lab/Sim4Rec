import pandas as pd
import numpy as np
import re
import itertools
import tqdm

import seaborn as sns
import tqdm
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from nltk.tokenize import TreebankWordTokenizer, WhitespaceTokenizer

import nltk
nltk.download('stopwords')
nltk.download('words')
words = set(nltk.corpus.words.words())
words = set([w.lower() for w in words])

from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download("wordnet")

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))


from nltk.tokenize import sent_tokenize

import gensim
from gensim.downloader import load
from gensim.models import Word2Vec
w2v_model = gensim.downloader.load('word2vec-google-news-300')

from typing import Dict, List, Optional, Tuple


def title_prep(title: str) -> str:
    
    """
    The function of cleaning the title of the movie from extra spaces, reduction to lowercase.ch of methods to create
    vector embeddings from original data

    :param title: the title of the movie
    :type title: str
    :return: cleaned title of the movie
    :rtype: str
    """
    
    title = re.sub(r'\s+', r' ', title)
    title = re.sub(r'($\s+|^\s+)', '', title)
    title = title.lower()
    
    return title

def extract_year(title: str) -> Optional[str]:
    
    """
    Extracting year from the movie title

    :param title: the cleaned title of the movie
    :type title: str
    :return: movie year
    :rtype: float, optional
    """
    
    one_year = re.findall(r'\(\d{4}\)', title)
    two_years = re.findall(r'\(\d{4}-\d{4}\)', title)
    one_year_till_today = re.findall(r'\(\d{4}[-–]\s?\)', title)
    if len(one_year) == 1:
        return int(one_year[0][1:-1])
    
    elif len(two_years) == 1:
        return round((int(two_years[0][1:5]) + int(two_years[0][6:-1]))/2)
    
    elif len(one_year_till_today) == 1:
        return int(one_year_till_today[0][1:5])
    else:
        return np.nan
    
def genres_processing(movies: pd.DataFrame) -> pd.DataFrame:   
    
    """
    Processing movie genres by constructing a binary vector of length n, where n is the number of all possible genres.
    For example a string like 'genre1|genre3|...' will be transformed into a vector [0,1,0,1,...].

    :param movies: DataFrame with column 'genres'
    :type title: pd.DataFrame
    :return: DataFrame with processed genres
    :rtype: pd.DataFrame
    """
    
    genre_lists = [set(item.split('|')).difference(set(['(no genres listed)'])) for item in movies['genres']]
    genre_lists = pd.DataFrame(genre_lists)
    
    genre_dict = {token: idx for idx, token in enumerate(set(itertools.chain.from_iterable([item.split('|') 
                for item in movies['genres']])).difference(set(['(no genres listed)'])))}
    genre_dict = pd.DataFrame(genre_dict.items())
    genre_dict.columns = ['genre', 'index']
    
    dummy = np.zeros([len(movies), len(genre_dict)])
    
    for i in range(dummy.shape[0]):
        for j in range(dummy.shape[1]):
            if genre_dict['genre'][j] in list(genre_lists.iloc[i, :]):
                dummy[i, j] = 1
    
    df_dummy = pd.DataFrame(dummy, columns = ['genre' + str(i) for i in range(dummy.shape[1])])
    
    movies_return = pd.concat([movies, df_dummy], 1)
    return movies_return

def fill_null_years(movies: pd.DataFrame) -> pd.DataFrame:
    
    """
    Processing null years

    :param movies: DataFrame with processed years
    :type title: pd.DataFrame
    :return: DataFrame with processed not null years
    :rtype: pd.DataFrame
    """
    
    df_movies = movies.copy()
    genres_columns = [item for item in movies.columns.tolist() if item[:5]=='genre' and item !='genres']
    df_no_year = movies[movies.year.isna()][['movieId', *genres_columns]]

    years_mean = {}
    for i in df_no_year.index:
    
        row = np.asarray(df_no_year.loc[i, :][genres_columns])
        years = []
        for j in np.asarray(movies[['year', *genres_columns]]):
            if np.sum(row == j[1:]) == len(genres_columns):
                try:
                    years.append(int(j[0]))
                except:
                    pass
            
        years_mean[i] = round(np.mean(years))
    
    for i in years_mean:
        df_movies.loc[i, 'year'] = years_mean[i]
    
    df_movies.year=df_movies.year.astype('int')
    
    return df_movies

def clean_text(text: str) -> str:
    """
    Cleaning text: remove extra spaces and non-text characters

    :param text: tag row text
    :type title: str
    :return: tag cleaned text
    :rtype: str
    """
    
    text = re.sub("[^a-zA-Z]", " ",text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+$", "", text)
    text = re.sub(r"^\s+", "", text)
    text = text.lower()

    return text


def procces_text(text):
    """
    Processing text: lemmatization, tokenization, removing stop-words

    :param text: tag cleaned text
    :type title: str
    :return: tag processed text
    :rtype: str
    """
    lemmatizer = WordNetLemmatizer() 

    text = [word for word in nltk.word_tokenize(text) if not word in stop_words]
    text = [lemmatizer.lemmatize(token) for token in text]
    text = [word for word in text if word in words]

    text = " ".join(text)
    
    return text

def string_embedding(string: str) -> np.ndarray:
    """
    Processing text: lemmatization, tokenization, removing stop-words

    :param string: cleaned and processed tags
    :type title: str
    :return: average vector of the string words embeddings
    :rtype: np.ndarray, optional
    """
    
    arr = string.split(' ')
    vec = 0
    cnt = 0
    for i in arr:
        try:
            vec += w2v_model[i]
            cnt += 1
        except:
            pass
    if cnt == 0:
        vec = np.zeros((300, 1))
    else:
        vec /= cnt
    return vec


def data_processing(df_movie: pd.DataFrame, 
                    df_rating: pd.DataFrame, 
                    df_tags: pd.DataFrame
) -> List[pd.DataFrame]:
    
    print("------------------------ Movie processing ------------------------")
    #Extraction of the movies' years and transform genres lists to genres vector
    df_movies_procc = df_movie.copy()
    df_movies_procc.title = df_movies_procc.title.apply(title_prep) #title processing
    df_movies_procc['year'] = df_movies_procc.title.apply(extract_year) #year processing
    df_movies_procc = genres_processing(df_movies_procc) #genres processing
    df_movies_procc = fill_null_years(df_movies_procc) #fillimg null year values
    
    #Creating rating_avg column 
    print("------------------------ Rating processing ------------------------")
    df_movies_procc = pd.merge(df_movies_procc, df_rating.groupby('movieId', as_index=False).rating.mean(), on='movieId', how='left')
    df_movies_procc.rating = df_movies_procc.rating.fillna(0.0)
    df_movies_procc = df_movies_procc.rename(columns={'rating' : 'rating_avg'})
    df_movies_clean = df_movies_procc.drop(['title', 'genres'], axis=1)[['movieId', 'year', 'rating_avg', *['genre' + str(i) for i in range(19)]]]
    
    print("------------------------ Tags processing ------------------------")
    df_tags_ = df_tags.drop(df_tags[df_tags.tag.isna()].index)
    df_movie_tags = df_tags_.sort_values(by=['movieId', 'timestamp'])[['movieId', 'tag', 'timestamp']]    
    df_movie_tags['clean_tag'] = df_movie_tags.tag.apply(lambda x : procces_text(clean_text(x)))
    df_movie_tags = df_movie_tags[df_movie_tags.clean_tag.str.len()!=0]
    
    print("------------------------ Tags embedding ------------------------")
    #tags text gathering
    docs_movie_tags = df_movie_tags.sort_values(["movieId", "timestamp"]).groupby("movieId", as_index=False).agg({"clean_tag":lambda x: " ".join(x)})
    df_movies_tags = pd.concat([docs_movie_tags.movieId, pd.DataFrame(docs_movie_tags.clean_tag.apply(string_embedding).to_list(), columns = ['w2v_' + str(i) for i in range(300)])], axis = 1)
    df_movies_clean = pd.merge(df_movies_clean, df_movies_tags, on = "movieId", how = "left").fillna(0.0)
    
    print("------------------------ Users processing ------------------------")
    #users procc
    df_users = df_rating.copy()
    df_users = df_users.groupby(by=['userId'], as_index=False).rating.mean().rename(columns = {'rating' : 'rating_avg'})
    df_users_genres = pd.merge(df_movies_clean[['movieId', *df_movies_clean.columns[3:22]]], pd.merge(df_rating, df_users, on = 'userId')[['userId', 'movieId']],
        on = 'movieId')

    df_users_genres = df_users_genres.groupby(by = ['userId'], as_index = False)[df_movies_clean.columns[3:22]].mean()
    df_users_genres = pd.merge(df_users_genres, df_users, on = 'userId')
    df_pairs = pd.merge(df_rating, df_users, on = 'userId')[['userId', 'movieId']]
    
    print("------------------------ Users embedding ------------------------")
    users_id = []
    vect_space = []
    for Id in tqdm.tqdm(df_pairs.userId.unique()):
        movie_list = df_pairs[df_pairs.userId == Id].movieId.tolist()
        vect = np.asarray(df_movies_clean[df_movies_clean.movieId.isin(movie_list)][[*df_movies_clean.columns[22:]]].mean().tolist())
        users_id.append(Id)
        vect_space.append(vect)
        
    df_users_w2v = pd.DataFrame(vect_space, columns = ['w2v_' + str(i) for i in range(len(df_movies_clean.columns[22:]))])
    df_users_w2v['userId'] = users_id
    df_users_clean = pd.merge(df_users_genres, df_users_w2v, on = 'userId')
    df_rating_clean = df_rating[['userId', 'movieId', 'rating', 'timestamp']]
    
    """
    cat_dict_movies = pd.Series(df_movies_clean.movieId.astype("category").cat.codes.values, index=df_movies_clean.movieId).to_dict()
    cat_dict_users = pd.Series(df_users_clean.userId.astype("category").cat.codes.values, index=df_users_clean.userId).to_dict()
    
    df_movies_clean.movieId = df_movies_clean.movieId.apply(lambda x: cat_dict_movies[x])
    df_users_clean.userId = df_users_clean.userId.apply(lambda x: cat_dict_users[x])
    df_rating_clean.movieId = df_rating.movieId.apply(lambda x: cat_dict_movies[x])
    df_rating_clean.userId = df_rating.userId.apply(lambda x: cat_dict_users[x])
    """
    
    df_movies_clean = df_movies_clean.rename(columns={'movieId': 'item_idx'})
    df_users_clean = df_users_clean.rename(columns={'userId': 'user_idx'})
    df_rating_clean = df_rating_clean.rename(columns={'movieId': 'item_idx', 'userId': 'user_idx', 'rating': 'relevance'})
    
    return [df_movies_clean, df_users_clean, df_rating_clean]