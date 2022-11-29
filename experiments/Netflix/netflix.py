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
    :type text: str
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
    """"
    Processing text: lemmatization, tokenization, removing stop-words
    :param string: cleaned and processed tags
    :type string: str
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

def group_w2v(history: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """"
    Aggregate embedded data for users partitions.
    :param history: users movies history data .
    :type history: pd.DataFrame
    :param movies: movies data.
    :type movies: pd.DataFrame
    :return: average vector of the string words embeddings
    :rtype: np.ndarray, optional
    """
    
    """
    Aggregation (mean) embedded data for users watch history partitions.
    
    Arguments:
    --history: data frame of users movies history.
    --movies: data frame of movies. 
    
    Return:
    --df: data frame of users with aggregation embedded movies data.
    """
    users_id_arr = history.user_Id.unique()
    
    id_arr = []
    vec_arr = np.zeros((len(users_id_arr), 300))
    
    for user_id in tqdm.tqdm_notebook(range(len(users_id_arr))):
        vec = np.asarray(movies[movies.movie_Id.isin(history[history.user_Id == users_id_arr[user_id]].movie_Id)].iloc[:, 4:]).mean(axis=0) 
        
        id_arr.append(users_id_arr[user_id])
        vec_arr[user_id] = vec
    
    df = pd.DataFrame(vec_arr)
    df['user_Id'] = id_arr
    
    return df

def data_processing(df_movies: pd.DataFrame, 
                    df_rating: pd.DataFrame, 
                    rename: bool = True
) -> List[pd.DataFrame]:
    
    df_movies = df_movies.drop(["rating_cnt", "rating_avg"], axis=1)
    df_movies['clean_title'] = df_movies.title.apply(lambda x : procces_text(clean_text(x)))
    df_movies.drop("title", axis = 1, inplace = True)
    df_movies_clean = pd.concat([df_movies.drop("clean_title", axis=1), 
                                 pd.DataFrame(df_movies.clean_title.apply(string_embedding).to_list(), columns = ['w2v_' + str(i) for i in range(300)])], axis = 1)

    movies_vector = df_movies_clean.drop(['year'], axis=1)
    for col in movies_vector.drop("movie_Id", axis=1).columns:
        movies_vector[col] = movies_vector[col].astype('float')
    
    agg_columns = []
    df_result = pd.DataFrame()
    
    chunksize=10000
    chunk_count = (df_rating.shape[0] // chunksize) + 1 if df_rating.shape[0]%chunksize!=0 else df_rating.shape[0] // chunksize
    for idx in tqdm.tqdm_notebook(range(chunk_count)):
        chunk = df_rating.iloc[idx*chunksize:(idx+1)*chunksize, :]
        df_history = pd.merge(chunk[['user_Id', 'movie_Id', 'rating']], movies_vector.movie_Id, on = 'movie_Id', how = 'left')
        df_history = pd.merge(df_history, movies_vector, how='left', on='movie_Id').drop('movie_Id', axis=1)
        df_history['cnt'] = 1

        if idx == 0:
            agg_columns = df_history.drop(['user_Id'], axis=1).columns
        df_history_aggregated = df_history.groupby("user_Id", as_index=False)[agg_columns].sum()
        df_result = df_result.append(df_history_aggregated, ignore_index=True)

        if idx % 20 == 0:
            df_result = df_result.groupby("user_Id", as_index=False)[agg_columns].sum()

    df_result = df_result.groupby("user_Id", as_index=False)[agg_columns].sum()
    for col in agg_columns:
        if col != "cnt":
            df_result[col] = df_result[col] / df_result["cnt"]
    df_result = df_result.rename(columns={"rating": "rating_avg", "cnt": "rating_cnt"})
    df_users_clean = df_result
    
    df_movies_clean = pd.merge(df_rating.groupby("movie_Id", as_index=False)["rating"]\
                               .agg(['mean', 'count'])\
                               .rename(columns={"mean": "rating_avg", "count": "rating_cnt"}), df_movies_clean,how='left', on='movie_Id').fillna(0.0)
    
    df_rating_clean = df_rating
    if rename:
        cat_dict_movies = pd.Series(df_movies_clean.movie_Id.astype("category").cat.codes.values, index=df_movies_clean.movie_Id).to_dict()
        cat_dict_users = pd.Series(df_users_clean.user_Id.astype("category").cat.codes.values, index=df_users_clean.user_Id).to_dict()
        df_movies_clean.movie_Id = df_movies_clean.movie_Id.apply(lambda x: cat_dict_movies[x])
        df_users_clean.user_Id = df_users_clean.user_Id.apply(lambda x: cat_dict_users[x])
        df_rating_clean.movie_Id = df_rating_clean.movie_Id.apply(lambda x: cat_dict_movies[x])
        df_rating_clean.user_Id = df_rating_clean.user_Id.apply(lambda x: cat_dict_users[x])
    
    df_movies_clean = df_movies_clean.rename(columns={'movie_Id': 'item_idx'})
    df_users_clean = df_users_clean.rename(columns={'user_Id': 'user_idx'})
    df_rating_clean = df_rating_clean.rename(columns={'movie_Id': 'item_idx', 'user_Id': 'user_idx', 'rating': 'relevance'})
    
    return [df_movies_clean, df_users_clean, df_rating_clean]