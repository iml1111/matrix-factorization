import os
import pandas as pd


MOVIELENS_PATH = "./data/movielens"


def get_movielens(file='ratings.csv'):
    return pd.read_csv(os.path.join(MOVIELENS_PATH, file), encoding='utf-8')

def make_sparse_matrix(df):
    """
    Make sparse matrix
    :param df: train_df [userId, movieId, rating, ...]
    :return: sparse_matrix (movie_n) * (user_n)
    """
    sparse_matrix = (
        df
        .groupby('movieId')
        .apply(lambda x: pd.Series(x['rating'].values, index=x['userId']))
        .unstack()
    )
    sparse_matrix.index.name = 'movieId'
    return sparse_matrix
