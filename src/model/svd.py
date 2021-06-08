import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class SVD:

    def __init__(self, sparse_matrix, K):
        self.sparse_matrix = sparse_matrix
        self.K = K
        self.init_sparse_matrix()

    def init_sparse_matrix(self):
        self.train_matrix = self.sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=1)
        self.sparse_matrix = self.sparse_matrix.fillna(0)

    def train(self):
        print("Factorizing...")
        item_factors, user_factors = self.get_svd(self.train_matrix, self.K)
        result_df = pd.DataFrame(
            np.matmul(item_factors, user_factors),
            columns=self.sparse_matrix.columns.values,
            index=self.sparse_matrix.index.values
        )
        self.item_factors = item_factors
        self.user_factors = user_factors
        self.pred_matrix = result_df

    @staticmethod
    def get_svd(sparse_matrix, K):
        U, s, VT = np.linalg.svd(sparse_matrix.transpose())
        # U = (user_n, user_n), s = (user_n,), VT = (item_n, item_n)

        U = U[:, :K] # (user_n, K)
        s = s[:K] * np.identity(K, np.float) # (K, K)
        VT = VT[:K, :] # (K, item_n)

        item_factors = np.transpose(np.matmul(s, VT)) # (item_n, K)
        user_factors = np.transpose(U) # (K, user_n)

        return item_factors, user_factors

    def evaluate(self):
        print("Evaluating...")
        idx, jdx = self.sparse_matrix.to_numpy().nonzero()
        ys, preds = [], []
        for i, j in zip(idx, jdx):
            ys.append(self.sparse_matrix.iloc[i, j])
            preds.append(self.pred_matrix.iloc[i, j])

        error = mean_squared_error(ys, preds)
        return np.sqrt(error)

