from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import trange


class SGD:

    def __init__(self, sparse_matrix, K, lr, beta, n_epochs):
        """
        Arguments
        - sparse_matrix (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - lr (float) : learning rate
        - beta (float)  : regularization parameter
        """
        # convert ndArray
        self.sparse_matrix = sparse_matrix.to_numpy()
        self.item_n, self.user_n = sparse_matrix.shape
        self.K = K
        self.lr = lr
        self.beta = beta
        self.n_epochs = n_epochs

    def train(self):
        # Initialize user and item latent feature matrice
        self.I = np.random.normal(scale=1./self.K, size=(self.item_n, self.K))
        self.U = np.random.normal(scale=1./self.K, size=(self.user_n, self.K))

        # Init biases
        self.item_bias = np.zeros(self.item_n)
        self.user_bias = np.zeros(self.user_n)
        self.total_mean = np.mean(self.sparse_matrix[np.where(self.sparse_matrix != 0)])

        # Create training Samples
        self.samples = [
            (i, j)
            for i in range(self.item_n)
            for j in range(self.user_n)
            if self.sparse_matrix[i, j] > 0
        ]

        training_log = []
        for i in range(self.n_epochs):
            np.random.shuffle(self.samples)

            for i, u in self.samples:
                # get error
                y = self.sparse_matrix[i, u]
                pred = self.predict(i, u)
                error = y - pred
                # update bias
                self.item_bias[i] += self.lr * (error - self.beta * self.item_bias[i])
                self.user_bias[u] += self.lr * (error - self.beta * self.user_bias[u])
                # update latent factors
                I_i = self.I[i:][:]
                self.I[i, :] += self.lr * (error * self.U[u,:] - self.beta * self.I[i,:])
                self.U[u, :] += self.lr * (error * I_i - self.beta * self.U[u,:])

            rmse = self.evaluate()
            print(f"epoch {i} RMSE:", rmse)
            training_log.append((i, rmse))

        self.pred_matrix =  self.get_pred_matrix()

    def predict(self, i, u):
        """
        :param i: item index
        :param u: user index
        :return: predicted rating
        """
        return (
            self.total_mean
            + self.item_bias[i],
            + self.user_bias[u],
            + self.U[u,:].dot(self.I[i,:].T)
        )

    def get_pred_matrix(self):
        return (
            self.total_mean
            + self.item_bias[:,np.newaxis]
            + self.user_bias[np.newaxis:,]
            + self.I.dot(self.U.T)
        )

    def evaluate(self):
        idx, jdx = self.sparse_matrix.nonzero()
        pred_matrix = self.get_pred_matrix()
        ys, preds = [], []
        for i, j in zip(idx, jdx):
            ys.append(self.sparse_matrix.iloc[i, j])
            preds.append(pred_matrix.iloc[i, j])

        error = mean_squared_error(ys, preds)
        return np.sqrt(error)
