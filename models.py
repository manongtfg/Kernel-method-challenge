import numpy as np
import scipy
from scipy.optimize import minimize
from kernels import SpectrumKernel, GaussianKernel, LinearKernel

class kernel_ridge_regression():

    def __init__(self, lambda_=0.5, kernel='RBF', num_classes = 2, **kwargs):
        self.lambda_ = lambda_
        self.kernel = kernel
        self.kernel_class = None
        self.kernel_params = kwargs
        self.num_classes = num_classes
        self.X_train = list()
        self.alpha_matrix = list()

        
    def kernel_function(self, X):
        """ Computes the matrix associated to the kernel function
        Params
        ------
        X: the input sequence, which rows can be real vectors or strings
        """

        if self.kernel == 'RBF':

            sigma = self.kernel_params.get('sigma', 1) 

            self.kernel_class = GaussianKernel(sigma)
            K = self.kernel_class.compute_kernel_matrix(X)

            return K

        elif self.kernel == 'linear':

            self.kernel_class = LinearKernel()
            K = self.kernel_class.compute_kernel_matrix(X)

            return K

        elif self.kernel == 'spectrum':

            k = self.kernel_params.get('k', 3) 

            self.kernel_class = SpectrumKernel(k)
            K = self.kernel_class.compute_kernel_matrix(X)

            return K


    def kernel_vector(self, X1, x2):
        """  Compute the spectrum kernel vector between a test sequence and a set of training sequences.
        Params
        ------
        X1 : Training rows
        x2 : Testing row 
        Returns
        -------
        kernel vector in order to predict 
        """

        K = self.kernel_class.compute_kernel_vector(X1, x2)

        return K



    def fit(self, X, y):
        """ Computes the value of alpha that minimizes the problem of 1/n (K alpha - y)^T(K alpha - y) + lambda alpha^T K alpha for each class.

        Params
        ----
        X : features for each observation.
        y : labels associated, vectorised, meaning that y[i, j] = 1 if X[i] belongs to class j and 0 otherwise. 

        """
        n, _ = X.shape
        self.X_train = X
        K_matrix = self.kernel_function(X)

        self.alpha_matrix = np.zeros((n, self.num_classes))

        for i in range(self.num_classes):

            alpha_i = np.linalg.inv((K_matrix + self.lambda_ * n * np.eye(n))) @ y[:, i]
            self.alpha_matrix[:, i] = alpha_i

        
    def predict(self, X):
        """ Computes the prediction for a vector x with the function f(x) = ∑ alpha_i K(x_i, x)"""
        
        m, _ = X.shape
        prediction = np.zeros((m, self.num_classes))

        if self.kernel == 'spectrum':
            x_test = X.loc[:, 'seq'].values

        for k in range(m):

            if self.kernel == 'spectrum':
                K_vector = self.kernel_vector(self.X_train, x_test[k])

            elif self.kernel == 'linear':
                K_vector = self.kernel_vector(self.X_train, X[k, :])

            else:
                K_vector = self.kernel_vector(self.X_train, X.iloc[k, :])

            for i in range(self.num_classes):

                prediction[k, i] = np.sum(self.alpha_matrix[:, i] * K_vector)
        
        y_predict = np.argmax(prediction, axis=1)


        return y_predict
    

    def get_params(self, deep=True):

        params = {
            'lambda_': self.lambda_,
            'k': self.k
        }
        params.update(self.kernel_params)
        return params  

    def _set_kernel_params(self, kwargs):

        return {
            'sigma': kwargs.get('sigma', 1),  # usefull for 'rbf'
            'k': kwargs.get('k', 3),  # size of substrings for 'spectrum' 
        }
    def set_params(self, **params):

        for param, value in params.items():
            if param == 'kernel':  
                self.kernel = value
                self.kernel_params = self._set_kernel_params(params)
            elif param in self.kernel_params:
                self.kernel_params[param] = value  
            else:
                setattr(self, param, value) 
        return self



class kernel_spectral_clustering():

    def __init__(self, kernel='RBF', k_neighbors = 2, **kwargs):

        self.kernel = kernel 
        self.k_neighbors = k_neighbors
        self.kernel_class = None 
        self.kernel_params = kwargs
        self.Z_star = list()
        self.X_train = list()


    def kernel_function(self, X):
        """ Computes the matrix associated to the kernel function"""

        if self.kernel == 'RBF':

            sigma = self.kernel_params.get('sigma', 1) 

            self.kernel_class = GaussianKernel(sigma)
            K = self.kernel_class.compute_kernel_matrix(X)

            return K

        elif self.kernel == 'linear':

            self.kernel_class = LinearKernel()
            K = self.kernel_class.compute_kernel_matrix(X)

            return K

        elif self.kernel == 'spectrum':
            
            k = self.kernel_params.get('k', 3) 

            self.kernel_class = SpectrumKernel(k)
            K = self.kernel_class.compute_kernel_matrix(X)

            return K



    def kernel_vector(self, X1, x2):
        """ Computes the matrix associated to the kernel function"""

        K = self.kernel_class.compute_kernel_vector(X1, x2)

        return K
        
    def classification_train_set(self, X):
        """ Compute the spherical Kmeans for classification"""

        K = self.kernel_function(X)

        eigvals, eigvects = np.linalg.eigh(K)
        indices = np.argsort(eigvals)[-self.k_neighbors:][::-1]

        X_train = X
        Z_star = eigvects[:, indices]

        classes = np.argmax(Z_star, axis=1)

        self.Z_star = Z_star
        self.X_train = X_train 

        return classes
    
    def predict_class_test(self, X_test):

        n, _ = X_test.shape
        test_classes = np.zeros(n)


        if self.kernel == 'spectrum':
            x_test = X_test.loc[:, 'seq'].values

        for i in range(n):

            if self.kernel == 'spectrum':
                K_vector = self.kernel_vector(self.X_train, x_test[i])
            
            else:
                K_vector = self.kernel_vector(self.X_train, X_test.iloc[i, :])

            Z_test = self.Z_star.T @ K_vector 
            test_classes[i] = np.argmax(Z_test)

        return test_classes 


class kernel_logistic_regression():

    def __init__(self, alpha0_coeff=1, kernel='RBF', lambda_=1, n_iter=20, threshold=0, **kwargs):
        
        self.alpha0_coeff = alpha0_coeff
        self.kernel = kernel 
        self.kernel_class = None 
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.threshold  = threshold
        self.X_train = list()
        self.alpha_final = list()
        self.kernel_params = kwargs

    def kernel_function(self, X):
        """ Computes the matrix associated to the kernel function"""

        if self.kernel == 'RBF':

            sigma = self.kernel_params.get('sigma', 1) 

            self.kernel_class = GaussianKernel(sigma)
            K = self.kernel_class.compute_kernel_matrix(X)

            return K

        elif self.kernel == 'linear':

            self.kernel_class = LinearKernel()
            K = self.kernel_class.compute_kernel_matrix(X)

            return K

        elif self.kernel == 'spectrum':

            k = self.kernel_params.get('k', 3) 
            self.kernel_class = SpectrumKernel(k)

            K = self.kernel_class.compute_kernel_matrix(X)

            return K


    def kernel_vector(self, X1, x2):
        """  Compute the spectrum kernel vector between a test sequence and a set of training sequences."""

        K = self.kernel_class.compute_kernel_vector(X1, x2)

        return K
        
    def sigmoid(self, u):
        """ Compute the sigmoid function """
        u = np.clip(u, -500, 500)

        return 1 / (1 + np.exp(-u))
    

    def solveWKRR(self, K, W_t, z_t, n, tol=1e-6, max_iter=100):
        """ Compute gradient descent in order to find the optimal alpha """
        alpha_0 = self.alpha_final  # Initialisation


        def loss(alpha):
            """ Objective function """

            product_K_alpha = (K @ alpha)
            product_with_Wt = W_t @ (product_K_alpha - z_t)

            return self.lambda_ * alpha.T @ product_K_alpha + 1/(2*n) * (product_K_alpha - z_t).T @ product_with_Wt

        def grad(alpha):
            """ Gradient of the objective function """
            product_K_alpha = (K @ alpha)
            product_with_Wt = W_t @ (product_K_alpha - z_t)

            return 2 * self.lambda_ * product_K_alpha + 1/n * product_with_Wt @ K

        def hessian(alpha):
            """ Hessian of the objective function """
            return 2 * self.lambda_ * K + 2/n * K @ W_t @ K 
        
        #Use of L-BFGS-B method in order to approximate the hessian 
        result = minimize(loss, alpha_0, method='L-BFGS-B', jac=grad, tol=tol, options={'maxiter': max_iter})

        return result.x
    

    
    def fit(self, X, y):
        
        n = len(y)
        t = 0
        self.X_train = X
        y[y == 0] = - 1
        alpha_t = self.alpha0_coeff * np.ones(self.X_train.shape[0])

        self.alpha_final = alpha_t

        K = self.kernel_function(X)
        m = K @ alpha_t

        W_t = np.diag(self.sigmoid(m) * self.sigmoid(-m))
        z_t = m + y / self.sigmoid(y * m)

        while t < self.n_iter:
            t += 1
            alpha_t = self.solveWKRR(K, W_t, z_t, n)
            self.alpha_final = alpha_t
            m = K @ alpha_t

            W_t = np.diag(self.sigmoid(m) * self.sigmoid(-m))
            z_t = m + y / self.sigmoid(y * m)


    
    def predict(self, X):

        n, _ = X.shape
        y_predict = np.zeros(n)

        if self.kernel == 'spectrum':
            x_test = X.loc[:, 'seq'].values

        for i in range(n):

            if self.kernel == 'spectrum':
                K_vector = self.kernel_vector(self.X_train, x_test[i])

            else :
                K_vector = self.kernel_vector(self.X_train, X.iloc[i, :])

            y_predict[i] = K_vector @ self.alpha_final

        y_predict[y_predict > self.threshold] = 1
        y_predict[y_predict < self.threshold] = -1
        
        return y_predict


    def _set_kernel_params(self, kwargs):
        """ Récupère tous les paramètres du noyau (même s'ils ne sont pas utilisés) """
        return {
            'sigma': kwargs.get('sigma', 1),  # usefull for 'rbf'
            'k': kwargs.get('k', 3),  # size of substrings for 'spectrum' 
        }


    def get_params(self, deep=True):

        params =  {

            'alpha0_coeff': self.alpha0_coeff,
            'lambda_': self.lambda_,
        }
        params.update(self.kernel_params)
        return params  


    def set_params(self, **params):

        for param, value in params.items():
            if param == 'kernel':  
                self.kernel = value
                self.kernel_params = self._set_kernel_params(params)
            elif param in self.kernel_params:
                self.kernel_params[param] = value  
            else:
                setattr(self, param, value) 
        return self


