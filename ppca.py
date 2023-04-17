import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class PPCA:
    def __init__(self, n_components, max_iter=1000, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X):
        self.X_fit = X.astype(np.float16)
        self.n_samples, self.n_features = X.shape
        
        self.W = np.random.normal(size=(self.n_features, self.n_components))
        self.sigma = np.random.normal()
        self.mu = np.nanmean(self.X_fit, axis=0)
        self.std = np.nanstd(self.X_fit, axis=0)
        self.X_fit -= self.mu
        self.X_fit /= self.std
        
        self.X_fit = np.nan_to_num(self.X_fit)
        I = np.identity(self.n_components)
        
        for i in range(self.max_iter):
            # E step
            M = np.dot(self.W.T, self.W) + self.sigma * I
            M_inv = np.linalg.inv(M)
            Z_exp = np.dot(np.dot(self.X_fit, self.W), M_inv)
            Z_cov = self.sigma * M_inv + np.matmul(Z_exp[:, :, np.newaxis], Z_exp[:, np.newaxis, :])
            
            # M step
            self.W = np.dot(np.dot(self.X_fit.T, Z_exp), np.linalg.inv(np.sum(Z_cov, axis=0)))            
            first = np.linalg.norm(self.X_fit, axis=1)**2
            second = -2 * np.sum(np.dot(Z_exp, self.W.T) * self.X_fit, axis=1)
            third = np.trace(np.dot(Z_cov, np.dot(self.W.T, self.W)), axis1=1, axis2=2)
            self.sigma = np.sum((first + second + third) / (self.n_samples * self.n_features))
            
            # Check convergence
            if i > 0 and np.isclose(sigma_old, self.sigma, rtol=self.tol):
                print(f'Converged after {i} steps. Sigma: {self.sigma:.5f}.')
                break
            if i == self.max_iter - 1:
                print(f"EM didn't converge after the maximum number of iterations ({self.max_iter}).")
            sigma_old = self.sigma
        
        # sort W by explained variance
        variance_ratio = self.explained_variance()
        sorted_idx = np.argsort(variance_ratio)[::-1]
        self.W = self.W[:, sorted_idx]
        
    def transform(self, X):
        X = X.astype(np.float16)
        X -= self.mu
        X /= self.std
        X = np.nan_to_num(X)
        return np.dot(X, self.W)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def explained_variance(self):
        total_variance = np.var(self.X_fit, axis=0).sum()
        variance_ratio = np.var(self.transform(self.X_fit), axis=0) / total_variance
        variance_ratio /= np.sum(variance_ratio)
        return variance_ratio
    
    def plot_biplot(self, components):
        fig, ax = plt.subplots()
        ax.scatter(components[:, 0], components[:, 1], alpha=0.8, color='royalblue')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("PPCA")
        plt.show()