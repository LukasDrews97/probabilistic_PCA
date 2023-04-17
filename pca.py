import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        X = X.astype(np.float16)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # Calculate Z-Score
        X -= self.mean
        X /= self.std
        X = np.nan_to_num(X)
        
        # calculate covariance matrix
        cov_mat = np.cov(X.T)
        
        # eigenvectors, eigenvalues
        self.eig_vals, self.eig_vecs = np.linalg.eig(cov_mat)
        self.eig_vecs = np.transpose(self.eig_vecs)
        
        # sort eigenvectors
        sorted_idx = np.argsort(self.eig_vals)[::-1]
        self.eig_vals = self.eig_vals[sorted_idx]
        self.eig_vecs = self.eig_vecs[sorted_idx]
        self.eig_vecs = self.eig_vecs.astype(np.float16)
        
        # save n components
        self.components = self.eig_vecs[:self.n_components]
        
    def transform(self, X):
        X = X.astype(np.float16)
        # Calculate Z-Score
        X -= self.mean
        X /= self.std
        X = np.nan_to_num(X)
        
        # calculate projection onto lower-dimensional space
        return np.dot(X, self.components.T).real
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    @property
    def explained_variance(self):
        return self.eig_vals / np.sum(self.eig_vals)
    
    def plot_explained_variance(self, n_components=10):
        fig, ax = plt.subplots()
        ax.bar(range(len(self.explained_variance[:n_components])), self.explained_variance[:n_components])
        ax.plot(range(len(self.explained_variance[:n_components])), np.cumsum(self.explained_variance[:n_components]), c='black', marker='o')
        ax.set_xticks(range(len(self.explained_variance[:n_components])))
        ax.set_xticklabels(range(1, len(self.explained_variance[:n_components])+1))
        ax.set_xlabel('Component')
        ax.set_ylabel('Explained Variance')
        ax.set_title('Explained Variance by Component')
        plt.show()
        
    def plot_biplot(self, components, n_components, scaling=1.0):
        fig, ax = plt.subplots()
        ax.scatter(components[:, 0], components[:, 1], alpha=0.8, color='royalblue')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("PCA")
        
        sorted_arrows = []
        for i in range(self.eig_vecs.shape[0]):
            arrow_end = (self.eig_vecs[i, 0], self.eig_vecs[i, 1])
            sorted_arrows.append((np.sqrt(arrow_end[0]**2 + arrow_end[1]**2), self.eig_vecs[i, 0], self.eig_vecs[i, 1]))
        sorted_arrows = list(sorted(sorted_arrows, key=lambda x: x[0], reverse=True))
        
        for i in range(min(len(sorted_arrows), n_components)):
            arrow_start = (0, 0)
            arrow_end = (scaling*sorted_arrows[i][1], scaling*sorted_arrows[i][2])
            ax.arrow(*arrow_start, *arrow_end, head_width=0.02 * scaling, head_length=0.03 * scaling, length_includes_head=True, color='orangered', alpha=0.8)
            plt.text(scaling*sorted_arrows[i][1] * 1.15, scaling*sorted_arrows[i][2] * 1.15, i+1, color='darkblue', ha ='center', va='center')

        plt.show()
