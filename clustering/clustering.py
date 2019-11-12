import time

import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['Kmeans', 'GMM', 'Spectral', 'Agglomerative']


def preprocess_features(npdata, pca_dim=256, whitening=False, L2norm=False):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca_dim (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')
    pca = PCA(pca_dim, whiten=whitening)
    npdata = pca.fit_transform(npdata)
    # L2 normalization
    if L2norm:
        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / row_sums[:, np.newaxis]
    return npdata

class Clustering:
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        self.k = k
        self.pca_dim = pca_dim
        self.whitening = whitening
        self.L2norm = L2norm
        
    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data, self.pca_dim, self.whitening, self.L2norm)
        # cluster the data
        I = self.run_method(xb, self.k)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)
        return None
    
    def run_method():
        print('Define each method')
    
class Kmeans(Clustering):
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        super().__init__(k, pca_dim, whitening, L2norm)

    def run_method(self, x, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters)
        I = kmeans.fit_predict(x)
        return I
       
class GMM(Clustering):
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        super().__init__(k, pca_dim, whitening, L2norm)

    def run_method(self, x, n_clusters):
        kmeans = GaussianMixture(n_clusters=n_clusters)
        I = kmeans.fit_predict(x)
        return I

class Spectral(Clustering):
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        super().__init__(k, pca_dim, whitening, L2norm)

    def run_method(self, x, n_clusters):
        spectral = SpectralClustering(n_clusters=n_clusters)
        I = spectral.fit_predict(x)
        return I
    
class Agglomerative(Clustering):
    def __init__(self, k, pca_dim=256, whitening=False, L2norm=False):
        super().__init__(k, pca_dim, whitening, L2norm)

    def run_method(self, x, n_clusters):
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        I = agg.fit_predict(x)
        return I