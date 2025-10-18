from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
import torch

def build_laplacian(coords, k=6):
    A = kneighbors_graph(coords, n_neighbors=k, mode='connectivity')
    L = csgraph.laplacian(A, normed=True)
    return torch.tensor(L.toarray(), dtype=torch.float32)
