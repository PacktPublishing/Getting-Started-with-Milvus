# Before running the script, you need to install the gensim python third-party library in advance.
# Execute: pip install gensim

import numpy as np

class LSH:
    def __init__(self, n_hashes, n_buckets):
        self.n_hashes = n_hashes
        self.n_buckets = n_buckets
        self.hash_tables = [{} for _ in range(n_hashes)]
        self.random_vectors = np.random.randn(n_hashes, 2)

    def _pairwise_distances(self, X, Y=None):
        X = np.array(X)
        Y = np.array(Y)

        dist_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                dist_matrix[i, j] = np.linalg.norm(X[i] - Y[j])
        return dist_matrix

    def _hash(self, point):
        return tuple((np.dot(self.random_vectors, np.array(point)) > 0).astype(int))

    def insert(self, point):
        for i in range(self.n_hashes):
            hash_value = self._hash(point)
            if hash_value not in self.hash_tables[i]:
                self.hash_tables[i][hash_value] = []
            self.hash_tables[i][hash_value].append(tuple(point))

    def query(self, query_point):
        candidates = set()
        for i in range(self.n_hashes):
            hash_value = self._hash(query_point)
            if hash_value in self.hash_tables[i]:
                candidates.update(self.hash_tables[i][hash_value])

        candidates = np.array([np.array(candidate) for candidate in candidates])
        if candidates.size == 0:
            return None
        distances = self._pairwise_distances(candidates, [query_point])
        nearest_index = np.argmin(distances)
        return candidates[nearest_index]

data = [[1, 2], [3, 4], [5, 6], [7, 8]]
query_point = [4, 5]

lsh = LSH(n_hashes=5, n_buckets=10)
for point in data:
    lsh.insert(point)

result = lsh.query(query_point)
print("lsh query result:", result)