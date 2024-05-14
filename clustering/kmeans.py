import torch
import torch.nn as nn

from einops import rearrange, repeat


class KMeans(nn.Module):

    def __init__(self, num_clusters: int, num_features: int, num_iterations: int):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.num_iterations = num_iterations

    @torch.no_grad()
    def forward(self, data, centers):
        orig_shape = data.shape
        if len(orig_shape) != 3:
            data = rearrange(data, 'n d -> () n d')

        # Step 2: Compute the distance of each data point from the nearest cluster center
        for i in range(self.num_iterations):
            old_centers = centers.clone()
            # print(old_centers.shape)  # torch.Size([3, 4, 2])

            # do the cluster assignment step
            cluster_assignment = torch.cdist(data, centers).argmin(dim=2)

            cluster_assignment = repeat(cluster_assignment, 'b n -> b n c', c=self.num_features)
            centers.scatter_reduce_(dim=1, index=cluster_assignment, src=data, reduce='mean', include_self=False)

            # check for convergence
            if torch.allclose(centers, old_centers):
                break

        return centers if len(orig_shape) == 3 else rearrange(centers, '() c d -> c d')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    from clustering.kmeans_pp import KMeansPP

    # Generate some data
    X, y = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=42)
    X = torch.tensor(X).float()

    X = torch.stack([X, X + 10, X - 10], dim=0)  # dummy batch dimension
    print(X.shape)

    _num_clusters = 4
    _num_features = 2
    _num_iterations = 10

    kmeans = KMeans(_num_clusters, _num_features, _num_iterations)
    _centers = KMeansPP(_num_clusters, _num_features)(X)

    _centers = kmeans(X, _centers)

    # plot the clusters and the cluster centers
    X = X[0]
    _centers = _centers[0]

    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(_centers[:, 0], _centers[:, 1], c='red', marker='x')
    plt.show()
