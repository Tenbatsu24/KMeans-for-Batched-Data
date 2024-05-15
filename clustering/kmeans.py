import torch
import torch.nn as nn

from einops import rearrange, repeat


class KMeans(nn.Module):
    """
    KMeans clustering algorithm for batched data

    Constructor for KMeans class
    :param num_clusters: number of clusters to initialize
    :param dim: number of features in each data point. if data is 2D, num_features=2
    :num_iterations: number of iterations to run the KMeans algorithm
    """

    def __init__(self, num_clusters: int, dim: int, num_iterations: int):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_features = dim
        self.num_iterations = num_iterations

    @torch.no_grad()
    def forward(self, data, centers):
        orig_shape = data.shape

        if len(orig_shape) != 3:
            data = rearrange(data, 'n d -> () n d')
            centers = rearrange(centers, 'c d -> () c d')

        # Step 2: Compute the distance of each data point from the nearest cluster center
        for i in range(self.num_iterations):
            old_centers = centers.clone()

            # do the cluster assignment step
            cluster_assignment = torch.cdist(data, centers).argmin(dim=2)

            cluster_assignment = repeat(cluster_assignment, 'b n -> b n c', c=self.num_features)
            centers.scatter_reduce_(dim=1, index=cluster_assignment, src=data, reduce='mean', include_self=False)

            # check for convergence
            if torch.allclose(centers, old_centers):
                break

        return (
            centers if len(orig_shape) == 3 else rearrange(centers, '() c d -> c d'),
            cluster_assignment if len(orig_shape) == 3 else rearrange(cluster_assignment, '() b n -> b n')
        )
