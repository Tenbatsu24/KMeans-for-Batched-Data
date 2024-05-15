import torch
import torch.nn as nn

from einops import rearrange


class KMeansPP(nn.Module):
    """
    KMeans++ initialization for KMeans clustering
    Constructor for KMeans++ class
    :param n_clusters: number of clusters to initialize
    :param dim: number of features in each data point. if data is 2D, num_features=2
    """

    def __init__(self, n_clusters: int, dim: int):
        super().__init__()
        self.centers = n_clusters
        self.num_features = dim

    @torch.no_grad()
    def forward(self, data):
        """
        Forward pass for KMeans++ initialization
        :param data: input data to initialize clusters. Shape: (B, N, D)
        :return: initialized cluster centers
        """

        orig_shape = data.shape
        if len(orig_shape) != 3:
            data = rearrange(data, 'n d -> () n d')

        b = data.size(0)
        b_range = torch.arange(b)

        # Step 1: Randomly select the first cluster center
        centers = torch.zeros(b, self.centers, self.num_features, device=data.device)

        # Step 2: Compute the distance of each data point from the nearest cluster center
        for i in range(self.centers):
            if i == 0:
                # sample random data point as the first cluster center for each batch
                centers[:, i] = data[b_range, torch.randint(data.size(1), (b,))]
            else:
                dist = torch.cdist(data, centers[:, :i]).min(dim=2).values

                # get the data point which is farthest from the nearest cluster center
                centers[:, i] = data[b_range, torch.argmax(dist, dim=1)]

        return centers if len(orig_shape) == 3 else rearrange(centers, '() c d -> c d')
