import timeit

import torch
import matplotlib.pyplot as plt

from clustering import KMeansPP, KMeans


def create_dummy_data(n_samples=1000, centers=4, n_features=2, random_state=42):
    from sklearn.datasets import make_blobs

    # Generate some data
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=random_state)
    X = torch.tensor(X).float()
    X = torch.stack([X, X + 10, X - 10], dim=0)  # dummy batch dimension
    return X


def plot_kmeans_result(X, _centers):
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu())
    plt.scatter(_centers[:, 0].cpu(), _centers[:, 1].cpu(), c='red', marker='x')
    plt.show()


def main():
    X = create_dummy_data()
    print('data shape:', X.shape)

    _num_clusters = 4
    _num_features = 2
    _num_iterations = 10

    start = timeit.default_timer()

    kmeans = KMeans(_num_clusters, _num_features, _num_iterations)
    _centers = KMeansPP(_num_clusters, _num_features)(X)
    _centers = kmeans(X, _centers)

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    plot_kmeans_result(X[0], _centers[0])


def main_cuda():
    print('CUDA is available:', torch.cuda.is_available())
    X = create_dummy_data(n_samples=2500)
    print('data shape:', X.shape)

    _num_clusters = 50
    _num_dims = 2
    _num_iterations = 10

    start = timeit.default_timer()

    X = X.cuda()
    kmeans = KMeans(_num_clusters, _num_dims, _num_iterations).cuda()
    _centers = KMeansPP(_num_clusters, _num_dims).cuda()(X)
    _centers = kmeans(X, _centers)

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    # plot the clusters and the cluster centers
    plot_kmeans_result(X[0], _centers[0])


if __name__ == '__main__':
    print("============================================")
    main()
    print("============================================")

    if torch.cuda.is_available():
        print("============================================")
        main_cuda()
        print("============================================")
