import timeit

import torch
import matplotlib.pyplot as plt

from clustering import KMeansPP, KMeans


def create_dummy_data(n_samples=10000, centers=4, n_features=2, batch_size=8):
    from sklearn.datasets import make_blobs

    # Generate some data
    batch_list = []
    for _ in range(batch_size):
        _x, _ = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features)
        _x = torch.tensor(_x).float()
        batch_list.append(_x)
    x = torch.stack(batch_list, dim=0)  # dummy batch dimension
    return x


def plot_kmeans_result(data, _centers, _assignment, title='KMeans with KMeans++ Initialization'):
    data = data.cpu().numpy()
    _centers = _centers.cpu().numpy()
    _assignment = _assignment.cpu().numpy()

    max_plot = min(data.shape[0], 10)

    _, axs = plt.subplots(2, round(max_plot / 2), figsize=(20, 10))
    for i in range(max_plot):
        ax = axs[i % 2, i // 2]
        ax.scatter(data[i][:, 0], data[i][:, 1], c=_assignment[i], cmap='viridis')
        ax.scatter(_centers[i][:, 0], _centers[i][:, 1], c='red', marker='x')

    plt.suptitle(title)
    plt.tight_layout()
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
    _centers, _assignment = kmeans(X, _centers)

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    plot_kmeans_result(
        X, _centers, _assignment[:, :, 0],
        title=f'KMeans with KMeans++ Initialization (Time: {stop - start:.2f} sec)'
    )


def main_cuda():
    print('CUDA is available:', torch.cuda.is_available())
    X = create_dummy_data()
    print('data shape:', X.shape)

    _num_clusters = 4
    _num_dims = 2
    _num_iterations = 10

    start = timeit.default_timer()

    X = X.cuda()
    kmeans = KMeans(_num_clusters, _num_dims, _num_iterations).cuda()
    _centers = KMeansPP(_num_clusters, _num_dims).cuda()(X)
    _centers, _assignment = kmeans(X, _centers)

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    # plot the clusters and the cluster centers
    plot_kmeans_result(
        X, _centers, _assignment[:, :, 0],
        title=f'KMeans with KMeans++ Initialization using CUDA (Time: {stop - start:.2f} sec)'
    )


if __name__ == '__main__':
    print("============================================")
    main()
    print("============================================")

    if torch.cuda.is_available():
        print("============================================")
        main_cuda()
        print("============================================")
