from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def noSupervizado():
    data = {'x': [23, 34, 45, 56, 67, 78, 89, 90, 98, 87, 76, 65, 54, 43, 23, 44, 56, 67, 78, 87, 45, 12],
            'y': [34, 54, 56, 76, 78, 23, 23, 43, 78, 98, 65, 43, 56, 98, 21, 34, 56, 21, 56, 87, 98, 32]}
    df = DataFrame(data, columns=['x', 'y'])
    print(df)
    plt.scatter(df['x'], df['y'])
    plt.show()

    kmeans = KMeans(n_clusters=4).fit(df)

    centroides = kmeans.cluster_centers_

    print(centroides)

    plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', s=50)
    plt.show()


if __name__ == '__main__':
    noSupervizado()
