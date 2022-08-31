
import os, math
import numpy as np
import matplotlib.pyplot as plt

UNCLASSIFIED = False
NOISE = 0

def dist(a, b):
    t = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
    return math.sqrt(t)

def eps_neighbor(a, b, eps):
    return dist(a, b) < eps

def region_querym(data, pointId, eps, clusterResult):
    nPoints = data.shape[1]
    seeds = []

    for i in range(nPoints):
        if not clusterResult[i] in [UNCLASSIFIED, NOISE]:
            continue
        a = data[:, pointId]
        b = data[:, i]
        if eps_neighbor(a, b, eps):
            seeds.append(i)

    del_list = []
    for i in range(len(seeds)):
        if seeds[i] in del_list:
            continue

        for j in range(i + 1, len(seeds)):
            a = data[:, seeds[i]]
            b = data[:, seeds[j]]
            if not eps_neighbor(a, b, eps):
                del_list.append(seeds[j])

    new_seeds = list(set(seeds) - set(del_list))
    return new_seeds


def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    seeds = region_querym(data, pointId, eps, clusterResult)
    if len(seeds) < minPts:
        clusterResult[pointId] = NOISE
        return False
    else:
        clusterResult[pointId] = clusterId
        for seedId in seeds:
            clusterResult[seedId] = clusterId

    return True

def dbscan_best(data, eps, minPts):
    """
    in：point(x,y) data, radius, min points num
    out：data groups, group num
    """
    nPoints = data.shape[1]
    min_group = 100000
    clusterResultLast = None

    for ii in range(nPoints):
        clusterResult = [UNCLASSIFIED] * nPoints
        clusterId = 1
        for pointId in range(ii ,nPoints):
            if clusterResult[pointId] == UNCLASSIFIED:
                if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                    clusterId = clusterId + 1
        for pointId in range(0 ,ii):
            if clusterResult[pointId] == UNCLASSIFIED:
                if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                    clusterId = clusterId + 1

        if clusterId - 1 < min_group:
            min_group = clusterId - 1
            clusterResultLast = clusterResult
            if min_group == 1:
                break

    return clusterResultLast, min_group


def plotFeature(data, clusters, clusterNum, file, result_path, w, h):
    matClusters = np.mat(clusters).transpose()
    fig, ax = plt.subplots()
    plt.axis('off')
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown', 'greenyellow', 'khaki',
                     'plum', 'violet']
    fig.add_subplot(111, aspect='equal')
    plt.xlim(xmin=0, xmax=w)
    plt.ylim(0, h)
    plt.margins(0, 0)

    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()

    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
        ax.scatter(subCluster[0, :][0].flatten().A[0], subCluster[1, :][0].flatten().A[0], c=colorSytle, s=50)

    file_base, _ = os.path.splitext(file)
    fig.savefig(os.path.join(result_path, file_base + '_dbscan.jpg'), format='jpg', transparent=True, dpi=100, pad_inches=0)
    plt.close()

def dbscan_handle(dataSet, filename, result_path, w, h):
    clusterNum = 0

    if len(dataSet):
        dataSet = np.mat(dataSet).transpose()
        clusters, clusterNum = dbscan_best(dataSet, int(math.sqrt(w * h / 6.5)), 4)
        plotFeature(dataSet, clusters, clusterNum, filename, result_path, w, h)

    return clusterNum
