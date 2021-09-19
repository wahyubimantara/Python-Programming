import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import glob

def runKmeans():
    df = pd.read_excel(r'dataset/gabung.xlsx')
    data = (df[["latitude", "longitude"]])
    X = data.iloc[:, [0, 1]].values
    silhouette_avg_after = 0
    silhouette_avg_before = 0
    def K_Means(X, K):
        max_iter = 10000
        Toleransi_Perubahan_Centroid = 0.000001
        HitungJarak = {}
        result = np.zeros(X.shape[0], dtype=np.int)
        centroids = np.zeros((K, X.shape[1]))
        maxLat = np.amax(X[:, 0])
        minLat = np.amin(X[:, 0])
        maxLon = np.amax(X[:, 1])
        minLon = np.amin(X[:, 1])
        for n in range(K):
            centroids[n][0] = rd.uniform(minLat, maxLat)#max
            centroids[n][1] = rd.uniform(minLon, maxLon)#min
        print("centroid awal")
        print(centroids)
        for iteration in range(max_iter):
            for n in range(K):
                HitungJarak[n] = np.sqrt(np.sum(np.power((X - centroids[n]), 2), axis=1))
            clss = list(HitungJarak.values())
            result = np.argmin(clss, axis = 0)
            prev_centroids = centroids.copy()
            print("Iterasi", iteration + 1)
            for n in range(K):
                centroids[n] = np.average(X[result == n], axis=0)       
            print("centroid baru", centroids,"\n")
        
            nilai_batas = np.sum(abs(centroids - prev_centroids))/np.sum(prev_centroids) * 100.0
            if nilai_batas < Toleransi_Perubahan_Centroid:
                count = np.zeros(K)
                for n in range(K):
                    count[n] = len(result[result == n])
                print("Gap Cluster :", (np.max(count) - np.min(count)))
                if (np.max(count) - np.min(count)) > 10:
                    print("total per cluster :", count)
                    centroids[np.argmax(count)][0] = rd.uniform(minLat, maxLat)#min
                    centroids[np.argmax(count)][1] = rd.uniform(minLon, maxLon)#min
                else:
                    break
        return result, centroids
    K = 3
    Output, centroid = K_Means(X, K)
    silhouette_avg_after = silhouette_score(X, Output)
    nama=['clusterdistMat1','clusterdistMat2','clusterdistMat3']
    for i in range(K):
        df = pd.DataFrame(X[Output == i])
        filepath = nama[i] + '.xlsx'
        df.rename({0: 'latitude', 1: 'longitude'}, axis=1, inplace=True)
        df.to_excel(filepath, index=False)
    print("Centroid Akhir", centroid)

    color = ['red','blue','green']
    labels = ['cluster1','cluster2','cluster3']
    for i in range(K):
        plt.scatter(X[Output == i][:, 0], X[Output == i][:, 1], c=color[i], label=labels[i])
    plt.scatter(centroid[:,0], centroid[:,1], s=200,c = 'magenta', label = 'Centroids')
    plt.title("Visualisasi Hasil K-Means")
    plt.xlabel('Lat')
    plt.ylabel('Long')
    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    maxGap = np.max(np.abs(max - min))
    plt.xlim(min[0] - 0.01, min[0] + maxGap + 0.01)
    plt.ylim(min[1] - 0.01, min[1] + maxGap + 0.01)
    plt.legend()
    plt.show()

    dsAsli = []
    outAsli = []
    labelSales = ['Sales1','Sales2','Sales3']
    for i, dataAsli in enumerate(glob.glob("Data_Sales_Asli/*.xlsx")):
        df = pd.read_excel(dataAsli)
        data = (df[["latitude", "longitude"]])
        Xbaru = data.iloc[:, [0, 1]].values
        if len(dsAsli) > 0:
            dsAsli = np.vstack((dsAsli, Xbaru))
        else:
            dsAsli = Xbaru.copy()
        outAsli = outAsli + [i] * len(Xbaru)
        plt.scatter(Xbaru[:, 0], Xbaru[:, 1], c=color[i], label=labelSales[i])
    plt.title("Visualisasi Data Perjalanan Sales Riil")
    plt.xlabel('Lat')
    plt.ylabel('Long')
    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    maxGap = np.max(np.abs(max - min))
    plt.xlim(min[0] - 0.01, min[0] + maxGap + 0.01)
    plt.ylim(min[1] - 0.01, min[1] + maxGap + 0.01)
    plt.legend()
    plt.show()
    silhouette_avg_before = silhouette_score(dsAsli, outAsli)
    print("\nThe average original silhouette_score is :", silhouette_avg_before)
    print("The average current silhouette_score is :", silhouette_avg_after)
    print("The average silhouette_score gap is :", np.abs(silhouette_avg_after - silhouette_avg_before) / silhouette_avg_before * 100, "%")
