import numpy as np
import pandas as pd
from geopy.distance import great_circle
def runSales():
    def PerjalananSalesAsli(file_name):
        def Data():
            df = pd.read_excel(file_name)
            print("\nHasil Perjalanan Sales PT Indomarco Adi Prima Nganjuk")
            print(file_name)
            df=(df[["latitude", "longitude"]])
            df=np.array(df)
            return df
        X=Data()
        m, n = X.shape
        distMat = np.zeros((m,m))
        for i in range(len(X)):
            for j in range(len(X)):
                start = X[i]
                end = X[j]
                dist = great_circle(start, end).kilometers
                distMat[i, j] = dist
        paths= np.arange(0, len(distMat), dtype=int)
        print ("Jalur Penjalanan Sales")
        print(paths)
        paths_len=np.zeros(len(distMat))
        def Hitung_PanjangLintasan(paths, distMat):
            tourLen = 0.0
            for i in range(len(distMat)-1):
                tourLen += distMat[paths[i], paths[i + 1]]
            tourLen += distMat[paths[-1], paths[0]]
            return tourLen

        paths_len=Hitung_PanjangLintasan(paths,distMat)
        print("Hitung Jarak= ",paths_len,"\n")
        return paths_len
    import glob
    file_name = glob.glob("Data_Sales_Asli/" + "/*.xlsx")
    Hasil_Penghitungan= []
    totalJarakRiil = []
    for i in file_name:
        paths_len = PerjalananSalesAsli(i)
        totalJarakRiil.append(paths_len)
    print("Hasil Perjalanan Sales Riil")
    print("Fitness", totalJarakRiil)
    all_totalJarakRiil=np.sum(totalJarakRiil)
    print("Total Fitness : ",all_totalJarakRiil )
    return all_totalJarakRiil