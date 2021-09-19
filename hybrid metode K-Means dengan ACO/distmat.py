import numpy as np
import pandas as pd
from geopy.distance import great_circle
import os
def runDistmat():
    file_iterate=1
    file_name = [x for x in os.listdir() if x.endswith('.xlsx')]
    for a in file_name:
        df = pd.read_excel(a)
        df=(df[["latitude", "longitude"]])
        X = np.array(df)
        m, n = X.shape
        hasil = np.zeros((m,m))
        for i in range(len(X)):
            for j in range(len(X)):
                start = X[i]
                end = X[j]
                dist = great_circle(start, end).kilometers
                hasil[i, j] = dist    
        DF = pd.DataFrame(hasil)
        DF.to_csv("distMat"+ str(file_iterate)+".csv", index=False)
        file_iterate+=1
    