import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
def runAco(NCmax,AntsNum,rho,al,b):
    def Penghitungan(file_name,AntsNum):
        print("\n     Optimasi Rute Distribusi Produk ")
        print("  PT Indomarco Adi Prima Stock Point Nganjuk")
        print("----------------------------------------------")
        print("Nama File : ",file_name)
        Q=1
        nilai_tau_awal=0.021
        def Data():
            df=pd.read_csv(file_name)
            df=np.array(df)
            return df

        distMat=Data()
        print("Distance Matrix")
        print(distMat,"\n")
        nCities= distMat.shape[0]
        if AntsNum == 0:
            AntsNum= nCities
 
        def Hitung_Visibilitas(distMat, nCities):
            for i in range(nCities):
                distMat[i, i] = np.inf
            visibilitas = np.true_divide(1, distMat)
            return visibilitas

        def Hitung_PanjangLintasan(paths, distMat):
            tourLen = 0.0
            for i in range(len(paths) - 1):
                tourLen += distMat[paths[i], paths[i + 1]]
            tourLen += distMat[paths[-1], paths[0]]
            return tourLen
            
        def Penyusunan_Jalur(start, alpha, beta,visibilitas,distMat,tau):
            memory = np.zeros(1, dtype=int)
            memory[0] = start
            bantu_prob = np.power(tau, alpha) * np.power(visibilitas, beta)
            prob = bantu_prob / np.sum(bantu_prob, axis = 1, keepdims = True)
            prob[:, start] = 0
            for i in range(nCities - 1):
                nextNode = np.argmax(prob[start, :])
                random = np.random.uniform(0, 1) / 100
                if (random >= prob[start, nextNode]):
                    prob[start, nextNode] = random
                    idx = np.argwhere(prob[start, :] > 0.0).flatten()
                    if len(idx) > 0:
                        randIdx = rd.randrange(0, len(idx))
                        nextNode = idx[randIdx]
                prob[:, nextNode] = 0
                start = nextNode
                memory = np.append(memory, nextNode)
            return memory
            
        def pheromone_Evaporate(tau):
            for i in range(len(tau)):
                for j in range(len(tau)):
                    tau[i, j] = (1 - rho) * tau[i, j]
            return tau

        def update_Pheromoneamount(paths,distMat,tau):
            for i in range(len(paths)):
                Delta = Q / Hitung_PanjangLintasan(paths[i], distMat)
                for j in range(len(paths[i]) - 1):
                    tau[paths[i][j], paths[i][j + 1]] += Delta
                    tau[paths[i][j + 1], paths[i][j]] += Delta
                j += 1
                tau[paths[i][0], paths[i][j]] += Delta
                tau[paths[i][j], paths[i][0]] += Delta
            return tau

        def ACO(NCmax,AntsNum,alpha,beta):
            Hitung_Fitness = np.zeros(NCmax)
            best_path = np.zeros((NCmax, nCities),dtype=int)
            tau = np.zeros((nCities, nCities))
            visibilitas=Hitung_Visibilitas(distMat,nCities)
            print("Matrix Visibilitas")
            print(visibilitas,"\n")

            for i in range(nCities):
                for j in range(nCities):
                    tau[i,j]=nilai_tau_awal
            print("Tau Awal")
            print(tau,"\n")
            
            for t in range(NCmax):
                paths =np.zeros((nCities, nCities), dtype=int)
                paths_len=np.zeros(nCities)
                start = np.random.choice(range(AntsNum), AntsNum, replace=False)
                for k in range(AntsNum):
                    memory = Penyusunan_Jalur(start[k], alpha, beta,visibilitas,distMat,tau)
                    paths[k]=memory
                for a in range(nCities):
                    paths_len[a] = Hitung_PanjangLintasan(paths[a], distMat)
                    
                print('Hasil penyusunan jalur kunjungan semut pada Iterasi ke- ',t+1 )
                a= paths_len.shape[0]
                jarak=paths_len.reshape(a,1)
                Matrix_Perjalanan=np.hstack((paths,jarak))
                print(Matrix_Perjalanan)

                best_path_index=np.argmin(paths_len)
                best_path[t]=paths[best_path_index]
                print("Local Best : ",best_path[t])
                print("min fitness :", np.amin(jarak))
                  
                tau = pheromone_Evaporate(tau) #tau*rho
                tau = update_Pheromoneamount(paths,distMat,tau)
                print('\nTau Baru')
                print(tau,'\n')
                np.set_printoptions(suppress=True) #Biar tidak keluar nilai e+

            for a in range(NCmax):
                Hitung_Fitness[a] = Hitung_PanjangLintasan(best_path[a], distMat)
            Terpendek_All_Iterasi=best_path[np.argmin(Hitung_Fitness)]
            return Terpendek_All_Iterasi
            
        Jalur=ACO(NCmax,AntsNum,alpha=al,beta=b)
        fitnes=Hitung_PanjangLintasan(Jalur, distMat)
        print("\nHasil ACO diperoleh jalur terpendek(Global Best) ")
        print(Jalur)
        print("Dengan Nilai Fitness: ",fitnes)

        plt.figure(figsize =(20, 20))
        latLong = pd.read_excel("cluster" + file_name.split(".")[0] + ".xlsx").values
        latLongPlot = np.zeros(latLong.shape)
        for i in range(nCities):
            latLongPlot[i] = latLong[Jalur[i]]
            plt.text(latLongPlot[i, 0], latLongPlot[i, 1], Jalur[i], horizontalalignment='center', verticalalignment='center', size=10)
        plt.title("Total Jarak: "+str(fitnes)+" km")
        plt.plot(latLongPlot[:, 0], latLongPlot[:, 1], 'b->', label='line 1', linewidth=2)
        plt.plot([latLongPlot[0][0], latLongPlot[-1][0]], [latLongPlot[0][1], latLongPlot[-1][1]], 'r--', label='line 1', linewidth=2)
        plt.plot(latLongPlot[:, 0], latLongPlot[:, 1], 'oc', label='line 2', markersize=14)
        max = np.amax(latLongPlot, axis=0)
        min = np.amin(latLongPlot, axis=0)
        gap = np.amax(np.abs(max - min))
        plt.xlim(min[0] - gap * 0.2, min[0] + gap * 1.2)
        plt.ylim(min[1] - gap * 0.2, min[1] + gap * 1.2)
        return Jalur, fitnes

    import os
    file_name = [x for x in os.listdir() if x.endswith('.csv')]
    Hasil_Penghitungan= []
    allFitness = []
    for i in file_name:
        Jalur, fitness = Penghitungan(i,AntsNum)
        allFitness.append(fitness)
        Hasil_Penghitungan.append(Jalur.flatten())
    print("\n     Hasil Optimasi Rute Distribusi Produk ")
    print("  PT Indomarco Adi Prima Stock Point Nganjuk")
    print("  Dengan K-Means dan Ant Colony Optimization")
    print("----------------------------------------------")
    print("Fitness", allFitness)
    print("Sum Fitness", np.sum(allFitness))
    for i, Jalur in enumerate(Hasil_Penghitungan):
        print("Sales ke-", i+1, " : ", " -> ".join(map(str, Jalur)), " | ", allFitness[i])
    print("Total Fitness : ", np.sum(allFitness))

    import salesAsli
    selisih=salesAsli.runSales()-np.sum(allFitness)
    print("\nSelisih Jarak = Hasil Perjalanan Sales Riil - Hasil K-ACO= "+str(selisih)+" km")
    plt.show()