"""
Code by Wahyu Bimantara
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import classification_report

#=========== Membaca Excel ==========#
data = pd.read_excel (r'data2.xlsx')

#=========== Membaca Setiap Baris ==========#
PelvicIncidence = pd.DataFrame(data, columns= ['pelvic_incidence'])
PelvicTilt = pd.DataFrame(data, columns= ['pelvic_tilt'])
LumbarLordosisAngle = pd.DataFrame(data, columns= ['lumbar_lordosis_angle'])
SacralSlope = pd.DataFrame(data, columns= ['sacral_slope'])
PelvicRadius = pd.DataFrame(data, columns= ['pelvic_radius'])
DegreeSpondylolisthesis = pd.DataFrame(data, columns= ['degree_spondylolisthesis'])
Class = pd.DataFrame(data, columns= ['class'])
matrixClassa = Class.as_matrix()
#=========== Value bawah sendiri tidak dipanggil karena berupa data testing dan kelasnya dicari ==========#
matrixClass = matrixClassa[0:len(Class)-1]

#=========== Mendapatkan D ==========#
matrixD = np.hstack((PelvicIncidence, PelvicTilt, LumbarLordosisAngle, SacralSlope, PelvicRadius, DegreeSpondylolisthesis))
print ("\nMatrix D \n", matrixD)

#=========== Mencari Mean Setiap Fitur ==========#
meanPelvicIncidence = float(PelvicIncidence.mean())
meanPelvicTilt = float(PelvicTilt.mean())
meanLumbarLordosisAngle = float(LumbarLordosisAngle.mean())
meanSacralSlope = float(SacralSlope.mean())
meanPelvicRadius = float(PelvicRadius.mean())
meanDegreeSpondylolisthesis = float(DegreeSpondylolisthesis.mean())

average = np.hstack((meanPelvicIncidence, meanPelvicTilt, meanLumbarLordosisAngle, meanSacralSlope, meanPelvicRadius, meanDegreeSpondylolisthesis))
print ("\nRata-Rata setiap fitur\n", average)

#=========== Mencari ZeroMean ==========#
zeroMean = np.subtract(matrixD, average)
print("\nZeroMean\n",zeroMean)

#=========== Menghitung Covarian ==========#
n = len(zeroMean[0])
covarian = 1/(n-1)*(np.transpose(zeroMean).dot(zeroMean))
print("\ncovarian\n",covarian)

#=========== Menghitung Nilai Eigen dan Eigen Vector ==========#
w, v = np.linalg.eig(covarian)
print("\nEigen Value\n",w) # w = eigen value
print("\nEigen Vector\n",v) #v = eigen vector

#=========== Mengurutkan Eigen Value ==========#
wSort = sorted(w, reverse = True)
print("\nEigen Value Sorted\n",wSort)

#=========== Mempertahankan 85% data ==========#
i=0
lamb=0
for i in range(len(w)):
    lamb += wSort[i]
    i += 1

keep = (85/100) * lamb

i2 = 0
j2 = 0
for i2 in range(len(w)):
    j2 += wSort[i2]
    i2 += 1
    if j2 > keep:
        break

newEgVec = v[0:len(v),0:i2]

fiturBaru = np.transpose(np.transpose(newEgVec).dot(np.transpose(zeroMean)))
print("\nFitur Baru 85%\n",fiturBaru)
print("\n================== END OF PCA ==================\n")

print("\n================== WEIGHTED-KNN ==================\n")

#=========== Memisahkan antara data testing dan data set dari data PCA ==========#
dataSet = fiturBaru[0:len(fiturBaru)-1,:]
dataTesting = fiturBaru[len(fiturBaru)-1:,:]

print("\nData Set\n",dataSet)
print("\nData Testing\n",dataTesting)

#=========== Perhitungan Eucledian ==========#
hitungSetTes  = np.subtract(dataTesting, dataSet)
powerSetTes = np.power(hitungSetTes, 2)
splitSetTes = np.hsplit(powerSetTes, len(powerSetTes[0]))

addSetTes = 0
for i in range(len(powerSetTes[0])):
    addSetTes =+ splitSetTes[i]

sqrtSetTes = np.sqrt(addSetTes)
balikan = np.hstack((sqrtSetTes, matrixClass))

print("\nJarak dari data testing\n",balikan)

#=========== Weighted-KNN ==========#
hernia = 0
spondylolisthesis = 0
normal = 0

for i in range(len(balikan)):
    if balikan[i,len(balikan[0])-1] == 'Hernia':
        hernia =+ (1/np.power(balikan[i,0],2))
    elif balikan[i,len(balikan[0])-1] == 'Spondylolisthesis':
        spondylolisthesis =+ (1/np.power(balikan[i,0],2))
    else:
        normal =+ (1/np.power(balikan[i,0],2))

print("\nVote Hernia            = ", hernia)
print("Vote Spondylolisthesis = ", spondylolisthesis)
print("Vote Normal            = ", normal)

#=========== Voting untuk menentukan kelas ==========#

hasil = max(hernia, spondylolisthesis, normal)

if hasil == hernia:
    predicted = ["hernia"]
elif hasil == spondylolisthesis:
    predicted = ["spondylolisthesis"]
else:
    predicted = ["normal"]
print("\nHASIL VOTE = ", predicted)

print("\n================== END OF WEIGHTED-KNN ==================\n")

print("\n================== CONFUSION MATRIX ==================\n")
#=========== Mencari Confusion Matrix ==========#

dataUji = 20
if balikan[dataUji,len(balikan[0])-1] == 'Hernia':
    expected = ["hernia"]
elif balikan[dataUji,len(balikan[0])-1] == 'Spondylolisthesis':
    expected = ["spondylolisthesis"]
elif balikan[dataUji,len(balikan[0])-1] == 'Normal':
    expected = ["normal"]

confussionMatrix = classification_report(expected, predicted)

print("\nexpected = ", expected, "\npredicted = ", predicted)
print(confussionMatrix)