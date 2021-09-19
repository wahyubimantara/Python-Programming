import k_means_fix
import distmat
import aco_fix
pilihan = "y"
NCmax=20
rho=0.3
al=1
b=1
AntsNum=0
if pilihan == "t":
    aco_fix.runAco(NCmax,AntsNum,rho,al,b)
else:
    k_means_fix.runKmeans()
    distmat.runDistmat()
    aco_fix.runAco(NCmax,AntsNum,rho,al,b)