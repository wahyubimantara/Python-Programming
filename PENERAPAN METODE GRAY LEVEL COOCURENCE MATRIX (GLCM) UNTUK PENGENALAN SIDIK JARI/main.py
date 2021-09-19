import matplotlib.pyplot as plt
import numpy as np
import skimage.feature
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from pandas import ExcelWriter
from pandas import ExcelFile
# from openpyxl import load_workbook

def Greyscale(im):
    result = 0.33 * im[:, :, 0] + 0.59 * im[:, :, 1] + 0.07 * im[:, :, 2]
    a = result.astype(int)
    return(a)

def Greycomatrix(im_greycomatrix):
    a = im_greycomatrix.astype(int)
    print("max: ", np.max(a), " min : ", np.min(a))
    print(a.shape)
    result = greycomatrix(a, [0, 1], [0, np.pi*1/4,], levels=256)
    print(result.shape)
    # result = greycomatrix(a, [0, 1], [0], levels=256)
    return(result)

def Contrast(im_contrast):
    result = greycoprops(Greycomatrix(im_contrast), 'contrast')
    return(result)

def Energy(im_energy):
    result = greycoprops(Greycomatrix(im_energy), 'energy')
    return(result)

def Homogeneity(im_homogeneity):
    result = greycoprops(Greycomatrix(im_homogeneity), 'homogeneity')
    return(result)

def Dissimilarity(im_dissimilarity):
    result = greycoprops(Greycomatrix(im_dissimilarity), 'dissimilarity')
    return(result)

def ASM(im_asm):
    result = greycoprops(Greycomatrix(im_asm), 'ASM')
    return(result)

def simpan(contrast, energy, homogeneity, dissimilarity, asm):
    data = pd.DataFrame(
        {
            'contrast_1':[contrast[0,0]],
            'contrast_2':[contrast[0,1]],
            'contrast_3':[contrast[1,0]],
            'contrast_4':[contrast[1,1]],

            'energy_1':[energy[0,0]],
            'energy_2':[energy[0,1]],
            'energy_3':[energy[1,0]],
            'energy_4':[energy[1,1]],

            'homogeneity_1':[homogeneity[0,0]],
            'homogeneity_2':[homogeneity[0,1]],
            'homogeneity_3':[homogeneity[1,0]],
            'homogeneity_4':[homogeneity[1,1]],

            'dissimilarity_1':[dissimilarity[0,0]],
            'dissimilarity_2':[dissimilarity[0,1]],
            'dissimilarity_3':[dissimilarity[1,0]],
            'dissimilarity_4':[dissimilarity[1,1]],

            'asm_1':[asm[0,0]],
            'asm_2':[asm[0,1]],
            'asm_3':[asm[1,0]],
            'asm_4':[asm[1,1]]
        }
    )

    # book = load_workbook('dataset.xlsx')
    tulis = ExcelWriter('dataset.xlsx')
    # tulis.book = book
    # tulis.sheets = dict((ws.title, ws) for ws in book.worksheets)
    data.to_excel(tulis,'Sheet1',index=False)
    tulis.save()

def run():
    im_gray = plt.imread('loop3.jpg')
    # im_gray = Greyscale(im_gray)

    greycomatrix_result = Greycomatrix(im_gray)
    # print(greycomatrix_result)

    contrast_feature = Contrast(im_gray)
    print("\ncontrast_feature")
    print(contrast_feature)

    energy_feature = Energy(im_gray)
    print("\nenergy_feature")
    print(energy_feature)

    homogeneity_feature = Homogeneity(im_gray)
    print("\nhomogeneity_feature")
    print(homogeneity_feature)

    dissimilarity_feature = Dissimilarity(im_gray)
    print("\ndissimilarity_feature")
    print(dissimilarity_feature)

    asm_feature = ASM(im_gray)
    print("\nasm_feature")
    print(asm_feature)
    simpan(contrast_feature, energy_feature, homogeneity_feature, dissimilarity_feature, asm_feature)

if __name__ == '__main__':
    run()