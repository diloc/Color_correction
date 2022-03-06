# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:00:44 2018

@author: DLozano
"""

import matplotlib.pylab as plt
import numpy as np
import time
import cv2
import sys, os

#%%
# dirParent = "D:/OneDrive - LA TROBE UNIVERSITY/PhD - Datasets"
# dirParent = "C:/Users/dlozanoclaro/OneDrive - LA TROBE UNIVERSITY/PhD - Datasets/"
# dirParent = "C:/Users/17600112/OneDrive - LA TROBE UNIVERSITY/PhD - Datasets/colour_constancy/"

# path_trash  = os.path.join(dirParent, 'trash')

# if not os.path.exists(path_trash): os.makedirs(path_trash) 

#%%
tplInitial = (1.0, 0.0, 1.0)
eps = np.finfo(np.float).eps
colrLabl = ['blue', 'green', 'red']
corrLabl = ['lightcoral', 'mediumseagreen', 'cornflowerblue']

#%%
def im(img,name):
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img[np.where((img<[1,1,1]).all(axis = 2))] = [255,255,255]

    plt.imshow(img)
    plt.title(name)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def imT(img1,img2,name1,name2):
    if len(img1.shape)==3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1[np.where((img1<[1,1,1]).all(axis = 2))] = [255,255,255]
    
    if len(img2.shape)==3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2[np.where((img2<[1,1,1]).all(axis = 2))] = [255,255,255]
    
    plt.plot([1,2,3])
    plt.subplot(121)
    plt.imshow(img1)
    plt.title(name1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(img2)
    plt.title(name2)
    plt.xticks([]), plt.yticks([])
    plt.show()
    #time.sleep(3)
    #plt.close()

#%%

def im3(img1,img2,img3, name1,name2, name3):
    if len(img1.shape)==3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1[np.where((img1<[1,1,1]).all(axis = 2))] = [255,255,255]
    
    if len(img2.shape)==3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2[np.where((img2<[1,1,1]).all(axis = 2))] = [255,255,255]
    
    if len(img3.shape)==3:
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        img3[np.where((img3<[1,1,1]).all(axis = 2))] = [255,255,255]
    
    plt.plot([1,2,3])
    plt.subplot(131)
    plt.imshow(img1)
    plt.title(name1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(132)
    plt.imshow(img2)
    plt.title(name2)
    plt.xticks([]), plt.yticks([])
    plt.subplot(133)
    plt.imshow(img3)
    plt.title(name3)
    plt.xticks([]), plt.yticks([])
    plt.show()
    #time.sleep(3)
    #plt.close()
#%%
def im4(img1, img2, name1, name2, img3, img4, name3, name4, posn):
    colrLabl = ['blue', 'green', 'red']
    
    H1, S1, V1 = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV))
    H2, S2, V2 = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV))
    H3, S3, V3 = cv2.split(cv2.cvtColor(img3, cv2.COLOR_BGR2HSV))
    H4, S4, V4 = cv2.split(cv2.cvtColor(img4, cv2.COLOR_BGR2HSV))
    
    if len(img1.shape)==3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1[np.where((img1<[1,1,1]).all(axis = 2))] = [255,255,255]
    
    if len(img2.shape)==3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2[np.where((img2<[1,1,1]).all(axis = 2))] = [255,255,255]
    
    if len(img3.shape)==3:
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        img3[np.where((img3<[1,1,1]).all(axis = 2))] = [255,255,255]
    
    if len(img4.shape)==3:
        img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
        img4[np.where((img4<[1,1,1]).all(axis = 2))] = [255,255,255]
    
    
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    # img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    
    fig, axs = plt.subplots(2, 2)
    
    fig.suptitle(posn)
    
    axs[0, 0].imshow(img1)
    axs[0, 0].set_title(name1)
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(img2)
    axs[0, 1].set_title(name2)
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(img3)
    axs[1, 0].set_title(name3)
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(img4)
    axs[1, 1].set_title(name4)
    axs[1, 1].axis('off')
    
    # axs[2, 0].hist(H3.flatten(), bins=180, color=colrLabl[0], label='L')
    # axs[2, 0].hist(H4.flatten(), bins=180, color=colrLabl[1], label='R')
    # # axs[2, 0].set_title(name3)
    # axs[2, 0].set_xlabel("Hue")
    # axs[2, 0].legend()
    
    
    # axs[2, 1].axis('off')
    
    
    
    plt.show()
#%%
def hist(gray,name):
    plt.hist(gray.flatten(), bins=180, color='r', label=name)
    plt.xlim([0, 260])
    plt.legend()
    plt.title(name)
    plt.show()
    
def histT(gray1, gray2, name1, name2):
    plt.plot([1,2,3])
    
    plt.subplot(121)
    plt.hist(gray1.flatten(), bins=180, color='r', label=name1)
    plt.xlim([0, 260])
    plt.ylim([0, 10000])
    plt.legend()
    plt.title(name1)

    plt.subplot(122)
    plt.hist(gray2.flatten(), bins=180, color='r', label=name2)
    plt.xlim([0, 260])
    plt.ylim([0, 10000])
    plt.legend()
    plt.title(name2)
    
    plt.show()
    
    
    
#%%
def hist1(im1, mask, label):
    
    if len(mask[mask < 1]) / len(mask.flatten()) != 0:
        img1 = im1.copy()

        img1B = img1[:, :, 0].flatten()
        img1G = img1[:, :, 1].flatten()
        img1R = img1[:, :, 2].flatten()
        
        img1B = img1B[np.where(img1B!=0)]
        img1G = img1G[np.where(img1G!=0)]
        img1R = img1R[np.where(img1R!=0)]
        
        #Statistics
        img1Bstcs = np.round(np.mean(img1B)), np.round(np.std(img1B))
        img1Gstcs = np.round(np.mean(img1G)), np.round(np.std(img1G))
        img1Rstcs = np.round(np.mean(img1R)), np.round(np.std(img1R))
        
    
        fig, Adj = plt.subplots(1, 3, figsize=(12,5))
        fig.suptitle(label, fontsize=16)
        
        Adj[0].hist(img1B, bins=255, color='b', label="orig: " + 'mean =' + str(img1Bstcs[0]) + ' str= ' + str(img1Bstcs[1]))
        Adj[0].set_xlim([0, 260])
        Adj[0].legend()
        Adj[0].set_title("blue")
        
        Adj[1].hist(img1G, bins=255, color='g', label="orig: " + 'mean =' + str(img1Gstcs[0]) + ' str= ' + str(img1Gstcs[1]))
        Adj[1].set_xlim([0, 260])
        Adj[1].legend()
        Adj[1].set_title("green")
        
        Adj[2].hist(img1R, bins=255, color='r', label="orig: " + 'mean =' + str(img1Rstcs[0]) + ' str= ' + str(img1Rstcs[1]))
        Adj[2].set_xlim([0, 260])
        Adj[2].legend()
        Adj[2].set_title("red")
        
        plt.show()
        
        print('-'*20)
        print(label + ' (mean)', 
              " "*5, 'blue  = ', img1Bstcs[0], 
              " "*5, 'green = ', img1Gstcs[0], 
              " "*5, 'red = ', img1Rstcs[0])
        
        print('-'*20)
        print(label + ' (std)', 
              " "*5, 'blue  = ', img1Bstcs[1], 
              " "*5, 'green = ', img1Gstcs[1], 
              " "*5, 'red = ', img1Rstcs[1])
        


#%%
def hist2(im1, im2, mask, label):
    
    if len(mask[mask < 1]) / len(mask.flatten()) != 0:

        img1 = im1.copy()
        img2 = im2.copy()
        
        img1B = img1[:, :, 0].flatten()
        img1G = img1[:, :, 1].flatten()
        img1R = img1[:, :, 2].flatten()
        
        img2B = img2[:, :, 0].flatten()
        img2G = img2[:, :, 1].flatten()
        img2R = img2[:, :, 2].flatten()
        
        img1B = img1B[np.where(img1B!=0)]
        img1G = img1G[np.where(img1G!=0)]
        img1R = img1R[np.where(img1R!=0)]
        
        img2B = img2B[np.where(img2B!=0)]
        img2G = img2G[np.where(img2G!=0)]
        img2R = img2R[np.where(img2R!=0)]
        
        #Statistics
        img1Bstcs = np.round(np.mean(img1B)), np.round(np.std(img1B))
        img1Gstcs = np.round(np.mean(img1G)), np.round(np.std(img1G))
        img1Rstcs = np.round(np.mean(img1R)), np.round(np.std(img1R))
        
        img2Bstcs = np.round(np.mean(img2B)), np.round(np.std(img2B))
        img2Gstcs = np.round(np.mean(img2G)), np.round(np.std(img2G))
        img2Rstcs = np.round(np.mean(img2R)), np.round(np.std(img2R))
        
    
        fig, Adj = plt.subplots(1, 3, figsize=(12,5))
        fig.suptitle(label, fontsize=16)
        
        Adj[0].hist(img1B, bins=255, color='g', label="orig: " + 'mean =' + str(img1Bstcs[0]) + ' str= ' + str(img1Bstcs[1]))
        Adj[0].hist(img2B, bins=255, color='r', label="corr: " + 'mean =' + str(img2Bstcs[0]) + ' str= ' + str(img2Bstcs[1]))
        Adj[0].set_xlim([0, 260])
        Adj[0].legend()
        Adj[0].set_title("blue")
        
        Adj[1].hist(img1G, bins=255, color='g', label="orig: " + 'mean =' + str(img1Gstcs[0]) + ' str= ' + str(img1Gstcs[1]))
        Adj[1].hist(img2G, bins=255, color='r', label="corr: " + 'mean =' + str(img2Gstcs[0]) + ' str= ' + str(img2Gstcs[1]))
        Adj[1].set_xlim([0, 260])
        Adj[1].legend()
        Adj[1].set_title("green")
        
        Adj[2].hist(img1R, bins=255, color='g', label="orig: " + 'mean =' + str(img1Rstcs[0]) + ' str= ' + str(img1Rstcs[1]))
        Adj[2].hist(img2R, bins=255, color='r', label="corr: " + 'mean =' + str(img2Rstcs[0]) + ' str= ' + str(img2Rstcs[1]))
        Adj[2].set_xlim([0, 260])
        Adj[2].legend()
        Adj[2].set_title("red")
        
        plt.show()
        
        print('-'*20)
        print(label + ' (mean)',
            "    ",
            'blue  = ', img1Bstcs[0], '|', img2Bstcs[0],
            "    ",
            'green = ', img1Gstcs[0], '|', img2Gstcs[0],
            "    ",
            'red = ', img1Rstcs[0], '|', img2Rstcs[0])
        
        print(label + ' (std)',
            "    ",
            'blue  = ', img1Bstcs[1], '|', img2Bstcs[1],
            "    ",
            'green = ', img1Gstcs[1], '|', img2Gstcs[1],
            "    ",
            'red = ', img1Rstcs[1], '|', img2Rstcs[1])

#%%
# def histAllHsv(imgOri, df1):
#     df = df1.copy()
    
#     fig, axs =  plt.subplots(len(df), 4)
    
#     for cnt in range(len(df)):
#         imgCorr = []
        
        
        
        
        
    

#%%
def plotPotsBGR(imOrigBGR, imCorrBGR, crop, df_Main, suffix, ):   
    imOrig = imOrigBGR.copy()
    imCorr = imCorrBGR.copy()

    df = df_Main.copy()
    
    for cntP in range(len(df)):
    # for cntP in (list(np.random.choice(160, 30))):
        potCorr, potOrig = [], []
        soilCorr, soilOrig = [], []
        maskJoin, mask_inv = [], []
        
        posn, name, top, left, wd, ht = df.loc[cntP,
                                               ['position', 'name', 'top', 'left', 'width', 'height']].values
        
        # if name != 'Checker': continue
    
        trayNum = posn.split('_')[0]
        
        # if trayNum != 'T16': continue
    
        potCorr = imCorr[top + crop : top + ht - crop, 
                  left + crop: left + wd - crop]
        
        potOrig = imOrig[top + crop : top + ht - crop, 
                  left + crop: left + wd - crop]
        
        # mask_blk, _ = segmentation.segment(potCorr)
        # mask_inv = cv2.bitwise_not(mask_blk)
        # mask = mask_blk
        
        # soilCorr = cv2.bitwise_and(potCorr, potCorr, mask = mask)
        # soilOrig = cv2.bitwise_and(potOrig, potOrig, mask = mask)
        
        
        # potCorrBGR = cv2.cvtColor(potCorr, cv2.COLOR_RGB2BGR)
        # soilCorrBGR = cv2.cvtColor(soilCorr, cv2.COLOR_RGB2BGR)
        # soilOrigBGR = cv2.cvtColor(soilOrig, cv2.COLOR_RGB2BGR)

        # # im3(potOrigBGR, soilOrigBGR, soilCorrBGR, posn, 'soilOrig', 'soilCorr')
        # # hist2(soilOrig, soilCorr, mask, posn)
        
        # imT(potOrigBGR, potCorrBGR ,'potOrig', 'potCorr')
        # hist2(potOrig, potCorr, np.zeros((mask.shape)), posn)
        
        im(potOrig, posn)
        potName = suffix + '_' + str(cntP) + '_' + posn + '.jpg'
        cv2.imwrite(os.path.join(path_trash, potName), potOrig)
    

#%%
def plotColorVal(sRed, sGrn, sBlu, potName):
    redSamp, grnSamp, bluSamp = sRed.copy(), sGrn.copy(), sBlu.copy()
    alpha, beta = [], []
    MSEb, MSEa = [], []

    colorTrue = np.array([redSamp.columns,
                          grnSamp.columns,
                          bluSamp.columns])

    mednColr =  np.array([redSamp.median(axis =0).values,
                          grnSamp.median(axis =0).values,
                          bluSamp.median(axis =0).values])
    
    fig1, ax1 = plt.subplots(1, 2, figsize=(12,12))

    for cnt in range(3):
        y, x = [], []
        y = colorTrue[cnt, :]
        x = mednColr[cnt, :]
        y_ind = np.argsort(y) 
        y, x  = y[y_ind], x[y_ind]
        
        ax1[0].plot(y, label = colrLabl[cnt], color=colrLabl[cnt], linewidth=2)
        ax1[1].plot(x, label = colrLabl[cnt], color=colrLabl[cnt], linewidth=2)
        ax1[1].plot(y, color = corrLabl[cnt], linewidth=1, linestyle= '--')
    
    ax1[0].set_ylim([0, 300])
    ax1[0].legend(loc='upper left')
    ax1[0].set_title('True Image colors')
    ax1[0].set_ylabel('Pixel value')
    ax1[0].set_xlabel('Number of samples')
    ax1[0].grid()
    
    ax1[1].set_ylim([0, 300])
    ax1[1].legend(loc='upper left')
    ax1[1].set_title('Card Image colors')
    ax1[1].set_ylabel('Pixel value')
    ax1[1].set_xlabel('Number of samples')
    ax1[1].grid()
    
    plt.show() 
    
#%%
def plotColorVal2(colorTrue, colorImg, colorCorr, plotName, MSEb, MSEa):
    fig1, ax1 = plt.subplots(1, 2, figsize=(14,5))

    for cnt in range(3):
        y0, y1, y1c = [], [], []
        y0 = colorTrue[cnt, :]
        y1 = colorImg[cnt, :]
        y1c = colorCorr[cnt, :]
        
        ax1[0].plot(y0, label = colrLabl[cnt] + 'True', color=colrLabl[cnt], linewidth=2) 
        ax1[0].plot(y1, label = colrLabl[cnt] + 'Img', color=colrLabl[cnt], linewidth=2, linestyle= '--')
        
        ax1[1].plot(y1c, label = colrLabl[cnt]+ 'Corr', color=colrLabl[cnt], linewidth=2)
        # ax1[1].plot(y1, label = colrLabl[cnt] + 'Img', color=colrLabl[cnt],  linewidth=2, linestyle= '--')
        ax1[1].plot(y0, label = colrLabl[cnt] + 'True', color=colrLabl[cnt], linewidth=1, linestyle= '--')
        
        
        
        
    ax1[0].set_ylim([0, 300])
    ax1[0].legend(loc='upper left')
    ax1[0].set_title(plotName + '\n' +'True Image colors' + '\n' + 'RGB')
    ax1[0].set_ylabel('Pixel value')
    ax1[0].set_xlabel('Number of samples')
    ax1[0].grid()
    
    ax1[1].set_ylim([0, 300])
    ax1[1].legend(loc='upper left')
    ax1[1].set_title('Card Image colors')
    ax1[1].set_ylabel('Pixel value')
    ax1[1].set_xlabel('Number of samples')
    # ax1[1].set_title(potName + ': ' + colrLabl[0] + ' ' + str(MSEa[0]))
    ax1[1].grid()
    ax1[1].set_title(plotName + '\n' + 
                     'Mean Square Error:   after / before' '\n' +
                     'Blue=' + str(int(MSEa[0])) +'/' + str(int(MSEb[0])) + '  '
                     'Green=' + str(int(MSEa[1])) +'/' + str(int(MSEb[1])) + '  '
                     'Red='+ str(int(MSEa[2])) +'/' + str(int(MSEb[2]))                     )
    plt.show() 


def plotIlluminant(df1):
    df = df1.copy()
    df.sort_values(by='Ymean', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print()
    print()
    print("The difference of color illuminant")
    
    for cntP in range(len(df)):
        
         
        posn = df.loc[cntP, 'position']
        Ymean = round(df.loc[cntP, 'Ymean'], 2)
        bluIllum, grnIllum, redIllum  = df.loc[cntP, ['bluIllum', 'grnIllum', 'redIllum']].values
        bluAlpha, grnAlpha, redAlpha  = df.loc[cntP, ['bluAlpha', 'grnAlpha', 'redAlpha']].values
        
        bluIl = np.uint8(round(bluIllum/2))
        grnIl = np.uint8(round(grnIllum/2))
        redIl = np.uint8(round(redIllum/2))
        
        colrImg = []
        colrImg = np.ones((20,20,3)).astype('uint8')
        colrImg[:, :, 0] = bluIl
        colrImg[:, :, 1] = grnIl
        colrImg[:, :, 2] = redIl
        
        Him, Sim, Vim = cv2.split(cv2.cvtColor(colrImg, cv2.COLOR_BGR2HSV))
        H, S, V = np.mean(Him), np.mean(Sim), np.mean(Vim)
        
        bluCorr = np.uint8(round(bluIllum* bluAlpha))
        grnCorr = np.uint8(round(grnIllum* grnAlpha))
        redCorr = np.uint8(round(redIllum* redAlpha))
        
        colrCorr = []
        colrCorr = np.ones((20,20,3)).astype('uint8')
        colrCorr[:, :, 0] = bluCorr
        colrCorr[:, :, 1] = grnCorr
        colrCorr[:, :, 2] = redCorr
        
        colrsLabl = (posn + '  Ymean=' + str(Ymean) + '\n' + 'Blue=' + str(bluIl) +' ' 
                        + 'Grn=' + str(grnIl) +' ' + 'Red=' +str(redIl) + '\n'
                        + 'H=' + str(H) +' ' + 'S=' +str(S) +' ' + 'V=' +str(V))
        
        if cntP == 1:
            imT(colrImg, colrCorr,colrsLabl, "corrected")
        
        print('-'*80)
        print(posn)
        print('Ymean=', Ymean, ' ', 
              'Blue=', bluIl, ' ','Grn=', grnIl,' ', 'Red=',redIl,
              'H=', H,' ', 'S=',S,' ', 'V=',V)
        
        print('bluAlpha=', bluAlpha,' ', 'grnAlpha=', grnAlpha,' ', 'redAlpha=',redAlpha)
        
        
        
    print('='*80)
    print('bAlphamean=', df.loc[:, 'bluAlpha'].mean(),' ', 
          'gAlphamean=', df.loc[:, 'grnAlpha'].mean(),' ',
          'rAlphamean=', df.loc[:, 'redAlpha'].mean())
    
    print('bAlphastd=', df.loc[:, 'bluAlpha'].std(),' ', 
          'gAlphastd=', df.loc[:, 'grnAlpha'].std(),' ',
          'rAlphastd=', df.loc[:, 'redAlpha'].std())
#%%

def alpha(Alpha):
    AlphaL = Alpha.loc[(Alpha['name']=='CheckerL'), :]
    AlphaR = Alpha.loc[(Alpha['name']=='CheckerR'), :]
    colrLabl = ['blue', 'green', 'red']
    bluAlphaL = np.sort(AlphaL.loc[:, 'bluAlpha'].values)
    grnAlphaL = np.sort(AlphaL.loc[:, 'grnAlpha'].values)
    redAlphaL = np.sort(AlphaL.loc[:, 'redAlpha'].values)
    
    bluAlphaR = np.sort(AlphaR.loc[:, 'bluAlpha'].values)
    grnAlphaR = np.sort(AlphaR.loc[:, 'grnAlpha'].values)
    redAlphaR = np.sort(AlphaR.loc[:, 'redAlpha'].values)
    
    fig, axs = plt.subplots(1, 2, figsize=(14,5))
    axs[0].plot(bluAlphaL, color=colrLabl[0], linewidth=1, linestyle= '--', label= 'bluAlpha' + '_Left_Img')
    axs[0].plot(np.mean(bluAlphaL))
    axs[0].plot(grnAlphaL, color=colrLabl[1], linewidth=1, linestyle= '--', label= 'grnAlpha' + '_Left_Img')
    axs[0].plot(redAlphaL, color=colrLabl[2], linewidth=1, linestyle= '--', label= 'redAlpha' + '_Left_Img')
    
    axs[0].plot(bluAlphaR, color=colrLabl[0], linewidth=1, label= 'bluAlpha' + '_Right_Img')
    axs[0].plot(grnAlphaR, color=colrLabl[1], linewidth=1, label= 'grnAlpha' + '_Right_Img')
    axs[0].plot(redAlphaR, color=colrLabl[2], linewidth=1, label= 'redAlpha' + '_Right_Img')
    
    axs[0].legend()
    axs[0].set_title("Alpha comparison (Left vs. Right)")
    axs[0].set_ylim([0.5, 1])
    axs[0].grid()
    
    bluAll = np.sort(np.append(bluAlphaL, bluAlphaR))
    grnAll = np.sort(np.append(grnAlphaL, grnAlphaR))
    redAll = np.sort(np.append(redAlphaL, redAlphaR))
    
    bluMin = np.round(bluAll[0], decimals = 2)
    bluMax = np.round(bluAll[-1], decimals = 2)
    bluDiff = np.round(bluAll[-1] - bluAll[0], decimals = 2)
    bluMean = np.round(np.mean(bluAll), decimals = 2)
    grnMin = np.round(grnAll[0], decimals = 2)
    grnMax = np.round(grnAll[-1], decimals = 2)
    grnDiff = np.round(grnAll[-1] - grnAll[0], decimals = 2)
    grnMean = np.round(np.mean(grnAll), decimals = 2)
    
    redMin = np.round(redAll[0], decimals = 2)
    redMax = np.round(redAll[-1], decimals = 2)
    redDiff = np.round(redAll[-1] - redAll[0], decimals = 2)
    redMean = np.round(np.mean(redAll), decimals = 2)
    
    axs[1].plot(bluAll, color=colrLabl[0], linewidth=1, label= 'bluAlpha from ' + str(bluMin) + ' to ' + str(bluMax) + ' (mean= ' +str(bluMean) + ')')
    axs[1].plot(grnAll, color=colrLabl[1], linewidth=1, label= 'grnAlpha from ' + str(grnMin) + ' to ' + str(grnMax) + ' (mean= ' +str(grnMean) + ')')
    axs[1].plot(redAll, color=colrLabl[2], linewidth=1, label= 'redAlpha from ' + str(redMin) + ' to ' + str(redMax) + ' (mean= ' +str(redMean) + ')')
    
    axs[1].legend()
    axs[1].set_title("Alpha comparison (All)")
    axs[1].set_ylim([0.5, 1])
    axs[1].grid()
    
    plt.show()
    
#%% Plot 3D surface
def plot3Dsurf(x, y, z, chan, title):
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)
    ax.set_xlabel('rows')
    ax.set_ylabel('columns')
    ax.set_zlabel(chan)
    ax.set_title(chan + ' ' + title, fontsize = 20)
    plt.show()
            
