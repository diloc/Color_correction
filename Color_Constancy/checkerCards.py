# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:55:32 2021

@author: Diloz
"""
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import optimize
import matplotlib.pylab as plt
from joblib import Parallel, delayed
from Detect_colorChecker import paralell_search

import plotting

# colrLable = ['blu', 'grn', 'red']
# lbl_illum = ['blu_L', 'grn_L', 'red_L']
eps = np.finfo(float).eps
#%% Obtain Checker
def getChecker(potCrp, cardBW, search_scale, search_degree, num_cores):

    (H_card, W_card) = cardBW.shape[:2]
    gray = cv2.cvtColor(potCrp, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray,30,30)
    results = Parallel(n_jobs = num_cores, backend = "threading")(delayed(paralell_search)(search_scale, degree, edged, H_card, W_card, cardBW) for degree in search_degree)
    maxVal_all, _, _, _  = zip(*results)
    
    ind = np.argmax(maxVal_all)
    maxVal, maxLoc, r, SCALE  = results[ind]
    deg = search_degree[ind]
    
    (startX, startY) = (int(round(maxLoc[0]*r)), int(round(maxLoc[1]*r)))
    (endX, endY) = (int(round((maxLoc[0] + W_card)*r)), int(round((maxLoc[1] + H_card) * r)))
    
    if deg != 0:
        checkerImg = ndimage.rotate(potCrp, deg)
    else:
        checkerImg = potCrp
    
    return checkerImg[startY:endY,startX:endX,:],startY, endY, startX, endX, deg

#%% 3.1 function:Samples the 70% of the area of every patch on the Macbeth colorChercker
# Calculate the mean RGB color of each patch area
# Calculate the color difference between the sample and ideal value in the LAB color space
# the mode of the sample for the three image channels (RGB)
def cardSampler(colorChecker, checkIMG, colrLable, posn, rows, cols):
    colorTable = colorChecker.loc[:, ['position', 'label', 'red', 'grn', 'blu']]
    rowSiz = checkIMG.shape[0] /rows
    colSiz = checkIMG.shape[1] /cols
    winSiz = int((rowSiz * 0.70)/2)
    
    cardImg =  np.zeros(((2*4*winSiz), (2*6*winSiz), 3), dtype=np.uint8)
    cardLabel = np.ones_like(cardImg)
    
    
    for cntRow in range(rows):
        crow = int((cntRow + 0.5) * rowSiz)
        
        for cntCol in range(cols):
            ccol = int((cntCol + 0.5) * colSiz)
            img_patch = checkIMG[crow-winSiz : crow+winSiz, ccol-winSiz : ccol+winSiz]
            cardImg[int(2*winSiz*cntRow): int(2*winSiz*(1+cntRow)),
                      int(2*winSiz*cntCol): int(2*winSiz*(1+cntCol))] = img_patch
            indx = int((cntRow*6) + cntCol)
            
            labl_patch = np.ones_like(img_patch)

            img_LAB = cv2.cvtColor(img_patch, cv2.COLOR_BGR2LAB)
            L = np.mean(img_LAB[:, :, 0])
            A = np.mean(img_LAB[:, :, 1])
            B = np.mean(img_LAB[:, :, 2])
            
            diff = colorChecker.loc[indx, ['L', 'A', 'B']].values - [L, A, B]
            colorTable.loc[indx, 'delta' + '_' + posn] = np.sqrt(np.sum(diff**2))
            

            for cnt2 in range(len(colrLable)):
                y0_patch = []
                # y0_patch = colorChecker[cnt2, indx]
                y0_patch = colorChecker.loc[indx, colrLable[cnt2]]
                labl_patch[:, :, cnt2] = labl_patch[:, :, cnt2]*y0_patch 
                
                colorTable.loc[indx, colrLable[cnt2] + '_' + posn] = np.mean(img_patch[:, :, cnt2])
                            
            cardLabel[int(2*winSiz*cntRow): int(2*winSiz*(1+cntRow)),
                      int(2*winSiz*cntCol): int(2*winSiz*(1+cntCol))] = labl_patch

    return cardImg, cardLabel, colorTable



#%%
def illum_check(colorChecker, df_main, imgSRC, check_rot, colrLable, imgName):
    tableLable = colorChecker.loc[:, ['position', 'label', 'red', 'grn', 'blu']].copy()
    colorTableAll = tableLable.copy()
    # lbl_illum = [sub + "_L" for sub in colrLable]

    height = imgSRC.shape[0]
    width = imgSRC.shape[1]
    padLeft = 20
    
    imgPad = np.zeros((height + 100, width + 100 + padLeft, 3), np.uint8) # Image padding
    imgPad[0:height, padLeft:width+padLeft] = imgSRC.copy()
    
    # winCol, winRow = 150, 100
    winRow, winCol = 70, 200
    
    df = df_main.copy()
    checkers = df.loc[(df['name']=='Checker')].reset_index(drop=True)
    checkers_fit = checkers.copy()

    for cnt2 in range(len(checkers)):
        colorTable = pd.DataFrame([])
        posn, name, top, left, wd, ht = checkers.loc[cnt2, ['position', 'name', 
                                                  'top', 'left', 'width', 'height']].values
        bottom = top + winRow + ht
        right = left+winCol + wd
        potCrp = imgPad[top : bottom, left : right]
        
        sqrsiz = 48
        cardBW =  np.zeros(((sqrsiz * 4) +1 , (sqrsiz *6) + 1), dtype=np.uint8)
        
        for row in range(4 + 1):
            lineRow = int(sqrsiz  * row)
            cardBW[lineRow, :] = 255
            
            for col in range(6 + 1):
                lineCol = int(sqrsiz  * col)
                cardBW[:, lineCol] = 255
        
        
        search_scale =  np.linspace(0.8, 1.6, 5)
        search_degree = np.linspace(-2.5,2.5,11)
        num_cores = 1

        checkerImg, _, _, _, _, _ = getChecker(potCrp, cardBW, 
                                         search_scale, search_degree, num_cores)
        checkerImg = ndimage.rotate(checkerImg, check_rot)
            
        cardImg, cardLabel, colorTable = cardSampler(colorChecker, checkerImg, colrLable, posn, rows=4, cols=6)
        colorTableAll = pd.concat([colorTableAll, colorTable], axis=1)
        colorTableAll = colorTableAll.loc[:, ~colorTableAll.columns.duplicated()] 
        
        illum_img = illum_greyWorld(colorTable, colrLable, posn)
        # illum_fit, offset_fit = illum_fitting(cardImg, cardLabel, colrLable)
        illum_fit = illum_fitting(cardImg, cardLabel, colrLable)

        checkers.loc[checkers["position"] ==posn, colrLable] = illum_img
        checkers_fit.loc[checkers_fit["position"] ==posn, colrLable] = illum_fit

    # magClor = checkers_fit.loc[:, ['blu', 'grn', 'red']].sum(axis=1).values
    # checkers_fit.loc[:, "magClor"] = magClor
    
    # for cnt3 in colrLable:
    #     checkers_fit.loc[:, cnt3 + "_perc"] = checkers_fit.loc[:, cnt3] / checkers_fit.loc[:, "magClor"]
    
    
    
    checkers_fit = addXYZ(checkers_fit)
    return checkers, checkers_fit, colorTableAll

#%%
def illum_greyWorld(colorTable, colrLable, posn):
    BGR_labl = [sub + "_" + posn for sub in colrLable]
    greyPatch_img = colorTable.loc[colorTable['position'] >=19, BGR_labl]
    illum_img = greyPatch_img.mean().values
    
    return illum_img

#%%
def illum_fitting(imgColr, lablColr, colrLable):
    cardImg = imgColr.copy()
    cardLabel = lablColr.copy()
    illum = []
    
    for cnt in range(len(colrLable)):
        illuInitial = (1.0, 0.0)
        
        y0_card = cardLabel[:,:, cnt].flatten()
        y1_card = cardImg[:,:, cnt].flatten()
        
        indx = np.argsort(y0_card) 
        y0_card = y0_card[indx]
        y1_card = y1_card[indx]
        
        # Estimate the illuminant of the scene
        ErrorFunc = lambda tpl,y0,y1 : (tpl[0]/255) * y0 - y1
        l1Final, success = optimize.leastsq(ErrorFunc,illuInitial, args=(y0_card,y1_card), maxfev = 20000)
        illum.append(l1Final[0])
        
    return illum

#%%

def addXYZ(dfColor):
    df = dfColor.copy()
    
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]])
    
    for sq in range(len(df)):
        RGB_val = df.loc[sq, ['red', 'grn', 'blu']].values
        df.loc[sq, ['X', 'Y', 'Z']] = np.matmul(M, RGB_val)
        mag = df.loc[sq, ['X', 'Y', 'Z']].sum()
               
        if mag == 0: mag = eps
        
        df.loc[sq, ['x', 'y', 'z']] = df.loc[sq, ['X', 'Y', 'Z']].values/mag
        # df.loc[sq, ['X', 'Y', 'Z']] = cv2.cvtColor( np.uint8([[RGB_val]] ), cv2.COLOR_RGB2XYZ)[0][0]
        df.loc[sq, ['L', 'A', 'B']] = cv2.cvtColor( np.uint8([[RGB_val]] ), cv2.COLOR_RGB2LAB)[0][0]
    
    # plt.scatter(df.loc[:, 'x'], df.loc[:, 'y'])
    # plt.xlim(0, 0.8)
    # plt.ylim(0, 0.8)
    # plt.show()
    
    return df