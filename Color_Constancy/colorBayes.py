# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:37:58 2021

@author: Diloz
"""
import os
import sys
import cv2
import scipy
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import optimize
from scipy.stats import normaltest
import matplotlib.pylab as plt

from scipy.optimize import curve_fit

import imalgo
import plotting
import checkerCards
import segmentation

#%%
eps = np.finfo(float).eps
colrLable = ['blu', 'grn', 'red']
# lbl_illum = ['blu_L', 'grn_L', 'red_L']

expID, daySowing, cam = 'exp05', '2018-01-05-11-00', 'cam03' # Color Constancy
dirParent = "C:/Users/Diego Lozano/OneDrive - LA TROBE UNIVERSITY/PhD - Datasets"
# dirParent   = "C:/Users/17600112/OneDrive - LA TROBE UNIVERSITY/PhD - Datasets"
root            = os.path.join(dirParent, expID)
path_potTempl = os.path.join(root, 'potTempl.png')
potTemp = cv2.imread(path_potTempl)[:, :,0]

folder_trash    = os.path.join(dirParent, 'trash')

if not os.path.exists(folder_trash): os.makedirs(folder_trash) 

#%%  5. Estimate the illuminant on pots by interpolating the Macbeth cards
# This function estimates the illuminant on the centre of the pot using interpolation
def illum_interp(df_coor, df_Known, imBGR_src, imNum, illumLabl):
    delt_int = (imBGR_src.shape[0]/100)
    # Known illuminant at the colorCheckers
    df_data = df_Known.copy().dropna() 
    # df_data = df_data.drop(columns=['left', 'width', 'top', 'height', 'light'])
    df_data = df_data.loc[:, ['position', 'name', 'col_centre', 'row_centre', 'blu', 'grn', 'red']]
    df_data.sort_values(by=["row_centre"], inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    row_known1 = df_data.loc[:, "row_centre"].values
    col_known1 = df_data.loc[:, "col_centre"].values
    
    row_max = np.max(row_known1)
    row_min = np.min(row_known1)
    col_max = np.max(col_known1)
    col_min = np.min(col_known1)
    
     # Unknown illuminant at each pot centre coordinates
    df_All = df_coor.drop(columns=['left', 'width', 'top', 'height', 'light']).copy()
    df_All = df_All.drop(df_All[df_All.name == 'Checker'].index)
    df_All.loc[df_All["row_centre"] >= row_max,"row_centre"] = row_max - delt_int
    df_All.loc[df_All["row_centre"] <= row_min,"row_centre"] = row_min + delt_int
    df_All.loc[df_All["col_centre"] >= col_max, "col_centre"] = col_max - delt_int
    df_All.loc[df_All["col_centre"] <= col_min, "col_centre"] = col_min + delt_int
    # df_All.sort_values(by=["position"], inplace=True)
    df_All.sort_values(by=["row_centre"], inplace=True)
    df_All.reset_index(drop=True, inplace=True)
    row_All = df_All.loc[:, "row_centre"].values
    col_All = df_All.loc[:, "col_centre"].values
    
    print(df_All.describe())
    print(df_data.loc[0:5, :])
    # Zip coordinates to train Model interp1
    zipCoord1 = list(zip(row_known1, col_known1)) 
    
    # Double interpolation (NaN)
    # for cnt in range(len(illumLabl)):
    for cnt in range(len(illumLabl)):
        row_known2, col_known2, interp1, illum_known2 = [], [], [], []
        illum_chan = illumLabl[cnt]
        
        # First Interpolation: Using a linear interpolator
        illum_known1 = df_data.loc[:, illum_chan].values
        interp1 = scipy.interpolate.LinearNDInterpolator(zipCoord1, illum_known1)
        illum_estim1 = interp1(row_All, col_All)
        
        # Second Interpolation for misssing values: Nearest Interpolator
        indxEmpty = np.argwhere(np.isnan(illum_estim1)).flatten()

        if len(indxEmpty) > 0:
            indxNoEmpty = np.argwhere(~np.isnan(illum_estim1)).flatten()
            row_known2 = row_All[indxNoEmpty]
            col_known2 = col_All[indxNoEmpty]
            illum_known2 = illum_estim1[indxNoEmpty]
        
     	    # Zip coordinates to train Model interp2
            zipCoord2 = list(zip(row_known2, col_known2))
            interp2 = scipy.interpolate.NearestNDInterpolator(zipCoord2, illum_known2)
            illum_estim2 = interp2(row_All, col_All)
            df_All.loc[:, illum_chan] = illum_estim2

    df_All = pd.concat([df_All, df_data])
    df_All.reset_index(drop=True, inplace=True)
    


    return df_All
#%%
# Generate the illuminant prior surface
def illum_surface(df_illum, Siz_src, imNum, illumLabl):
    width = Siz_src[1]
    height = Siz_src[0]
    dim = (width, height)
    wth = int(width/10)
    hth = int(height /10)
    
    df = df_illum.copy()
    df.sort_values(by=["row_centre"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    row_point = df.loc[:, "row_centre"].values
    row_diff = np.append(row_point[0], (row_point[1:] - row_point[0:-1]))
    row_chang = row_point[row_diff>100]

    row_num = int(len(row_chang))
    col_num = int(round((len(row_point) / row_num)))
    
    im_aux = np.ones((row_num, col_num, 3), dtype=np.float32)
    bott = 0
    top = col_num - 1
    
    for row in range(row_num):
        df_aux = df.loc[bott:top, :].copy()
        bott = top + 1
        top+= col_num
        df_aux.sort_values(by=["col_centre"], inplace=True)
        df_aux.reset_index(drop=True, inplace=True)
        
        for col in range(len(df_aux)):
            im_aux[row, col, 0] = df_aux.loc[col, illumLabl[0]]
            im_aux[row, col, 1] = df_aux.loc[col, illumLabl[1]]
            im_aux[row, col, 2] = df_aux.loc[col, illumLabl[2]]
    
    im_ill_resiz =  cv2.resize(im_aux, dim, interpolation = cv2.INTER_AREA)

    # Smooth the Surface using a kernel size equal to 10%  of the image
    blur1 = cv2.blur(im_ill_resiz,(wth,wth))
    blur2 = cv2.blur(blur1,(101,101))
    prior_surf = np.clip(blur2, 0, 511)
    
    return prior_surf

#%% Correct an input image using the illuminant surface
def correctIllumGrey(img_src, illum):
    imgBGR = np.float32(img_src.copy())
    illumBGR = illum.copy()
    # Convert RGB illuminant values into sRBG, Values [0, 1]
    illumXYZ = np.float32(cv2.cvtColor(illumBGR, cv2.COLOR_BGR2XYZ))
    illumXYZ[illumXYZ[:, :, :]<=0] = eps
    
    # Normalize illuminant with respect of Y, where Y=1
    illumXYZ_norm = np.zeros_like(illumXYZ)
    illumXYZ_norm[:, :, 0] = np.divide(illumXYZ[:, :, 0], illumXYZ[:, :, 1])
    illumXYZ_norm[:, :, 1] = np.divide(illumXYZ[:, :, 1], illumXYZ[:, :, 1])
    illumXYZ_norm[:, :, 2] = np.divide(illumXYZ[:, :, 2], illumXYZ[:, :, 1])
    illumXYZ_norm[illumXYZ_norm[:, :, :]<=0] = eps
    
    illumBGR_aux = cv2.cvtColor(illumXYZ_norm * 255, cv2.COLOR_XYZ2BGR)
    illumBGR_aux[illumBGR_aux[:, :, :]<=0] = eps
    
    imgBGR[:, :, 0] = np.divide(imgBGR[:, :, 0], illumBGR_aux[:, :, 0])*255
    imgBGR[:, :, 1] = np.divide(imgBGR[:, :, 1], illumBGR_aux[:, :, 1])*255
    imgBGR[:, :, 2] = np.divide(imgBGR[:, :, 2], illumBGR_aux[:, :, 2])*255
    
    imgBGR = np.uint8(np.clip(np.round(imgBGR), 0, 255))
 
    return imgBGR
#%%
def correctIllumFit(img_src, illum):
    imgBGR = img_src.copy()
    imgBGR_aux1 = np.ones_like(np.float32(imgBGR))
    illumBGR = np.float32(illum.copy())
    illumBGR[illumBGR[:, :, :] == 0] = eps
    
    illumBGR_aux = np.ones_like(illumBGR)
    illumBGR_aux[:, :, 0] = np.divide(255, illumBGR[:, :, 0])
    illumBGR_aux[:, :, 1] = np.divide(255, illumBGR[:, :, 1])
    illumBGR_aux[:, :, 2] = np.divide(255, illumBGR[:, :, 2])

    imgBGR_aux1[:, :, 0] = np.multiply(imgBGR[:, :, 0], illumBGR_aux[:, :, 0])
    imgBGR_aux1[:, :, 1] = np.multiply(imgBGR[:, :, 1], illumBGR_aux[:, :, 1])
    imgBGR_aux1[:, :, 2] = np.multiply(imgBGR[:, :, 2], illumBGR_aux[:, :, 2])
    
    imgBGR[:, :, 0] =np.uint8(np.clip(np.round(imgBGR_aux1[:, :, 0]), 0, 255))
    imgBGR[:, :, 1] =np.uint8(np.clip(np.round(imgBGR_aux1[:, :, 1]), 0, 255))
    imgBGR[:, :, 2] =np.uint8(np.clip(np.round(imgBGR_aux1[:, :, 2]), 0, 255))
    
    return imgBGR 

#%% 6. Calculate the cloth Reflectance using Bayesian Inference
# Cloth reflectance Xc = Yc/Lc
def imgFeature(img_src, illum, coor_df, potTemp, imNum):
    illBGR = illum.copy()
    imgBGR = img_src.copy()
    df = coor_df.loc[coor_df["name"]!="Checker", :].copy()
    df.sort_values(by=["position"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df_ref = df.copy()
    df_ill = df.copy()
    df_pix = df.copy()
    
    sampl = np.random.randint(len(df), size=1)

    
    for cnt in range(len(df)):
        posn, name, top, left, wd, ht = df.loc[cnt, ['position', 'name', 
                                                  'top', 'left', 'width', 'height']].values
        
        if name == "Checker": continue
    
        bottom = top + ht
        right = left+ wd
        off = 10
        potBGRCrp = imgBGR[top - off  : bottom + off, left - off : right + off]
        search_scale =  np.linspace(1.00, 1.0040, 10)
        search_degree = np.linspace(-2.5,2.5,15)
        num_cores = 2

        potImgAdj, startY, endY, startX, endX, deg = checkerCards.getChecker(potBGRCrp, potTemp, 
                                                                      search_scale, search_degree, num_cores)
               
        off2 = 30
        potImg = potImgAdj[off2:-off2, off2:-off2]
        
        potMask, _  = segmentation.segment(potImg, 0.15, posn)
        kernel = np.ones((3,3),np.uint8)
        potMask = cv2.erode(potMask, kernel, iterations=7)
        potSeg = cv2.bitwise_and(potImg, potImg, mask = potMask)
        pixel_leaf = potImg[np.where(potMask >0)]
        
        if cnt == sampl:
            plotting.imT(potImg, potSeg, imNum + '_' + posn, 'potSeg')

        potIllCrp = illBGR[top - off  : bottom + off, left - off : right + off]
        potIllAdj = potIllCrp[startY:endY,startX:endX,:]
        potIll = potIllAdj[off2:-off2, off2:-off2]
        illum_leaf = potIll[np.where(potMask >0)]
    

        for cnt2 in range(len(colrLable)):
            chan = colrLable[cnt2]
            
            refl_leaf = pixel_leaf[:, cnt2]/illum_leaf[:, cnt2]

            hist_pix, bins_pix = np.histogram(pixel_leaf[:, cnt2], bins=np.arange(256))
            hist_ill, bins_ill = np.histogram(illum_leaf[:, cnt2], bins=np.arange(512))
            hist_ref, bins_ref = np.histogram(refl_leaf, bins=np.arange(256)/256)
            
            
            pixFit, _, pixMean, pixStd = gausfit(hist_pix, bins_pix[:-1])
            refFit, _, refMean, refStd = gausfit(hist_ref, bins_ref[:-1])
            illMean = np.mean(illum_leaf[:, cnt2])
            
            df_pix.loc[cnt, chan] = pixMean
            df_ill.loc[cnt, chan] = illMean
            df_ref.loc[cnt, chan] = refMean

            if cnt == sampl:
                plt.plot(bins_pix[:-1], hist_pix, label = chan + "_pix" )
                plt.plot(bins_pix[:-1], pixFit, label= chan + "_fit" + "_" + str(np.around(pixMean, decimals =3)))
                plt.legend()
                plt.title(imNum + '_' + posn)
                plt.show()

          
    df_pix.sort_values(by=["name", "position"], inplace=True)
    df_ill.sort_values(by=["name", "position"], inplace=True)
    df_ref.sort_values(by=["name", "position"], inplace=True)
    
    df_pix = checkerCards.addXYZ(df_pix)
    df_ill = checkerCards.addXYZ(df_ill)
    df_ref = checkerCards.addXYZ(df_ref)
                                 
    df_pix.reset_index(drop=True, inplace=True)
    df_ill.reset_index(drop=True, inplace=True)
    df_ref.reset_index(drop=True, inplace=True)
    
    return df_pix, df_ill, df_ref

#%%
def corrIllum(img_src, illum):
    imgBGR = img_src.copy()
    imgBGR_aux1 = np.ones_like(np.float32(imgBGR))
    illumBGR = np.float32(illum.copy())

    
    illumBGR_aux = np.ones_like(illumBGR)
    illumBGR_aux[0] = np.divide(255, illumBGR[0])
    illumBGR_aux[1] = np.divide(255, illumBGR[1])
    illumBGR_aux[2] = np.divide(255, illumBGR[2])

    imgBGR_aux1[:, :, 0] = np.multiply(imgBGR[:, :, 0], illumBGR_aux[0])
    imgBGR_aux1[:, :, 1] = np.multiply(imgBGR[:, :, 1], illumBGR_aux[1])
    imgBGR_aux1[:, :, 2] = np.multiply(imgBGR[:, :, 2], illumBGR_aux[2])
    
    imgBGR[:, :, 0] =np.uint8(np.clip(np.round(imgBGR_aux1[:, :, 0]), 0, 255))
    imgBGR[:, :, 1] =np.uint8(np.clip(np.round(imgBGR_aux1[:, :, 1]), 0, 255))
    imgBGR[:, :, 2] =np.uint8(np.clip(np.round(imgBGR_aux1[:, :, 2]), 0, 255))
    
    return imgBGR 
    
#%%

def gausfit(chan, bins):
    hist = np.copy(chan)
    mids = 0.5*(bins[1:] + bins[:-1])
    mu = np.mean(mids)  
    sigma = np.std(mids)
    
    try:   
        popt,pcov = curve_fit(gaussian, bins, hist, p0=[1, mu, sigma])
        
    except RuntimeError:
        popt = [1, mu, sigma]
        print('error')
    
    if popt[1] > (mu + 2*sigma):
        popt = [1, mu, sigma]
        print('error2')
    
    fit = gaussian(bins,*popt)
    
    return  fit, *popt

#%%
def gaussian(x, k, mu,sigma):
    return k*np.exp(-0.5*((x-mu)**2/(sigma**2)))
# def gaussian(x,a,x0,sigma):
#     return a*np.exp(-(x-x0)**2/(2*sigma**2))

#%%
#%%  5. Estimate the illuminant on pots by interpolating the Macbeth cards
# This function estimates the illuminant on the centre of the pot using interpolation
def pixel_interp(df_coor, df_Known, imBGR_src, imNum, illumLabl):
    delt_int = (imBGR_src.shape[0]/100)
    
    df_All = df_coor.drop(columns=['left', 'width', 'top', 'height', 'light']).copy()
    # Known illuminant at the colorCheckers
    df_data = df_Known.copy().dropna() 
    df_data = df_data.loc[:, ['position', 'name', 'col_centre', 'row_centre', 'blu', 'grn', 'red']]
    df_data.sort_values(by=["row_centre"], inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    row_known1 = df_data.loc[:, "row_centre"].values
    col_known1 = df_data.loc[:, "col_centre"].values
    
    row_max = np.max(row_known1)
    row_min = np.min(row_known1)
    col_max = np.max(col_known1)
    col_min = np.min(col_known1)
    
     # Unknown illuminant at each pot centre coordinates
    posn_all = df_All.loc[:, "position"].tolist()
    posn_data = df_data.loc[:, "position"].tolist()
    df_unk = pd.DataFrame(columns = df_All.columns.tolist())
    
    cntAux = 0
    for cnt in range(len(df_All.loc[:, "position"])):
        x = posn_all[cnt]
        if not x in posn_data:
            df_unk.loc[cntAux, :] = df_All.loc[cnt, :]
            cntAux += 1 

    df_unk.loc[df_unk["row_centre"] >= row_max,"row_centre"] = row_max - delt_int
    df_unk.loc[df_unk["row_centre"] <= row_min,"row_centre"] = row_min + delt_int
    df_unk.loc[df_unk["col_centre"] >= col_max, "col_centre"] = col_max - delt_int
    df_unk.loc[df_unk["col_centre"] <= col_min, "col_centre"] = col_min + delt_int
    df_unk.sort_values(by=["row_centre"], inplace=True)
    df_unk.reset_index(drop=True, inplace=True)

    row_unk = df_unk.loc[:, "row_centre"].values
    col_unk = df_unk.loc[:, "col_centre"].values

    # Zip coordinates to train Model interp1
    zipCoord1 = list(zip(row_known1, col_known1)) 
    
    # Double interpolation (NaN)
    for cnt in range(len(illumLabl)):
        row_known2, col_known2, interp1, illum_known2 = [], [], [], []
        illum_chan = illumLabl[cnt]
        
        # First Interpolation: Using a linear interpolator
        illum_known1 = df_data.loc[:, illum_chan].values
        interp1 = scipy.interpolate.LinearNDInterpolator(zipCoord1, illum_known1)
        illum_estim1 = interp1(row_unk, col_unk)
        
        df_unk.loc[:, illum_chan] = illum_estim1
        
        # Second Interpolation for misssing values: Nearest Interpolator
        indxEmpty = np.argwhere(np.isnan(illum_estim1)).flatten()

        if len(indxEmpty) > 0:
            indxNoEmpty = np.argwhere(~np.isnan(illum_estim1)).flatten()
            row_known2 = row_unk[indxNoEmpty]
            col_known2 = col_unk[indxNoEmpty]
            illum_known2 = illum_estim1[indxNoEmpty]
        
     	    # Zip coordinates to train Model interp2
            zipCoord2 = list(zip(row_known2, col_known2))
            interp2 = scipy.interpolate.NearestNDInterpolator(zipCoord2, illum_known2)
            illum_estim2 = interp2(row_unk, col_unk)
            df_unk.loc[:, illum_chan] = illum_estim2

    df_out = pd.concat([df_unk, df_data])
    df_out.reset_index(drop=True, inplace=True)
    
    return df_out
    
#%%  5. Estimate the illuminant on pots by interpolating the Macbeth cards
# This function estimates the illuminant on the centre of the pot using interpolation
def interp(df_coor, df_Known, imBGR_src, imNum, illumLabl):
    delt_int = (imBGR_src.shape[0]/100)
    
    df_All = df_coor.drop(columns=['left', 'width', 'top', 'height', 'light']).copy()
    # Known illuminant at the colorCheckers
    df_data = df_Known.copy().dropna() 
    df_data = df_data.loc[:, ['position', 'name', 'col_centre', 'row_centre', 'blu', 'grn', 'red']]
    df_data.sort_values(by=["row_centre"], inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    row_known1 = df_data.loc[:, "row_centre"].values
    col_known1 = df_data.loc[:, "col_centre"].values
    
    row_max = np.max(row_known1)
    row_min = np.min(row_known1)
    col_max = np.max(col_known1)
    col_min = np.min(col_known1)
    
     # Unknown illuminant at each pot centre coordinates
    posn_all = df_All.loc[:, "position"].tolist()
    posn_data = df_data.loc[:, "position"].tolist()
    df_unk = pd.DataFrame(columns = df_All.columns.tolist())
    
    cntAux = 0
    for cnt in range(len(df_All.loc[:, "position"])):
        x = posn_all[cnt]
        if not x in posn_data:
            df_unk.loc[cntAux, :] = df_All.loc[cnt, :]
            cntAux += 1 

    df_unk.loc[df_unk["row_centre"] >= row_max,"row_centre"] = row_max - delt_int
    df_unk.loc[df_unk["row_centre"] <= row_min,"row_centre"] = row_min + delt_int
    df_unk.loc[df_unk["col_centre"] >= col_max, "col_centre"] = col_max - delt_int
    df_unk.loc[df_unk["col_centre"] <= col_min, "col_centre"] = col_min + delt_int
    df_unk.sort_values(by=["row_centre"], inplace=True)
    df_unk.reset_index(drop=True, inplace=True)

    row_unk = df_unk.loc[:, "row_centre"].values
    col_unk = df_unk.loc[:, "col_centre"].values

    # Zip coordinates to train Model interp1
    zipCoord1 = list(zip(row_known1, col_known1)) 
    
    # Double interpolation (NaN)
    for cnt in range(len(illumLabl)):
        row_known2, col_known2, interp1, illum_known2 = [], [], [], []
        illum_chan = illumLabl[cnt]
        
        # First Interpolation: Using a linear interpolator
        illum_known1 = df_data.loc[:, illum_chan].values
        interp1 = scipy.interpolate.LinearNDInterpolator(zipCoord1, illum_known1)
        illum_estim1 = interp1(row_unk, col_unk)
        
        df_unk.loc[:, illum_chan] = illum_estim1
        
        # Second Interpolation for misssing values: Nearest Interpolator
        indxEmpty = np.argwhere(np.isnan(illum_estim1)).flatten()

        if len(indxEmpty) > 0:
            indxNoEmpty = np.argwhere(~np.isnan(illum_estim1)).flatten()
            row_known2 = row_unk[indxNoEmpty]
            col_known2 = col_unk[indxNoEmpty]
            illum_known2 = illum_estim1[indxNoEmpty]
        
     	    # Zip coordinates to train Model interp2
            zipCoord2 = list(zip(row_known2, col_known2))
            interp2 = scipy.interpolate.NearestNDInterpolator(zipCoord2, illum_known2)
            illum_estim2 = interp2(row_unk, col_unk)
            df_unk.loc[:, illum_chan] = illum_estim2

    df_out = pd.concat([df_unk, df_data])
    df_out.reset_index(drop=True, inplace=True)
    
    return df_out
    
#%%
# Generate the illuminant prior surface
def pixel_surface(df_illum, Siz_src, imNum, illumLabl):
    width = Siz_src[1]
    height = Siz_src[0]
    dim = (width, height)
    wth = int(width/10)
    hth = int(height /10)
    
    df = df_illum.copy()
    df.sort_values(by=["row_centre"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    row_point = df.loc[:, "row_centre"].values
    row_diff = np.append(row_point[0], (row_point[1:] - row_point[0:-1]))
    row_chang = row_point[row_diff>100]

    row_num = int(len(row_chang))
    col_num = int(round((len(row_point) / row_num)))
    
    im_aux = np.ones((row_num, col_num, 3), dtype=np.float32)
    bott = 0
    top = col_num - 1
    
    for row in range(row_num):
        df_aux = df.loc[bott:top, :].copy()
        bott = top + 1
        top+= col_num
        df_aux.sort_values(by=["col_centre"], inplace=True)
        df_aux.reset_index(drop=True, inplace=True)
        
        for col in range(len(df_aux)):
            im_aux[row, col, 0] = df_aux.loc[col, illumLabl[0]]
            im_aux[row, col, 1] = df_aux.loc[col, illumLabl[1]]
            im_aux[row, col, 2] = df_aux.loc[col, illumLabl[2]]
    
    im_ill_resiz =  cv2.resize(im_aux, dim, interpolation = cv2.INTER_AREA)

    # Smooth the Surface using a kernel size equal to 10%  of the image
    blur1 = cv2.blur(im_ill_resiz,(wth,wth))
    blur2 = cv2.blur(blur1,(101,101))
    # prior_surf = np.clip(blur2, 0, 511)
    
    return blur2

#
#%%
def correctPixelOff(img_src, illum):
    imgBGR = img_src.copy()
    imgBGR_aux1 = np.ones_like(np.float32(imgBGR))
    illumBGR = np.float32(illum.copy())
    illumBGR[illumBGR[:, :, :] == 0] = eps

    imgBGR_aux1[:, :, 0] = np.subtract(imgBGR[:, :, 0], illumBGR[:, :, 0])
    imgBGR_aux1[:, :, 1] = np.subtract(imgBGR[:, :, 1], illumBGR[:, :, 1])
    imgBGR_aux1[:, :, 2] = np.subtract(imgBGR[:, :, 2], illumBGR[:, :, 2])

    
    imgBGR[:, :, 0] =np.uint8(np.clip(np.round(imgBGR_aux1[:, :, 0]), 0, 255))
    imgBGR[:, :, 1] =np.uint8(np.clip(np.round(imgBGR_aux1[:, :, 1]), 0, 255))
    imgBGR[:, :, 2] =np.uint8(np.clip(np.round(imgBGR_aux1[:, :, 2]), 0, 255))
    
    return imgBGR

#%%
def calibration(imScr, illumChart0, colorTable0):
    imBGR = imScr.copy()
    illumChart = illumChart0.copy()
    colorTable = colorTable0.copy()
    checkers = illumChart.loc[:, "position"].tolist()
    
    deltaName = ["delta_" + sub for sub in checkers]
    
    deltaSum = colorTable.loc[:, deltaName].sum(axis = 0)
    deltaIndx = deltaSum.index.tolist()
    deltaMin = deltaIndx[deltaSum.argmin()][6:]
    
    alpha = 255/illumChart.loc[illumChart["position"]==deltaMin, ["blu", "grn", "red"]].to_numpy()[0]
    
    imOut = correct(imBGR, alpha)
    
    return imOut

#%%
def correct(img, alpha, beta=[]):
    res = np.zeros_like(img)
    imgFloat = np.float32(img)
    
    res[:, :, 0] = np.uint8(np.int32(np.clip(imgFloat[:,:, 0] * alpha[0], 0, 255)))
    res[:, :, 1] = np.uint8(np.int32(np.clip(imgFloat[:,:, 1] * alpha[1], 0, 255)))
    res[:, :, 2] = np.uint8(np.int32(np.clip(imgFloat[:,:, 2] * alpha[2], 0, 255)))
        
    return res 

#%%
def correct_bayes(bgrSrc, colorChecker, coor_df, checker_angle, priorIll, imgNumb, colrLable):
    BGR_src = bgrSrc.copy()
    sizSrc = BGR_src.shape
       
    illumSurf = illum_surface(priorIll, sizSrc, imgNumb, colrLable)

    BGR_aux = correctIllumFit(BGR_src, illumSurf)
    _, illumChart1, colorTable1 = checkerCards.illum_check(colorChecker, coor_df, 
                                                        BGR_aux, checker_angle, 
                                                        colrLable, "original")
    BGR_bayes = calibration(BGR_aux, illumChart1, colorTable1) 
    
    plotting.imT(bgrSrc, BGR_bayes,"bgrSrc", "BGR_bayes")
  
    return BGR_bayes
