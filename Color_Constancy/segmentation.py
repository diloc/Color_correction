# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:10:37 2020

@author: DLozano
"""
import numpy as np
import cv2
import imalgo
import sys
import plotting
import matplotlib.pylab as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

eps = np.finfo(float).eps

#%%

def segment(pot, holeFact, posn):
    maskSeg1, potSeg1 = shapeanalysis(np.copy(pot), 0.0005)
    maskSeg2, potSeg2 = histoanalysis(np.copy(pot), 0.00, holeFact)
    
    green1 = 2.0* potSeg1[:, :, 1] - 1.0* potSeg1[:, :, 0] - 1.0* potSeg1[:, :, 2]
    green2 = 2.0* potSeg2[:, :, 1] - 1.0* potSeg2[:, :, 0] - 1.0* potSeg2[:, :, 2]
    
    area1 = len(green1[green1>30])
    area2 = len(green2[green2>0])
    
    if area1 == maskSeg1.shape[0] * maskSeg1.shape[1]: area1 = 0
    if area2 == maskSeg2.shape[0] * maskSeg2.shape[1]: area2 = 0

    if (area1 > area2):
        maskSeg = maskSeg1
        # print('area1 > area2', area1, area2)
    else:
        maskSeg = maskSeg2
        # print('area1 < area2', area1, area2)
    
    
    # if np.allclose(pot[:, :, 0], potSeg[:, :, 0]): 
    #     print('problem')
    
    potOut = cv2.bitwise_and(pot, pot, mask = maskSeg)
    
    # plotting.imT(pot, potOut, 'pot.','potOut ' +posn)
    return maskSeg, potOut
    


def histoanalysis(pot, offP, holeFact):
    row, col, _ = np.shape(pot)
    # hole = row * col * holeFact
    kernel = np.ones((3,3),np.uint8)
    krnl = np.int(np.mean([row, col]) * 0.010)
    
    if krnl % 2 == 0 : krnl = krnl + 1
    
    potAux = np.copy(pot) 
    
    gaus = cv2.GaussianBlur(np.copy(potAux), (3, 3), 0)   
    green = 2.0 * gaus[:,:,1].astype(float) - gaus[:,:,0].astype(float) - gaus[:,:,2].astype(float)  
    green =  np.uint8(np.clip(green, 0, 255))
    green[green <= 15] = 0
    
    _,otsu2 = imalgo.otsu(np.copy(green), potAux)  
    
    # roi2 = imalgo.roiFilter(otsu2, hole, 0.26, True)
    dilate2 = cv2.dilate(np.copy(otsu2),kernel,iterations = 1)
    maskP, _, _, _ = imalgo.spcFilter(dilate2, [], holeFact)

    pot6 = cv2.bitwise_and(potAux, potAux, mask = maskP)
    # plotting.im3(pot, green, pot6, 'pot', 'green', 'pot6 Histo')
    return maskP, pot6

def shapeanalysis(pot,offP):
    row, col = pot.shape[0:2]
    kernel = np.ones((3,3),np.uint8)
    
    off = np.int(np.mean([row, col])*offP)
    potAux = pot[off:row - off, off:col - off] 
    
    mask = imalgo.grosSegm(potAux, 21)
    pot4  = cv2.bitwise_and(potAux,potAux, mask= mask)
    
    H, S, V = cv2.split(cv2.cvtColor(pot4, cv2.COLOR_BGR2HSV))
    
    H[np.where(H > 100)] = 0
    H[np.where(H < 10)] = 0
    H[(np.where((H>=10) & (H <= 100 )))] = 255
    pot5  = cv2.bitwise_and(potAux,potAux, mask= H)
    
    sMask2 = H


    src = np.copy(pot5)
    areaNum = 10
    area = len(H[H>0])
    if area > 0:
        iterations = int(-10 * np.log(area /(row*col))) + 15
        
        for cntC in range(iterations):
        # for cntC in range(int(expDay) * (-4) + 140):
            tHSV2,mskHSV2 = [], []
            sMask1, perAll1, xyMeanAll1, areaAll1 = [], [], [], []
            masked3 = []
            tHSV2,mskHSV2 = imalgo.thrHSV(src,[5,  3,  3] , [150, 255, 255])
            sMask1, perAll1, xyMeanAll1, areaAll1 = imalgo.spcFilter(mskHSV2, [], 0.05)
            
            if areaAll1 == 0: break
            if len(areaAll1) == 2 and cntC == 0: flag = 'two plants'
            if len(areaAll1)> areaNum: break
            
            areaNum = len(areaAll1)
            masked3 = imalgo.gamma(cv2.bitwise_and(tHSV2,tHSV2, mask= sMask1), 0.9995)
            src = masked3
            sMask2, perAll2, xyMeanAll2, areaAll2 = sMask1, perAll1, xyMeanAll1, areaAll1
    
        potSeg = cv2.bitwise_and(potAux, potAux, mask = sMask2)
    
    else:
        sMask2 = np.zeros_like(H)
        potSeg = np.zeros_like(potAux)
    

    return sMask2, potSeg

