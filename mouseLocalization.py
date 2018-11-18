#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:18:28 2018

@author: yangx
"""

import os
import glob
import numpy as np
import pandas as pd
import csv
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from scipy import io
import time
from mouseTouch import dataProcessing
from skimage.morphology import skeletonize as skn

matPath = '/data/up+down.mat'
d1 = io.loadmat(matPath)
up = d1['up']
down = d1['down']
regr = dataProcessing.randomForest4up(up,down,100)
def batchfnc():
    import tkinter.filedialog
    defaultFolder = '/media/yangx/YXHD-01/data_shared/hangryvideo/'
    currentpath = os.getcwd()
    os.chdir(defaultFolder)
    folderPath = tkinter.filedialog.askdirectory()
    vidList = glob.glob(os.path.join(folderPath,'*.mp4'))
    vidList.sort()
#    time.sleep(4200)
    for vidPath in vidList:
        if os.path.isdir(folderPath):
            mouseLocalization(vidPath)
    os.chdir(currentpath)
    return(0) 
    
def mouseLocalization(vidPath, vidShow=True):
    fcnBegin = time.time()
    folderPath = os.path.split(vidPath)[0]
    filename = os.path.split(vidPath)[1]
    filename = os.path.splitext(filename)[0]
    if filename[:3]=='ViV':
        filename = filename[3:-6]
    mat2write = filename+'-rect'
    matpath = os.path.join(folderPath,mat2write)
#    txt2write = filename+'-imr.txt'
#    txtPath= os.path.join(folderPath,txt2write)
    csv2write = filename+'-imr.csv'
    csvPath= os.path.join(folderPath,csv2write)
    
    background = backgroundCaculation(vidPath)
    grayBack = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    vid = cv2.VideoCapture(vidPath)
    nFrames = np.int0(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = np.int0(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = np.int0(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = np.int0(vid.get(cv2.CAP_PROP_FPS))

    video2write = filename+'-imr.mp4'
    vidpath = os.path.join(folderPath,video2write)
    if os.path.exists(vidpath):
        os.remove(vidpath)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vidpath,fourcc, fps, (100,100), isColor=0)
    
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    MouseThreshold = 8
    
    darkBack = grayBack.copy()
    cv2.floodFill(darkBack,None,(50,270),0,[2,2,2],[2,2,2])
    cv2.floodFill(darkBack,None,(900,270),0,[1,1,1],[1,1,1])
    darkBack = cv2.morphologyEx(darkBack,cv2.MORPH_CLOSE,kernel,iterations=3)
    darkBack = 255*np.uint8(darkBack<MouseThreshold)
    cv2.floodFill(darkBack,None,(480,270),128)
    roi = 255*np.uint8(darkBack==128)
    plt.imshow(roi)
    
    l,t,w,h = np.int(width/3),np.int(height/3),np.int(width/3),np.int(height/3)  # simply hardcoded the values
    trackWindow = (l,t,w,h)
    centerBack = (480,270)
#    rectBack = ((0,0),(0,0),0)
    
    ispause = False
    rectArray = -np.ones([nFrames,9],dtype='float32')
    begin = True
#    ind = 0
#    imrMat = np.zeros([100,100,100])
    theta = np.pi*np.arange(8)/4
    shift = 10
    xshift = np.int0(shift*np.cos(theta))
    yshift = np.int0(shift*np.sin(theta))
    fsumArray = np.zeros([nFrames,3])
#    f = open(txtPath,'w')
    f = open(csvPath,'w',newline='')
    writer = csv.writer(f)
#    fgbg = cv2.createBackgroundSubtractorMOG2(10*fps,8,False)
                
    for iloop in range(nFrames-120*fps):
        ret, frameMat = vid.read()
        if (ret == True):
#            fgmask = fgbg.apply(frameMat)
            grayFrame = cv2.cvtColor(frameMat, cv2.COLOR_BGR2GRAY)
            grayDiff = np.int0(grayBack) - np.int0(grayFrame)
            grayDiff[grayDiff<0] = 0
            mouseDark = 255 * np.uint8((grayFrame<MouseThreshold)&(grayDiff>MouseThreshold))
#            mouseDark = cv2.morphologyEx(mouseDark,cv2.MORPH_OPEN,kernel,iterations=3) 
            mouseDark = mouseDark&roi
            
            md, contours, hierarchy = cv2.findContours(mouseDark,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)>0:
                contour = contours[0]
                area = cv2.contourArea(contour)
                for cnt in contours:
                    areaNow = cv2.contourArea(cnt)
                    if areaNow>area:
                        area = areaNow
                        contour = cnt
            else:
                continue
#            mouseRect = cv2.minAreaRect(contour) 
            mouseDark = np.zeros([height,width],dtype='uint8')
            cv2.drawContours(mouseDark,[contour],0,255,cv2.FILLED)

            mo = cv2.moments(contour)
            cx = mo['m10']/mo['m00']
            cy = mo['m01']/mo['m00']
            centerPoint = (np.int0(cx),np.int0(cy))  
           
#            mouseRect, trackWindow = cv2.CamShift(mouseDark, trackWindow, term_crit) 
            
#            print(mouseRect)
#            if not roi[centerPoint[1],centerPoint[0]]==255:
#                centerPoint = centerBack
            
            mouseDark = cv2.dilate(mouseDark,kernel,iterations=3)
            
            skeleton = 255 * np.uint8(skn(mouseDark/255))
            mouseDst = cv2.distanceTransform(mouseDark,1,5)
            mouseSkeleton = np.uint8(skeleton * mouseDst / mouseDst.max())
            retVal,mouseSkeleton = cv2.threshold(mouseSkeleton,180,255,cv2.THRESH_BINARY)

            image,mouseContours,hierarchy = cv2.findContours(mouseSkeleton, 1, 2)
#            mouseContours = np.vstack(mouseContours)
#            mouseRect = cv2.minAreaRect(mouseContours)
            
            
            frame = frameMat.copy()
#            cv2.floodFill(frame,fmask,centerPoint,[0,255,0])
            fmaskMat = 255*np.ones([height+2,width+2],dtype='uint8')
            fmaskMat[1:-1,1:-1] = 255-mouseDark
            fmask = fmaskMat.copy()
            cv2.floodFill(frame,fmask,centerPoint,[0,255,0],[2,2,2],[2,2,2])
            farea = (frame[...,0]==0) & (frame[...,1]==255) & (frame[...,2]==0)
            fsum = farea.sum()         

            cv2.polylines(frame, mouseContours, True, 255, 2)
#            fsum = (255-fmask).sum()/255
#            print(fsum)

            if fsum<500 or fsum>10000:
                print(fsum)
#                fsum,fmask = fmaskCheck(frameMat,centerPoint)

                for jloop in range(8):
                    frame = frameMat.copy()
                    fmask = fmaskMat.copy()
                    seedPoint = (centerPoint[0]+xshift[jloop],
                                 centerPoint[1]+yshift[jloop])
                    if seedPoint[0]<0 or seedPoint[0]>960 or seedPoint[1]<0 or seedPoint[1]>540:
                        continue
                    if not roi[seedPoint[1],seedPoint[0]]==255:
                        continue
                    cv2.floodFill(frame,fmask,seedPoint,[0,255,0],[5,5,5],[5,5,5])
                    farea = (frame[...,0]==0) & (frame[...,1]==255) & (frame[...,2]==0)
                    fsum = farea.sum() 
#                    fsum = (255-fmask).sum()/255
                    if fsum>500 and fsum<10000:
                        break
                print('fmask sum is:%d'%fsum)               
            
            if fsum<500 or fsum>10000:
                fmask = mouseDark
            else:
#                fmask = fmask[1:-1,1:-1]
#                fmask = 255-fmask
                fmask = 255*np.uint8(farea)

            fmask = cv2.dilate(fmask,kernel,iterations=3)
            
            fsumArray[iloop,0] = centerPoint[0]
            fsumArray[iloop,1] = centerPoint[1]
            fsumArray[iloop,2] = fmask.sum()/255
                        
            if iloop%(60*fps)==0:
                print('processed %d frame'%iloop)
#            if iloop%(15*fps)==0:
#                imr = rotation((255-grayFrame)*fmask,centerPoint,mouseRect[2])
#                (h, w) = imr.shape[:2]
#                imr = imr[np.int0(h/2)-50:np.int0(h/2)+50,np.int0(w/2)-50:np.int0(w/2)+50]
#                imrMat[:,:,ind] = imr
#                fig2write = 'imr'+'%02d'%ind+'.jpg'
#                figpath = os.path.join(origPath,fig2write)
#                cv2.imwrite(figpath,imr)
#                ind+=1                       
#            if ind==100:
#                break

            mouseRect, trackWindow = cv2.CamShift(fmask, trackWindow, term_crit)
            rectArray[iloop,0] = mouseRect[0][0]
            rectArray[iloop,1] = mouseRect[0][1]
            rectArray[iloop,2] = mouseRect[1][0]
            rectArray[iloop,3] = mouseRect[1][1]
            rectArray[iloop,4] = mouseRect[2]
            
            imr = Image.fromarray((255-grayFrame)&fmask)
            imr = imr.rotate(mouseRect[2],center=centerPoint)
            imr = imr.crop([centerPoint[0]-50,centerPoint[1]-50,centerPoint[0]+50,centerPoint[1]+50])
            imr = np.asarray(imr)
#            imr = rotation((255-grayFrame)&fmask,centerPoint,mouseRect[2])
#            (h, w) = imr.shape[:2]
#            imr = imr[np.int0(h/2)-50:np.int0(h/2)+50,np.int0(w/2)-50:np.int0(w/2)+50]
            if regr.predict(imr.reshape([1,-1]))<0.5:
                rectArray[iloop,4] += 180
                rectArray[iloop,4] %= 360
                imr = imr[::-1,::-1]
                
            imd = dataProcessing.waveletDecomposition(imr)
            imd = imd.flatten()
#            for jloop in range(len(imd)-1):
#                f.write('%.4f'%imd[jloop]+' ')
#            f.write('%f'%imd[-1]+'\n')
            writer.writerow(tuple(imd))
#            imd = imd.reshape((1,-1))
#            imd = pd.DataFrame(imd)
#            imd.to_csv(csvPath,mode='a',header=0,index=0)
            
#            if begin:
#                imdmat = imd
#                begin = False
#            else:
#                imdmat = np.concatenate((imdmat,imd))
                
                                       
#            centerBack = centerPoint 

#            mousePts = np.int0(cv2.boxPoints(mouseRect))  
#            mouseBox = cv2.polylines(frameMat, [mousePts], True, 255, 2)


            if vidShow:         
                cv2.imshow('FrameMat', frame)
#                cv2.imshow('imr',imr)
        
                k = cv2.waitKey(1)#-1048576
                if k==ord(' ') and not ispause:
                    ispause = True;
                    while True:
                        k = cv2.waitKey(5)#-1048576
                        if k==ord(' ') and ispause:
                            ispause = False
                            break
                elif k==ord('\x1b'): # 'Esc'
                    break
                # 在播放视频时，按esc键，可强制退出
#            else:
#                out.write(imr)
            out.write(imr)
        else:
            break
    cv2.destroyAllWindows()
    # 删除建立的全部窗口
    vid.release()
    out.release()
#    np.save(npyPath,imrMat)
#    imdmatr = dataProcessing.dimensionReduction(imdmat,10)
#    plt.plot(imdmatr)
#    np.savetxt(txtpath,rectArray,fmt='%.4f',newline='\n')
#    mdict = {'rectArray':rectArray,'fsumArray':fsumArray,'imdmatr':imdmatr}
#    mdict = {'rectArray':rectArray,'fsumArray':fsumArray,'imdmat':imdmat}
    f.close()
#    io.savemat(matpath,mdict)
    fcnEnd = time.time()
    telapse = (fcnEnd-fcnBegin)/60
    print('time elapse is %.1f min\n' %telapse)
    return(rectArray,fsumArray)

def backgroundCaculation(vidPath):   
    vid = cv2.VideoCapture(vidPath)
    nFrames = np.int0(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = np.int0(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = np.int0(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = np.int0(vid.get(cv2.CAP_PROP_FPS))
    background = np.zeros([height,width,3],dtype='float32')
    for iloop in range(nFrames):
        ret, frameMat = vid.read()
        if iloop<fps*60:
            continue
        if iloop>=fps*120:
            break
        background = background + frameMat
    background = np.uint8(background/(60*fps))
    vid.release()
    return(background)

def rotation(img,center,angle,scale=1):
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D(center,angle,scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    imr = cv2.warpAffine(img, M, (nW, nH))
    return(imr)
    
def makeImgSet(imgPath):
    dataPath = os.path.join(imgPath,'orig.npy')
    rotPath = os.path.join(imgPath,'rotation')
    disPath = os.path.join(imgPath,'discard')
    img = np.load(dataPath)
    downset = list(np.int0(np.loadtxt(rotPath)))
    disset = list(np.loadtxt(disPath))
    totalset = set(list(np.arange(100)))
    upset = list(totalset-set(downset)-set(disset))
#    upset = list(totalset-set(downset))
    up0 = img[...,upset]
    down0 = img[...,downset]
    up = np.concatenate((up0,down0[::-1,:]),axis=2)
    up = np.concatenate((up,up[:,::-1,:]),axis=2)
    down = np.concatenate((down0,up0[::-1,:]),axis=2)
    down = np.concatenate((down,down[:,::-1,:]),axis=2)
    return(up,down)

def saveImgSet(up,down):
    upPath='/media/yangx/YXHD-01/data_shared/miceImage/up'
    downPath='/media/yangx/YXHD-01/data_shared/miceImage/down'
    npyPath = os.path.join(upPath,'up.npy')
    np.save(npyPath,up)
    npyPath = os.path.join(downPath,'down.npy')
    np.save(npyPath,down)
    (m,n,num) = up.shape
    for iloop in range(num):
        fig2write = 'up'+'%03d'%iloop+'.jpg'
        figpath = os.path.join(upPath,fig2write)
        cv2.imwrite(figpath,up[...,iloop])
        fig2write = 'down'+'%03d'%iloop+'.jpg'
        figpath = os.path.join(downPath,fig2write)
        cv2.imwrite(figpath,down[...,iloop])
    return