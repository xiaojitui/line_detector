#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import os
import numpy as np
import copy
from scipy.stats import mode


# In[ ]:





# In[2]:


def find_horz(cnts, x_width = 50):
    boxes = []
    lines = []
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if x >= x_width:
            boxes.append([x, y, x+w, y+h])
            if y not in lines:
                lines.append(y)
    return boxes, lines

def find_ver(cnts, y_height = 50):
    boxes = []
    lines = []
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if x >= x_width:
            boxes.append([x, y, x+w, y+h])
            if y not in lines:
                lines.append(y)
    return boxes, lines

def grouplines(cols, min_sep, method = 'min'):
    grouped = []
    i = 0
    while i < len(cols):
        checked = [i]
        cur_group = [cols[i]]
        j = i+1
        while j < len(cols):
            if cols[j] - cols[i] <= min_sep:
                cur_group.append(cols[j])
                checked.append(j)
                i = j
                j = j+1
            else:
                j +=1
        grouped.append(cur_group)
        i = checked[-1] + 1

    cols_clean = []
    for i in range(len(grouped)):
        #col = np.mean(grouped[i])
        #col = np.max(grouped[i])
        if method == 'min':
            col = np.min(grouped[i])
            
        if method == 'max':
            col = np.max(grouped[i])
            
        if method == 'mean':
            col = np.mean(grouped[i])
            
        if method == 'mode':
            col = mode(grouped[i])[0][0]
        
        
        cols_clean.append(int(col))
        
    return cols_clean

def remove_ver(img, lines):
    image = copy.deepcopy(img)

    # Remove vertical lines
    for c in lines:
        cv2.line(image, (c, 0), (c, image.shape[0]), (255,255,255), 5)
     
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray

def remove_hor(img, lines):
    image = copy.deepcopy(img)

    # Remove hor lines
    for c in lines:
        cv2.line(image, (0, c), (image.shape[1], c), (255,255,255), 5)
     
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray


def vis_lines(img, ver_line, hor_line):
    image = copy.deepcopy(img)

    # Remove vertical lines
    for c in ver_line:
        cv2.line(image, (c, 0), (c, image.shape[0]), (0,255,0), 2)

    for c in hor_line:
        cv2.line(image, (0, c), (image.shape[1], c), (255,0,0), 2)
     
    return image


def vis_lines_1(img, ver_line, hor_line):
    image = copy.deepcopy(img)

    # Remove vertical lines
    for c in ver_line:
        cv2.line(image, (c, ver_line[c][0]), (c, ver_line[c][1]), (0,255,0), 2)

    for c in hor_line:
        cv2.line(image, (hor_line[c][0], c), (hor_line[c][1], c), (255,0,0), 2)
     
    return image


def find_max_ele(eles):
    
    max_len = []
    max_ele = -1
    
    i = 0
    while i < len(eles):
        checked = [i]
        cur_len = 1
        j = i+1
        while j < len(eles):
            if eles[j] == eles[i]:
                checked.append(j)
                cur_len += 1
                i = j
                j += 1
            else:
                max_len.append(cur_len)
                #j += 1
                break
                
        if j == len(eles):
            max_len.append(cur_len)
            break
        else:
            i = j
        
            
    return np.array(max_len)


# In[ ]:





# In[3]:


def get_hough_lines(img, minline = 100, maxgap = 10, resmax = 20, hor_tol = 20, ver_tol = 15, method = 'mean'):
    img_t = img.copy()
    gray = cv2.cvtColor(img_t,cv2.COLOR_BGR2GRAY)
    ##gray = cv2.medianBlur(gray,5)
    ##gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    #minLineLength = 100
    #maxLineGap = 10 # 10, 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,resmax,minline,maxgap) # 50, 20, 1
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(img_t,(x1,y1),(x2,y2),(0,255,0),1)
            
    hor_lines = {}
    ver_lines = {}
    for line in lines:
        for ele in line:
            # hor
            if ele[1] == ele[3]:
                if ele[1] not in hor_lines:
                    hor_lines[ele[1]] = 1
                else:
                    hor_lines[ele[1]] += 1
            if ele[0] == ele[2]:
                if ele[0] not in ver_lines:
                    ver_lines[ele[0]] = 1
                else:
                    ver_lines[ele[0]] += 1
                    
                    
    hor_clean = []
    ver_clean = []
    for line in lines:
        for ele in line:
            # hor
            if ele[1] == ele[3]:
                if ele[1] not in hor_clean and ( abs(ele[2] - ele[0]) > hor_tol and hor_lines[ele[1]] >= 1): # 20 and >= 1
                    hor_clean.append(ele[1])
            if ele[0] == ele[2]:
                if ele[0] not in ver_clean and ( abs(ele[3] - ele[1]) > ver_tol and ver_lines[ele[0]] >= 1): # 15 for test5, 10 for short
                    ver_clean.append(ele[0])
    hor_clean = sorted(hor_clean)
    hor_clean = grouplines(hor_clean, 5, method = method)
    ver_clean = sorted(ver_clean)
    ver_clean = grouplines(ver_clean, 10, method = method)
    
    return hor_clean, ver_clean


# In[4]:


def get_pix_lines(img, block = 11, C = 0, page_ratio = 0.95, hor_ratio = 0.5, hor_tol = 10, ver_ratio = 0.5, ver_tol = 10):
    img_s = img.copy()
    img_s = cv2.cvtColor(img_s,cv2.COLOR_BGR2GRAY)
    img_s = cv2.adaptiveThreshold(img_s,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, block, C) # 11, 0
    #img_s = cv2.dilate(img_s,None,iterations = 1)
    img_s = cv2.erode(img_s,None,iterations = 1) # use for test5
    
    
    hor_line_cor = []
    # 255 is white, 0 is black
    for j in range(img_s.shape[0]):
        cur_img = img_s[j, :]
        eles = find_max_ele(cur_img)

        #if np.sum(cur_img == 255) >= page_ratio * img_s.shape[1]:
            #hor_line_cor.append(j)
        #elif np.sum(cur_img == 0) >= page_ratio * img_s.shape[1]:
            #hor_line_cor.append(j)
        if max(eles) >= hor_ratio * img_s.shape[1]: # or np.sum(eles >= 50) >= 5: # 50 for others, 100 for test5
            hor_line_cor.append(j)
        else:
            continue
            
    hor_line_cor = sorted(hor_line_cor)
    hor_line_cor = grouplines(hor_line_cor, hor_tol, method = 'mean') 
    
        
    ver_line_cor = []
    # 255 is white, 0 is black
    for i in range(img_s.shape[1]):
        cur_img = img_s[:, i]
        eles = find_max_ele(cur_img)
        #if np.sum(cur_img == 255) == img_s.shape[0]:
        if np.sum(cur_img == 255) >= page_ratio * img_s.shape[0]:
            ver_line_cor.append(i)
        #elif np.sum(cur_img == 0) == img_s.shape[0]:
        elif np.sum(cur_img == 0) >= page_ratio * img_s.shape[0]:
            ver_line_cor.append(i)
        elif max(eles) >= ver_ratio * img_s.shape[0]: #and np.sum(eles >= 100) > 2: # 2.74%, 25; 0.3 or 0.5
            ver_line_cor.append(i)
        else:
            continue
            
    ver_line_cor = sorted(ver_line_cor)
    ver_line_cor = grouplines(ver_line_cor, ver_tol, method = 'mean') 
    
    
    return hor_line_cor, ver_line_cor


# In[ ]:





# In[ ]:





# In[ ]:




