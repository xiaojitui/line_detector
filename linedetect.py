#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
from imgprocess import *


# In[ ]:





# In[6]:


def read_img(filepath, box = None, show_img = False):
    img = plt.imread(filepath)
 

    if box:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        img = img[y1:y2, x1:x2, :]

    img = cv2.resize(img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    if show_img:
        plt.imshow(img)
        
    return img


# In[7]:


def parse_lines(img, method = 'hough'):

    if method == 'pix':
        hor_clean, ver_clean = get_pix_lines(img, block = 11, C = 1, 
                                             page_ratio = 0.95, hor_ratio = 0.95, 
                                             hor_tol = 5, ver_ratio = 0.5, 
                                             ver_tol = 10)
        
    if method == 'hough':
        hor_clean, ver_clean = get_hough_lines(img, minline = 100, 
                                               maxgap = 10, resmax = 20, 
                                               hor_tol = 20, ver_tol = 10, 
                                               method = 'mean')
    return hor_clean, ver_clean


# In[8]:


def vis_results(img, hor_clean, ver_clean):
    
    img1 = vis_lines(img, [], hor_clean)
    img2 = vis_lines(img, ver_clean, [])
    
    fig, ax = plt.subplots(1, 2, figsize = (15, 15))
    ax[0].imshow(img1)
    ax[1].imshow(img2)


# In[ ]:





# In[11]:


## test
'''
box = [10, 110, 800, 600]
filepath = './test.jpg'
img = read_img(filepath, box)
hor_clean, ver_clean = parse_lines(img, method = 'hough')
vis_results(img, hor_clean, ver_clean)
'''

# In[ ]:


if __name__ == '__main__':
    
    box = [10, 110, 800, 600]
    filepath = './test.jpg'
    img = read_img(filepath, box)
    hor_clean, ver_clean = parse_lines(img, method = 'hough')
    #vis_results(img, hor_clean, ver_clean)

