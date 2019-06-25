import glob, os, re
import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.cluster import MeanShift
import pickle as pkl
import time
from multiprocessing import Pool
import shutil

#Chops columns into entries
#Script contains unused functions and needs heavy editing.

"""Sorts based on natural ordering of numbers, ie. "12" > "2" """
def naturalSort(String_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

def cropEntries(image, file, padding):
    #t1 = time.time()
    croppedImages = []
    crop_points = []
    img = image.copy()
    height, width = img.shape[:2]
    sf = float(width)/float(2611)
    pad = int(padding/float(height)*float(11675))
    histogram  = pd.Series([width - cv2.countNonZero(img[i,:]) for i in list(range(height))])
    # do plots. 
    #fig = plt.figure()
    #ax = histogram.plot()
    #ax.set_ylim([0,150])
    #ax.set_xlim([10500,11500])
    #plt.savefig('histogram' + file + '.pdf', bbox_inches='tight')
    #plt.close(fig)


    dip_df = histogram[histogram < sf*25].to_frame().rename(columns = {0:'count'})
    indices = np.array(dip_df.index.tolist()).reshape(-1,1)
    #pkl.dump(indices, open('indices.pkl', 'wb'))
    #t2 = time.time()
    #print('Prep time: ' + str(round(t2-t1, 2)) + ' s')

    # find indices to cut the entries
    #tf1 = time.time()
    ms = MeanShift(bandwidth = sf*50, bin_seeding=True)
    ms.fit(indices)
    dip_group = ms.predict(indices)
    #tf2 = time.time()
    #print('Fit time: ' + str(round(tf2-tf1, 2)) + ' s')
    
    # add new column 
    #t1 = time.time()
    dip_df = dip_df.assign(group = dip_group)
    #cut_points = [0] + sorted(dip_df.groupby('group').apply(lambda x: int(np.mean(x.index))).tolist())[1:-1] + [height]
    
    #calculate where to cut
    cut_points = [0] + sorted(dip_df.groupby('group').idxmin()['count'].tolist())[1:-1] + [height]
    median_height = np.median([cut_points[i+1] - cut_points[i] for i in list(range(len(cut_points) - 1))])
    #t2 = time.time()
    #print('Sort time: ' + str(round(t2-t1, 2)) + ' s')

    #for each pair of cut points found
    for i in list(range(len(cut_points)-1)):
        start,end = cut_points[i],cut_points[i+1]

        # if we suspect an entry is too large
        if end-start > 1.5*median_height:
            # do the algorithm over again
            entry_hist = pd.DataFrame(data={'count':[float(width - cv2.countNonZero(img[j,:])) for j in list(range(start,end))]}, index=list(range(start,end)))
            entry_dip_df = entry_hist[entry_hist['count'] < sf*100]
            entry_indices = np.array(entry_dip_df.index.tolist()).reshape(-1,1)
            entry_ms = MeanShift(bandwidth = sf*50, bin_seeding=True)
            entry_ms.fit(entry_indices)
            entry_dip_group = entry_ms.predict(entry_indices)
            entry_dip_df = entry_dip_df.assign(entry_group = entry_dip_group)
            entry_cut_points = [start] + sorted(entry_dip_df.groupby('entry_group').idxmin()['count'].tolist())[1:-1] + [end]
            
            # if you have too many cut points for one entry
            if len(entry_cut_points) > 2 :
                #print(entry_cut_points)
                #fig2 = plt.figure()
                #ax = entry_hist['count'].plot()
                #for xval in entry_cut_points:
                    #ax2 = plt.axvline(x = xval, linestyle = ':', color = 'r')
                #ax.set_ylim([0,300])
                #plt.savefig('entry_hist' + file + str(i+1) + '.pdf', bbox_inches='tight')
                #plt.close(fig2)

                
                for entry_i in list(range(len(entry_cut_points)-1)):
                    # adjust the cut points
                    if histogram.iloc[entry_cut_points[entry_i]:entry_cut_points[entry_i+1]].sum() > sf*20:
                        adjusted_start = entry_cut_points[entry_i]
                        adjusted_end = entry_cut_points[entry_i+1]
                        while (histogram.iloc[adjusted_start] == 0) and (adjusted_start < (adjusted_end-1)):
                            adjusted_start += 1
                        while (histogram.iloc[adjusted_end-1] == 0) and ((adjusted_end-1) > adjusted_start):
                            adjusted_end -= 1
                        adjusted_start = max(adjusted_start - pad, 0)
                        adjusted_end = min(adjusted_end + pad, height)
                        croppedImages.append(img[adjusted_start:adjusted_end, 0:width])
                        crop_points.append([adjusted_start,adjusted_end])
            else:
                if entry_hist['count'].sum() > sf*20:
                    # adjust cut points
                    adjusted_start = start + 0
                    adjusted_end = end - 0
                    while (histogram.iloc[adjusted_start] == 0) and (adjusted_start < (adjusted_end-1)):
                        adjusted_start += 1
                    while (histogram.iloc[adjusted_end-1] == 0) and ((adjusted_end-1) > adjusted_start):
                        adjusted_end -= 1
                    adjusted_start = max(adjusted_start - pad, 0)
                    adjusted_end = min(adjusted_end + pad, height)
                    croppedImages.append(img[adjusted_start:adjusted_end, 0:width])
                    crop_points.append([adjusted_start,adjusted_end])
        else:
            # if the cut points end up possibly cutting words
            if histogram.iloc[start:end].sum() > sf*20:
                # adjust cut points
                adjusted_start = start + 0
                adjusted_end = end - 0
                while (histogram.iloc[adjusted_start] == 0) and (adjusted_start < (adjusted_end-1)):
                    adjusted_start += 1
                while (histogram.iloc[adjusted_end-1] == 0) and ((adjusted_end-1) > adjusted_start):
                    adjusted_end -= 1
                adjusted_start = max(adjusted_start - pad, 0)
                adjusted_end = min(adjusted_end + pad, height)
                croppedImages.append(img[adjusted_start:adjusted_end, 0:width])
                crop_points.append([adjusted_start,adjusted_end])
    #pkl.dump(crop_points, open('crop_points.' + file + '.pkl', 'wb'))
    return croppedImages, crop_points

 
"""
cleans the image of noise
"""
def cleanImage(image):
    inv = cv2.bitwise_not(image)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    closing = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opening)
    #return opening

"""
first cropping pass. 
crops the left and right by seeing if there are any black pixels in a given column
and stopping when there is one. 
"""
def fineCrop(image):
    height, width = image.shape[:2]
    clean = cleanImage(image)
    sWidth = int(width/5.0)
    cropLeft = sWidth
    cropRight = sWidth
    histogram  = pd.Series([height - cv2.countNonZero(clean[:,i]) for i in list(range(width))])
    def hasBlackPixel(i):
        if histogram[i] == 0:
            return False
        else:
            return True
    while cropLeft > 0 and hasBlackPixel(cropLeft):
        cropLeft -= 1
    while cropRight < width - 1 and hasBlackPixel(cropRight):
        cropRight += 1
    return image[0 : height, cropLeft  : cropRight]

"""
high level wrapper function that takes in one image (+params) and outputs
a crop points dictionary 
"""
def entry_wrapper(file_param_tuple):
    crop_points_dict = {}
    file, params = file_param_tuple
    #print 'Chopping: ' + file
    fileN = file[:-4].split("/")[-1]
    ext = file[-4:]

    # read in the img
    #t1 = time.time()
    original = cv2.imread(file, 0)
    #t2 = time.time()
    #print('Read time: ' + str(round(t2-t1, 2)) + ' s')
    
    #crop it once
    #t1 = time.time()
    original = fineCrop(original)
    #t2 = time.time()
    #print('fineCrop time: ' + str(round(t2-t1, 2)) + ' s')
    
    # make border 
    #t1 = time.time()
    original = cv2.copyMakeBorder(original,4,4,10,10, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    #t2 = time.time()
    #print('copyMakeBorder time: ' + str(round(t2-t1, 2)) + ' s')
    
    # crop again with better algorithm
    #t1 = time.time()
    #clean = cleanImage(original)
    entries, points = cropEntries(original, file, params['padding'])
    #t2 = time.time()
    #print('Entry crop time: ' + str(round(t2-t1, 2)) + ' s')
    
    #write entries to files
    #print('Saving...')
    #t1 = time.time()
    i = 1
    for image in entries:
        w_image = cv2.copyMakeBorder(image,15,15,15,15, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
        cv2.imwrite(os.path.join('entry', fileN + "_" + str(i) + ext), w_image)
        crop_points_dict[fileN + "_" + str(i) + ext] = points[i-1]
        i += 1
    #t2 = time.time()
    #print('Done in: ' + str(round(t2-t1, 3)) + ' s')
    return crop_points_dict

def entryChop(params):
    # make entry folder.
    nDirectory = 'entry'
    if os.path.exists(nDirectory):
        shutil.rmtree(nDirectory)
    os.mkdir(nDirectory)

    # create file/param lists
    x = []
    if 'img_name' in params:
        x = sorted(glob.glob(os.getcwd() + "/columns/" + params['img_name'] + " *.png"), key=naturalSort)
    else:
        x = sorted(glob.glob(os.getcwd() + "/columns/*.png"), key=naturalSort)
    files_and_params = [(i, params) for i in x]

    # map files/params to entry_wrapper
    # entry_wrapper outputs a crop points dict, which is thrown into results 
    result = []
    if params['do_multiprocessing']:
        pool = Pool(params['pool_num'])
        result = pool.map(entry_wrapper, files_and_params)
    for file_param_tuple in files_and_params:
        result.append(entry_wrapper(file_param_tuple))

    # merge all the dictionaries into one
    crop_points_dict = { k: v for d in result for k, v in d.items() }

    # dump crop points dict into pickle file. 
    pkl.dump(crop_points_dict, open('crop_points_dict.pkl', 'wb'))



