import glob, os, re
import numpy as np
import cv2
from multiprocessing import Pool

"""Sorts based on natural ordering of numbers, ie. "12" > "2" """
def naturalSort(String_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

"""
croptop:
takes in 
image matrix 
p_cutoff number

chooses an index at the top of the image to cut the top margin off
allows for one pixel mistake and then keeps going
"""
def cropTop(image, p_cutoff):
    h, w = image.shape[:2]
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    y = 0
    while w - cv2.countNonZero(image[y,:]) < sfw*p_cutoff/2.0:
        y += 1
    #print(y)
    if cv2.countNonZero(image[y+int(sf*50):y+int(sf*100),:]) < float(int(sf*50.0))*sfw*p_cutoff/2.0:
        y += int(sf*50)
        while w - cv2.countNonZero(image[y,:]) < sfw*p_cutoff/2.0:
            y += 1
    return y - int(sf*25)

"""
cropbottom:
takes in 
image matrix 
p_cutoff number

chooses an index at the bottom of the image to cut the bottom margin off
allows for one pixel mistake and then keeps going
"""
def cropBottom(image, p_cutoff):
    h, w = image.shape[:2]
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    y = h - 1
    while w - cv2.countNonZero(image[y,:]) < sfw*p_cutoff/2.0:
        y -= 1
    if cv2.countNonZero(image[y-int(sf*100):y-int(sf*50),:]) < float(int(sf*50.0))*sfw*p_cutoff/2.0:
        y -= int(sf*50)
        while w - cv2.countNonZero(image[y,:]) < sfw*p_cutoff/2.0:
            y -= 1
    return y + int(sf*35)

"""
cropleft:
takes in 
image matrix 
p_cutoff number

chooses an index at the left of the image to cut the left margin off
allows for one pixel mistake and then keeps going
"""
def cropLeft(image, p_cutoff):
    h, w = image.shape[:2]
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    x = 0
    while h - cv2.countNonZero(image[:,x]) < sf*p_cutoff:
        x += 1
    if cv2.countNonZero(image[:,x+int(sfw*50):x+int(sfw*100)]) < float(int(sfw*50.0))*sf*p_cutoff:
        x += int(sfw*50)
        while h - cv2.countNonZero(image[:,x]) < sf*p_cutoff:
            x += 1
    return x - int(50*sfw)

"""
cropRight:
takes in 
image matrix 
p_cutoff number

chooses an index at the right of the image to cut the right margin off
allows for one pixel mistake and then keeps going
"""
def cropRight(image, p_cutoff):
    h, w = image.shape[:2]
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    x = w - 1
    while h - cv2.countNonZero(image[:,x]) < sf*p_cutoff:
        x -= 1
    if cv2.countNonZero(image[:,x-int(sfw*200):x-int(sfw*100)]) < float(int(sfw*100.0))*sf*p_cutoff:
        x -= int(100*sfw)
        while h - cv2.countNonZero(image[:,x]) < sf*p_cutoff:
            x -= 1
    return x + int(100*sfw)

"""
cropMargins:
takes in a filename_param_tuple
wrapper function that crops the file
"""
def cropMargins(filename_param_tuple):
    file, params = filename_param_tuple
    p_cutoff = params['p_cutoff']
    print(file + '-margins cropped')

    #read in img
    image = cv2.imread(file, 0)

    # find where the margins are on each side
    top = cropTop(image, p_cutoff)
    bottom = cropBottom(image, p_cutoff)
    left = cropLeft(image, p_cutoff)
    right = cropRight(image, p_cutoff)

    # crop the image
    h, w = image.shape[:2]
    sf = float(h)/13524.0
    sfw = float(w)/9475.0
    cropped = image[top-int(sf*5.0) : bottom+int(sf*5.0), left-int(sfw*5.0) : right+int(sfw*10.0)]
    
    # save to file
    nDirectory = 'margins'
    filename = file.split("/")[-1]
    cv2.imwrite(os.path.join(nDirectory, filename), cropped)
    
    return

"""
cleanImage:
takes in an image, does various operations to remove noise
"""
def cleanImage(image):
    inv = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,5))
    closing = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opening)


def marginCrop(params):

    #make margins dir.
    if not os.path.exists('margins'):
        os.mkdir('margins')

    # parses single image
    if 'img_name' in params:
        cropMargins((os.getcwd() + "/no_ads/" + params['img_name'] + ".png", params))
        return

    #create list of image/param tuples.
    x = sorted(glob.glob(os.getcwd() + "/no_ads/*.png"), key=naturalSort)
    params_and_files = [(i, params) for i in x]

    # map params_and_files to cropMargins.
    if params['do_multiprocessing']:
        pool = Pool(params['pool_num'])
        pool.map(cropMargins, params_and_files)
    else:
        for filename_param_tuple in params_and_files:
            cropMargins(filename_param_tuple)
    

