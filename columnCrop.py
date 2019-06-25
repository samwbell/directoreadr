import glob, os, re
import numpy as np
from numpy import ndarray
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.cluster import MeanShift
from multiprocessing import Pool

#Chops the pages into columns

"""
Sorts based on natural ordering of numbers, ie. "12" > "2" 
"""
def naturalSort(String_): 
	return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

"""
does the bulk of actual image cropping into columns

input: 
	image: ??, actual img (i think it's a matrix of pixel vals actually?)
	file: str, filename
	do_plots: bool, whether to make the diagnostic plots
"""
def cropImage(image, file, do_plots):
	croppedImages = []
	img = image.copy()
	height, width = img.shape[:2]
	sf = float(height)/11675.0
	sfw = float(width)/7820.0

	# list of rolling means of black pixels
	histogram  = pd.Series([height - cv2.countNonZero(img[:,i]) for i in list(range(width))]).rolling(5, center=True).mean()

	# prints out plots of the pixel count histogram and a smoothed version of the histogram
	if do_plots:
		fig = plt.figure()
		ax = histogram.plot()
		ax.set_ylim([0,200])
		fig.savefig(file.partition('.png')[0] + '.histogram.pdf', bbox_inches='tight')
		plt.close(fig)
		fig = plt.figure()
		ax = histogram.rolling(50,center=True).mean().rolling(10,center=True).mean().plot()
		ax.set_ylim([0,200])
		fig.savefig(file.partition('.png')[0] + '.histogram.smooth.pdf', bbox_inches='tight')
		plt.close(fig)
	
	# takes all instances where black pixel count < 150
	dip_df = histogram[histogram < sf*150].to_frame().rename(columns = {0:'count'})
	
	# sets all instances of just 50 (factored to scale) to 0.
	dip_df.loc[dip_df['count']<sf*50,'count'] = 0
	histogram.iloc[0] = 0
	indices = np.array(dip_df.index.tolist()).reshape(-1,1)

	# predicts the best place to cut the columns
	ms = MeanShift()
	ms.fit(indices)
	dip_group = ms.predict(indices)
	dip_df = dip_df.assign(group = dip_group)

	# picks the rightmost place to cut the columns. might not work if image is tilted.
	try:
		cut_points = [0] + sorted(dip_df.groupby('group').apply(lambda x: max(x[x['count']==0].index - int(sfw*35.0))).tolist())[1:-1] + [width]
	except:
		cut_points = [0]

	# returns points to cut. 
	for i in list(range(len(cut_points)-1)):
		croppedImages.append(img[0:height, cut_points[i]:cut_points[i+1]])
	return croppedImages

def crop_file(file_param_tuple):
	file, params = file_param_tuple
	img = cv2.imread(file, 0)

	crop = cropImage(img, file, params['do_plots'])

	name = file[:-4].partition('.chop')[0].split("/")[-1]
	ext = file[-4:]
	i = 1
	for image in crop:
		cv2.imwrite(os.path.join('columns', name + " ("+ str(i) + ")" + ext), image)
		i += 1
	if len(crop) == 1:
		print("ERROR: " + file)
	#print file + '-cropped to columns'
	return

def doCrop(params):
	#make columns dir
	if not os.path.exists('columns'):
		os.mkdir('columns')

	# parses single image
	if 'img_name' in params:
		print(params)
		crop_file((os.getcwd() + "/margins/" + params['img_name'] + ".png", params))
		return

	#find chopped files. 
	file_list = glob.glob(os.getcwd() + "/margins/*.png")
	for chop_file in file_list:
		if re.match('.*\.chop\.png', chop_file):
			unchopped_file = chop_file.partition('.chop.png')[0] + '.png'
			file_list.remove(unchopped_file)
			print('ALERT: Chop file override!\nInstead of ' + unchopped_file + ', using: ' + chop_file)
	file_list.sort(key=naturalSort) 
	file_list = [(i, params) for i in file_list]

	# map list of files/params to crop_file
	if params['do_multiprocessing']:
		pool = Pool(params['pool_num'])
		pool.map(crop_file, file_list)
	for file_param_tuple in file_list:
		try:
			crop_file(file_param_tuple)
		except:
			print('WARNING: File ' + file_param_tuple[0] + ' failed!!!')
	
	
