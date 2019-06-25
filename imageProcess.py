import time
mt1 = time.time()
import ads, margins, columnCrop, entryChop, parse
import os
import sys
import json

if not sys.argv[1]:
	raise Exception('You need to pass in a parameter file.')
inputParams = str(sys.argv[1])

#This is the driver script for all the image processing.

if __name__ == '__main__':

	# opens parameter file 
	with open(inputParams) as json_data:
		d = json.load(json_data)

	# moves into the folder in question, 
	# and extracts the parameters for each filter
	os.chdir(d['year_folder'])
	img_p = d['image_process']
	all_params = [d['no_ads'], d['margins'],d['columns'],d['entries']]
	
	# checks if you only wish to parse a single image. 
	if img_p['single_image']:
		for p in all_params:
			p.update({'img_name': img_p['img_name']})
	
	# performs img filtering operations
	if img_p['ads']:
		print('Removing ads...')
		t1 = time.time()
		ads.rmAds(all_params[0]) 
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	if img_p['margins']:
		print('Cropping margins...')
		t1 = time.time()
		margins.marginCrop(all_params[1])
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	if img_p['columns']:
		print('Cropping columns...')
		t1 = time.time()
		columnCrop.doCrop(all_params[2]) 
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	if img_p['entries']:
		print('Chopping entries...')
		t1 = time.time()
		entryChop.entryChop(all_params[3])
		t2 = time.time()
		print('Done in: ' + str(round(t2-t1, 2)) + ' s')
	if img_p['parse']:
		os.chdir("..")
		parse.main(d)

mt2 = time.time()
print('Full runtime: ' + str(round(mt2-mt1, 2)) + ' s')