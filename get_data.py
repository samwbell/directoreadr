import os
import glob
import sys
from shutil import copyfile

"""
Writes a string to a file
"""
def write_file(path, text):
	with open(path, 'w') as path_stream:
		path_stream.write(text)
		path_stream.close()

"""
Reads a file to a string
"""
def read_file(path):
	with open(path, 'r') as path_stream:
		rstr = path_stream.read()
		path_stream.close()
	return rstr

dir_list = sorted(glob.glob('cd????/'))

if not os.path.isdir('results'):
	os.mkdir('results')

for directory in dir_list:
	copyfile(directory + 'FOutput.csv', 'results/' + directory.rstrip('/') + 'FOutput.csv')
	copyfile(directory + 'drops_address.csv', 'results/' + directory.rstrip('/') + 'drops_address.csv')
	copyfile(directory + 'geocoder_errors.csv', 'results/' + directory.rstrip('/') + 'geocoder_errors.csv')
	copyfile(directory + 'wrong_city.csv', 'results/' + directory.rstrip('/') + 'wrong_city.csv')
	