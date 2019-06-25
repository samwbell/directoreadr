import os
import glob
import sys

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


if not sys.argv[1]:
	raise Exception('You need to pass in a parameter file.')
inputParams = str(sys.argv[1])

dir_list = sorted(list(set(glob.glob('cd????/')) - set(['cd1980/', 'cd1985/', 'cd1990/'])))[10:]

#dir_list = ['cd1937/', 'cd1950/', 'cd1952/', 'cd1954/',  'cd1960/', 'cd1962/', 'cd1964/', 'cd1966/', 'cd1968/', 'cd1970/', 'cd1976/', 'cd1978/', 'cd1980/']

file = read_file('inputParams.json')

for directory in dir_list:
	part0,part1,part2 = file.partition('\"year_folder\":')
	wfile = part0 + part1 + ' \"' + directory.rstrip('/') + '\",' + part2.partition(',')[2]
	print('Running directory: ' + directory)
	write_file('temp.json', wfile)
	os.system('python imageProcess.py temp.json')
	
