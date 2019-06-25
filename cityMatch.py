from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import re
import pandas as pd

#Matches string to database of cities

city_dict = {
	'Prv': 'PROVIDENCE',
	'Prov': 'PROVIDENCE',
	'Providence': 'PROVIDENCE',
	'Paw': 'PAWTUCKET',
	'Pawt': 'PAWTUCKET',
	'Pawtucket': 'PAWTUCKET',
	'NP': 'NORTH PROVIDENCE',
	'N Prv': 'NORTH PROVIDENCE',
	'N Prov': 'NORTH PROVIDENCE',
	'N Providence': 'NORTH PROVIDENCE',
	'E Prv': 'EAST PROVIDENCE',
	'E Prov': 'EAST PROVIDENCE',
	'E Providence': 'EAST PROVIDENCE',
	'East Providence': 'EAST PROVIDENCE',
	'Wrwk': 'WARWICK',
	'Warwick': 'WARWICK',
	'W Wrwk': 'WEST WARWICK',
	'West Wrwk': 'WEST WARWICK',
	'W Warwick': 'WEST WARWICK',
	'West Warwick': 'WEST WARWICK',
	'Smithfield': 'SMITHFIELD',
	'Sfld': 'SMITHFIELD',
	'N Smithfield': 'NORTH SMITHFIELD',
	'N Sfld': 'NORTH SMITHFIELD',
	'Cumberland': 'CUMBERLAND',
	'Cmd': 'CUMBERLAND',
	'Seekonk': 'SEEKONK',
	'Seek': 'SEEKONK',
	'Cranston': 'CRANSTON',
	'Crns': 'CRANSTON',
	'Cf': 'CENTRAL FALLS',
	'C F': 'CENTRAL FALLS',
	'Central Falls': 'CENTRAL FALLS',
	'Attleboro': 'ATTLEBORO',
	'Attl': 'ATTLEBORO',
	'N Attleboro': 'NORTH ATTLEBORO',
	'N Attl': 'NORTH ATTLEBORO',
	'S Attleboro': 'SOUTH ATTLEBORO',
	'S Attl': 'SOUTH ATTLEBORO',
	'Woon': 'WOONSOCKET',
	'Woonsocket': 'WOONSOCKET',
	'Lincoln': 'LINCOLN',
	'Lcln': 'LINCOLN',
	'Kingston': 'SOUTH KINGSTOWN',
	'S Kingstown': 'SOUTH KINGSTOWN',
	'South Kingstown': 'SOUTH KINGSTOWN',
	'SKgtwn': 'SOUTH KINGSTOWN',
	'S Kgtwn': 'SOUTH KINGSTOWN',
	'N Kingstown': 'NORTH KINGSTOWN',
	'North Kingstown': 'NORTH KINGSTOWN',
	'NKgtwn': 'NORTH KINGSTOWN',
	'N Kgtwn': 'NORTH KINGSTOWN',
	'Johnston': 'JOHNSTON',
	'Jstn': 'JOHNSTON',
	'Narr': 'NARRAGANSETT',
	'Narragansett': 'NARRAGANSETT',
	'Newport': 'NEWPORT',
	'Bris': 'BRISTOL',
	'Bristol': 'BRISTOL',
	'Tiverton': 'TIVERTON',
	'Little Compton': 'LITTLE COMPTON',
	'Portsmouth': 'PORTSMOUTH',
	'Middletown': 'MIDDLETOWN',
	'Warren': 'WARREN',
	'Barrington': 'BARRINGTON',
	'Burrillville': 'BURRILLVILLE',
	'Foster': 'FOSTER',
	'Glocester': 'GLOCESTER',
	'Gloc': 'GLOCESTER',
	'Coventry': 'COVENTRY',
	'Scituate': 'SCITUATE',
	'Sct': 'SCITUATE',
	'East Greenwich': 'EAST GREENWICH',
	'E Greenwich': 'EAST GREENWICH',
	'E Grn': 'EAST GREENWICH',
	'West Greenwich': 'WEST GREENWICH',
	'W Greenwich': 'WEST GREENWICH',
	'W Grn': 'WEST GREENWICH',
	'Richmond': 'RICHMOND',
	'Exeter': 'EXETER',
	'Hopkinton': 'HOPKINTON',
	'Charlestown': 'CHARLESTOWN',
	'New Shoreham': 'NEW SHOREHAM',
	'Block Island': 'NEW SHOREHAM',
	'Jamestown': 'JAMESTOWN',
	'N/A': 'N/A'
}

"""
given two cities, scores how likely they are to be the same city
"""
def city_scorer(str1, str2):
	if str1[0] == str2[0]:
		fscore = 100.0
	elif str1[0].lower() == str2[0].lower():
		fscore = 75.0
	else:
		fscore = 25.0
	rscore = (fscore + 3 * fuzz.ratio(str1, str2)) / 4.0
	return rscore
"""
given a city, extracts the city in the city dict that most closely matches it
"""
def city_match(city):
	results = process.extractOne(city, city_dict.keys(), scorer = city_scorer)
	cty,score = results[0],results[1]
	if score > 85:
		#print(cty + ', score = ' + str(score))
		return city_dict[cty]
	else:
		#print('Did not match a city, will guess PROVIDENCE')
		return 'PROVIDENCE'

