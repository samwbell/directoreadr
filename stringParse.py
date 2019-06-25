import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from address import sts

#Parses the text into street, city, phone number, and company name.


"""
split_on_st:
makes sure that if a string contains a street that it becomes a separate word
"""
def split_on_st(string, st):
	words = string.split()
	i = words.index(st)
	add_str = ''
	if i<2:
		return 'Search','failed',True
	else:
		j = i-2
		while j>0 and (not (re.match('\d+', words[j]))):
			j -= 1
		if j < 0:
			return 'Search','failed',True
		else:
			rtuple = ' '.join(words[:i+1]).partition(' ' + words[j] + ' ')
			if len(words) > (i + 1):
				if words[i+1][0] == '(':
					add_str = ' ' + ' '.join(words[i+1:]).partition(';')[0].partition(')')[0]
					if add_str[-1] != ')':
						add_str += ')'
			return rtuple[0],(rtuple[1] + rtuple[2] + add_str), False

"""
multiline_parse:

Args:
	cut_text_words: re.search by digit on string
	text_words: its just string.split()
	string: string

returns street parsed out, and the company name
"""
def multiline_parse(cut_text_words, text_words, string):

	words = string.replace(',',' ').replace(' and ',' ').split()
	street_strings = []
	street_string = ''

	# for each word in string
	# create street_string
	for i in list(range(len(words))):
		if (i == 0 or re.match('\D', words[i])) and i != len(words) - 1:
			street_string += ' ' + words[i]
		elif i == len(words) - 1:
			street_strings.append(street_string + ' ' + words[i])
		else:
			street_strings.append(street_string)
			street_string = words[i]

	companyName = street_strings[0]
	street = street_strings[1:]

	for i in list(range(len(street))):
		# if there are prices
		if re.match('\d+$',street[i]):
			j = i+1
			while j < len(street)-1 and re.match('\d+$',street[j]):
				j += 1
			parts = re.search('(\d+)(.+)', street[j])
			street[i] += parts.group(2)

	return street, companyName

"""
line_parse:
parses each line individual line
"""
def line_parse(string):
	do_regex = True
	regex = '(\D+)(\s\d+\s)(.+)'

	# if string is a street, split on the street
	for st in sts:
		if re.match('.+\s' + st + '\s.*', string) or re.match('.+\s' + st + '$', string):
			companyName, street, do_regex = split_on_st(string, st)
	
	if do_regex:
		# remove 0's
		str_list = string.split()
		if '0' in str_list:
			str_list.remove('0')
		if '00' in str_list:
			str_list.remove('00')

		string = ' '.join(str_list)
		parts = re.search(regex, string)

		# if the string matches the format of words / space / digits / space / stuff
		# extract the company name and street from the capturing groups
		if parts:
			street = (parts.group(2) + parts.group(3)).strip()
			companyName = parts.group(1).strip()

		# if the string matches words / digits / stuff
		elif re.search('(\D+)(\d+)(.+)', string):
			parts = re.search('(\D+)(\d+)(.+)', string)
			street = (parts.group(2) + parts.group(3)).strip()
			companyName = parts.group(1).strip()

		# regex failure: no number found
		else:
			street, companyName = 'N/A', 'N/A'

		cut_strs = [' rm ', ' Rm ', ' suites ', ' Suites ', ' fl ', ' Fl ', '1st fl', '2d fl', '3d fl', '4th fl', '5th fl', 'th fl']

		# cut out any of the address associated with specific rooms/floors/suites/etc
		for cut_str in cut_strs:
			street = street.partition(cut_str)[0]

		# if the address has bldg in it
		if re.match('.+\sbldg\s.+', street):
			street = street.partition(' bldg ')[2]
		if re.match('.+\sBldg\s.+', street):
			street = street.partition(' Bldg ')[2]

	return street, companyName

"""
search:

given a string from a city directory, it returns the street and company name 
"""
def search(str_param_tuple):
	input_string, params = str_param_tuple

	string = input_string.strip()

	# removes any phone numbers
	string = string.partition(' tel ')[0].partition(' tels ')[0].partition(' Tel ')[0].partition(' Tels ')[0]
	string = string.partition(' TEL ')[0].partition(' TELS ')[0].partition('(See ')[0]
	string = string.partition(' telephone')[0].partition(' Telephone')[0].partition('-See ')[0].partition('See page')[0]
	string = string.partition(' phone')[0].partition(' phone')[0].partition(' TELEPHONE')[0].partition(' PHONE')[0]

	text_words = string.split()
	cut_text = re.search(' \d+ ', string)

	# if there are numbers in the string
	if cut_text:
		cut_text_words = string[cut_text.span()[0]:].split()
	else:
		cut_text_words = text_words

	# if the string is multiple lines
	if 'and' in cut_text_words and text_words[min(text_words.index('and') + 1, len(text_words) - 1)] not in ['page', 'pages']:
		street, companyName = multiline_parse(cut_text_words, text_words, string)
	else: 
		street, companyName = line_parse(string)
	
	return street, companyName

