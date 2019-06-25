import pandas as pd
import re
import numpy as np
from cityMatch import city_match
from fuzzywuzzy import fuzz, process
import time 
import pickle as pkl

# read in pd datafram of addresses
ri_streets_table = pd.read_csv('StreetZipCity.csv')
street_patt = re.compile(r"(^\d+)(.+)")
renamed_street_dict = {'BALBO AVE':'DE PASQUALE AVE', 'ALLEN\'S AVE':'ALLENS AVE', 'POTTER\'S AVE':'POTTERS AVE', 'BROADWAY ST':'BROADWAY'}

# try to load the pickle
try:
    street_name_dict = pkl.load(open('street_name_dict.pkl', 'rb'))
except:
    street_name_dict = {}

# mapping abbreviated streets to actual streets
abbreviations = pd.DataFrame({'Street':['BWAY','WASH'], 'Zip_Code':['BROADWAY', 'WASHINGTON ST'], 'City':['PROVIDENCE','PROVIDENCE']})
historical_streets = pd.read_csv('historical_streets.csv').dropna()

# adds abbreviations and streets that no longer exist
ri_streets_table = ri_streets_table.append(abbreviations, ignore_index=True).append(historical_streets, ignore_index=True)
addr_options = ri_streets_table[ri_streets_table['City'] == 'PROVIDENCE']

# all the possible street endings
sts = ['St','Ave','Av','Ct','Dr','Rd','Ln']
sts += [st.lower() for st in sts]
sts += [st.upper() for st in sts]


"""
turns direction abbreviations to their full names
"""
def substitute_directions(inwords):
    outwords = inwords[:]
    for i in list(range(len(inwords))):
        if outwords[i] == 'W':
            outwords[i] = 'WEST'
        if outwords[i] == 'E':
            outwords[i] = 'EAST'
        if outwords[i] == 'N':
            outwords[i] = 'NORTH'
        if outwords[i] == 'S':
            outwords[i] = 'SOUTH'
    return outwords

"""
takes in two streets, and tries to see if they're similar
ie. so that East America Ave matches with America Ave.
"""
def street_scorer(istr1, istr2):
    str1 = istr1.upper()
    str2 = istr2.upper()
    # turns any direction abbrevs. into their full names
    words1 = substitute_directions(str1.split())
    words2 = substitute_directions(str2.split())
    # if they don't have the same last word
    # and if it's a street name, remove the last word
    if words1[-1] != words2[-1]:
        if words2[-1] in sts:
            words2 = words2[:-1]
        if words1[-1] in sts:
            words1 = words1[:-1]
    # sort by size of the word
    word1 = sorted(words1, key=len, reverse=True)[0]
    word2 = sorted(words2, key=len, reverse=True)[0]

    # takes the average of the Levenshtein ratio between the 
    # longest words of each string and between the strings as a whole
    return (fuzz.ratio(word1, word2) + fuzz.ratio(' '.join(words1), ' '.join(words2)))/2

"""
Class to hold address information.
"""
class Address(object):

    def __init__(self, street=None, city='PROVIDENCE', streets_table=ri_streets_table):

        self.city = city
        self.street = street
        self.streets_table = streets_table
        self.addr_matches = []

    """
    Find (street, city, score) fuzzy matches for addresses in streets_table.
    """
    def set_addr_matches(self, cutoff, limit):
        
        self.addr_matches = []
        street = self.street.strip()
        # if address parse fails
        if street == 'N/A':
            #print('N/A')
            self.addr_matches.append((street, 'N/A', 'FAILED TO PARSE AN ADDRESS'))
            return

        # if there's a number in street
        # try to guess what the city is in the address
        if re.match('.+\(.{1,20}\)$', street):
            #print(street)
            city_guess = street.partition('(')[2].partition(')')[0].partition(';')[0]
            city_guess = re.sub('/d', '', city_guess)
            city_guess = re.sub(';', '', city_guess)
            if city_guess != '':
                #print(city_guess)
                self.city = city_match(city_guess.strip())
        
        street = street.partition('(')[0]
        #print('Matching: ' + street + ', ' + self.city)
        
        # Get all valid addresses within the matches cities.
        addr_options = self.streets_table[self.streets_table['City'] == self.city]
        
        # Separate street number from street name.
        sepr = re.search(street_patt, street)
        if sepr:
            stnum = sepr.group(1).strip()
            stnam = sepr.group(2).strip()
        else:
            stnum = ''
            stnam = street.strip()
        
        # handle empty street
        if stnam == '':
            self.addr_matches.append((street, 'N/A', 'EMPTY STREET'))
            return
        #print('stnam', stnam)

        # match street to street name ending
        for st in sts:
            if re.match('.*\s' + st + '\s.*', stnam):
                stnam = stnam.partition(' ' + st + ' ')[0] + ' ' + st
                break


        stnam = re.sub(' Av$', ' Ave', stnam).replace('- ','')
        stnam = stnam.partition(' cor ')[0].partition(' corner ')[0].partition(' at ')[0]
        #print('stnam')
        #print(stnam)

        # Look for best fuzzy matches with a score > cutoff.
        t1 = time.time()

        # for perfect match
        if stnam.upper() in addr_options['Street'].tolist():
            #print('Perfect match')
            street_matches = (stnam.upper(), 100.0)
            if stnam.upper() in abbreviations['Street'].tolist():
                street_matches = (abbreviations[abbreviations['Street'] == street_matches[0]]['Zip_Code'].values[0], 100)
        # for street in dictionary
        elif stnam.upper() in street_name_dict.keys():
            #print('Street in dictionary')
            street_matches = street_name_dict[stnam.upper()]
        else:
            try:
                street_matches = process.extractOne(stnam, addr_options['Street'], scorer=street_scorer)
                if street_matches[0] in abbreviations['Street'].tolist():
                    street_matches = list(street_matches)
                    street_matches[0] = abbreviations[abbreviations['Street'] == street_matches[0]]['Zip_Code'].values[0]
                    street_matches = tuple(street_matches)
                street_name_dict[stnam.upper()] = street_matches
            except:
                #print('ERROR IN STREET MATCHING')
                pcity = 'N/A'
                if self.city != 'PROVIDENCE':
                    pcity = self.city
                self.addr_matches.append((street, pcity, 'ERROR IN STREET MATCHING'))
                return
        if not street_matches:
            #print('No match')
            self.addr_matches.append((street, 'N/A', 'NO MATCH: ' + stnam + ',' + self.city))
            return
        street, score = (street_matches[0],street_matches[1]) # removes the third dummy value that sometimes shows up in the tuple
        t2 = time.time()
        #print('Search time: ' + str(round(t2-t1, 6)) + ' s')

        # Add to addr_matches if score reaches cutoff:
        if score < cutoff:
            #print('SCORE LESS THAN CUTOFF: ' + street + ',' + str(score) + ',' + self.city)
            pcity = 'N/A'
            if self.city != 'PROVIDENCE':
                pcity = self.city
            self.addr_matches.append((street, pcity, 'SCORE LESS THAN CUTOFF: ' + stnam + ',' + str(score) + ',' + self.city))
        else:
            if street in renamed_street_dict.keys():
                street = renamed_street_dict[street]
            addr = stnum + ' ' + street
            addr_match = (addr, self.city, score)
            #print(addr_match)
            self.addr_matches.append(addr_match)

        pkl.dump(street_name_dict, open('street_name_dict.pkl', 'wb'))



#address = Address(street='79 Bway', city='PROVIDENCE')
#address.set_city_matches(cutoff=80)
#address.set_addr_matches(cutoff=45, limit=1)
#print('Address matches: ')
#print(address.addr_matches)





