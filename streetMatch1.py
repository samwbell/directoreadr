import pandas as pd
import re
import pickle as pkl

from address import Address
from multiprocessing import Pool

# Matches string to a database of streets

def search_street(query):
    address = Address(street=query, city='PROVIDENCE')
    address.set_addr_matches(cutoff=80, limit=1)
    rtuple = address.addr_matches[0]
    return query,rtuple[0],rtuple[1],rtuple[2]

def streetMatcher(dataFrame, dir_dir):
    final = []
    mistakes = []
    notpvd = []
    #dataFrame = pd.read_pickle('ccities')
    #streetTable()
    try:
        street_dict = pkl.load(open('street_dict.pkl', 'rb'))
    except:
        street_dict = {}

    search_list = []

    # Check to see if any street strings are not already in the dictionary.
    street_set = set(dataFrame['Street'])
    search_list = list(street_set - set(street_dict.keys()))

    # If there are street strings missing from the dictionary, do the address matching, and add them.
    if search_list:
        do_multiprocessing = True
        if do_multiprocessing:
            pool = Pool(3)
            search_results = pool.map(search_street, search_list)
        else:
            search_results = [search_street(street_i) for street_i in search_list]
        for query,addr,city,score in search_results:
            street_dict[query] = (addr, city, score)

    #Get each row of dataframe with corrected cities
    for row in dataFrame.itertuples():

        street = row.Street

        # Get valid addresses from city and street info.
        addr, city, score = street_dict[street]
        if city == 'N/A':
            mistakes.append({
                'Street': addr,
                'Drop_Reason': score,
                'File_List': row.File_List,
                'Text': row.Text,
                })
        elif city != 'PROVIDENCE':
            notpvd.append({
                'Address': addr,
                'City': city,
                'Conf_Score': score,
                'Header': row.Header,
                'Clean_Header': row.clean_header,
                'File_List': row.File_List,
                'Text': row.Text,
                'Company_Name': row.Company_Name
            })
        else:
            final.append({
                'Address': addr,
                'City': city,
                'Conf_Score': score,
                'Header': row.Header,
                'Clean_Header': row.clean_header,
                'File_List': row.File_List,
                'Text': row.Text,
                'Company_Name': row.Company_Name
            })    

    final = pd.DataFrame(final)
    drops = pd.DataFrame(mistakes)
    wrong_city = pd.DataFrame(notpvd)
    drops.to_csv(dir_dir + '/drops_address.csv', sep = ',', encoding = 'utf-8-sig')
    wrong_city.to_csv(dir_dir + '/wrong_city.csv', sep = ',', encoding = 'utf-8-sig')
    pkl.dump(street_dict, open('street_dict.pkl', 'wb'))

    return final