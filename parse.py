import time
mt1 = time.time()
import stringParse, arcgeocoder, address
import streetMatch1
import sys, glob, os, re, datetime
import pandas as pd
import numpy as np
import cv2
import pickle as pkl
from PIL import Image

# necessary for using tesserocr
import locale
locale.setlocale(locale.LC_ALL, 'C')
from tesserocr import PyTessBaseAPI, RIL
import multiprocessing
import json
from fuzzywuzzy import fuzz, process
from header_match import generate_dict, match_headers


#This is the driver script for pulling the data out of the images, parsing them, matching them, and geocoding them.

dir_dir = ""

def naturalSort(String_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', String_)]

"""
turns a pandas dataframe into a csv. strips various characters off.
writes to FOutput.csv 

Args:
    dataFrame: the pandas datafram
"""
def makeCSV(dataFrame):
    # creates the csv FOutput
    today = datetime.date.today()
    dataFrame.set_index('Query')
    dataFrame['Address - From Geocoder'] = dataFrame['Address - From Geocoder'].astype('str').str.rstrip(',').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
    dataFrame['Company_Name'] = dataFrame['Company_Name'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
    dataFrame['File_List'] = dataFrame['File_List'] #.apply(lambda paths: [path.rpartition('/')[2] for path in paths[0]]).astype('str')
    dataFrame['Header'] = dataFrame['Header'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]').str.lstrip('>')
    dataFrame['Text'] = dataFrame['Text'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
    dataFrame['Query'] = dataFrame['Query'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
    dataFrame['Latitude'] = dataFrame['Latitude'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
    dataFrame['Longitude'] = dataFrame['Longitude'].astype('str').str.strip('[[]]').str.lstrip('u\'').str.rstrip('\'').str.strip('[\\n ]')
    dataFrame.to_csv(dir_dir + '/FOutput.csv', sep = ',', encoding = 'utf-8-sig')


def local_search(df, location_dict):
    return df

def dfProcess(dataFrame, params):
    # this processes the dataframe to match streets and geocode
    print('Matching city and street...')
    t1 = time.time()
    # street matching
    frame = streetMatch1.streetMatcher(dataFrame, dir_dir)
    t2 = time.time()
    print('Done in: ' + str(round(t2-t1, 3)) + ' s')
    print('Geocoding...')
    # Geocoding
    t1 = time.time()
    #frame.to_pickle('frame.pkl')
    #frame = pd.read_pickle('frame.pkl')
    if params["geocode"] == True:
        fDF = arcgeocoder.geocode(frame, dir_dir)
    else:
        fDF = local_search(frame, location_dict)

    #print(str(len(fDF)) + ' addresses')
    t2 = time.time()
    print('Done in: ' + str(round(t2-t1, 3)) + ' s')
    return fDF

"""
gets horizontal histogram of the number of white pixels in each column
"""
def getHorzHist(image):
    height, width = image.shape[:2]
    i=0
    histogram = [0]*width
    while i<width:
        histogram[i] = height - cv2.countNonZero(image[:, i])
        # print(cv2.countNonZero(image[:, i]))
        i=i+1
    return histogram

"""
returns the location of the first black pixel in the image from the left
"""
def getFBP(image_file, sf):
    # Gets the first black pixel
    fbp_thresh = 3
    im = cv2.imread(image_file, 0)
    h,w = im.shape[:2]
    hlow = int(float(h)*0.25)
    hhigh = int(float(h)*0.75)
    hhist = getHorzHist(im[hlow:hhigh,:])
    histstr = ','.join([str(li) for li in hhist])
    strpart = histstr.partition('0,')
    listStringPart = strpart[2].split(',')
    listIntPart = list(map(int, listStringPart))
    i=0
    while ((listIntPart[min(i,len(listIntPart)-1)] < fbp_thresh) or (listIntPart[min(i+2,len(listIntPart)-1)] < fbp_thresh)) and (i < len(listIntPart)):
        i+=1
    blackindx = i
    cut = len(strpart[0].split(',')) + len(strpart[1].split(','))
    firstBlackPix = cut + blackindx - fbp_thresh
    return sf*float(firstBlackPix)

# returns the number of alphabetic chars in the string
def count_alpha(text):
    return len([l for l in str(text) if l.isalpha()])

# returns the number of alphanumeric chars in the string
def count_alnum(text):
    return len([l for l in str(text) if l.isalnum()])

# returns the number of uppercase chars in the string
def count_upper(text):
    return len([l for l in str(text) if l.isupper()])

"""
Determines if the text is a header entry

Algorithm is as follows:
        1954 or under: 
                - no capitalization
                - indented
                - starts with star and indented
        1955-1962:
                - there are letters in it
                - indented
                - indented a bit and 90%+ capital letters
                - less than 3 entries and w h a t
                - starts with * and lightly indented
        1964: 
                - has letters
                - indented
                - starts with star and indented

        ... and so on.
"""
def is_header(fbp, text, file, entry_num):
    year = int(file.partition('/')[0].lstrip('cd'))
    # divides logic by year
    if year <= 1954:
        if int(count_alpha(text.strip())) == 0:
            return False
        elif (fbp > 40):
            return True
        elif (text.lstrip()[0] == '*') and (fbp > 30):
            return True
        else:
            return False
    elif year <= 1962:
        if len([l for l in text if l.isalpha()]) == 0:
            return False
        elif (fbp > 40):
            return True
        elif (fbp > 35) and ((float(count_upper(text))/float(count_alpha(text))) > 0.9):
            return True
        elif (entry_num < 3) and ((float(count_upper(text))/float(count_alpha(text))) > 0.95):
            return True
        elif (text.lstrip()[0] == '*') and (fbp > 30):
            return True
        else:
            return False
    elif year == 1964:
        if int(count_alpha(text)) == 0:
            return False
        elif (fbp > 40):
            return True
        elif (text.lstrip()[0] == '*') and (fbp > 30):
            return True
        else:
            return False
    elif year <= 1968:
        if int(count_alpha(text)) == 0:
            return False
        elif (fbp > 40):
            return True
        elif (fbp > 30) and (count_upper(text)/count_alnum(text) > 0.9):
            return True
        elif (entry_num < 3) and (fuzz.partial_ratio(text.partition('-')[2], 'Contd') >= 80):
            return True
        elif (text.lstrip()[0] == '*') and (fbp > 30):
            return True
        else:
            return False
    elif year <= 1990:
        if int(count_alpha(text)) == 0:
            return False
        elif (fbp > 22) and (count_upper(text)/count_alnum(text) > 0.9):
            return True
        elif (entry_num < 3) and ((fuzz.partial_ratio(text.partition('-')[2], 'Contd') >= 80) or (count_upper(text)/count_alnum(text) > 0.95)):
            return True
        else:
            return False
    else:
        if int(count_alpha(text)) == 0:
            return False
        elif (fbp > 22) and (count_upper(text)/count_alnum(text) > 0.9):
            return True
        elif (entry_num < 3) and ((fuzz.partial_ratio(text.partition('-')[2], 'Contd') >= 80) or (count_upper(text)/count_alnum(text) > 0.95)):
            return True
        else:
            return False

"""
Performs the ocr for an entry

returns:
        the file that was ocr'd
        the text outputted by ocr
        the first black pixel in the image
        the scale factor for the entry
        the entry's number (in terms of the number of entries in a page)
"""
def ocr_file(file, api):
    
    image = Image.open(file)
    api.SetImage(image)
    api.SetVariable("tessedit_char_whitelist", "()*,'&.;-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    boxes = api.GetComponentImages(RIL.TEXTLINE, True)
    outStr = api.GetUTF8Text()
    text = outStr
    im = cv2.imread(file, 0)
    width = im.shape[1]
    sf = float(width)/float(2611)
    fbp = getFBP(file, sf)
    entry_num = int(file.rpartition('_')[2].rpartition('.png')[0])
    return file,text,fbp,sf,entry_num


"""
processes the OCR in chunks to avoid having to reload the API each time.
"""
def chunk_process_ocr(chunk_files):
    rlist = []
    with PyTessBaseAPI(lang="eng") as api:
        for file in chunk_files:
            #print(file)
            rlist.append(ocr_file(file, api))
    return rlist


"""
Given a list of strings and a threshold length, it concatenates strings such that small strings join big strings
"""
def join_small_str(parts, thresh):
    root = None
    joined = []
    for i in range(len(parts)):
        small = len(parts[i]) < thresh
        if root and small:
            root = root + " " + parts[i]
        elif root and not small:
            if len(root) < thresh:
                root = root + " " + parts[i]
                joined.append(root)
                root = None
            else:
                joined.append(root)
                root = parts[i]
        else:
            root = parts[i]
    if root:
        joined.append(root)
    return joined


"""
handles newlines in the entries by splitting if the newline occurs part way thru the string
"""
def replace_newline(text):
    if "\\n" in text:
        parts = [part for part in join_small_str(re.split(r"\\n", text), 15) if len(part) != 0]
    elif "\n" in text:
        parts = [part for part in join_small_str(re.split(r"\n", text), 15) if len(part) != 0]
    else:
        parts = [text]
    return parts


"""
Main processing/driver script
"""
def process_data(folder, params):
        
    do_OCR = params['do_ocr']
    make_table = params['make_table']
    #Make the zip code to city lookup table
    if make_table:
            streetTable()

    # only triggers for single image runs
    if do_OCR and 'img' in params:
        file_list = sorted(glob.glob(folder +"/" +  params['img'] + "*.png"), key = naturalSort)
        texts = []
        first_black_pixels = []
        sfs = []
        entry_nums = []
        flat_ocr_results = []

        # do ocr for each entry, append resulting tuple to flat_ocr+results
        with PyTessBaseAPI(lang='eng') as api:
            for file in file_list:
                flat_ocr_results.append(ocr_file(file, api))

        # turn it into a pandas df
        single_raw_data = pd.DataFrame(flat_ocr_results, columns = ['file','text','first_black_pixel','sf','entry_num'])
        
        # add the image to the raw_data pkl file
        raw_data = pd.read_pickle(dir_dir + '/raw_data.pkl')
        raw_data = pd.concat([raw_data[~raw_data.file.isin(file_list)], single_raw_data], ignore_index = True)
        raw_data.to_pickle(dir_dir + '/raw_data.pkl')

    elif do_OCR:
        files = []
        texts = []
        first_black_pixels = []
        sfs = []
        entry_nums = []
        print('Doing OCR')
        t1 = time.time()
        file_list = sorted(glob.glob(folder.rstrip('/') + '/*.png'), key = naturalSort)
        print('Processing ' + str(len(file_list)) + ' files...')

        # performs the ocr on all entries, appends tuple results to flat_ocr_results
        if params['do_multiprocessing']:
            pool = multiprocessing.Pool(params['pool_num'])
            chunk_size = min(max(int(len(file_list)/50.0), 1), 20)
            chunk_list = [file_list[i:i + chunk_size] for i in list(range(0, len(file_list), chunk_size))]
            ocr_results = pool.map(chunk_process_ocr, chunk_list)
            flat_ocr_results = [item for sublist in ocr_results for item in sublist]
        else:
            flat_ocr_results = []
            with PyTessBaseAPI(lang='eng') as api:
                for file in file_list:
                    print(file)
                    flat_ocr_results.append(ocr_file(file, api))

        # writes results to the raw data pkl file
        raw_data = pd.DataFrame(flat_ocr_results, columns = ['file','text','first_black_pixel','sf','entry_num'])
        t2 = time.time()
        print('Done in: ' + str(round(t2-t1, 3)) + ' s')
        print('Saving...')
        t1 = time.time()
        raw_data.to_pickle(dir_dir + '/raw_data.pkl')
        t2 = time.time()
        print('Done in: ' + str(round(t2-t1, 3)) + ' s')
    
    # doesn't perform ocr, just reads the data in from the pkl
    else:
        print('Reading raw data from raw_data.pkl...')
        t1 = time.time()
        raw_data = pd.read_pickle(dir_dir + '/raw_data.pkl')
        t2 = time.time()
        print('Done in: ' + str(round(t2-t1, 3)) + ' s')

    # finds the first entry for each page and builds an index list based on that
    print('Concatenating entries...')
    t1 = time.time()

    page_breaks = raw_data[raw_data['entry_num'] == 1].index.tolist()
    ilist = list(range(0,raw_data.shape[0]))


    # dict where the key is each page's index, val is the # of entries on that page
    page_break = {i:max([num for num in page_breaks if i>=num]) for i in ilist}
    # dict where key = pg index, val = first black pixel locale
    fbp_dict = {index:value for index,value in raw_data['first_black_pixel'].iteritems()}

    # get relative first black pixel- so like accounts for page slanting
    def get_relative_fbp(i):
        pbi = page_break[i]
        if i <= pbi + 8:
            rval = fbp_dict[i] - min([fbp_dict[j] for j in list(range(i,min(i+10,len(fbp_dict)-1)))])
        else:
            rval = fbp_dict[i] - min([fbp_dict[j] for j in list(range(i-8,min(i+2,len(fbp_dict)-1)))])
        return rval

    # adds new relative fbp column to the dataframe
    # adds new is_header column to the df which is true if an entry is a header
    raw_data = raw_data.assign(relative_fbp = [get_relative_fbp(i) for i in ilist])
    raw_data = raw_data.assign(is_header = raw_data.apply(lambda row: is_header(row['relative_fbp'], row['text'], row['file'], row['entry_num']), axis=1))
    is_header_dict = {index:value for index,value in raw_data['is_header'].iteritems()}
    entry_num_dict = {index:value for index,value in raw_data['entry_num'].iteritems()}

    raw_data_length = raw_data.shape[0]
    
    #determines if subsequent header lines should be concantenated into one
    def concatenateQ(i):
        if i==raw_data_length - 1:
            return False
        elif i==0 and is_header_dict[i]:
            return False
        elif is_header_dict[i] and (not is_header_dict[i-1]):
            return False
        elif is_header_dict[i] and is_header_dict[i-1]:
            return True
        elif (not is_header_dict[i]) and is_header_dict[i+1]:
            return False
        elif (not is_header_dict[i]) and (entry_num_dict[i+1] == 1):
            return False
        elif raw_data.iloc[i+1]['relative_fbp'] > 9.0:
            return True
        else:
            return False

    # adds new cq column to df
    raw_data = raw_data.assign(cq = raw_data.index.map(concatenateQ))

    # saves raw data as a csv
    raw_data.to_csv(dir_dir + '/raw_data.csv', encoding = 'utf-8-sig')
    
    file_lists = []
    file_list = []
    texts = []
    text = ''
    headers = []
    header = ''
    cq_dict = {index:value for index,value in raw_data['cq'].iteritems()}
    text_dict = {index:value for index,value in raw_data['text'].iteritems()}
    file_dict = {index:value for index,value in raw_data['file'].iteritems()}
    
    for index in raw_data.index:
        #raw_row = raw_data.iloc[i]
        row_text = text_dict[index]
        cq = cq_dict[index]
        file = file_dict[index]

        # concat headers
        if is_header_dict[index]:
            if cq:
                header += ' ' + row_text.strip()
                #print(header)
            else:
                header = row_text.strip()

        # if it's a page number
        elif entry_num_dict[index] == 1 and row_text == file.rpartition('_Page_')[2].rpartition(' ')[0]:
            pass
        
        # concat
        elif cq:
            file_list.append(file)
            text += ' ' + row_text.strip()
        # add the headers, text, files to their lists
        else:
            file_list.append(file)
            text += ' ' + row_text.strip()
            file_lists.append(file_list)
            headers.append(header)
            texts.append(text.strip())
            file_list = []
            text = ''

    # throw it all into a dataframe
    data = pd.DataFrame(data={'Header':headers, 'Text':texts, 'File_List':file_lists})
    
    t2 = time.time()
    print('Done in: ' + str(round(t2-t1, 3)) + ' s')

    print('Matching headers...')
    t1 = time.time()

    # try to load pickle file for headers, otherwise make it yourself
    try:
        raw_header_match_dict = pkl.load(open("header_match_dict.pkl", 'rb'))
    except:
        raw_header_match_dict = {}
    true_headers = list(pd.read_csv("true_headers.csv")['Headers'].dropna())
    header_match_dict = generate_dict(data, true_headers, raw_header_match_dict)
    print('match dict built')
    
    # see if you can match headers
    # matched, match_failed, all_headers = match_headers(data, header_match_dict)
    data = match_headers(data, header_match_dict)
    

    t2 = time.time()
    print('Done in: ' + str(round(t2-t1, 3)) + ' s')

    print('Writing data to data.csv...')
    t1 = time.time()
    data.to_csv(dir_dir + '/data.csv', encoding = 'utf-8-sig')
    t2 = time.time()
    print('Done in: ' + str(round(t2-t1, 3)) + ' s')


    print('Expanding newlines...')
    t1 = time.time()
    expanded_data_list = []
    for index,row in data.iterrows():
        parts = replace_newline(row['Text'])
        new_row = row.copy()
        for text_part in parts:
            new_row['Text'] = text_part
            expanded_data_list.append(new_row.copy())
    data = pd.DataFrame(expanded_data_list)
    t2 = time.time()
    print('Done in: ' + str(round(t2-t1, 3)) + ' s')

    print('Parsing text...')
    t1 = time.time()
    if params['do_multiprocessing']:
        pool = multiprocessing.Pool(params['pool_num'])
        search_list = [(i, params['stringParse']) for i in data['Text'].tolist()]
        output_tuples = pool.map(stringParse.search, search_list)
    else:
        output_tuples = [stringParse.search(search_text) for search_text in data['Text'].tolist()]
    
    # add streets and company names to the datagrame
    #streets,company_names = zip(*output_tuples)
    streets = [output_tuple[0] for output_tuple in output_tuples]
    company_names = [output_tuple[1] for output_tuple in output_tuples]
    data = data.assign(Street=streets, Company_Name=company_names)
    t2 = time.time()
    print('Done in: ' + str(round(t2-t1, 3)) + ' s')


    # make sure that row_streets isn't a list of streets
    # if it is, every street gets its own row
    print('Expanding...')
    t1 = time.time()
    #data_list = [row for row in data.iterrows()]
    expanded_data_list = []
    for index,row in data.iterrows():
        if type(row['Street']) == list:
            row_streets = row['Street']
            new_row = row.copy()
            for street in row_streets:
                new_row['Street'] = street
                expanded_data_list.append(new_row.copy())
        else:
                expanded_data_list.append(row)
    data = pd.DataFrame(expanded_data_list)
    t2 = time.time()
    print('Done in: ' + str(round(t2-t1, 3)) + ' s')

    print('Matching city and street and geocoding...')
    t2 = time.time()
    result = dfProcess(data, params)
    t2 = time.time()
    print('Collective runtime: ' + str(round(t2-t1, 3)) + ' s')

    # save to csv
    if not result.empty:
        print('Saving to FOutput.csv...')
        t1 = time.time()
        makeCSV(result)
        t2 = time.time()
        print('Done in: ' + str(round(t2-t1, 3)) + ' s')


"""
takes in input params, starts off the parse process according to those params
"""
def main(inputParams):
    global dir_dir
    dir_dir = "./" + inputParams['year_folder']
    
    # runs on single images if specificed in params
    if inputParams['image_process']['single_image']:
        inputParams['parse']['img'] = inputParams['image_process']['img_name']

    process_data(inputParams['year_folder'] + '/entry', inputParams['parse'])

    mt2 = time.time()
    print('Full runtime: ' + str(round(mt2-mt1, 3)) + ' s')


"""
for terminal input
"""
if __name__ == '__main__':
    if not sys.argv[1]:
        raise Exception('You need to input a parameters file. try inputParams.json.')
    inputParams = str(sys.argv[1])
    with open(inputParams) as json_data:
        d = json.load(json_data)
    main(d)
