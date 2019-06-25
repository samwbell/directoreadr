This is a pipeline for extracting data from city directories


## Installing Packages

This is tested on Python 3.5.3.

**Note:** You will need to install two specific packages, libtesseract and libleptonica, due to the tesserocr package. These are system specific and instructions will vary depending on your OS or distro. If any issues arise, please look here https://github.com/sirfz/tesserocr and Google as necessary.

First start by setting up the Python virtualenv folder:

``` 
    python3 -m venv /path/to/env
```

Then activate the environment by running:

```
    source /path/to/env/bin/activate
```

### MacOS

```
    pip3 install -r requirements_py3_macos.txt
```

Install homebrew here: https://brew.sh/

Then run:

``` 
    brew install leptonica tesseract
```

Then install tesserocr by this command (tested on MacOS 10.14.1)


```CC=clang XCC=clang++ CPPFLAGS="-stdlib=libc++ -DUSE_STD_NAMESPACE -mmacosx-version-min=10.8" pip3 install tesserocr```

### Linux

```
    pip3 install -r requirements_py3_linux.txt
```

Then install the following packages based on your distro. 

For Ubuntu:

``` 
    sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev
```

Then install tesserocr

```
    pip3 install tesserocr
```

### Windows

Follow the instructions given here: https://pypi.org/project/tesserocr/



Once finished, you can deactivate your environment by typing: `deactivate`. 

### Setting up the Brown Geocoder (instructions deprecated)

#### New instructions (to be modified)

Once the env_vars.sh script has been gotten, run it when you activate the environment. That should work at the moment. We are in the process of shifting to an open source geocoder which will change the install instructions. 

#### Older instructions

Within ~/anaconda3/envs/georeg/etc/conda/activate.d (might be anaconda2), create an env_vars.sh script that sets the passwords for the Brown ArcGIS geocoder.  You may have to ask for this file.

Within ~/anaconda3/envs/georeg/etc/conda/deactivate.d (might be anaconda2), create an env_vars.sh script that unsets the passwords for the Brown ArcGIS geocoder.# georeg-pipeline

### Miscellaneous 

You will need to produce a StreetZipCity.csv file for your area.  It can be missing the zipcode data, which are not necessary to the code.

#### Possible TESSDATA problem. 

If the tesseract ocr only works in one window, in that window, run this command:

```` 
python
import tesserocr
print(tesserocr.get_languages())
````

This will result in something like:

````
('/usr/local/share/tessdata/', ["eng", "osd", "snum"])
````

The first path is the one you need to use for TESSDATA_PREFIX environment variable. 

In a new window, before you run the python code, do this:

```` export TESSDATA_PREFIX="/usr/local/share/tessdata" ````

Use whatever path the first command resulted in. 



