import glob
import os
import fnmatch

path = '/home/moisio/r-index/KEVO/test'

files = os.listdir(path)
for file in files:
    if '29' in file:
        print(file)

