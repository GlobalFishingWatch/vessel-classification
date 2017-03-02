import numpy as np
import os
from glob import glob

this_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.abspath(os.path.join(this_dir, "../.."))



src_path = os.path.join(code_dir, 
    'training-data-source/data/time-ranges')

dest_path = os.path.join(this_dir, 'classification/data/combined_fishing_ranges.csv')

fp_upweight = 100

header = None
with open(dest_path, 'w') as output:
    for pth in glob(os.path.join(src_path, '*.csv')):
        with open(pth) as f:
            lines = f.readlines()
        if header:
            assert header == lines[0]
        else:
            header = lines[0]
            output.write(header)
        count = fp_upweight if pth.endswith('false_positives.csv') else 1
        print('using weight {} for {}'.format(count, pth))
        for i in range(count):
            output.writelines(lines[1:])

# TODO: add flag to select source directory and refactor.
# TODO: Create Blank vessel_list if absent
# TODO: automatically run augment
# TODO: have augment split items between test and training.
# TODO: remove current vessel lists from git and add instruction on how to create training lists
# using augment.
# Consolidate this and augment into single file with functions (gasp!) and move to classification/data/