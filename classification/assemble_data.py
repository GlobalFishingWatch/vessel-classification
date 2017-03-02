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

# TODO: automatically run augment