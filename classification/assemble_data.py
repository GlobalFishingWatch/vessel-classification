import numpy as np
import os
import argparse
from glob import glob



def assemble_fishing_data(src_dir, output_path):
    header = None
    with open(output_path, 'w') as output:
        for pth in glob(os.path.join(src_dir, '*.csv')):
            name = os.path.basename(pth)
            with open(pth) as f:
                lines = f.readlines()
            if header:
                assert header == lines[0]
            else:
                header = lines[0]
                output.write(header)
            output.writelines(lines[1:])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Assemble Fishing Data.')
    # ../../training-data-source/data/time-ranges
    parser.add_argument('--src-dir', required=True, help='source directory for time range files')
    # classification/data/combined_fishing_ranges.csv'
    parser.add_argument('--output-path', required=True, help='location to save generated file to')
    args = parser.parse_args()

    assemble_fishing_data(args.src_dir, args.output_path)
"""
python assemble_data.py --src-dir ../../training-data-source/data/time-ranges --output-path classification/data/combined_fishing_ranges.csv
"""

# TODO: Create Blank vessel_list if absent
# TODO: automatically run augment
# TODO: have augment split items between test and training.
# TODO: remove current vessel lists from git and add instruction on how to create training lists
# using augment.
# Consolidate this and augment into single file with functions (gasp!) and move to classification/data/