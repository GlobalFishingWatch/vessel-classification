import numpy as np
import os
import argparse
from glob import glob
import pandas as pd
import logging
import argparse


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


def augment_training_list(input_path, output_path, fishing_range_path):
    logging.basicConfig(level=logging.INFO)
    vessel_list = pd.read_csv(input_path)
    range_mmsi = pd.read_csv(fishing_range_path)
    missing_fishing = (set(range_mmsi['mmsi']) - set(vessel_list['mmsi']))

    template = vessel_list.iloc[-1].copy()
    template.label = "unknown"
    template.length = np.nan
    template.tonnage = np.nan
    template.engine_power = np.nan
    template.split = 'Training'
    template.source = "fishing-ranges"

    extra = []
    for mmsi in sorted(missing_fishing):
        new = template.copy()
        new['mmsi'] = mmsi
        extra.append(new)

    if extra:
        logging.info('adding %s new vessels', len(extra))
        extended_vessel_list = vessel_list.append(extra)

    extended_vessel_list.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assemble Fishing Data.')
    # ../../training-data-source/data/time-ranges
    parser.add_argument(
        '--fishing-range-dir',
        required=True,
        help='source directory for time range files')
    # classification/data/combined_fishing_ranges.csv'
    parser.add_argument(
        '--fishing-range-output-path',
        required=True,
        help='location to save generated file to')
    parser.add_argument(
        '--training-class-input-path',
        default=None,
        help='path to training classes to augment')
    parser.add_argument(
        '--training-class-output-path',
        default=None,
        help='output path for augmented training classes')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.fishing_range_output_path) and not args.force:
        print(
            'fishing ranges already exist, exiting (use `--force` to overwrite)')
        raise SystemExit()

    assemble_fishing_data(args.fishing_range_dir,
                          args.fishing_range_output_path)

    if args.training_class_output_path is None:
        logging.info('no `training-class-output-path` skipping augmentation')
    else:
        if not args.training_class_input_path:
            print('`training-class-input-path` required for augmentation')
            raise SystemExit()
        if os.path.exists(args.training_class_output_path) and not args.force:
            print(
                'classes already exist, exiting (use `--force` to overwrite)')
            raise SystemExit()
        augment_training_list(args.training_class_input_path,
                              args.training_class_output_path,
                              args.fishing_range_output_path)

    # vessel_list = pd.read_csv("classification/data/training_classes.csv")
"""
python assemble_data.py \
        --fishing-range-dir ../../training-data-source/data/time-ranges \
        --fishing-range-output-path classification/data/combined_fishing_ranges.csv \
        --training-class-input-path classification/data/training_classes_base.csv \
        --training-class-output-path classification/data/training_classes.csv \
        --force 
"""

# TODO: remove current vessel lists from git and add instruction on how to create training lists
# using augment.
