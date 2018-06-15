"""

Typical Usage:

python -m train.assemble_data \
        --fishing-range-dir ../training-data-source/data/time-ranges \
        --fishing-range-output-path classification/data/combined_fishing_ranges.csv \
        --training-class-input-path classification/data/training_classes.csv \
        --training-class-output-path classification/data/fishing_classes.csv \
        --force 
"""
import numpy as np
import os
import argparse
from glob import glob
import pandas as pd
import logging
import argparse
from classification.utility import VESSEL_CATEGORIES, TRAINING_SPLIT, TEST_SPLIT


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


def augment_training_list(input_path, output_path, fishing_range_path, false_positive_paths, seed=0):
    np.random.seed(seed)
    logging.basicConfig(level=logging.INFO)
    vessel_list = pd.read_csv(input_path)
    range_mmsi = sorted(set(pd.read_csv(fishing_range_path).mmsi))
    fp_mmsi = set()
    for pth in false_positive_paths:
        fp_mmsi |= set(pd.read_csv(pth)['mmsi'])

    class_name_of = {x.mmsi : x.label for x in vessel_list.itertuples()}.get

    template = vessel_list.iloc[-1].copy()
    template.label = "unknown"
    template.length = np.nan
    template.tonnage = np.nan
    template.engine_power = np.nan
    template.split = 'Training'
    template.source = "fishing-ranges"

    new_list = []
    used_mmsi = set()
    for name in dict(VESSEL_CATEGORIES)['fishing']:
        candidates = [x for x in range_mmsi if class_name_of(x) == name and x not in fp_mmsi]
        if not candidates:
            continue
        np.random.shuffle(candidates)
        n = len(candidates) // 2
        print name, "TEST", n, "TOTAL", len(candidates)
        for i, mmsi in enumerate(candidates):
            assert mmsi not in used_mmsi
            new = template.copy()
            new.mmsi = mmsi
            new.split = TEST_SPLIT if (i < n) else TRAINING_SPLIT
            new.label = name
            new_list.append(new)
        used_mmsi |= set(candidates)

    for mmsi in range_mmsi:
        if mmsi not in used_mmsi:        
            new = template.copy()
            new.mmsi = mmsi
            new.split = TRAINING_SPLIT
            new_list.append(new)

    new_vessel_list = vessel_list.iloc[:0].append(new_list)

    new_vessel_list.to_csv(output_path, index=False)


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
        false_positive_paths = [x for x in glob(os.path.join(args.fishing_range_dir, '*.csv')) if os.path.basename(x).startswith('false_')]
 
        augment_training_list(args.training_class_input_path,
                              args.training_class_output_path,
                              args.fishing_range_output_path,
                              false_positive_paths)

