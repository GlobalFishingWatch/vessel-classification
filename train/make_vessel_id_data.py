from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import pandas as pd
import assemble_data
from glob import glob


query = """
#StandardSql
# This could still have some duplicates if there are multiple
# things per range.
SELECT 
 vessel_id as mmsi, start_time, end_time, is_fishing
FROM (
SELECT * FROM 
`machine_learning_dev_ttl_120d.combined_fishing_ranges_by_mmsi` a
JOIN
`world-fishing-827.pipe_production_b.position_messages_*` b
ON cast(a.mmsi as string) = b.ssvid
WHERE b.timestamp BETWEEN a.start_time AND a.end_time
)
GROUP BY mmsi, start_time, end_time, is_fishing
"""

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
base_dir = os.path.join(parent_dir, "classification/data")

# '9ce4d61b-394f-c237-1fb9-dd1c724db2c1', 367780410 -- Just wrong.... # Add to training (7/2 - 7/29 at least
# '9ce4d61b-394f-c237-79cd-92491e418fc', 312854000, -- Just wrong.... # Add to training (7/25-7/27)
# 9ce4d61b-394f-c237-79cd-92491e418fc,2017-07-25T12:00:00Z,2017-07-27T12:00:00Z,0.0
# 9ce4d61b-394f-c237-1fb9-dd1c724db2c1,2017-07-02T12:00:00Z,2017-07-29T12:00:00Z,0.0

def load_ranges(path):
    p = os.path.join(path, "combined_fishing_ranges_vessel_id.csv")
    df = pd.read_csv(p)
    return df


map_rel_path = "temp/ssvid_to_multiple_vessel_ids.csv"


def read_ssvid_map(path):
    map_path = os.path.join(path, map_rel_path)
    map_df = pd.read_csv(map_path)
    ssivd_map = {x.ssvid: [] for x in map_df.itertuples()}
    for x in map_df.itertuples():
        ssivd_map[x.ssvid].append(x.vessel_id)
    return ssivd_map


def process_ranges(path, ranges):
    map_path = os.path.join(path, map_rel_path)
    map_df = pd.read_csv(map_path)
    vid_map = {x.vessel_id: x.ssvid for x in map_df.itertuples()}
    label_path = os.path.join(path, "training_classes.csv")
    label_df = pd.read_csv(label_path)
    label_map = {x.mmsi : x.label for x in label_df.itertuples()}

    out_path = os.path.join(path, "fishing_classes_vessel_id.csv")
    with open(out_path, 'w') as f:
        f.write("mmsi,label,length,tonnage,engine_power,split\n")
        for v in sorted(set(ranges.mmsi)):
            split = "Test" if ((hash(v) & 3) == 0) else "Training" 
            label = label_map.get(vid_map.get(v), '')
            f.write("{},{},,,,{}\n".format(v, label, split))

def process_classes(path):
    ssvid_map = read_ssvid_map(path)
    label_path = os.path.join(path, "training_classes.csv")
    label_df = pd.read_csv(label_path)
    relabelled = []
    for x in label_df.itertuples():
        for vid in ssvid_map.get(x.mmsi, []):
            relabelled.append(x._replace(mmsi=vid))
    relabelled_df = pd.DataFrame(relabelled)
    # Creating as above adds an extra index column, which break the input, so remove it now.
    del relabelled_df['Index']
    out_path = os.path.join(path, "training_classes_vessel_id.csv")
    relabelled_df.to_csv(out_path, index=False)


def create_fishing_ranges(path):
    ssvid_map = read_ssvid_map(path)
    in_path = os.path.abspath(os.path.join(path, 'combined_fishing_ranges.csv'))
    out_path = os.path.abspath(os.path.join(path, 'combined_fishing_ranges_vessel_id.csv'))
    print("{} -> {}".format(in_path, out_path))
    df = pd.read_csv(in_path)
    ssvid_map = read_ssvid_map(path)
    relabelled = []
    for x in df.itertuples():
        for vid in ssvid_map.get(x.mmsi, []):
            relabelled.append(x._replace(mmsi=vid))    
    relabelled_df = pd.DataFrame(relabelled)
    # Creating as above adds an extra index column, which break the input, so remove it now.
    del relabelled_df['Index']
    relabelled_df.to_csv(out_path, index=False)


def augment_classes(path):
    input_path = os.path.abspath(os.path.join(path, 'training_classes_vessel_id.csv'))
    output_path = os.path.abspath(os.path.join(path, 'fishing_classes_vessel_id.csv'))
    fishing_range_dir = os.path.join(this_dir, "../../data/time-ranges")
    fishing_range_path = os.path.abspath(os.path.join(base_dir, 'combined_fishing_ranges_vessel_id.csv'))
    false_positive_paths = [x for x in glob(os.path.join(fishing_range_dir, '*.csv')) if os.path.basename(x).startswith('false_')]
    assemble_data.augment_training_list(input_path, output_path, fishing_range_path, false_positive_paths)



def process(path):
    ranges = load_ranges(path)
    process_ranges(path, ranges)
    process_classes(path)
    create_fishing_ranges(path)
    augment_classes(path)


if __name__ == "__main__":
    print("""
    Steps: 
      1. upload combined_fishing_ranges_by_mmsi.csv to big query as 
         machine_learning_dev_ttl_120d.combined_fishing_ranges_by_mmsi

      2. Run the query above

      3. Place the resulting output into `classfication/data/fishing_classes_vessel_id.csv"

      4. Then run this (if you are running this without the above -- start over!)

    """)
    process(base_dir)