from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import os
from collections import defaultdict

"""
#StandardSql
# This could still have some duplicates if there are multiple
# things per range.
SELECT 
 vessel_id as mmsi, start_time, end_time, is_fishing
FROM (
SELECT * FROM 
`machine_learning_dev_ttl_120d.combined_fishing_ranges_by_mmsi` a
JOIN
`world-fishing-827.pipe_staging_a.position_messages_*` b
ON a.mmsi = b.ssvid
WHERE b.timestamp BETWEEN a.start_time AND a.end_time
)
GROUP BY mmsi, start_time, end_time, is_fishing
"""

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
base_dir = os.path.join(parent_dir, "classification/data")

def load_ranges(path):
    p = os.path.join(path, "combined_fishing_ranges_vessel_id.csv")
    df = pd.read_csv(p)
    return df

# def process_fishing(path, idmap):
#     # TODO: we should really keep TEST / TRAIN from crossing between vessels with the same MMIS
#     # even if they have different VESSEL_IDS
#     in_path = os.path.join(path, "combined_fishing_ranges.csv")
#     out_path = os.path.join(path, "combined_fishing_vessel_id.csv")
#     active_vids = set()
#     missing = set()
#     initial_rows = 0
#     transcriped_rows = 0
#     with open(out_path, 'w') as f:
#         f.write("mmsi,start_time,end_time,is_fishing\n")
#         for x in pd.read_csv(in_path).itertuples():
#             initial_rows += 1
#             _, mmsi, start_time, end_time, is_fishing = x
#             if mmsi not in idmap:
#                 if mmsi not in missing:
#                     print(mmsi, "missing; skipping")
#                     missing.add(mmsi)
#                 continue
#             transcriped_rows += 1
#             for vid in idmap[mmsi]:
#                 f.write(','.join([str(x) for x in (vid, start_time, end_time, is_fishing)]) + '\n')
#                 active_vids.add(vid)
#     print("Transcribed", transcriped_rows, "out of", initial_rows)
#     return sorted(active_vids)

def process_ranges(path, ranges):
    out_path = os.path.join(path, "fishing_classes_vessel_id.csv")
    with open(out_path, 'w') as f:
        f.write("mmsi,label,length,tonnage,engine_power,split\n")
        for v in sorted(set(ranges.mmsi)):
            split = "Test" if ((hash(v) & 3) == 0) else "Training" 
            f.write("{},,,,,{}\n".format(v, split))




def process(path):
    ranges = load_ranges(path)
    process_ranges(path, ranges)


if __name__ == "__main__":
    process(base_dir)