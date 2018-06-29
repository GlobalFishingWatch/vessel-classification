from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import pandas as pd


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
    print("""
    Steps: 
      1. upload combined_fishing_ranges_by_mmsi.csv to big query as 
         machine_learning_dev_ttl_120d.combined_fishing_ranges_by_mmsi

      2. Run the query above

      3. Place the resulting output into `classfication/data/fishing_classes_vessel_id.csv"

      4. Then run this (if you are running this without the above -- start over!)

    """)
    process(base_dir)