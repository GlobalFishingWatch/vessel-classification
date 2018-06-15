import os
import pandas as pd
import assemble_data
from glob import glob

this_dir = os.path.dirname(__file__)
base_dir = os.path.join(this_dir, "../classification/data")

training = 'training_classes'
fishing = 'fishing_classes'
combined = 'combined_fishing_ranges'

map_path = os.path.abspath(os.path.join(this_dir, "ssvid_to_vessel_id.csv"))
map_df = pd.read_csv(map_path)
id_map = {x.ssvid: x.vessel_id for x in map_df.itertuples()}

# Create combined fishing classes with vessel id
in_path = os.path.abspath(os.path.join(base_dir, combined + '.csv'))
out_path = os.path.abspath(os.path.join(base_dir, combined + '_vessel_id.csv'))
print("{} -> {}".format(in_path, out_path))
df = pd.read_csv(in_path)
mask = [(x in id_map) for x in df.mmsi]
df = df[mask]
mmsi = [id_map[x] for x in df.mmsi]
df.mmsi = mmsi
df.to_csv(out_path, index=False)


# Augment training list
input_path = os.path.abspath(os.path.join(base_dir, training + '_vessel_id.csv'))
output_path = os.path.abspath(os.path.join(base_dir, fishing + '_vessel_id.csv'))
fishing_range_dir = os.path.join(this_dir, "../../data/time-ranges")
fishing_range_path = os.path.abspath(os.path.join(base_dir, combined + '_vessel_id.csv'))
false_positive_paths = [x for x in glob(os.path.join(fishing_range_dir, '*.csv')) if os.path.basename(x).startswith('false_')]


print("=>", output_path)
assemble_data.augment_training_list(input_path, output_path, fishing_range_path, false_positive_paths)





