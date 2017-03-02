import numpy as np
import pandas as pd
from glob import glob


likely = set()
for path in glob("/Users/timothyhochberg/Documents/SkyTruth/Code/treniformis/treniformis/_assets/GFW/FISHING_MMSI/KNOWN_AND_LIKELY/*"):
    with open(path) as f:
        likely |= set([int(x.strip()) for x in f.readlines()])


vessel_list = pd.read_csv("classification/data/training_classes.csv")
range_mmsi = pd.read_csv("classification/data/combined_fishing_ranges.csv")
missing_fishing = (set(range_mmsi['mmsi']) - set(vessel_list['mmsi'])) & likely

template = vessel_list.iloc[-1].copy()
template.is_fishing = "unknown_fishing"
template.length = np.nan
template.tonnage = np.nan
template.engine_power = np.nan
template.split = 'Training'
template.list_sources = "fishing-ranges"

extra = []
for mmsi in sorted(missing_fishing):
    new = template.copy()
    new['mmsi'] = mmsi
    extra.append(new)

if extra:
    print('augmenting')
    extended_vessel_list = vessel_list.append(extra)

    extended_vessel_list.to_csv("classification/data/training_classes.csv", index=False)