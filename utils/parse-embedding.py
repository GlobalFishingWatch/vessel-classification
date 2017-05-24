import json
import numpy as np

width = 19200
labels = ['cargo', 'tanker', 'trawlers', 'sailing', 'set_gillnets', 'pole_and_line', 'motor_passenger', 'reefer', 'tug', 'set_longlines', 'pots_and_traps', 'other_not_fishing', 'gear', 'squid_jigger', 'seismic_vessel', 'purse_seines', 'drifting_longlines', 'other_fishing', 'trollers']
label_to_idx = {label: idx for idx, label in enumerate(labels)}
idx_to_label = {idx: label for idx, label in enumerate(labels)}

count = 0
with open("vessels_gear_embedding.json") as f:
   line = f.readline()
   while line:
      count += 1
      line = f.readline()


data = np.zeros((count, width + 7 + len(labels)))
idx = 0
with open("vessels_gear_embedding.json") as f:
   line = f.readline()
   while line:
      line = json.loads(line)

      for colidx, col in enumerate(line['embedding']['embedding']):
         data[idx, colidx] = col

      data[idx, width+0] = line['engine_power']['value']
      data[idx, width+1] = line['length']['value']
      data[idx, width+2] = line['tonnage']['value']

      data[idx, width+3] = line['mmsi']

      data[idx, width+4] = label_to_idx[line['Multiclass' ]['max_label']]
      data[idx, width+5] = line['Multiclass' ]['max_label_probability']

      for label, value in line['Multiclass' ]['label_scores'].iteritems():
         data[idx, width+6 + label_to_idx[label]] = value

      line = f.readline()
      idx += 1

np.savez_compressed("vessels_gear_embedding.npz", data = data)
