# Copyright 2017 Google Inc. and Skytruth Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, namedtuple
import csv
import datetime
import dateutil.parser
import pytz
import logging
import os
import sys
import tensorflow as tf
import yaml
import numpy as np

# Upweight false positives to strongly discourage transits
FALSE_POSITIVE_UPWEIGHT = 1
MAX_UPWEIGHT = 100
""" The main column for vessel classification. """
PRIMARY_VESSEL_CLASS_COLUMN = 'label'

#TODO: (bitsofbits) think about extracting to config file

# The 'real' categories for multihotness are the fine categories, which 'coarse' and 'fishing' 
# are defined in terms of. Any number of coarse categories, even with overlapping values can 
# be defined in principle, although at present the interaction between the mulithot and non multihot
# versions makes that more complicated.

schema = yaml.load('''
unknown:
  non_fishing:
    passenger:
    gear:
    fish_factory:
    cargo_or_tanker:
      bunker_or_tanker:
        bunker:
        tanker:
      cargo_or_reefer:
        cargo:
        reefer:
          specialized_reefer:
          container_reefer:
        fish_tender:
          well_boat:
    patrol_vessel:
    research:
    dive_vessel:
    submarine:
    dredge_non_fishing:
    supply_vessel:
    tug:
    seismic_vessel:
    helicopter:
    other_not_fishing:

  fishing:
    squid_jigger:
    drifting_longlines:
    pole_and_line:
    other_fishing:
    trollers:
    fixed_gear:
      pots_and_traps:
      set_longlines:
      set_gillnets:
    trawlers:
    dredge_fishing:
    seiners:
     purse_seines:
      tuna_purse_seines:
      other_purse_seines:
     other_seines:
    driftnets:
''')


def atomic(obj):
    for k, v in obj.items():
        if v is None:
            yield k
        else:
            for x in atomic(v):
                yield x

def categories(obj, include_atomic=True):
    for k, v in obj.items():
        if v is None:
            if include_atomic:
                yield k, [k]
        else:
            yield (k, list(atomic(v)))
            for x in categories(v, include_atomic=include_atomic):
                yield x




#TODO: Better names
VESSEL_CLASS_DETAILED_NAMES = sorted(atomic(schema))

VESSEL_CATEGORIES = sorted(categories(schema))

TEST_SPLIT = 'Test'
TRAINING_SPLIT = 'Training'

FishingRange = namedtuple('FishingRange',
                          ['start_time', 'end_time', 'is_fishing'])


def int_or_hash(x):
    try:
        return int(x)
    except:
        return hash(x)

class VesselMetadata(object):
    def __init__(self,
                 metadata_dict,
                 fishing_ranges_map,
                 fishing_range_training_upweight=1.0):
        self.metadata_by_split = metadata_dict
        self.metadata_by_mmsi = {}
        self.fishing_ranges_map = fishing_ranges_map
        self.fishing_range_training_upweight = fishing_range_training_upweight
        for split, vessels in metadata_dict.iteritems():
            for mmsi, data in vessels.iteritems():
                self.metadata_by_mmsi[mmsi] = data
        self.mmsi_map_int2str = {int_or_hash(k) : k for k in self.metadata_by_mmsi}


        intersection_mmsis = set(self.metadata_by_mmsi.keys()).intersection(
            set(fishing_ranges_map.keys()))
        logging.info("Metadata for %d mmsis.", len(self.metadata_by_mmsi))
        logging.info("Fishing ranges for %d mmsis.", len(fishing_ranges_map))
        logging.info("Vessels with both types of data: %d",
                     len(intersection_mmsis))

    def vessel_weight(self, mmsi):
        if mmsi in self.fishing_ranges_map:
            fishing_range_multiplier = self.fishing_range_training_upweight
        else:
            fishing_range_multiplier = 1.0
        return self.metadata_by_mmsi[mmsi][1] * fishing_range_multiplier

    def vessel_label(self, label_name, mmsi):
        return self.metadata_by_mmsi[mmsi][0][label_name]

    def mmsis_for_split(self, split):
        assert split in [TRAINING_SPLIT, TEST_SPLIT]
        # Check to make sure we don't have leakage
        assert not (set(self.metadata_by_split[TRAINING_SPLIT].keys()) &
                    set(self.metadata_by_split[TEST_SPLIT].keys())
                    ), 'mmsi in both training and test split'
        return self.metadata_by_split[split].keys()

    def weighted_training_list(self,
                               random_state,
                               split,
                               max_replication_factor,
                               row_filter=lambda row: True,
                               boundary=1):
        replicated_mmsis = []
        logging.info("Training mmsis: %d", len(self.mmsis_for_split(split)))
        fishing_ranges_mmsis = []
        for mmsi, (row, weight) in self.metadata_by_split[split].iteritems():
            if row_filter(row):
                if mmsi in self.fishing_ranges_map:
                    fishing_ranges_mmsis.append(mmsi)
                    weight = weight * self.fishing_range_training_upweight  # TODO: rip this out.

                weight = min(weight, max_replication_factor)

                int_n = int(weight)
                replicated_mmsis += ([mmsi] * int_n)
                frac_n = weight - float(int_n)
                if (random_state.uniform(0.0, 1.0) <= frac_n):
                    replicated_mmsis.append(mmsi)
        missing = (-len(replicated_mmsis)) % boundary
        if missing:
            replicated_mmsis = np.concatenate(
                [replicated_mmsis,
                 np.random.choice(replicated_mmsis, missing)])
        random_state.shuffle(replicated_mmsis)
        logging.info("Replicated training mmsis: %d", len(replicated_mmsis))
        logging.info("Fishing range mmsis: %d", len(fishing_ranges_mmsis))

        return replicated_mmsis

    def fishing_range_only_list(self, random_state, split,
                                max_replication_factor):
        replicated_mmsis = []
        fishing_mmsi_set = set(
            [k for (k, v) in self.fishing_ranges_map.items() if v])
        fishing_range_only_mmsis = [mmsi
                                    for mmsi in self.mmsis_for_split(split)
                                    if mmsi in fishing_mmsi_set]
        logging.info("Fishing range training mmsis: %d / %d",
                     len(fishing_range_only_mmsis),
                     len(self.mmsis_for_split(split)))
        for mmsi in fishing_range_only_mmsis:
            weight = min(self.vessel_weight(mmsi), max_replication_factor)
            assert mmsi in self.fishing_ranges_map
            if weight == 0:
                logging.info('skipping %s due to zero weight', mmsi)
                continue
            int_n = int(weight)
            logging.info(
                "mmis: %s, max_repl_factor: %s, weight: %s, int_n: %s", mmsi,
                max_replication_factor, self.vessel_weight(mmsi), int_n)
            replicated_mmsis += ([mmsi] * int_n)
            frac_n = weight - float(int_n)
            if (random_state.uniform(0.0, 1.0) <= frac_n):
                replicated_mmsis.append(mmsi)

        random_state.shuffle(replicated_mmsis)
        logging.info("Replicated training mmsis: %d", len(replicated_mmsis))

        return replicated_mmsis


def read_vessel_time_weighted_metadata_lines(available_mmsis, lines,
                                             fishing_range_dict):
    """ For a set of vessels, read metadata; use flat weights

    Args:
        available_mmsis: a set of all mmsis for which we have feature data.
        lines: a list of comma-separated vessel metadata lines. Columns are
            the mmsi and a set of vessel type columns, containing at least one
            called 'label' being the primary/coarse type of the vessel e.g.
            (Longliner/Passenger etc.).
        fishing_range_dict: dictionary of mapping mmsi to lists of fishing ranges

    Returns:
        A VesselMetadata object with weights and labels for each vessel.
    """

    metadata_dict = defaultdict(lambda: {})

    # Build a list of vessels + split + and vessel type. Calculate the split on
    # the fly, but deterministically.
    min_time_per_mmsi = np.inf

    for row in lines:
        mmsi = row['mmsi'].strip()
        if mmsi in available_mmsis:
            if mmsi not in fishing_range_dict:
                continue
            # Is this mmsi included only to supress false positives
            # Symptoms; fishing score for this MMSI never different from 0
            is_false_positive = True
            split = row['split']
            if split not in ('Training', 'Test'):
                logging.warning(
                    'MMSI %s has no valid split assigned (%s); using for Training',
                    mmsi, split)
                split = 'Training'
            time_for_this_mmsi = 0
            for rng in fishing_range_dict[mmsi]:
                time_for_this_mmsi += (
                    rng.end_time - rng.start_time).total_seconds()
                if rng.is_fishing > 0:
                    is_false_positive = False
            if time_for_this_mmsi and is_false_positive:
                logging.info('upweighting MMSI %s by %s as a false positive',
                             mmsi, FALSE_POSITIVE_UPWEIGHT)
                time_for_this_mmsi *= FALSE_POSITIVE_UPWEIGHT
            metadata_dict[split][mmsi] = (row, time_for_this_mmsi)
            if time_for_this_mmsi:
                min_time_per_mmsi = min(min_time_per_mmsi, time_for_this_mmsi)

    for split_dict in metadata_dict.values():
        for mmsi in split_dict:
            row, time = split_dict[mmsi]
            split_dict[mmsi] = (row, min(time / min_time_per_mmsi, MAX_UPWEIGHT))

    return VesselMetadata(dict(metadata_dict), fishing_range_dict, 1.0)


def read_vessel_time_weighted_metadata(available_mmsis,
                                       metadata_file,
                                       fishing_range_dict={}):
    reader = metadata_file_reader(metadata_file)

    return read_vessel_time_weighted_metadata_lines(available_mmsis, reader,
                                                    fishing_range_dict)


def read_vessel_multiclass_metadata_lines(available_mmsis, lines,
                                          fishing_range_dict,
                                          fishing_range_training_upweight):
    """ For a set of vessels, read metadata and calculate class weights.

    Args:
        available_mmsis: a set of all mmsis for which we have feature data.
        lines: a list of comma-separated vessel metadata lines. Columns are
            the mmsi and a set of vessel type columns, containing at least one
            called 'label' being the primary/coarse type of the vessel e.g.
            (Longliner/Passenger etc.).
        fishing_range_dict: dictionary of mapping mmsi to lists of fishing ranges
        fishing_range_training_upweight: amount to upweight mmsi with fishing
           ranges to assure adequate coverage.
    Returns:
        A VesselMetadata object with weights and labels for each vessel.
    """

    vessel_type_set = set()
    dataset_kind_counts = defaultdict(lambda: defaultdict(lambda: 0))
    vessel_types = []

    available_mmsis = set(available_mmsis)
    # Build a list of vessels + split + and vessel type. Calculate the split on
    # the fly, but deterministically. Count the occurrence of each vessel type
    # per split.
    for row in lines:
        mmsi = row['mmsi'].strip()
        coarse_vessel_type = row[PRIMARY_VESSEL_CLASS_COLUMN]
        if mmsi in available_mmsis and coarse_vessel_type:
            split = row['split'].strip()
            assert split in ('Training', 'Test'), repr(split)
            vessel_types.append((mmsi, split, coarse_vessel_type, row))
            dataset_kind_counts[split][coarse_vessel_type] += 1
            vessel_type_set.add(coarse_vessel_type)
        # else:
        #     logging.warning('No training data for %s, (%s) %s %s', mmsi, sorted(available_mmsis)[:10], 
        #         type(mmsi), type(sorted(available_mmsis)[0]))

    # Calculate weights for each vessel type per split, for
    # now use weights of sqrt(max_count / count), but eventually weight by prevalance
    # in AIS (as best as we can figure) <== TODO
    dataset_kind_weights = defaultdict(lambda: {})
    for split, counts in dataset_kind_counts.iteritems():
        max_count = max(counts.values())
        for coarse_vessel_type, count in counts.iteritems():
            dataset_kind_weights[split][coarse_vessel_type] = np.sqrt(max_count / float(count))

    metadata_dict = defaultdict(lambda: {})
    for mmsi, split, coarse_vessel_type, row in vessel_types:
        metadata_dict[split][mmsi] = (
            row, dataset_kind_weights[split][coarse_vessel_type])

    if len(vessel_type_set) == 0:
        logging.fatal('No vessel types found for training.')
        sys.exit(-1)

    logging.info("Vessel types: %s", list(vessel_type_set))

    return VesselMetadata(
        dict(metadata_dict), fishing_range_dict,
        fishing_range_training_upweight)


def metadata_file_reader(metadata_file):
    """


    """
    with open(metadata_file, 'r') as f:
        reader = csv.DictReader(f)
        logging.info("Metadata columns: %s", reader.fieldnames)
        for row in reader:
            yield row


def read_vessel_multiclass_metadata(available_mmsis,
                                    metadata_file,
                                    fishing_range_dict={},
                                    fishing_range_training_upweight=1.0):
    reader = metadata_file_reader(metadata_file)

    return read_vessel_multiclass_metadata_lines(
        available_mmsis, reader, fishing_range_dict,
        fishing_range_training_upweight)


def find_available_mmsis(feature_path):
    with tf.Session() as sess:
        logging.info('Reading mmsi list file.')
        root_output_path, _ = os.path.split(feature_path)
        # The feature pipeline stage that outputs the MMSI list is sharded to only
        # produce a single file, so no need to glob or loop here.
        mmsi_list_tensor = tf.read_file(root_output_path +
                                        '/mmsis/part-00000-of-00001.txt')
        els = sess.run(mmsi_list_tensor).split('\n')
        mmsi_list = [mmsi.strip() for mmsi in els if mmsi.strip() != '']

        logging.info('Found %d mmsis.', len(mmsi_list))
        return set(mmsi_list)


def parse_date(date):
    try:
        unix_timestamp = float(date)
        return datetime.datetime.utcfromtimestamp(unix_timestamp).replace(
            tzinfo=pytz.utc)
    except:
        try:
            return dateutil.parser.parse(date)
        except:
            logging.fatal('could not parse date "{}"'.format(date))
            raise


def read_fishing_ranges(fishing_range_file):
    """ Read vessel fishing ranges, return a dict of mmsi to classified fishing
        or non-fishing ranges for that vessel.
    """
    fishing_range_dict = defaultdict(lambda: [])
    with open(fishing_range_file, 'r') as f:
        for l in f.readlines()[1:]:
            els = l.split(',')
            mmsi = els[0].strip()
            start_time = parse_date(els[1])
            end_time = parse_date(els[2])
            is_fishing = float(els[3])
            fishing_range_dict[mmsi].append(
                FishingRange(start_time, end_time, is_fishing))

    return dict(fishing_range_dict)


def build_multihot_lookup_table():
    n_base = len(VESSEL_CLASS_DETAILED_NAMES)
    n_categories = len(VESSEL_CATEGORIES)
    #
    table = np.zeros([n_categories, n_base], dtype=np.int32)
    for i, (_, base_labels) in enumerate(VESSEL_CATEGORIES):
        for lbl in base_labels:
            j = VESSEL_CLASS_DETAILED_NAMES.index(lbl)
            table[i, j] = 1
    return table


multihot_lookup_table = build_multihot_lookup_table()


def multihot_encode(label):
    """Multihot encode based on fine, coarse and is_fishing label

    Args:
        label: Tensor (int)

    Returns:
        Tensor with bits set for every allowable vessel type based on the inputs


    """
    tf_multihot_lookup_table = tf.convert_to_tensor(multihot_lookup_table)
    return tf.gather(tf_multihot_lookup_table, label)
