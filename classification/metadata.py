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
import hashlib
import six
from .feature_generation.file_iterator import GCSFile


""" The main column for vessel classification. """
PRIMARY_VESSEL_CLASS_COLUMN = 'label'

#TODO: (bitsofbits) think about extracting to config file

# The 'real' categories for multihotness are the fine categories, which 'coarse' and 'fishing' 
# are defined in terms of. Any number of coarse categories, even with overlapping values can 
# be defined in principle, although at present the interaction between the mulithot and non multihot
# versions makes that more complicated.

try:
    yaml_load = yaml.safe_load
except:
    yaml_load = yaml.load


raw_schema = '''
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
'''


schema = yaml.safe_load(raw_schema)


def atomic(obj):
    for k, v in obj.items():
        if v is None or isinstance(v, str):
            yield k
        else:
            for x in atomic(v):
                yield x

def categories(obj, include_atomic=True):
    for k, v in obj.items():
        if v is None or isinstance(v, str):
            if include_atomic:
                yield k, [k]
        else:
            yield (k, list(atomic(v)))
            for x in categories(v, include_atomic=include_atomic):
                yield x



VESSEL_CLASS_DETAILED_NAMES = sorted(atomic(schema))

VESSEL_CATEGORIES = sorted(categories(schema))

TRAINING_SPLIT = 'Training'
TEST_SPLIT = 'Test'

FishingRange = namedtuple('FishingRange',
                          ['start_time', 'end_time', 'is_fishing'])


def stable_hash(x):
    x = six.ensure_binary(x)
    digest = hashlib.blake2b(six.ensure_binary(x)).hexdigest()[-8:]
    return int(digest, 16)

class VesselMetadata(object):
    def __init__(self,
                 metadata_dict,
                 fishing_ranges_map):
        self.metadata_by_split = metadata_dict
        self.metadata_by_id = {}
        self.fishing_ranges_map = fishing_ranges_map
        self.id_map_int2bytes = {}
        for split, vessels in metadata_dict.items():
            for id_, data in vessels.items():
                id_ = six.ensure_binary(id_)
                self.metadata_by_id[id_] = data
                idhash = stable_hash(id_)
                self.id_map_int2bytes[idhash] = id_

        intersection_ids = set(self.metadata_by_id.keys()).intersection(
            set(fishing_ranges_map.keys()))
        logging.info("Metadata for %d ids.", len(self.metadata_by_id))
        logging.info("Fishing ranges for %d ids.", len(fishing_ranges_map))
        logging.info("Vessels with both types of data: %d",
                     len(intersection_ids))

    def vessel_weight(self, id_):
        return self.metadata_by_id[id_][1]

    def vessel_label(self, label_name, id_):
        return self.metadata_by_id[id_][0][label_name]

    def ids_for_split(self, split):
        assert split in (TRAINING_SPLIT, TEST_SPLIT)
        # Check to make sure we don't have leakage
        if (set(self.metadata_by_split[TRAINING_SPLIT].keys()) &
            set(self.metadata_by_split[TEST_SPLIT].keys())):
                    logging.warning('id in both training and test split')
        return self.metadata_by_split[split].keys()

    def weighted_training_list(self,
                               random_state,
                               split,
                               max_replication_factor,
                               row_filter=lambda row: True,
                               boundary=1):
        replicated_ids = []
        logging.info("Training ids: %d", len(self.ids_for_split(split)))
        fishing_ranges_ids = []
        for id_, (row, weight) in self.metadata_by_split[split].items():
            if row_filter(row):
                if id_ in self.fishing_ranges_map:
                    fishing_ranges_ids.append(id_)
                weight = min(weight, max_replication_factor)

                int_n = int(weight)
                replicated_ids += ([id_] * int_n)
                frac_n = weight - float(int_n)
                if (random_state.uniform(0.0, 1.0) <= frac_n):
                    replicated_ids.append(id_)
        missing = (-len(replicated_ids)) % boundary
        if missing:
            replicated_ids = np.concatenate(
                [replicated_ids,
                 np.random.choice(replicated_ids, missing)])
        random_state.shuffle(replicated_ids)
        logging.info("Replicated training ids: %d", len(replicated_ids))
        logging.info("Fishing range ids: %d", len(fishing_ranges_ids))

        return replicated_ids

    def fishing_range_only_list(self, random_state, split):
        replicated_ids = []
        fishing_id_set = set(
            [k for (k, v) in self.fishing_ranges_map.items() if v])
        fishing_range_only_ids = [id_
                                    for id_ in self.ids_for_split(split)
                                    if id_ in fishing_id_set]
        logging.info("Fishing range training ids: %d / %d",
                     len(fishing_range_only_ids),
                     len(self.ids_for_split(split)))

        return fishing_range_only_ids


def read_vessel_time_weighted_metadata_lines(available_ids, lines,
                                             fishing_range_dict, split):
    """ For a set of vessels, read metadata; use flat weights

    Args:
        available_ids: a set of all ids for which we have feature data.
        lines: a list of comma-separated vessel metadata lines. Columns are
            the id and a set of vessel type columns, containing at least one
            called 'label' being the primary/coarse type of the vessel e.g.
            (Longliner/Passenger etc.).
        fishing_range_dict: dictionary of mapping id to lists of fishing ranges

    Returns:
        A VesselMetadata object with weights and labels for each vessel.
    """

    metadata_dict = {TRAINING_SPLIT : {}, TEST_SPLIT : {}}

    min_time_per_id = np.inf

    for row in lines:
        id_ = six.ensure_binary(row['id'].strip())
        if id_ in available_ids:
            if id_ not in fishing_range_dict:
                continue
            # Is this id included only to supress false positives
            # Symptoms; fishing score for this id never different from 0
            item_split = raw_item_split = row['split']
            if raw_item_split in '0123456789':
                if int(raw_item_split) == split:
                    item_split = TEST_SPLIT
                else:
                    item_split = TRAINING_SPLIT
            if item_split not in (TRAINING_SPLIT, TEST_SPLIT):
                logging.warning(
                    'id %s has no valid split assigned (%s); using for Training',
                    id_, split)
                split = TRAINING_SPLIT
            time_for_this_id = 0
            for rng in fishing_range_dict[id_]:
                time_for_this_id += (
                    rng.end_time - rng.start_time).total_seconds()
            metadata_dict[item_split][id_] = (row, time_for_this_id)
            if split is None and raw_item_split in '0123456789':
                # Test on everything even though we are training on everything
                metadata_dict[TEST_SPLIT][id_] = (row, time_for_this_id)

            if time_for_this_id:
                min_time_per_id = min(min_time_per_id, time_for_this_id)

    # This weighting is fiddly. We are keeping it for now to match up
    # with older data, but should replace when we move to sets, etc.
    MAX_WEIGHT = 100.0
    for split_dict in metadata_dict.values():
        for id_ in split_dict:
            row, time = split_dict[id_]
            split_dict[id_] = (row, min(MAX_WEIGHT, time / min_time_per_id))

    return VesselMetadata(metadata_dict, fishing_range_dict)


def read_vessel_time_weighted_metadata(available_ids,
                                       metadata_file,
                                       fishing_range_dict={},
                                       split=0):
    reader = metadata_file_reader(metadata_file)

    return read_vessel_time_weighted_metadata_lines(available_ids, reader,
                                                    fishing_range_dict,
                                                    split)


def read_vessel_multiclass_metadata_lines(available_ids, lines,
                                          fishing_range_dict):
    """ For a set of vessels, read metadata and calculate class weights.

    Args:
        available_ids: a set of all ids for which we have feature data.
        lines: a list of comma-separated vessel metadata lines. Columns are
            the id and a set of vessel type columns, containing at least one
            called 'label' being the primary/coarse type of the vessel e.g.
            (Longliner/Passenger etc.).
        fishing_range_dict: dictionary of mapping id to lists of fishing ranges
    Returns:
        A VesselMetadata object with weights and labels for each vessel.
    """

    vessel_type_set = set()
    dataset_kind_counts = defaultdict(lambda: defaultdict(lambda: 0))
    vessel_types = []

    cat_map = {k: v for (k, v) in VESSEL_CATEGORIES}

    available_ids = set(available_ids)
    for row in lines:
        id_ = six.ensure_binary(row['id'].strip())
        if id_ not in available_ids:
            continue
        raw_vessel_type = row[PRIMARY_VESSEL_CLASS_COLUMN]
        if not raw_vessel_type:
            continue
        atomic_types = set()
        for kind in raw_vessel_type.split('|'):
            try:
                for atm in cat_map[kind]:
                    atomic_types.add(atm)
            except StandardError as err:
                logging.warning('unknown vessel type: {}\n{}'.format(kind, err))
        if not atomic_types:
            continue
        scale = 1.0 / len(atomic_types)
        split = row['split'].strip()
        assert split in ('Training', 'Test'), repr(split)
        vessel_types.append((id_, split, raw_vessel_type, row))
        for atm in atomic_types:
            dataset_kind_counts[split][atm] += scale
        vessel_type_set |= atomic_types
        # else:
        #     logging.warning('No training data for %s, (%s) %s %s', id_, sorted(available_ids)[:10], 
        #         type(id_), type(sorted(available_ids)[0]))

    # # Calculate weights for each vessel type per split, for
    # # now use weights of sqrt(max_count / count)
    dataset_kind_weights = defaultdict(lambda: {})
    for split, counts in dataset_kind_counts.items():
        max_count = max(counts.values())
        for atomic_vessel_type, count in counts.items():
            dataset_kind_weights[split][atomic_vessel_type] = np.sqrt(max_count / float(count))

    metadata_dict = defaultdict(lambda: {})
    for id_, split, raw_vessel_type, row in vessel_types:
        if split == 'Training':
            weights = []
            for kind in raw_vessel_type.split('|'):
                for atm in cat_map.get(kind, 'unknown'):
                    weights.append(dataset_kind_weights[split][atm])
            metadata_dict[split][id_] = (row, np.mean(weights))
        elif split == "Test":
            metadata_dict[split][id_] = (row, 1.0)
        else:
            logging.warning("unknown split {}".format(split))

    if len(vessel_type_set) == 0:
        logging.fatal('No vessel types found for training.')
        sys.exit(-1)

    logging.info("Vessel types: %s", list(vessel_type_set))

    return VesselMetadata(
        dict(metadata_dict), fishing_range_dict)


def metadata_file_reader(metadata_file):
    """


    """
    with open(metadata_file, 'r') as f:
        reader = csv.DictReader(f)
        logging.info("Metadata columns: %s", reader.fieldnames)
        for row in reader:
            yield row


def read_vessel_multiclass_metadata(available_ids,
                                    metadata_file,
                                    fishing_range_dict={}):
    reader = metadata_file_reader(metadata_file)

    return read_vessel_multiclass_metadata_lines(
        available_ids, reader, fishing_range_dict)


def find_available_ids(feature_path):
    with tf.Session() as sess:
        logging.info('Reading id list file.')
        root_output_path = os.path.dirname(feature_path)
        # The feature pipeline stage that outputs the id list is sharded to only
        # produce a single file, so no need to glob or loop here.
        id_path = os.path.join(root_output_path, 'ids/part-00000-of-00001.txt')
        logging.info('Reading id list file from {}'.format(id_path))
        with GCSFile(id_path) as f:
            els = f.read().split(b'\n')
        id_list = [id_.strip() for id_ in els if id_.strip() != '']

        logging.info('Found %d ids.', len(id_list))
        return set(id_list)


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
    """ Read vessel fishing ranges, return a dict of id to classified fishing
        or non-fishing ranges for that vessel.
    """
    fishing_range_dict = defaultdict(lambda: [])
    with open(fishing_range_file, 'r') as f:
        for l in f.readlines()[1:]:
            els = l.split(',')
            id_ = six.ensure_binary(els[0].strip())
            start_time = parse_date(els[1]).replace(tzinfo=pytz.utc)
            end_time = parse_date(els[2]).replace(tzinfo=pytz.utc)
            is_fishing = float(els[3])
            fishing_range_dict[id_].append(
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
