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
"""


Example:

    python -m classification.metrics.compute_fishing_metrics \
                --inference-table machine_learning_dev_ttl_120d.fishing_detection_vid_features_v20190509_  \
                --label-path classification/data/det_info_v20190507.csv \
                --dest-path ./test_fishing_inference_0509.html \
                --fishing-ranges classification/data/det_ranges_v20190507.csv

"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import csv
import subprocess
import numpy as np
import pandas as pd
import pandas_gbq
import dateutil.parser
import logging
import argparse
from collections import namedtuple, defaultdict
import sys
import yattag
from classification.metadata import VESSEL_CLASS_DETAILED_NAMES, VESSEL_CATEGORIES, schema, atomic
import gzip
import dateutil.parser
import datetime
import pytz
from .ydump import css, ydump_table
import six



coarse_categories = [
    'cargo_or_tanker', 'passenger', 'seismic_vessel', 'tug', 'other_fishing', 
    'drifting_longlines', 'seiners', 'fixed_gear', 'squid_jigger', 'trawlers', 
    'other_not_fishing']

coarse_mapping = defaultdict(set)
for k0, extra in [('fishing', 'other_fishing'), 
                  ('non_fishing', 'other_not_fishing')]:
    for k1, v1 in schema['unknown'][k0].items():
        key = k1 if (k1 in coarse_categories) else extra
        if v1 is None:
            coarse_mapping[key] |= {k1}
        else:
            coarse_mapping[key] |= set(atomic(v1))

coarse_mapping = [(k, coarse_mapping[k]) for k in coarse_categories]

fishing_mapping = [
    ['fishing', set(atomic(schema['unknown']['fishing']))],
    ['non_fishing', set(atomic(schema['unknown']['non_fishing']))],
]


fishing_category_map = {}
atomic_fishing = fishing_mapping[0][1]
for coarse, fine in coarse_mapping:
    for atomic in fine:
        if atomic in atomic_fishing:
            fishing_category_map[atomic] = coarse


# Faster than using dateutil
def _parse(x):
    if isinstance(x, datetime.datetime):
        return x
    # 2014-08-28T13:56:16+00:00
    # TODO: fix generation to generate consistent datetimes
    if x[-6:] == '+00:00':
        x = x[:-6]
    if x.endswith('.999999'):
        x = x[:-7]
    if x.endswith('Z'):
        x = x[:-1]
    try:
        dt = datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')
    except:
        logging.fatal('Could not parse "%s"', x)
        raise
    return dt.replace(tzinfo=pytz.UTC)


LocalisationResults = namedtuple('LocalisationResults',
                                 ['true_fishing_by_id',
                                  'pred_fishing_by_id', 'label_map'])

FishingRange = namedtuple('FishingRange',
    ['is_fishing', 'start_time', 'end_time'])


def ydump_fishing_localisation(doc, results):
    doc, tag, text, line = doc.ttl()

    y_true = np.concatenate(list(results.true_fishing_by_id.values()))
    y_pred = np.concatenate(list(results.pred_fishing_by_id.values()))

    header = ['Gear Type (id:true/total)', 'Precision', 'Recall', 'Accuracy', 'F1-Score']
    rows = []
    logging.info('Overall localisation accuracy %s',
                 accuracy_score(y_true, y_pred))
    logging.info('Overall localisation precision %s',
                 precision_score(y_true, y_pred))
    logging.info('Overall localisation recall %s',
                 recall_score(y_true, y_pred))

    for cls in sorted(set(fishing_category_map.values())) + ['other'] :
        true_chunks = []
        pred_chunks = []
        id_list = []
        for id_ in results.label_map:
            if id_ not in results.true_fishing_by_id:
                continue
            if fishing_category_map.get(results.label_map[id_], 'other') != cls:
                continue
            id_list.append(id_)
            true_chunks.append(results.true_fishing_by_id[id_])
            pred_chunks.append(results.pred_fishing_by_id[id_])
        if len(true_chunks):
            logging.info('ID for {}: {}'.format(cls, id_list))
            y_true = np.concatenate(true_chunks)
            y_pred = np.concatenate(pred_chunks)
            rows.append(['{} ({}:{}/{})'.format(cls, len(true_chunks), sum(y_true), len(y_true)),
                         precision_score(y_true, y_pred),
                         recall_score(y_true, y_pred),
                         accuracy_score(y_true, y_pred),
                         f1_score(y_true, y_pred), ])

    rows.append(['', '', '', '', ''])

    y_true = np.concatenate(list(results.true_fishing_by_id.values()))
    y_pred = np.concatenate(list(results.pred_fishing_by_id.values()))

    rows.append(['Overall',
                 precision_score(y_true, y_pred),
                 recall_score(y_true, y_pred),
                 accuracy_score(y_true, y_pred),
                 f1_score(y_true, y_pred), ])

    with tag('div', klass='unbreakable'):
        ydump_table(
            doc, header,
            [[('{:.2f}'.format(x) if isinstance(x, float) else x) for x in row]
             for row in rows])




def precision_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    true_pos = y_true & y_pred
    all_pos = y_pred

    return true_pos.sum() / all_pos.sum()


def recall_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    true_pos = y_true & y_pred
    all_true = y_true

    return true_pos.sum() / all_true.sum()


def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return 2 / (1 / prec + 1 / recall)

def accuracy_score(y_true, y_pred, weights=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if weights is None:
        weights = np.ones_like(y_pred).astype(float)
    weights = np.asarray(weights)

    correct = (y_true == y_pred)

    return (weights * correct).sum() / weights.sum()


def load_inferred_fishing(table, id_list, project_id, threshold=True):
    """Load inferred data and generate comparison data

    """
    query_template = """
    SELECT vessel_id as id, start_time, end_time, nnet_score FROM 
        TABLE_DATE_RANGE([{table}],
            TIMESTAMP('{year}-01-01'), TIMESTAMP('{year}-12-31'))
        WHERE vessel_id in ({ids})
    """
    ids = ','.join('"{}"'.format(x) for x in id_list)
    ranges = defaultdict(list)
    for year in range(2012, 2019):
        query = query_template.format(table=table, year=year, ids=ids)
        try:
            df = pd.read_gbq(query, project_id=project_id, dialect='legacy')
        except pandas_gbq.gbq.GenericGBQException as err:
            if 'matches no table' in err.args[0]:
                print('skipping', year)
                continue
            else:
                print(query)
                raise
        for x in df.itertuples():
            score = x.nnet_score
            if threshold:
                score = score > 0.5
            start = x.start_time.replace(tzinfo=pytz.utc)
            end = x.end_time.replace(tzinfo=pytz.utc)
            ranges[x.id].append(FishingRange(score, start, end))
    return ranges

def load_true_fishing_ranges_by_id(fishing_range_path,
                                     split_map,
                                     split,
                                     threshold=True):
    ranges_by_id = defaultdict(list)
    parse = dateutil.parser.parse
    with open(fishing_range_path) as f:
        for row in csv.DictReader(f):
            id_ = row['id'].strip()
            if not split_map.get(id_) == str(split):
                continue
            val = float(row['is_fishing'])
            if threshold:
                val = val > 0.5
            rng = (val, parse(row['start_time']).replace(tzinfo=pytz.UTC), 
                        parse(row['end_time']).replace(tzinfo=pytz.UTC))
            ranges_by_id[id_].append(rng)
    return ranges_by_id


def datetime_to_minute(dt):
    timestamp = (dt - datetime.datetime(
        1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
    return int(timestamp // 60)


def compare_fishing_localisation(inferred_ranges, fishing_range_path,
                                 label_map, split_map, split):

    logging.debug('loading fishing ranges')
    true_ranges_by_id = load_true_fishing_ranges_by_id(fishing_range_path,
                                                           split_map, split)
    print("TRUE", sorted(true_ranges_by_id.keys())[:10])
    print("INF", sorted(inferred_ranges.keys())[:10])
    print(repr(sorted(true_ranges_by_id.keys())[0]))
    print(repr(sorted(inferred_ranges.keys())[0]))
    true_by_id = {}
    pred_by_id = {}

    for id_ in sorted(true_ranges_by_id.keys()):
        id_ = six.ensure_text(id_)
        logging.debug('processing %s', id_)
        if id_ not in inferred_ranges:
            continue
        true_ranges = true_ranges_by_id[id_]
        if not true_ranges:
            continue

        # Determine minutes from start to finish of this id, create an array to
        # hold results and fill with -1 (unknown)
        logging.debug('processing %s true ranges', len(true_ranges))
        logging.debug('finding overall range')
        _, start, end = true_ranges[0]
        for (_, s, e) in true_ranges[1:]:
            start = min(start, s)
            end = max(end, e)
        start_min = datetime_to_minute(start)
        end_min = datetime_to_minute(end)
        minutes = np.empty([end_min - start_min + 1, 2], dtype=int)
        minutes.fill(-1)

        # Fill in minutes[:, 0] with known true / false values
        logging.debug('filling 0s')
        for (is_fishing, s, e) in true_ranges:
            s_min = datetime_to_minute(s)
            e_min = datetime_to_minute(e)
            for m in range(s_min - start_min, e_min - start_min + 1):
                minutes[m, 0] = is_fishing

        # fill in minutes[:, 1] with inferred true / false values
        logging.debug('filling 1s')
        for (is_fishing, s, e) in inferred_ranges[str(id_)]:
            s_min = datetime_to_minute(s)
            e_min = datetime_to_minute(e)
            for m in range(s_min - start_min, e_min - start_min + 1):
                if 0 <= m < len(minutes):
                    minutes[m, 1] = is_fishing

        mask = ((minutes[:, 0] != -1) & (minutes[:, 1] != -1))
        if mask.sum():
            accuracy = (
                (minutes[:, 0] == minutes[:, 1]) * mask).sum() / mask.sum()
            logging.debug('Accuracy for ID %s: %s', id_, accuracy)

            true_by_id[id_] = minutes[mask, 0]
            pred_by_id[id_] = minutes[mask, 1]

    return LocalisationResults(true_by_id, pred_by_id, label_map)


def compute_results(args):
    logging.info('Loading label maps')
    maps = defaultdict(dict)
    with open(args.label_path) as f:
        for row in csv.DictReader(f):
            id_ = row['id'].strip()
            if not row['split'] == str(args.split):
                continue
            for field in ['label', 'split']:
                if row[field]:
                    if field == 'label':
                        if row[field].strip(
                        ) not in VESSEL_CLASS_DETAILED_NAMES:
                            continue
                    maps[field][id_] = row[field]

    # Sanity check the attribute mappings
    for field in ['length', 'tonnage', 'engine_power', 'crew_size']:
        for id_, value in maps[field].items():
            assert float(value) > 0, (id_, value)

    logging.info('Loading inference data')
    ids = set([x for x in maps['split'] if maps['split'][x] == str(args.split)])

    fishing_ranges = load_inferred_fishing(args.inference_table, ids, args.project_id)
    logging.info('Comparing localisation')
    results = {}
    results['localisation'] = compare_fishing_localisation(
        fishing_ranges, args.fishing_ranges, maps['label'],
        maps['split'], args.split)


    return results


def dump_html(args, results):

    doc = yattag.Doc()

    with doc.tag('style', type='text/css'):
        doc.asis(css)

    logging.info('Dumping Localisation')
    doc.line('h2', 'Fishing Localisation')
    ydump_fishing_localisation(doc, results['localisation'])
    doc.stag('hr')

    with open(args.dest_path, 'w') as f:
        logging.info('Writing output')
        f.write(yattag.indent(doc.getvalue(), indent_text=True))


"""

python -m classification.metrics.compute_fishing_metrics \
--inference-table machine_learning_dev_ttl_120d.test_dataflow_2016_ \
--label-path classification/data/fishing_classes.csv \
--dest-path test_fishing.html \
--fishing-ranges classification/data/combined_fishing_ranges.csv \


"""


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
        description='Test fishing inference results and output metrics.\n')
    parser.add_argument(
        '--inference-table', help='table of inference results', required=True)
    parser.add_argument(
        '--project-id', help='Google Cloud project id', 
        default='world-fishing-827')
    parser.add_argument(
        '--label-path', help='path to test data', required=True)
    parser.add_argument('--fishing-ranges', help='path to fishing range data', required=True)
    parser.add_argument(
        '--dest-path', help='path to write results to', required=True)
    parser.add_argument('--split', type=int, default=0)


    args = parser.parse_args()

    results = compute_results(args)

    dump_html(args, results)

 