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

    python -m classification.metrics.compute_vessel_metrics \
        --inference-table machine_learning_dev_ttl_120d.mmsi_vessel_char_V20190411_  \
        --label-path classification/data/training_classes.csv \
        --dest-path ./test_inference.html

*Note: despite the table, name this table is in terms of vessel_id*

"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import csv
import subprocess
import numpy as np
import dateutil.parser
import logging
import argparse
from collections import namedtuple, defaultdict
import sys
import yattag
import newlinejson as nlj
from classification.metadata import VESSEL_CLASS_DETAILED_NAMES, VESSEL_CATEGORIES, TEST_SPLIT, schema, atomic
import gzip
import dateutil.parser
import datetime
import pytz
import pandas as pd

coarse_categories = [
    'cargo_or_tanker', 'passenger', 'tug',  'seismic_vessel','other_not_fishing', 
    'drifting_longlines', 'gear', 'purse_seines', 'set_gillnets', 'set_longlines', 'pots_and_traps',
     'trawlers', 'squid_jigger','other_fishing', 
    ]


all_classes = set(VESSEL_CLASS_DETAILED_NAMES)
categories = dict(VESSEL_CATEGORIES)
is_fishing = set(categories['fishing'])
not_fishing = set(categories['non_fishing'])

coarse_mapping = defaultdict(set)
used = set()
for cat in coarse_categories:
    atomic_cats = set(categories[cat])
    assert not atomic_cats & used
    used |= atomic_cats
    coarse_mapping[cat] = atomic_cats
unused = all_classes - used
coarse_mapping['other_fishing'] |= (is_fishing & unused)
coarse_mapping['other_not_fishing'] |= (not_fishing & unused)

coarse_mapping = [(k, coarse_mapping[k]) for k in coarse_categories]

fishing_mapping = [
    ['fishing', set(atomic(schema['unknown']['fishing']))],
    ['non_fishing', set(atomic(schema['unknown']['non_fishing']))],
]


# for k, v in coarse_mapping:
#     print(k, v)
# print()
# for k, v in fishing_mapping:
#     print(k, v)

# raise SystemExit

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


class InferenceResults(object):

    _indexed_scores = None

    def __init__(self, # TODO: Consider reordering args so that label_list is first
        mmsi, inferred_labels, true_labels, start_dates, scores,
        label_list,
        all_mmsi=None, all_inferred_labels=None, all_true_labels=None, all_start_dates=None, all_scores=None):

        self.label_list = label_list
        #
        self.all_mmsi = all_mmsi
        self.all_inferred_labels = all_inferred_labels
        self.all_true_labels = all_true_labels
        self.all_start_dates = np.asarray(all_start_dates)
        self.all_scores = all_scores
        #
        self.mmsi = mmsi
        self.inferred_labels = inferred_labels
        self.true_labels = true_labels
        self.start_dates = np.asarray(start_dates)
        self.scores = scores
        #

    def all_results(self):
        return InferenceResults(self.all_mmsi, self.all_inferred_labels,
                                self.all_true_labels, self.all_start_dates,
                                self.all_scores, self.label_list)

    @property
    def indexed_scores(self):
        if self._indexed_scores is None:
            logging.debug('create index_scores')
            iscores = np.zeros([len(self.mmsi), len(self.label_list)])
            for i, mmsi in enumerate(self.mmsi):
                for j, lbl in enumerate(self.label_list):
                    iscores[i, j] = self.scores[i][lbl]
            self._indexed_scores = iscores
            logging.debug('done')
        return self._indexed_scores


AttributeResults = namedtuple(
    'AttributeResults',
    ['mmsi', 'inferred_attrs', 'true_attrs', 'true_labels', 'start_dates'])

LocalisationResults = namedtuple('LocalisationResults',
                                 ['true_fishing_by_mmsi',
                                  'pred_fishing_by_mmsi', 'label_map'])

ConfusionMatrix = namedtuple('ConfusionMatrix', ['raw', 'scaled'])

CLASSIFICATION_METRICS = [
    ('fishing', 'Is Fishing'),
    ('coarse', 'Coarse Labels'),
    ('fine', 'Fine Labels'),
]

css = """

table {
    text-align: center;
    border-collapse: collapse;
}

.confusion-matrix th.col {
  height: 140px; 
  white-space: nowrap;
}

.confusion-matrix th.col div {
    transform: translate(16px, 49px) rotate(315deg); 
    width: 30px;
}

.confusion-matrix th.col span {
    border-bottom: 1px solid #ccc; 
    padding: 5px 10px; 
    text-align: left;
}

.confusion-matrix th.row {
    text-align: right;
}

.confusion-matrix td.diagonal {
    border: 1px solid black;
}

.confusion-matrix td.offdiagonal {
    border: 1px dotted grey;
}

.unbreakable {
    page-break-inside: avoid;
}




"""

# basic metrics


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


def weights(labels, y_true, y_pred, max_weight=200):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    weights = np.zeros([len(y_true)])
    for lbl in labels:
        trues = (y_true == lbl)
        if trues.sum():
            wt = min(len(trues) / trues.sum(), max_weight)
            weights += trues * wt

    return weights / weights.sum()


def base_confusion_matrix(y_true, y_pred, labels):
    n = len(labels)
    label_map = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros([n, n], dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt not in label_map:
            logging.warn('%s not in label_map', yt)
            continue
        if yp not in label_map:
            logging.warn('%s not in label_map', yp)
            continue
        cm[label_map[yt], label_map[yp]] += 1

    return cm

# Helper function formatting as HTML (using yattag)


def ydump_confusion_matrix(doc, cm, labels, **kwargs):
    """Dump an sklearn confusion matrix to HTML using yatag

    Args:
        doc: yatag Doc instance
        cm: ConfusionMatrix instance
        labels: list of str
            labels for confusion matrix
    """
    doc, tag, text, line = doc.ttl()
    with tag('table', klass='confusion-matrix', **kwargs):
        with tag('tr'):
            line('th', '')
            for x in labels:
                with tag('th', klass='col'):
                    with tag('div'):
                        line('span', x)
        for i, (l, row) in enumerate(zip(labels, cm.scaled)):
            with tag('tr'):
                line('th', str(l), klass='row')
                for j, x in enumerate(row):
                    if i == j:
                        if x == -1:
                            # No values present in this row, column
                            color = '#FFFFFF'
                        elif x > 0.5:
                            cval = np.clip(int(round(512 * (x - 0.5))), 0, 255)
                            invhexcode = '{:02x}'.format(255 - cval)
                            color = '#{}FF00'.format(invhexcode)
                        else:
                            cval = np.clip(int(round(512 * x)), 0, 255)
                            hexcode = '{:02x}'.format(cval)
                            color = '#FF{}00'.format(hexcode)
                        klass = 'diagonal'
                    else:
                        cval = np.clip(int(round(255 * x)), 0, 255)
                        hexcode = '{:02x}'.format(cval)
                        invhexcode = '{:02x}'.format(255 - cval)
                        color = '#FF{}{}'.format(invhexcode, invhexcode)
                        klass = 'offdiagonal'
                    with tag('td', klass=klass, bgcolor=color):
                        raw = cm.raw[i, j]
                        with tag('font',
                                 color='#000000',
                                 title='{0:.3f}'.format(x)):
                            text(str(raw))


def ydump_table(doc, headings, rows, **kwargs):
    """Dump an html table using yatag

    Args:
        doc: yatag Doc instance
        headings: [str]
        rows: [[str]]
            
    """
    doc, tag, text, line = doc.ttl()
    with tag('table', **kwargs):
        with tag('tr'):
            for x in headings:
                line('th', str(x))
        for row in rows:
            with tag('tr'):
                for x in row:
                    line('td', str(x))


def ydump_attrs(doc, results):
    """dump metrics for `results` to html using yatag

    Args:
        doc: yatag Doc instance
        results: InferenceResults instance

    """
    doc, tag, text, line = doc.ttl()

    def RMS(a, b):
        return np.sqrt(np.square(a - b).mean())

    def MAE(a, b):
        return abs(a - b).mean()

    # TODO: move computations out of loops for speed.
    # true_mask = np.array([(x is not None) for x in results.true_attrs])
    # infer_mask = np.array([(x is not None) for x in results.inferred_attrs])
    true_mask = ~np.isnan(results.true_attrs)
    infer_mask = ~np.isnan(results.inferred_attrs)
    rows = []
    for dt in np.unique(results.start_dates):
        mask = true_mask & infer_mask & (results.start_dates == dt)
        rows.append(
            [dt, RMS(results.true_attrs[mask], results.inferred_attrs[mask]),
             MAE(results.true_attrs[mask], results.inferred_attrs[mask])])

    with tag('div', klass='unbreakable'):
        line('h3', 'RMS Error by Date')
        ydump_table(doc, ['Start Date', 'RMS Error', 'Abs Error'],
                    [(a.date(), '{:.2f}'.format(b), '{:.2f}'.format(c))
                     for (a, b, c) in rows])

    logging.info('    Consolidating attributes')
    consolidated = consolidate_attribute_across_dates(results)
    # true_mask = np.array([(x is not None) for x in consolidated.true_attrs])
    # infer_mask = np.array([(x is not None) for x in consolidated.inferred_attrs])
    true_mask = ~np.isnan(consolidated.true_attrs)
    infer_mask = ~np.isnan(consolidated.inferred_attrs)

    logging.info('    RMS Error')
    with tag('div', klass='unbreakable'):
        line('h3', 'Overall RMS Error')
        text('{:.2f}'.format(
            RMS(consolidated.true_attrs[true_mask & infer_mask],
                consolidated.inferred_attrs[true_mask & infer_mask])))

    logging.info('    ABS Error')
    with tag('div', klass='unbreakable'):
        line('h3', 'Overall Abs Error')
        text('{:.2f}'.format(
            MAE(consolidated.true_attrs[true_mask & infer_mask],
                consolidated.inferred_attrs[true_mask & infer_mask])))

    def RMS_MAE_by_label(true_attrs, pred_attrs, true_labels):
        results = []
        labels = sorted(set(true_labels))
        for lbl in labels:
            mask = true_mask & infer_mask & (lbl == true_labels)
            if mask.sum():
                err = RMS(true_attrs[mask], pred_attrs[mask])
                abs_err = MAE(true_attrs[mask], pred_attrs[mask])
                count = mask.sum()
                results.append(
                    (lbl, count, err, abs_err, true_attrs[mask].mean(),
                     true_attrs[mask].std()))
        return results

    logging.info('    Error by Label')
    with tag('div', klass='unbreakable'):
        line('h3', 'RMS Error by Label')
        ydump_table(
            doc,
            ['Label', 'Count', 'RMS Error', 'Abs Error', 'Mean', 'StdDev'
             ],  # TODO: pass in length and units
            [
                (a, count, '{:.2f}'.format(b), '{:.2f}'.format(ab),
                 '{:.2f}'.format(c), '{:.2f}'.format(d))
                for (a, count, b, ab, c, d) in RMS_MAE_by_label(
                    consolidated.true_attrs, consolidated.inferred_attrs,
                    consolidated.true_labels)
            ])


def ydump_metrics(doc, results):
    """dump metrics for `results` to html using yatag

    Args:
        doc: yatag Doc instance
        results: InferenceResults instance

    """
    doc, tag, text, line = doc.ttl()

    rows = [
        (x, accuracy_score(results.true_labels, results.inferred_labels,
                           (results.start_dates == x)))
        for x in np.unique(results.start_dates)
    ]

    with tag('div', klass='unbreakable'):
        line('h3', 'Accuracy by Date')
        ydump_table(doc, ['Start Date', 'Accuracy'],
                    [(a.date(), '{:.2f}'.format(b)) for (a, b) in rows])

    consolidated = consolidate_across_dates(results)

    with tag('div', klass='unbreakable'):
        line('h3', 'Overall Accuracy')
        text('{:.2f}'.format(
            accuracy_score(consolidated.true_labels,
                           consolidated.inferred_labels)))

    cm = confusion_matrix(consolidated)

    with tag('div', klass='unbreakable'):
        line('h3', 'Confusion Matrix')
        ydump_confusion_matrix(doc, cm, results.label_list)

    with tag('div', klass='unbreakable'):
        line('h3', 'Metrics by Label')
        row_vals = precision_recall_f1(consolidated.label_list,
                                       consolidated.true_labels,
                                       consolidated.inferred_labels)
        ydump_table(doc, ['Label (mmsi:true/total)', 'Precision', 'Recall', 'F1-Score'], [
            (a, '{:.2f}'.format(b), '{:.2f}'.format(c), '{:.2f}'.format(d))
            for (a, b, c, d) in row_vals
        ])
        wts = weights(consolidated.label_list, consolidated.true_labels,
                      consolidated.inferred_labels)
        line('h4', 'Accuracy with equal class weight')
        text(
            str(
                accuracy_score(consolidated.true_labels,
                               consolidated.inferred_labels, wts)))

fishing_category_map = {
    'drifting_longlines' : 'drifting_longlines',
    'trawlers' : 'trawlers',
    'purse_seines' : 'purse_seines',
    'pots_and_traps' : 'stationary_gear',
    'set_gillnets' : 'stationary_gear',
    'set_longlines' : 'stationary_gear'
}


def ydump_fishing_localisation(doc, results):
    doc, tag, text, line = doc.ttl()

    y_true = np.concatenate(results.true_fishing_by_mmsi.values())
    y_pred = np.concatenate(results.pred_fishing_by_mmsi.values())

    header = ['Gear Type (mmsi:true/total)', 'Precision', 'Recall', 'Accuracy', 'F1-Score']
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
        mmsi_list = []
        for mmsi in results.label_map:
            if mmsi not in results.true_fishing_by_mmsi:
                continue
            if fishing_category_map.get(results.label_map[mmsi], 'other') != cls:
                continue
            mmsi_list.append(mmsi)
            true_chunks.append(results.true_fishing_by_mmsi[mmsi])
            pred_chunks.append(results.pred_fishing_by_mmsi[mmsi])
        if len(true_chunks):
            logging.info('MMSI for {}: {}'.format(cls, mmsi_list))
            y_true = np.concatenate(true_chunks)
            y_pred = np.concatenate(pred_chunks)
            rows.append(['{} ({}:{}/{})'.format(cls, len(true_chunks), sum(y_true), len(y_true)),
                         precision_score(y_true, y_pred),
                         recall_score(y_true, y_pred),
                         accuracy_score(y_true, y_pred),
                         f1_score(y_true, y_pred), ])

    rows.append(['', '', '', '', ''])

    y_true = np.concatenate(results.true_fishing_by_mmsi.values())
    y_pred = np.concatenate(results.pred_fishing_by_mmsi.values())

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

# Helper functions for computing metrics


def clean_label(x):
    x = x.strip()
    return x.replace('_', ' ')


def precision_recall_f1(labels, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    results = []
    for lbl in labels:
        trues = (y_true == lbl)
        positives = (y_pred == lbl)
        if trues.sum() and positives.sum():
            # Only return cases where there are least one vessel present in both cases
            results.append(
                (lbl, precision_score(trues, positives),
                 recall_score(trues, positives), f1_score(trues, positives)))
    return results


def consolidate_across_dates(results, date_range=None):
    """Consolidate scores for each MMSI across available dates.

    For each mmsi, we take the scores at all available dates, sum
    them and use argmax to find the predicted results.

    Optionally accepts a date range, which specifies half open ranges
    for the dates.
    """
    inferred_mmsi = []
    inferred_labels = []
    true_labels = []

    if date_range is None:
        valid_date_mask = np.ones([len(results.mmsi)], dtype=bool)
    else:
        # TODO: write out end date as well, so that we avoid this hackery
        end_dates = results.start_dates + datetime.timedelta(days=180)
        valid_date_mask = (results.start_dates >= date_range[0]) & (
            results.start_dates < date_range[1])

    mmsi_map = {}
    mmsi_indices = []
    for i, m in enumerate(results.mmsi):
        if valid_date_mask[i]:
            if m not in mmsi_map:
                mmsi_map[m] = len(inferred_mmsi)
                inferred_mmsi.append(m)
                true_labels.append(results.true_labels[i])
            mmsi_indices.append(mmsi_map[m])
        else:
            mmsi_indices.append(-1)
    mmsi_indices = np.array(mmsi_indices)

    scores = np.zeros([len(inferred_mmsi), len(results.label_list)])
    counts = np.zeros([len(inferred_mmsi)])
    for i, valid in enumerate(valid_date_mask):
        if valid:
            scores[mmsi_indices[i]] += results.indexed_scores[i]
            counts[mmsi_indices[i]] += 1

    inferred_labels = []
    for i, s in enumerate(scores):
        inferred_labels.append(results.label_list[np.argmax(s)])
        if counts[i]:
            scores[i] /= counts[i]

    return InferenceResults(
        np.array(inferred_mmsi), np.array(inferred_labels),
        np.array(true_labels), None, scores, results.label_list)


def consolidate_attribute_across_dates(results, date_range=None):
    """Consolidate scores for each MMSI across available dates.

    For each mmsi, we average the attribute across all available dates

    """
    inferred_attributes = []
    true_attributes = []
    true_labels = []
    indices = np.argsort(results.mmsi)
    mmsi = np.unique(results.mmsi)

    for m in np.unique(results.mmsi):
        start = np.searchsorted(results.mmsi, m, side='left', sorter=indices)
        stop = np.searchsorted(results.mmsi, m, side='right', sorter=indices)

        attrs_for_mmsi = results.inferred_attrs[indices[start:stop]]

        if date_range:
            start_dates = results.start_dates[indices[start:stop]]
            # TODO: This is kind of messy need to verify that date ranges and output ranges line up
            valid_date_mask = (start_dates >= date_range[0]) & (start_dates < date_range[1])
            attrs = attrs_for_mmsi[valid_date_mask]
        else:
            attrs = attrs_for_mmsi

        if len(attrs):
            inferred_attributes.append(attrs.mean())
        else:
            inferred_attributes.append(np.nan)

        trues = results.true_attrs[indices[start:stop]]
        has_true = ~np.isnan(trues)
        if has_true.sum():
            true_attributes.append(trues[has_true].mean())
        else:
            true_attributes.append(np.nan)

        labels = results.true_labels[indices[start:stop]]
        has_labels = (labels != "Unknown")
        if has_labels.sum():
            true_labels.append(labels[has_labels][0])
        else:
            true_labels.append("Unknown")

    return AttributeResults(
        mmsi, np.array(inferred_attributes), np.array(true_attributes),
        np.array(true_labels), None)


def harmonic_mean(x, y):
    return 2.0 / ((1.0 / x) + (1.0 / y))


def confusion_matrix(results):
    """Compute raw and normalized confusion matrices based on results.

    Args:
        results: InferenceResults instance

    Returns:
        ConfusionMatrix instance, with raw and normalized (`scaled`)
            attributes.

    """
    EPS = 1e-10
    cm_raw = base_confusion_matrix(results.true_labels,
                                   results.inferred_labels, results.label_list)

    # For off axis, normalize harmonic mean of row / col inverse errors.
    # The idea here is that this average will go to 1 => BAD, as
    # either the row error or column error approaches 1. That is, if this
    # off diagonal element dominates eitehr the predicted values for this 
    # label OR the actual values for this label.  A standard mean will only
    # go to zero if it dominates both, but these can become decoupled with 
    # unbalanced classes.
    row_totals = cm_raw.sum(axis=1, keepdims=True)
    col_totals = cm_raw.sum(axis=0, keepdims=True)
    inv_row_fracs = 1 - cm_raw / (row_totals + EPS)
    inv_col_fracs = 1 - cm_raw / (col_totals + EPS)
    cm_normalized = 1 - harmonic_mean(inv_col_fracs, inv_row_fracs)
    # For on axis, use the F1-score (also a harmonic mean!)
    for i in range(len(cm_raw)):
        recall = cm_raw[i, i] / (row_totals[i, 0] + EPS)
        precision = cm_raw[i, i] / (col_totals[0, i] + EPS)
        if row_totals[i, 0] == col_totals[0, i] == 0:
            cm_normalized[i, i] = -1  # Not values to compute from
        else:
            cm_normalized[i, i] = harmonic_mean(recall, precision)

    return ConfusionMatrix(cm_raw, cm_normalized)

# query to rebuild mmsi_to_vessel_id
"""
with pairs as (
  select ssvid, vessel_id, count(*) pair_count
  from `pipe_production_b.position_messages_*`
  group by ssvid, vessel_id
),

ranked as (
  select ssvid, vessel_id,
         row_number() over(partition by ssvid order by pair_count desc) as rank
  from pairs
)

select ssvid, vessel_id
from ranked
where rank = 1
"""

def load_inferred(inference_table, label_table, extractors, whitelist):
    """Load inferred data and generate comparison data

    """
    query = """

    SELECT c.* except(vessel_id), b.mmsi as vessel_id FROM 
    `machine_learning_dev_ttl_120d.mmsi_to_vessel_id` a
    JOIN
    `{}` b
    ON a.ssvid = cast(b.mmsi as string)
    JOIN
   `{}*` c
    USING (vessel_id)
    """.format(label_table, inference_table)
    print(query)
    df = pd.read_gbq(query, project_id='world-fishing-827', dialect='standard')

    for row in df.itertuples():
        if whitelist is not None and row.mmsi not in whitelist:
            continue
        for ext in extractors:
            ext.extract(row)
    for ext in extractors:
        ext.finalize()


class ClassificationExtractor(InferenceResults):
    # Conceptually an InferenceResult
    # TODO: fix to make true subclass or return true inference result at finalization time or something.
    def __init__(self, label_map):
        self.label_map = label_map
        #
        self.all_mmsi = []
        self.all_inferred_labels = []
        self.all_true_labels = []
        self.all_start_dates = []
        self.all_scores = []
        #
        self.mmsi = []
        self.inferred_labels = []
        self.true_labels = []
        self.start_dates = []
        self.scores = []
        #
        self.all_labels = set(label_map.values())

    def extract(self, row):
        mmsi = row.vessel_id
        lbl = self.label_map.get(mmsi)
        raw_label_scores = row.label_scores
        label_scores = {x['label'] : x['score'] for x in raw_label_scores}
        self.all_labels |= set(label_scores.keys())
        start_date = row.start_time
        # TODO: write out TZINFO in inference
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=pytz.utc)
        inferred = row.max_label
        # Every row that has inference values get stored in all_
        self.all_mmsi.append(mmsi)
        self.all_start_dates.append(start_date)
        self.all_true_labels.append(lbl)
        self.all_inferred_labels.append(inferred)
        self.all_scores.append(label_scores)
        # Only values that have a known component get stored in the not all_ arrays
        if lbl is not None and not (isinstance(lbl, float) and np.isnan(lbl)):
            self.mmsi.append(mmsi)
            self.start_dates.append(start_date)
            self.true_labels.append(lbl)
            self.inferred_labels.append(inferred)
            self.scores.append(label_scores)

    def finalize(self):
        self.inferred_labels = np.array(self.inferred_labels)
        self.true_labels = np.array(self.true_labels)
        self.start_dates = np.array(self.start_dates)
        self.scores = np.array(self.scores)
        self.label_list = sorted(
            self.all_labels, key=VESSEL_CLASS_DETAILED_NAMES.index)
        if len(self.true_labels) == 0:
            raise ValueError('no true labels')
        self.mmsi = np.array(self.mmsi)
        for lbl in self.label_list:
            true_count = (self.true_labels == lbl).sum()
            inf_count = (self.inferred_labels == lbl).sum()
            logging.info("%s true and %s inferred labels for %s", true_count,
                         inf_count, lbl)

    def __nonzero__(self):
        return len(self.mmsi) > 0


class AttributeExtractor(object):
    def __init__(self, key, attr_map, label_map):
        self.key = key
        self.attr_map = attr_map
        self.label_map = label_map
        self.mmsi = []
        self.inferred_attrs = []
        self.true_attrs = []
        self.true_labels = []
        self.start_dates = []

    def extract(self, row):
        mmsi = row.vessel_id
        if getattr(row, self.key) is None:
            return
        self.mmsi.append(mmsi)
        self.start_dates.append(row.start_time)
        self.true_attrs.append(
            float(self.attr_map[mmsi]) if (mmsi in self.attr_map) else np.nan)
        self.true_labels.append(self.label_map.get(mmsi, 'Unknown'))
        self.inferred_attrs.append(getattr(row, self.key))

    def finalize(self):
        self.inferred_attrs = np.array(self.inferred_attrs)
        self.true_attrs = np.array(self.true_attrs)
        self.start_dates = np.array(self.start_dates)
        self.mmsi = np.array(self.mmsi)
        self.true_labels = np.array(self.true_labels)

    def __nonzero__(self):
        return len(self.mmsi) > 0





def assemble_composite(results, mapping):
    """

    Args:
        results: InferenceResults instance
        mapping: sequence of (composite_key, {base_keys})

    Returns:
        InferenceResults instance

    Classes are remapped according to mapping.

    """

    label_list = [lbl for (lbl, base_labels) in mapping]
    inferred_scores = []
    inferred_labels = []
    true_labels = []
    start_dates = []

    inverse_mapping = {}
    for new_label, base_labels in mapping:
        for lbl in base_labels:
            inverse_mapping[lbl] = new_label
    base_label_map = {x: i for (i, x) in enumerate(results.label_list)}

    for i, mmsi in enumerate(results.all_mmsi):
        scores = {}
        for (new_label, base_labels) in mapping:
            scores[new_label] = 0
            for lbl in base_labels:
                scores[new_label] += results.all_scores[i][lbl]
        inferred_scores.append(scores)
        inferred_labels.append(max(scores, key=scores.__getitem__))
        old_label = results.all_true_labels[i]
        new_label = None if (old_label is None) else inverse_mapping[old_label]
        true_labels.append(new_label)
        start_dates.append(results.all_start_dates[i])

    def trim(seq):
        return np.array([x for (i, x) in enumerate(seq) if true_labels[i]])

    return InferenceResults(
        trim(results.all_mmsi), trim(inferred_labels), trim(true_labels),
        trim(start_dates), trim(inferred_scores), label_list,
        np.array(results.all_mmsi), np.array(inferred_labels),
        np.array(true_labels), np.array(start_dates),
        np.array(inferred_scores))





def datetime_to_minute(dt):
    timestamp = (dt - datetime.datetime(
        1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
    return int(timestamp // 60)




def compute_results(args):
    logging.info('Loading label maps')
    maps = defaultdict(dict)
    label_df = pd.read_gbq("select * from `{}`".format(training_data_table), project_id='world-fishing-827', dialect='standard')
    for row in label_df.itertuples():
        mmsi = row.mmsi
        if not row.split == TEST_SPLIT:
            continue
        for field in ['label', 'length', 'tonnage', 'engine_power', 'crew_size', 'split']:
            val = getattr(row, field)
            if val is not None and val != '' and not (isinstance(val, float) and np.isnan(val)):
                if field == 'label':
                    if row.label.strip(
                    ) not in VESSEL_CLASS_DETAILED_NAMES:
                        print("SHOULDNT HAPPEN!", field)
                        continue
                maps[field][mmsi] = getattr(row, field)
    results = {}

    # Sanity check the attribute mappings
    for field in ['length', 'tonnage', 'engine_power', 'crew_size']:
        for mmsi, value in maps[field].items():
            assert float(value) > 0, (mmsi, value)

    results['fine'] = ClassificationExtractor(maps['label'])

    ext = AttributeExtractor('length', maps['length'], maps['label'])
    results['length'] = ext
    ext = AttributeExtractor('tonnage', maps['tonnage'], maps['label'])
    results['tonnage'] = ext
    ext = AttributeExtractor('engine_power', maps['engine_power'],
                             maps['label'])
    results['engine_power'] = ext
    ext = AttributeExtractor('crew_size', maps['crew_size'],
                             maps['label']) 
    results['crew_size'] = ext       

    logging.info('Loading inference data')
    whitelist = None
    load_inferred(args.inference_table, training_data_table, results.values(), whitelist)

    # Sanity check attribute values after loading
    for field in ['length', 'tonnage', 'engine_power', 'crew_size']:
        if not all(results[field].inferred_attrs >= 0):
            logging.warning(
                'Inferred values less than zero for %s (%s, %s / %s)',
                field, min(results[field].inferred_attrs),
                (results[field].inferred_attrs < 0).sum(),
                len(results[field].inferred_attrs))

    # Assemble coarse and is_fishing scores:
    logging.info('Assembling coarse data')
    results['coarse'] = assemble_composite(results['fine'], coarse_mapping)
    logging.info('Assembling fishing data')
    results['fishing'] = assemble_composite(results['fine'],
                                            fishing_mapping)
    return results


def dump_html(args, results):

    doc = yattag.Doc()

    with doc.tag('style', type='text/css'):
        doc.asis(css)

    for key, heading in CLASSIFICATION_METRICS:
        if results[key]:
            logging.info('Dumping "{}"'.format(heading))
            doc.line('h2', heading)
            ydump_metrics(doc, results[key])
            doc.stag('hr')

    logging.info('Dumping Length')
    doc.line('h2', 'Length Inference')
    ydump_attrs(doc, results['length'])
    doc.stag('hr')
    logging.info('Dumping Tonnage')
    doc.line('h2', 'Tonnage Inference')
    ydump_attrs(doc, results['tonnage'])
    doc.stag('hr')
    logging.info('Dumping Engine Power')
    doc.line('h2', 'Engine Power Inference')
    ydump_attrs(doc, results['engine_power'])
    doc.stag('hr')
    logging.info('Dumping Crew Size')
    doc.line('h2', 'Crew Size Inference')
    ydump_attrs(doc, results['crew_size'])
    doc.stag('hr')


    with open(args.dest_path, 'w') as f:
        logging.info('Writing output')
        f.write(yattag.indent(doc.getvalue(), indent_text=True))



this_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(this_dir, 'temp')
training_data_table = 'machine_learning_dev_ttl_120d.temp_training_data_table'

def create_label_table(label_path):
    raw = pd.read_csv(label_path)
    filtered = raw[['mmsi',
                     'length',
                     'tonnage',
                     'engine_power',
                     'crew_size',
                     'label',
                     'split']]
    filtered.to_gbq(training_data_table, if_exists='replace',  project_id='world-fishing-827')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
        description='Test inference results and output metrics.\n')
    parser.add_argument(
        '--inference-table', help='table of inference results', required=True)
    parser.add_argument(
        '--label-path', help='path to test labels', required=True)
    parser.add_argument(
        '--dest-path', help='path to write results to', required=True)

    args = parser.parse_args()

    create_label_table(args.label_path)
    results = compute_results(args)

    dump_html(args, results)

 
