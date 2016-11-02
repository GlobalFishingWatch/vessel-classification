"""

Example:

python compute_metrics.py \
    --inference-path  gs://world-fishing-827/scratch/alex/test_vessel_classification_277947.json.gz \
    --label-path ../../mussidae/mussidae/data-precursors/time-range-sources/non-public-sources/mmsi_to_vessel_type.csv \
    --dest-path example_output.html

"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import csv
import subprocess
import numpy as np
from sklearn import metrics
import dateutil.parser
import logging
import argparse
from collections import namedtuple, defaultdict
import sys
import yattag
import newlinejson as nlj
from classification.utility import vessel_categorical_length_transformer, is_test
import gzip

InferenceResults = namedtuple('InferenceResults',
                              ['mmsi', 'inferred_labels', 'true_labels',
                               'start_dates', 'scores', 'label_list'])

ConfusionMatrix = namedtuple('ConfusionMatrix', ['raw', 'scaled'])

css = """

table {
    text-align: center;
    border-collapse: collapse;
}

.confmatrix th.col {
  height: 140px; 
  white-space: nowrap;
}

.confmatrix th.col div {
    transform: translate(25px, 51px) rotate(315deg); 
    width: 30px;
}

.confmatrix th.col span {
    border-bottom: 1px solid #ccc; 
    padding: 5px 10px; 
    text-align: left;
}

.confmatrix th.row {
    text-align: right;
}

.confmatrix td.diagonal {
    border: 1px solid black;
}

.confmatrix td.offdiagonal {
    border: 1px dotted grey;
}

.unbreakable {
    page-break-inside: avoid;
}




"""

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
    with tag('table', klass="confmatrix", **kwargs):
        with tag('tr'):
            line('th', '')
            for x in labels:
                with tag('th', klass="col"):
                    with tag('div'):
                        line('span', x)
        for i, (l, row) in enumerate(zip(labels, cm.scaled)):
            with tag('tr'):
                line('th', str(l), klass="row")
                for j, x in enumerate(row):
                    if i == j:
                        if x > 0.5:
                            cval = np.clip(int(round(512 * (x - 0.5))), 0, 255)
                            invhexcode = '{:02x}'.format(255 - cval)
                            color = '#{}FF00'.format(invhexcode)
                        else:
                            cval = np.clip(int(round(512 * x)), 0, 255)
                            hexcode = '{:02x}'.format(cval)
                            color = '#FF{}00'.format(hexcode)
                        klass = "diagonal"
                    else:
                        cval = np.clip(int(round(255 * x)), 0, 255)
                        hexcode = '{:02x}'.format(cval)
                        invhexcode = '{:02x}'.format(255 - cval)
                        color = '#FF{}{}'.format(invhexcode, invhexcode)
                        klass = "offdiagonal"
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


def ydump_metrics(doc, results):
    """dump metrics for `results` to html using yatag

    Args:
        doc: yatag Doc instance
        results: InferenceResults instance

    """
    doc, tag, text, line = doc.ttl()

    rows = [
        (x, accuracy_for_date(x, results.true_labels, results.inferred_labels,
                              results.scores, results.start_dates))
        for x in np.unique(results.start_dates)
    ]

    with tag('div', klass="unbreakable"):
        line('h3', 'Accuracy by Date')
        ydump_table(doc, ['Start Date', 'Accuracy'],
                    [(a.date(), '{:.2f}'.format(b)) for (a, b) in rows])

    consolidated = consolidate_across_dates(results)

    with tag('div', klass="unbreakable"):
        line('h3', 'Overall Accuracy')
        text('{:.2f}'.format(
            accuracy(consolidated.true_labels, consolidated.inferred_labels)))

    cm = confusion_matrix(consolidated)

    with tag('div', klass="unbreakable"):
        line('h3', 'Confusion Matrix')
        ydump_confusion_matrix(doc, cm, results.label_list)

    with tag('div', klass="unbreakable"):
        line('h3', 'Metrics by Label')
        ydump_table(doc, ['Label', 'Precision', 'Recall'],
                    [(a, '{:.2f}'.format(b), "{:.2f}".format(c))
                     for (a, b, c) in precision_recall(
                         consolidated.label_list, consolidated.true_labels,
                         consolidated.inferred_labels)])

# Helper functions for computing metrics


def clean_label(x):
    x = x.strip()
    return x.replace('_', ' ')


def precision_recall(labels, y_true, y_pred):
    results = []
    for lbl in labels:
        positives = (y_pred == lbl)
        trues = (y_true == lbl)
        true_positives = (positives & trues)
        precision = true_positives.sum() / (positives.sum() + 1e-10)
        recall = true_positives.sum() / (trues.sum() + 1e-10)
        results.append((lbl, precision, recall))
    return results


def accuracy(true_labels, inferred_labels):
    overall_true_positives = (inferred_labels == true_labels)
    return overall_true_positives.sum() / len(inferred_labels)


def accuracy_for_date(date, true_labels, inferred_labels, scores, dates):
    mask = (dates == date)
    return accuracy(true_labels[mask], inferred_labels[mask])


def consolidate_across_dates(results):
    """Consolidate scores for each MMSI across available dates.

    For each mmsi, we take the scores at all available dates, sum
    them and use argmax to find the predicted results.

    """
    inferred_labels = []
    true_labels = []
    mmsi = sorted(set(results.mmsi))
    label_map = {x: i for (i, x) in enumerate(results.label_list)}
    for m in mmsi:
        mask = (results.mmsi == m)
        scores = np.zeros(len(results.label_list), dtype=float)
        for i in np.arange(len(results.mmsi))[mask]:
            for lbl in results.scores[i]:
                try:
                    scores[label_map[lbl]] += results.scores[i][lbl]
                except:
                    print(label_map, lbl, results.scores[i].keys())
                    raise
        inferred_labels.append(results.label_list[np.argmax(scores)])
        true_labels.append(results.true_labels[mask][0])
    return InferenceResults(results.mmsi, np.array(inferred_labels),
                            np.array(true_labels), None, None,
                            results.label_list)


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
    cm_raw = metrics.confusion_matrix(
        results.true_labels, results.inferred_labels, results.label_list)

    # For off axis, normalize harmonic mean of row / col inverse errors.
    # The idea here is that this average will go to 1 => BAD, as
    # either the row error or column error approaches 1. That is, if this
    # off diagonal element dominates eitehr the predicted values for this 
    # label OR the actual values for this label.  A standard mean will only
    # go to zero if it dominates both, but these can become decoupled with 
    # unbalanced classes.
    row_totals = cm_raw.sum(axis=1, keepdims=True) + EPS
    col_totals = cm_raw.sum(axis=0, keepdims=True) + EPS
    inv_row_fracs = 1 - cm_raw / row_totals
    inv_col_fracs = 1 - cm_raw / col_totals
    cm_normalized = 1 - harmonic_mean(inv_col_fracs, inv_row_fracs)
    # For on axis, use the F1-score (also a harmonic mean!)
    for i in range(len(cm_raw)):
        recall = cm_raw[i, i] / row_totals[i, 0]
        precision = cm_raw[i, i] / col_totals[0, i]
        cm_normalized[i, i] = harmonic_mean(recall, precision)

    return ConfusionMatrix(cm_raw, cm_normalized)


def remap_lengths(len_map):
    map = {}
    for k, v in len_map.items():
        map[k] = vessel_categorical_length_transformer(v)
    return map


def open_gzip_or_regular(path):
    if path.endswith('.gz'):
        return gzip.GzipFile(path)
    else:
        return open(path)


def load_inferred(inference_path, label_map, field):
    """Load inferred data and generate comparison data

    """
    start_dates = []
    inferred_labels = []
    true_labels = []
    scores = []
    mmsi_list = []
    all_labels = set()
    with open_gzip_or_regular(inference_path) as f:
        with nlj.open(f) as src:
            for row in src:
                mmsi = row['mmsi']
                if mmsi in label_map:
                    lbl = label_map[mmsi]
                    if lbl == 'Unknown':
                        continue
                    mmsi_list.append(mmsi)
                    label_scores = row['labels'][field]['label_scores']
                    all_labels |= set(label_scores.keys())
                    start_dates.append(
                        dateutil.parser.parse(row['start_time']))
                    true_labels.append(label_map[mmsi])
                    inferred_labels.append(row['labels'][field]['max_label'])
                    scores.append(label_scores)
    inferred_labels = np.array(inferred_labels)
    true_labels = np.array(true_labels)
    start_dates = np.array(start_dates)
    scores = np.array(scores)
    label_list = sorted(all_labels)
    mmsi_list = np.array(mmsi_list)
    return InferenceResults(mmsi_list, inferred_labels, true_labels,
                            start_dates, scores, label_list)


def get_local_inference_path(args):
    """Return a local path to inference data.

    Data is downloaded to a temp directory if on GCS. 

    NOTE: if a correctly named local file is already present, new data
          will not be downloaded.
    """
    if args.inference_path.startswith('gs'):
        inference_path = os.path.join(temp_dir,
                                      os.path.basename(args.inference_path))
        if not os.path.exists(inference_path):
            subprocess.check_call(
                ['gsutil', 'cp', args.inference_path, inference_path])
    else:
        inference_path = args.inference_path
    #
    return inference_path


def compute_results(args):
    inference_path = get_local_inference_path(args)

    maps = defaultdict(dict)
    with open(args.label_path) as f:
        for row in csv.DictReader(f):
            mmsi = int(row['mmsi'].strip())
            if not is_test(mmsi):
                continue
            for field in ['is_fishing', 'label', 'sublabel', 'length']:
                if row[field]:
                    maps[field][mmsi] = clean_label(row[field])

    results = {}

    results['fishing'] = load_inferred(inference_path, maps['is_fishing'],
                                       'is_fishing')

    results['coarse'] = load_inferred(inference_path, maps['label'], 'label')

    results['fine'] = load_inferred(inference_path, maps['sublabel'],
                                    'sublabel')

    len_map = remap_lengths(maps['length'])
    results['length'] = load_inferred(inference_path, len_map, 'length')
    results['length'].label_list.sort(
        key=lambda x: float(x.split('-')[0].split('m+')[0]))

    return results


def dump_html(args, results):

    classification_metrics = [
        ('fishing', 'Is Fishing'), ('coarse', 'Coarse Labels'),
        ('fine', "Fine Labels"), ('length', 'Lengths')
    ]

    doc = yattag.Doc()

    with doc.tag("style", type="text/css"):
        doc.asis(css)

    for key, heading in classification_metrics:
        doc.line('h2', heading)
        ydump_metrics(doc, results[key])
        doc.stag('hr')

    with open(args.dest_path, 'w') as f:
        f.write(yattag.indent(doc.getvalue(), indent_text=True))

# TODO:
#    * Date range
#    * Thresholds
#    * Load all labels at once for better speed
#    * Use real temp directory (current approach good for development); remove `temp` from gitignore

this_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(this_dir, 'temp')

if __name__ == '__main__':
    logging.getLogger().setLevel('WARNING')

    parser = argparse.ArgumentParser(
        description='Test inference results and output metrics')
    parser.add_argument(
        '--inference-path', help='path to inference results', required=True)
    parser.add_argument(
        '--label-path', help='path to test data', required=True)
    parser.add_argument(
        '--dest-path', help='path to write results to', required=True)
    parser.add_argument(
        '--plot-confusion',
        help='plot confusion matrix (run with pythonw)',
        action='store_true')
    args = parser.parse_args()

    results = compute_results(args)

    dump_html(args, results)
