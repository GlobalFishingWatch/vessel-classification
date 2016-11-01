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
from classification.utility import vessel_categorical_length_transformer

InferenceResults = namedtuple("InferenceResults",
                              ["mmsi", "inferred_labels", "true_labels",
                               "start_dates", "scores", "label_list"])

ConfusionMatrix = namedtuple("ConfusionMatrix", ["raw", "scaled"])



# Helper function formatting as HTML (using yattag)

def ydump_confusion_matrix(doc, cm, labels, **kwargs):
    """Dump an sklearn confusiong matrix to HTML using yatag

    Args:
        doc: yatag Doc instance
        cm: ConfusionMatrix instance
        labels: list of str
            labels for confusion matrix
    """
    doc, tag, text, line = doc.ttl()
    with tag('table', **kwargs):
        with tag('tr'):
            line('th', "")
            for x in labels:
                with tag('th', style="height: 140px; white-space: nowrap;"):
                    with tag(
                            'div',
                            style="transform: translate(25px, 51px) rotate(315deg); width: 30px;"):
                        with tag(
                                'span',
                                style="border-bottom: 1px solid #ccc; padding: 5px 10px; text-align: left;"):
                            text(x)
        for i, (l, row) in enumerate(zip(labels, cm.scaled)):
            with tag('tr'):
                line('th', str(l), style="text-align: right;")
                for j, x in enumerate(row):
                    if i == j:
                        if x > 0.5:
                            cval = np.clip(int(round(512 * (x - 0.5))), 0, 255)
                            invhexcode = "{:02x}".format(255 - cval)
                            color = "#{}FF00".format(invhexcode)
                        else:
                            cval = np.clip(int(round(512 * x)), 0, 255)
                            hexcode = "{:02x}".format(cval)
                            color = "#FF{}00".format(hexcode)
                        style = 'border: 1px solid black; text-align: center;'
                    else:
                        cval = np.clip(int(round(255 * x)), 0, 255)
                        hexcode = "{:02x}".format(cval)
                        invhexcode = "{:02x}".format(255 - cval)
                        color = "#FF{}{}".format(invhexcode, invhexcode)
                        style = 'text-align: center;'
                    with tag('td', style=style, bgcolor=color):
                        raw = cm.raw[i, j]
                        with tag('font', color="#000000", title="{0:.3f}".format(x)):
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

    with tag('div', style="page-break-inside: avoid"):
        line('h3', "Accuracy by Date")
        ydump_table(
            doc, ["Start Date", "Accuracy"],
            [(a.date(), "{:.2f}".format(b)) for (a, b) in rows],
            border=1)

    consolidated = consolidate_across_dates(results)

    with tag('div', style="page-break-inside: avoid"):
        line('h3', 'Overall Accuracy')
        text(str(accuracy(consolidated.true_labels, consolidated.
                          inferred_labels)))

    cm = confusion_matrix(consolidated)

    with tag('div', style="page-break-inside: avoid"):
        line('h3', "Confusion Matrix")
        ydump_confusion_matrix(doc, cm, results.label_list)

    with tag('div', style="page-break-inside: avoid"):
        line('h3', 'Metrics by Label')
        ydump_table(doc, ["Label", "Precision", "Recall"], precision_recall(
            consolidated.label_list, consolidated.true_labels,
            consolidated.inferred_labels))


# Helper functions for computing metrics


def clean_label(x):
    x = x.strip()
    return x.replace("_", " ")


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
                scores[label_map[lbl]] += results.scores[i][lbl]
        inferred_labels.append(results.label_list[np.argmax(scores)])
        true_labels.append(results.true_labels[mask][0])
    return InferenceResults(results.mmsi, np.array(inferred_labels),
                            np.array(true_labels), None, None,
                            results.label_list)



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
    # For off axis, normalize using the false positive rate
    cm_normalized = cm_raw.astype('float') / (
        cm_raw.sum(axis=1, keepdims=True) + EPS)
    # For on axis, use the F1-score, what is already there is the recall
    col_totals = cm_raw.sum(axis=0) + EPS   
    for i in range(len(cm_raw)):
        recall = cm_normalized[i, i]
        prec = cm_raw[i, i] / col_totals[i]
        F1 = 2.0 / ((1.0 / recall) + (1.0 / prec))
        cm_normalized[i, i] = F1
    #
    return ConfusionMatrix(cm_raw, cm_normalized)



def remap_lengths(len_map):
    map = {}
    for k, v in len_map.items():
        map[k] = vessel_categorical_length_transformer(v)
    return map


def load_inferred(inference_path, label_map, field):
    """Load inferred data and generate comparison data

    """
    start_dates = []
    inferred_labels = []
    true_labels = []
    scores = []
    mmsi_list = []
    with nlj.open(inference_path) as src:
        for row in src:
            mmsi = row['mmsi']
            if mmsi in label_map:
                lbl = label_map[mmsi]
                if lbl == "Unknown":
                    continue
                mmsi_list.append(mmsi)
                start_dates.append(dateutil.parser.parse(row['start_time']))
                true_labels.append(label_map[mmsi])
                inferred_labels.append(row['labels'][field]['max_label'])
                scores.append(row['labels'][field]['label_scores'])
    inferred_labels = np.array(inferred_labels)
    true_labels = np.array(true_labels)
    start_dates = np.array(start_dates)
    scores = np.array(scores)
    label_list = sorted(
        set(true_labels) | set(inferred_labels) | set(label_map.values()))
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
        initial_inference_path = os.path.join(
            temp_dir, os.path.basename(args.inference_path))
        # Download labels if they don't already exist
        is_compressed = initial_inference_path.endswith('.gz')
        inference_path = initial_inference_path[:
                                                -3] if is_compressed else initial_inference_path
        if not os.path.exists(inference_path):
            subprocess.check_call(
                ["gsutil", "cp", args.inference_path, initial_inference_path])
            if is_compressed:
                subprocess.check_call(['gunzip', initial_inference_path])

    else:
        inference_path = args.inference_path  
    #
    return inference_path 


def compute_results(args):
    inference_path = get_local_inference_path(args)
    #
    maps = defaultdict(dict)
    with open(args.label_path) as f:
        for row in csv.DictReader(f):
            mmsi = int(row['mmsi'].strip())
            for field in ["is_fishing", "label", "sublabel", "length"]:
                if row[field]:
                    maps[field][mmsi] = clean_label(row[field])

    results = {}

    results['fishing'] = load_inferred(inference_path, maps['is_fishing'], 'is_fishing')

    results['coarse'] = load_inferred(inference_path, maps['label'], 'label')

    results['fine'] = load_inferred(inference_path, maps['sublabel'], 'sublabel')

    len_map = remap_lengths(maps['length'])
    results['length'] = load_inferred(inference_path, len_map, 'length')
    results['length'].label_list.sort(
        key=lambda x: float(x.split('-')[0].split('m+')[0]))

    return results


def dump_html(args, results):
    doc = yattag.Doc()

    doc.line('h2', "Is Fishing")
    ydump_metrics(doc, results['fishing'])
    doc.stag('hr')

    doc.line('h2', "Coarse Labels")
    ydump_metrics(doc, results['coarse'])
    doc.stag('hr')

    doc.line('h2', "Fine Labels")
    ydump_metrics(doc, results['fine'])
    doc.stag('hr')

    doc.line('h2', "Lengths")
    ydump_metrics(doc, results['length'])
    doc.stag('br')

    with open(args.dest_path, 'w') as f:
        f.write(yattag.indent(doc.getvalue(), indent_text=True))

# TODO:
#    * Date range
#    * Thresholds
#    * Load all labels at once for better speed
#    * Use real temp directory (current approach good for development); remove `temp` from gitignore
#    * Compute off by one accuracy for Length if we don't move to regression.

this_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(this_dir, "temp")

if __name__ == "__main__":
    logging.getLogger().setLevel("WARNING")
    #
    parser = argparse.ArgumentParser(
        description="Test inference results and output metrics")
    parser.add_argument(
        '--inference-path', help='path to inference results', required=True)
    parser.add_argument(
        '--label-path', help='path to test data', required=True)
    parser.add_argument(
        '--dest-path', help="path to write results to", required=True)
    parser.add_argument(
        '--plot-confusion',
        help='plot confusion matrix (run with pythonw)',
        action='store_true')
    args = parser.parse_args()
    #
    results = compute_results(args)

    dump_html(args, results)


