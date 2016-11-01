"""

Example:



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

InferenceResults = namedtuple("InferenceResults",
                              ["mmsi", "inferred_labels", "true_labels",
                               "start_dates", "scores", "label_set"])
ConfusionMatrix = namedtuple("ConfusionMatrix", ["raw", "scaled"])

# Helper function formatting as HTML (using yattage)


def repr_confusion_matrix(doc, cm, labels, **kwargs):
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
        for i, (l, row) in enumerate(zip(labels, cm)):
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
                    else:
                        cval = np.clip(int(round(255 * x)), 0, 255)
                        hexcode = "{:02x}".format(cval)
                        invhexcode = "{:02x}".format(255 - cval)
                        color = "#FF{}{}".format(invhexcode, invhexcode)
                    with tag('td', bgcolor=color):
                        line('font', "{0:.2f}".format(x), color="#000000")


def repr_table(doc, headings, rows, **kwargs):
    doc, tag, text, line = doc.ttl()
    with tag('table', **kwargs):
        with tag('tr'):
            for x in headings:
                line('th', str(x))
        for row in rows:
            with tag('tr'):
                for x in row:
                    line('td', str(x))


def repr_metrics(doc, results):
    doc, tag, text, line = doc.ttl()

    rows = [
        (x, accuracy_for_date(x, results.true_labels, results.inferred_labels,
                              results.scores, results.start_dates, THRESHOLD))
        for x in np.unique(results.start_dates)
    ]

    line('h3', "Accuracy by Date")
    repr_table(
        doc, ["Start Date", "Accuracy"],
        [(a.date(), "{:.2f}".format(b)) for (a, b) in rows],
        border=1)

    consolidated = consolidate_across_dates(results)

    line('h3', 'Overall Accuracy')
    text(str(accuracy(consolidated.true_labels, consolidated.
                      inferred_labels)))  #, consolidated.scores, THRESHOLD)))

    cm = confusion_matrix(consolidated)

    line('h3', "Confusion Matrix")
    repr_confusion_matrix(doc, cm.scaled, results.label_set)

    line('h3', 'Metrics by Label')
    # TODO MAKE threshold work
    # results.inferred_labels[consolidated.scores < THRESHOLD] = "Unknown"
    repr_table(doc, ["Label", "Precision", "Recall"], precision_recall(
        consolidated.label_set, consolidated.true_labels,
        consolidated.inferred_labels))

#
# Process the tests
#


def fix_label(x):
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


def accuracy(true_labels, inferred_labels):  #, scores, threshold):

    overall_true_positives = (
        inferred_labels == true_labels)  #& (scores >= threshold)
    return overall_true_positives.sum() / len(inferred_labels)


def accuracy_for_date(date, true_labels, inferred_labels, scores, dates,
                      threshold):
    mask = (dates == date)
    return accuracy(true_labels[mask],
                    inferred_labels[mask])  #, scores[mask], threshold)


def consolidate_across_dates(results):
    inferred_labels = []
    true_labels = []
    mmsi = sorted(set(results.mmsi))
    all_labels = set(results.label_set)
    for scores in results.scores:
        all_labels |= set(scores.keys())
    label_map = {x: i for (i, x) in enumerate(sorted(all_labels))}
    for m in mmsi:
        mask = (results.mmsi == m)
        scores = np.zeros(len(results.label_set), dtype=float)
        for i in np.arange(len(results.mmsi))[mask]:
            for lbl in results.scores[i]:
                scores[label_map[lbl]] += results.scores[i][lbl]
        inferred_labels.append(results.label_set[np.argmax(scores)])
        true_labels.append(results.true_labels[mask][0])
    return InferenceResults(results.mmsi, np.array(inferred_labels),
                            np.array(true_labels), None, None,
                            results.label_set)


this_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(this_dir, "temp")


def confusion_matrix(results):
    cm_raw = metrics.confusion_matrix(
        results.true_labels, results.inferred_labels, results.label_set)
    cm_normalized = cm_raw.astype('float') / (
        cm_raw.sum(axis=1) + 1e-10)[:, np.newaxis]
    return ConfusionMatrix(cm_raw, cm_normalized)


def find_length_cutpoints(inference_path):
    with nlj.open(inference_path) as src:
        for row in src:
            scores = row['labels']['length']['label_scores']
            if scores:
                raw_cutpoints = scores.keys()
                break
    cutpoints = []
    for rcp in raw_cutpoints:
        if rcp.endswith('m+'):
            cp = float(rcp[:-2])
        else:
            endstr = rcp.split('-')[0]
            cp = float(endstr)
        cutpoints.append((cp, rcp))
    cutpoints.sort()
    return cutpoints


def cut_length_at(cutpoints, len_map):
    map = {}
    cutpoints = sorted(cutpoints)
    for k, v in len_map.items():
        for cp, cut_name in reversed(cutpoints):
            if float(v) >= cp:
                map[k] = cut_name
                break
    return map


def load_inferred(inference_path, label_map, field):
    # load inferred_labels and generate comparison data
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
    label_set = sorted(
        set(true_labels) | set(inferred_labels) | set(label_map.values()))
    mmsi_list = np.array(mmsi_list)
    return InferenceResults(mmsi_list, inferred_labels, true_labels,
                            start_dates, scores, label_set)

# TODO:
#    * Date range
#    * Threshold

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
    THRESHOLD = 0.5

    #
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
    # Load stuff
    maps = defaultdict(dict)
    with open(args.label_path) as f:
        for row in csv.DictReader(f):
            mmsi = int(row['mmsi'].strip())
            for field in ["label", "sublabel", "length"]:
                if row[field]:
                    maps[field][mmsi] = fix_label(row[field])

    results = load_inferred(inference_path, maps['label'], 'label')

    subresults = load_inferred(inference_path, maps['sublabel'], 'sublabel')

    cutpoints = find_length_cutpoints(inference_path)

    len_map = cut_length_at(cutpoints, maps['length'])

    lenresults = load_inferred(inference_path, len_map, 'length')
    lenresults.label_set.sort(
        key=lambda x: float(x.split('-')[0].split('m+')[0]))

    # Dump out as HTML
    doc = yattag.Doc()

    doc.line('h2', "Coarse Labels")
    repr_metrics(doc, results)
    doc.stag('hr')

    doc.line('h2', "Fine Labels", style="page-break-before: always")
    repr_metrics(doc, subresults)
    doc.stag('hr')

    doc.line('h2', "Lengths", style="page-break-before: always")
    repr_metrics(doc, lenresults)
    doc.stag('br')

    with open(args.dest_path, 'w') as f:
        f.write(yattag.indent(doc.getvalue(), indent_text=True))
