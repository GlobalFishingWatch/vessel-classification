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
from collections import namedtuple
import sys
import yattag

InferenceResults = namedtuple("InferenceResults", ["mmsi", "inferred_labels", "true_labels", "start_dates", "scores", "label_set"])
ConfusionMatrix = namedtuple("ConfusionMatrix", ["raw", "scaled"])




# Helper function formatting as HTML (using yattage)


def repr_confusion_matrix(doc, cm, labels, **kwargs):
    doc, tag, text, line = doc.ttl()
    with tag('table', **kwargs):
        with tag('tr'):
            line('th', "")
            for x in labels:
                with tag('th', style="height: 140px; white-space: nowrap;"):
                    with tag('div', style="transform: translate(25px, 51px) rotate(315deg); width: 30px;"):
                        with tag('span', style="border-bottom: 1px solid #ccc; padding: 5px 10px; text-align: left;"):
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

    rows = [(x, accuracy_for_date(x, results.true_labels, results.inferred_labels, results.scores, results.start_dates, THRESHOLD))
                for x in np.unique(results.start_dates)]

    line('h2', "Accuracy by Date")
    repr_table(doc, ["Start Date", "Accuracy"], [(a.date(), "{:.2f}".format(b)) for (a, b) in rows],
        border=1)

    consolidated = consolidate_across_dates(results)

    line('h2', 'Overall Accuracy')
    text(str(accuracy(consolidated.true_labels, consolidated.inferred_labels)))#, consolidated.scores, THRESHOLD)))

    cm = confusion_matrix(results)

    line('h2', "Confusion Matrix")
    repr_confusion_matrix(doc, cm.scaled, results.label_set)

    line('h2', 'Metrics by Gear Type')
    results.inferred_labels[results.scores < THRESHOLD] = "Unknown"
    repr_table(doc, ["Label","Precision","Recall"],precision_recall(results.label_set, results.true_labels, results.inferred_labels))


#
# Process the tests
#

def fix_label(x):
    x = x.strip()
    if x == 'Tug_Pilot_Supply':
        x = "Tug/Pilot/Supply"
    elif x == "Cargo_Tanker":
        x = "Cargo/Tanker"
    elif x == "Seismic Vessel":
        x = "Seismic"
    else:
        x = x.replace("_", " ")
    return {'Dredger' : 'Trawler',
            'Pleasure Craft' : "Passenger",
            'Sail' : 'Passenger'}.get(x, x)

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

def accuracy(true_labels, inferred_labels):#, scores, threshold):

    overall_true_positives = (inferred_labels == true_labels) #& (scores >= threshold)
    return overall_true_positives.sum() / len(inferred_labels)

def accuracy_for_date(date, true_labels, inferred_labels, scores, dates, threshold):
    mask = (dates == date)
    return accuracy(true_labels[mask], inferred_labels[mask]) #, scores[mask], threshold)


def consolidate_across_dates(results):
    #TODO: once we have full scores, use those instead
    inferred_labels = []
    true_labels = []
    mmsi = sorted(set(results.mmsi))
    label_map = {x : i for (i, x) in enumerate(results.label_set)}
    for m in mmsi:
        mask = (results.mmsi == m)
        scores = np.zeros(len(results.label_set), dtype=float)
        for i in np.arange(len(results.mmsi))[mask]:
            scores[label_map[results.inferred_labels[i]]] += results.scores[i]
        inferred_labels.append(results.label_set[np.argmax(scores)])
        true_labels.append(results.true_labels[mask][0])
    return InferenceResults(results.mmsi, np.array(inferred_labels), np.array(true_labels), None, None, results.label_set)



this_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(this_dir, "temp")



def confusion_matrix(results):
    cm_raw = metrics.confusion_matrix(results.true_labels, results.inferred_labels, results.label_set)
    cm_normalized = cm_raw.astype('float') / (cm_raw.sum(axis=1) + 1e-10)[:, np.newaxis]
    return ConfusionMatrix(cm_raw, cm_normalized)


def load_inferred(inferrence_path, label_map):
    # load inferred_labels and generate comparison data
    start_dates = []
    inferred_labels = []
    true_labels = []
    scores = []
    mmsi_list = []
    with open(inferrence_path) as f:
        for row in csv.reader(f):
            mmsi, start, stop, label, score = [x.strip() for x in row]
            if mmsi in label_map:
                lbl = label_map[mmsi]
                if lbl == "Unknown":
                    continue
                mmsi_list.append(mmsi)
                start_dates.append(dateutil.parser.parse(start))
                true_labels.append(label_map[mmsi])
                inferred_labels.append(label)
                scores.append(float(score))
    inferred_labels = np.array(inferred_labels)
    true_labels = np.array(true_labels)
    start_dates = np.array(start_dates)
    scores = np.array(scores)
    label_set = sorted(set(true_labels) | set(inferred_labels))
    mmsi_list = np.array(mmsi_list)
    return InferenceResults(mmsi_list, inferred_labels, true_labels, start_dates, scores, label_set)




# TODO:
#    * Date range


if __name__ == "__main__":
    logging.getLogger().setLevel("WARNING")
    #
    parser = argparse.ArgumentParser(
        description="Test inference results and output metrics")
    parser.add_argument(
        '--inferrence-path',
        help='path to inferrence results',
        default="gs://world-fishing-827/scratch/alex/infered_labels.txt") # TODO: there is more recent data path (in JSON so will need adjusting)
    parser.add_argument(
        '--test-path',
        help='path to test data',
        default='classification/data/test_list.csv'
        )
    parser.add_argument(
        '--dest-path', help="path to write results to",
        required=True
        )
    parser.add_argument(
        '--plot-confusion', help='plot confusion matrix (run with pythonw)',
        action='store_true'
        )
    args = parser.parse_args()
    #
    THRESHOLD = 0.5



    #
    inferrence_path = os.path.join(temp_dir, os.path.basename(args.inferrence_path))
    # Download labels if they don't already exist
    if not os.path.exists(inferrence_path):
        subprocess.check_call(["gsutil", "cp", label_path, inferrence_path])
    #
    # Load stuff
    with open(args.test_path) as f:
        label_map = {x['mmsi'].strip(): fix_label(x['label']) for x in csv.DictReader(f)}

    results = load_inferred(inferrence_path, label_map)

    # Dump out as HTML
    doc= yattag.Doc()

    repr_metrics(doc, results)

    doc.line('br', '')

    with open(args.dest_path, 'w') as f:
        f.write(yattag.indent(doc.getvalue(), indent_text = True))

