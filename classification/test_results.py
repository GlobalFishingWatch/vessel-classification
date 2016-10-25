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

InferenceResults = namedtuple("InferenceResults", ["mmsi", "inferred_labels", "true_labels", "start_dates", "scores", "label_set"])


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


def plot_confusion_matrix(cm, labels, title, cmap=None):
    import matplotlib.pyplot as plt
    if cmap is None:
        cmap = plt.cm.Blues
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.3)
    plt.show()


def repr_confusion_matrix(cm, labels):
    lines = [","+",".join(labels)]
    for l, row in zip(labels, cm):
        lines.append("{},".format(l) + ",".join([str(x) for x in row]))
    return("\n".join(lines))


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


ConfusionMatrix = namedtuple("ConfusionMatrix", ["raw", "scaled"])

def confusion_matrix(results):
    cm_raw = metrics.confusion_matrix(results.true_labels, results.inferred_labels, label_set)
    cm_normalized = cm_raw.astype('float') / (cm_raw.sum(axis=1) + 1e-10)[:, np.newaxis]
    return ConfusionMatrix(cm_raw, cm_normalized)

# TODO:
#    * Date range

# Pass in field_name
# Pass in date_range
# pass in --confusion_matrix [PATH | --] defaults to --
# pass in --metrics [PATH | --] defaults to --
# pass 

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
        default = None
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
    # Load the test_labels
    with open(args.test_path) as f:
        label_map = {x['mmsi'].strip(): fix_label(x['label']) for x in csv.DictReader(f)}

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
    starts = sorted(set(start_dates))
    inferred_labels = np.array(inferred_labels)
    true_labels = np.array(true_labels)
    start_dates = np.array(start_dates)
    scores = np.array(scores)
    label_set = sorted(set(true_labels) | set(inferred_labels))
    mmsi_list = np.array(mmsi_list)
    results = InferenceResults(mmsi_list, inferred_labels, true_labels, start_dates, scores, label_set)
    #
    output = sys.stdout if (args.dest_path is None) else open(args.dest_path)
    def write(*args):
        for x in args:
            output.write(str(x) + " ")
        output.write("\n")

    try:
        # Find scoring values
        for x in starts:
            write("Accuracy for",
                x, accuracy_for_date(x, results.true_labels, results.inferred_labels, results.scores, results.start_dates, THRESHOLD))
        write()

        consolidated = consolidate_across_dates(results)

        write("Overall Accuracy", accuracy(consolidated.true_labels, consolidated.inferred_labels))#, consolidated.scores, THRESHOLD))
        write()
        # 
        cm = confusion_matrix(results)
        
        if args.plot_confusion:
            plot_confusion_matrix(cm.scaled, results.label_set, "Confusion Matrix")


        inferred_labels[scores < THRESHOLD] = "Unknown"
        write("Label,Precision,Recall")
        for lbl, prec, rec in precision_recall(results.label_set, results.true_labels, results.inferred_labels):
            write(lbl, prec, rec)
        write()

        write(repr_confusion_matrix(cm.scaled, results.label_set))
    finally:
        if output is not sys.stdout:        
            output.close()
