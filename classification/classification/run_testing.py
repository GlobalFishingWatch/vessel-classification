from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import csv
import subprocess
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import dateutil.parser

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


def plot_confusion_matrix(cm, labels, title, cmap=plt.cm.Blues):
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

def accuracy(true_labels, inferred_labels, scores, threshold):
    overall_true_positives = (inferred_labels == true_labels) & (scores >= threshold)
    return overall_true_positives.sum() / len(inferred_labels)

def accuracy_for_date(date, true_labels, inferred_labels, scores, dates, threshold):
    mask = (dates == date)
    return accuracy(true_labels[mask], inferred_labels[mask], scores[mask], threshold)


this_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(this_dir, "temp")

if __name__ == "__main__":
    # TODO[bitsofbits]: add argparse and pass in parameters instead of hardcoding
    label_path = "gs://world-fishing-827/scratch/alex/infered_labels.txt"
    test_path = "data/test_list.csv"
    THRESHOLD = 0.5
    # TODO:
    #    * Date range
    #    * Bug Alex about dumping all scores
    #    * Tickets
    #    * Exclude test from training


    #
    local_label_path = os.path.join(temp_dir, os.path.basename(label_path))
    # Download labels if they don't already exist
    if not os.path.exists(local_label_path):
        subprocess.check_call(["gsutil", "cp", label_path, local_label_path])
    # Load the test_labels
    with open(test_path) as f:
        label_map = {x['mmsi'].strip(): fix_label(x['label']) for x in csv.DictReader(f)}
    # load inferred_labels and generate comparison data
    start_dates = []
    inferred_labels = []
    true_labels = []
    scores = []
    with open(local_label_path) as f:
        for row in csv.reader(f):
            mmsi, start, stop, label, score = [x.strip() for x in row]
            if mmsi in label_map:
                lbl = label_map[mmsi]
                if lbl == "Unknown":
                    continue
                start_dates.append(dateutil.parser.parse(start))
                true_labels.append(label_map[mmsi])
                inferred_labels.append(label)
                scores.append(float(score))
    starts = sorted(set(start_dates))
    inferred_labels = np.array(inferred_labels)
    true_labels = np.array(true_labels)
    start_dates = np.array(start_dates)
    scores = np.array(scores)
    # Find scoring values
    labels = sorted(set(true_labels) | set(inferred_labels))
    for x in starts:
        print("Accuracy for", x, accuracy_for_date(x, true_labels, inferred_labels, scores, start_dates, THRESHOLD))
    print("Overall Accuracy", accuracy(true_labels, inferred_labels, scores, THRESHOLD))
    # true_positives
    cm = metrics.confusion_matrix(true_labels, inferred_labels, labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1) + 1e-10)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_normalized, labels, "Confusion Matrix")
    plt.show()
    inferred_labels[scores < THRESHOLD] = "Unknown"
    for lbl, prec, rec in precision_recall(labels, true_labels, inferred_labels):
        print(lbl, prec, rec)
