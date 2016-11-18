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
from classification.utility import is_test
import gzip
import dateutil.parser
import datetime
import pytz

InferenceResults = namedtuple('InferenceResults',
                              ['mmsi', 'inferred_labels', 'true_labels',
                               'start_dates', 'scores', 'label_list'])

LengthResults = namedtuple(
    'LengthResults',
    ['mmsi', 'inferred_lengths', 'true_lengths', 'true_labels', 'start_dates'])

LocalisationResults = namedtuple('LocalisationResults',
                                 ['true_fishing_by_mmsi',
                                  'pred_fishing_by_mmsi', 'label_map'])

ConfusionMatrix = namedtuple('ConfusionMatrix', ['raw', 'scaled'])

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
    with tag('table', klass="confusion-matrix", **kwargs):
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


def ydump_length(doc, results):
    """dump metrics for `results` to html using yatag

    Args:
        doc: yatag Doc instance
        results: InferenceResults instance

    """
    doc, tag, text, line = doc.ttl()

    def RMS(a, b):
        return np.sqrt(np.square(a - b).mean())

    rows = []
    for dt in np.unique(results.start_dates):
        mask = (results.start_dates == dt)
        rows.append([dt, RMS(results.true_lengths[mask],
                             results.inferred_lengths[mask])])

    with tag('div', klass="unbreakable"):
        line('h3', 'RMS Error (m) by Date')
        ydump_table(doc, ['Start Date', 'RMS Error'],
                    [(a.date(), '{:.2f}'.format(b)) for (a, b) in rows])

    consolidated = consolidate_length_across_dates(results)

    with tag('div', klass="unbreakable"):
        line('h3', 'Overall RMS Error')
        text('{:.2f}'.format(
            RMS(consolidated.true_lengths, consolidated.inferred_lengths)))

    def RMS_by_label(true_lengths, pred_lengths, true_labels):
        results = []
        labels = sorted(set(true_labels))
        for lbl in labels:
            mask = (lbl == true_labels)
            err = RMS(true_lengths[mask], pred_lengths[mask])
            results.append((lbl, err, true_lengths[mask].mean()))
        return results

    with tag('div', klass="unbreakable"):
        line('h3', 'RMS Error by Label')
        ydump_table(doc, ['Label', 'RMS Error (m)', 'Mean Length (m)'], [
            (a, '{:.2f}'.format(b), '{:.2f}'.format(c))
            for (a, b, c
                 ) in RMS_by_label(consolidated.true_lengths, consolidated.
                                   inferred_lengths, consolidated.true_labels)
        ])


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


def ydump_fishing_localisation(doc, results):
    doc, tag, text, line = doc.ttl()

    y_true = np.concatenate(results.true_fishing_by_mmsi.values())
    y_pred = np.concatenate(results.pred_fishing_by_mmsi.values())

    header = ["Gear Type", "Precision", "Recall", "Accuracy", "F1-Score"]
    rows = []
    logging.info("Overall localisation accuracy %s",
                 metrics.accuracy_score(y_true, y_pred))
    logging.info("Overall localisation precision %s",
                 metrics.precision_score(y_true, y_pred))
    logging.info("Overall localisation recall %s",
                 metrics.recall_score(y_true, y_pred))

    for cls in sorted(set(results.label_map.values())):
        true_chunks = []
        pred_chunks = []
        for mmsi in results.label_map:
            if mmsi not in results.true_fishing_by_mmsi:
                continue
            if results.label_map[mmsi] != cls:
                continue
            true_chunks.append(results.true_fishing_by_mmsi[mmsi])
            pred_chunks.append(results.pred_fishing_by_mmsi[mmsi])
        if len(true_chunks):
            y_true = np.concatenate(true_chunks)
            y_pred = np.concatenate(pred_chunks)
            rows.append([cls,
                         metrics.precision_score(y_true, y_pred),
                         metrics.recall_score(y_true, y_pred),
                         metrics.accuracy_score(y_true, y_pred),
                         metrics.f1_score(y_true, y_pred), ])

    rows.append(['', '', '', '', ''])

    y_true = np.concatenate(results.true_fishing_by_mmsi.values())
    y_pred = np.concatenate(results.pred_fishing_by_mmsi.values())

    rows.append(['Overall',
                 metrics.precision_score(y_true, y_pred),
                 metrics.recall_score(y_true, y_pred),
                 metrics.accuracy_score(y_true, y_pred),
                 metrics.f1_score(y_true, y_pred), ])

    with tag('div', klass="unbreakable"):
        ydump_table(
            doc, header,
            [[('{:.2f}'.format(x) if isinstance(x, float) else x) for x in row]
             for row in rows])

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
                    logging.error(label_map, lbl, results.scores[i].keys())
                    raise
        inferred_labels.append(results.label_list[np.argmax(scores)])
        true_labels.append(results.true_labels[mask][0])
    return InferenceResults(mmsi, np.array(inferred_labels),
                            np.array(true_labels), None, None,
                            results.label_list)


def consolidate_length_across_dates(results):
    """Consolidate scores for each MMSI across available dates.

    For each mmsi, we average the lengths across all available dates

    """
    inferred_lengths = []
    true_lengths = []
    true_labels = []
    mmsi = sorted(set(results.mmsi))
    for m in mmsi:
        mask = (results.mmsi == m)
        inferred_lengths.append(results.inferred_lengths[mask].mean())
        true_lengths.append(results.true_lengths[mask].mean())
        true_labels.append(results.true_labels[mask][0])
    return LengthResults(mmsi, np.array(inferred_lengths),
                         np.array(true_lengths), np.array(true_labels), None)


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


def load_inferred(inference_path, label_map, field):
    """Load inferred data and generate comparison data

    """
    start_dates = []
    inferred_labels = []
    true_labels = []
    scores = []
    mmsi_list = []
    all_labels = set()
    with gzip.GzipFile(inference_path) as f:
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


def load_lengths(inference_path, length_map, label_map):
    """Load inferred data and generate comparison data

    """
    start_dates = []
    inferred_lengths = []
    true_lengths = []
    true_labels = []
    mmsi_list = []
    with gzip.GzipFile(inference_path) as f:
        with nlj.open(f) as src:
            for row in src:
                mmsi = row['mmsi']
                if mmsi in length_map:
                    lbl = label_map[mmsi]
                    if lbl == 'Unknown':
                        continue
                    mmsi_list.append(mmsi)
                    start_dates.append(
                        dateutil.parser.parse(row['start_time']))
                    true_lengths.append(float(length_map[mmsi]))
                    true_labels.append(label_map.get(mmsi, 'Unknown'))
                    inferred_lengths.append(row['labels']['length']['value'])
    inferred_lengths = np.array(inferred_lengths)
    true_lengths = np.array(true_lengths)
    start_dates = np.array(start_dates)
    mmsi_list = np.array(mmsi_list)
    true_labels = np.array(true_labels)
    return LengthResults(mmsi_list, inferred_lengths, true_lengths,
                         true_labels, start_dates)


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


def load_true_fishing_ranges_by_mmsi(fishing_range_path):
    ranges_by_mmsi = defaultdict(list)
    parse = dateutil.parser.parse
    with open(fishing_range_path) as f:
        for row in csv.DictReader(f):
            mmsi = int(row['mmsi'].strip())
            if not is_test(mmsi):
                continue
            rng = ((float(row['is_fishing']) > 0.5), parse(row['start_time']),
                   parse(row['end_time']))
            ranges_by_mmsi[mmsi].append(rng)
    return ranges_by_mmsi


def load_predicted_fishing_ranges_by_mmsi(inference_path, mmsi_set):
    ranges_by_mmsi = defaultdict(list)
    coverage_by_mmsi = defaultdict(list)

    # Faster than using dateutil
    def parse(x):
        # 2014-08-28T13:56:16+00:00
        # TODO: fix generation to generate consistent datetimes
        if x[-6:] == "+00:00":
            x = x[:-6]
        dt = datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")
        return dt.replace(tzinfo=pytz.UTC)

    with gzip.GzipFile(inference_path) as f:
        with nlj.open(f) as src:
            for row in src:
                mmsi = row['mmsi']
                if mmsi not in mmsi_set:
                    continue
                if not is_test(mmsi):
                    info.warning("%s is not a test mmsi", mmsi)
                rng = [(parse(a), parse(b))
                       for (a, b) in row['labels']['fishing_localisation']]
                ranges_by_mmsi[mmsi].extend(rng)
                coverage_by_mmsi[mmsi].append(
                    (parse(row['start_time']), parse(row['end_time'])))
    return ranges_by_mmsi, coverage_by_mmsi


def datetime_to_minute(dt):
    timestamp = (dt - datetime.datetime(
        1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
    return int(timestamp // 60)


def compare_fishing_localisation(inference_path, fishing_range_path):

    logging.debug("loading fishing ranges")
    true_ranges_by_mmsi = load_true_fishing_ranges_by_mmsi(fishing_range_path)
    logging.debug("loading predicted fishing")
    pred_ranges_by_mmsi, pred_coverage_by_mmsi = load_predicted_fishing_ranges_by_mmsi(
        inference_path, set(true_ranges_by_mmsi.keys()))

    true_by_mmsi = {}
    pred_by_mmsi = {}

    for mmsi in sorted(true_ranges_by_mmsi.keys()):
        logging.debug("processing %s", mmsi)
        if mmsi not in pred_ranges_by_mmsi:
            continue
        true_ranges = true_ranges_by_mmsi[mmsi]
        if not true_ranges:
            continue

        # Determine minutes from start to finish of this mmsi, create an array to
        # hold results and fill with -1 (unknown)
        logging.debug("processing %s true ranges", len(true_ranges))
        logging.debug("finding overall range")
        _, start, end = true_ranges[0]
        for (_, s, e) in true_ranges[1:]:
            start = min(start, s)
            end = max(end, e)
        start_min = datetime_to_minute(start)
        end_min = datetime_to_minute(end)
        minutes = np.empty([end_min - start_min + 1, 2], dtype=int)
        minutes.fill(-1)

        # Fill in minutes[:, 0] with known true / false values
        logging.debug("filling 0s")
        for (is_fishing, s, e) in true_ranges:
            s_min = datetime_to_minute(s)
            e_min = datetime_to_minute(e)
            for m in range(s_min - start_min, e_min - start_min + 1):
                minutes[m, 0] = is_fishing

        # fill in minutes[:, 1] with 0 (default) in areas with coverage
        logging.debug("filling 1s")
        for (s, e) in pred_coverage_by_mmsi[mmsi]:
            s_min = datetime_to_minute(s)
            e_min = datetime_to_minute(e)
            for m in range(s_min - start_min, e_min - start_min + 1):
                if 0 <= m < len(minutes):
                    minutes[m, 1] = 0

        # fill in minutes[:, 1] with 1 where fishing is predicted
        logging.debug("filling in predicted values")
        for (s, e) in pred_ranges_by_mmsi[mmsi]:
            s_min = datetime_to_minute(s)
            e_min = datetime_to_minute(e)
            for m in range(s_min - start_min, e_min - start_min + 1):
                if 0 <= m < len(minutes):
                    minutes[m, 1] = 1

        mask = ((minutes[:, 0] != -1) & (minutes[:, 1] != -1))

        if mask.sum():
            accuracy = (
                (minutes[:, 0] == minutes[:, 1]) * mask).sum() / mask.sum()
            logging.debug("Accuracy for MMSI %s: %s", mmsi, accuracy)

            true_by_mmsi[mmsi] = minutes[mask, 0]
            pred_by_mmsi[mmsi] = minutes[mask, 1]

    return true_by_mmsi, pred_by_mmsi


def compute_results(args):
    inference_path = get_local_inference_path(args)

    true_fishing_by_mmsi, pred_fishing_by_mmsi = compare_fishing_localisation(
        inference_path, args.fishing_ranges)

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

    if args.compute_localisation_metrics:
        results['localisation'] = LocalisationResults(
            true_fishing_by_mmsi, pred_fishing_by_mmsi, maps['label'])

    if args.compute_class_metrics or args.dump_labels_to:
        results['fishing'] = load_inferred(inference_path, maps['is_fishing'],
                                           'is_fishing')

        results['coarse'] = load_inferred(inference_path, maps['label'],
                                          'label')

        results['fine'] = load_inferred(inference_path, maps['sublabel'],
                                        'sublabel')

    if args.compute_length_metrics:
        results['length'] = load_lengths(inference_path, maps['length'],
                                         maps['label'])

    return results


def dump_html(args, results):

    classification_metrics = [
        ('fishing', 'Is Fishing'),
        ('coarse', 'Coarse Labels'),
        ('fine', "Fine Labels")  #, ('length', 'Lengths')
    ]

    doc = yattag.Doc()

    with doc.tag("style", type="text/css"):
        doc.asis(css)

    if args.compute_class_metrics:
        for key, heading in classification_metrics:
            doc.line('h2', heading)
            ydump_metrics(doc, results[key])
            doc.stag('hr')

    if args.compute_length_metrics:
        doc.line('h2', 'Length Inference')
        ydump_length(doc, results['length'])
        doc.stag('hr')

    if args.compute_localisation_metrics:
        doc.line('h2', 'Fishing Localisation')
        ydump_fishing_localisation(doc, results['localisation'])
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
    logging.getLogger().setLevel('INFO')

    parser = argparse.ArgumentParser(
        description='Test inference results and output metrics.\n'
        'Specify one or more `--compute` options below.')
    parser.add_argument(
        '--inference-path', help='path to inference results', required=True)
    parser.add_argument(
        '--label-path', help='path to test data', required=True)
    parser.add_argument(
        '--fishing-ranges', help='path to fishing range data', required=True)
    parser.add_argument(
        '--dest-path', help='path to write results to', required=True)
    # Specify which things to dump to output file
    parser.add_argument('--compute-class-metrics', action='store_true')
    parser.add_argument('--compute-localisation-metrics', action='store_true')
    parser.add_argument('--compute-length-metrics', action='store_true')
    # It's convenient to be able to dump the consolidated gear types
    parser.add_argument(
        '--dump-labels-to',
        help='dump csv file mapping csv to consolidated gear-type labels')
    args = parser.parse_args()

    if not (args.compute_class_metrics or args.compute_localisation_metrics or
            args.compute_length_metrics):
        print(
            "Warning: no `--compute` option specified, not html file will be generated")

    results = compute_results(args)

    dump_html(args, results)

    if args.dump_labels_to:
        if not args.dump_labels_to.endswith('.csv'):
            logging.warn('dump-labels-to file does not end with ".csv"')
        with open(args.dump_labels_to, 'w') as f:
            f.write('mmsi,label\n')
            for mmsi, label in zip(results['coarse'].mmsi,
                                   results['coarse'].inferred_labels):
                f.write('{},{}\n'.format(mmsi, label))
