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