from __future__ import division, print_function
import os
import csv
import numpy as np
import tensorflow as tf
import compute_metrics
import datetime

class BasicMetricTests(tf.test.TestCase):
    y_true = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 0, 1, 0, 1, 0, 0]

    def test_precision_score(self):
        self.assertEqual(
            compute_metrics.precision_score(self.y_true, self.y_pred),
            0.66666666666666663)

    def test_recall_score(self):
        self.assertEqual(
            compute_metrics.recall_score(self.y_true, self.y_pred), 0.5)

    def test_f1_score(self):
        self.assertEqual(
            compute_metrics.f1_score(self.y_true, self.y_pred),
            0.5714285714285714)

    def test_accuracy_score(self):
        self.assertEqual(
            compute_metrics.accuracy_score(self.y_true, self.y_pred), 0.7)


class MultiClassMetrics(tf.test.TestCase):

    labels = [0, 1, 2]
    y_true = [0, 1, 1, 2, 2, 2, 2, 2]
    y_pred = [0,
              1,
              2,
              2,
              0,
              2,
              1,
              1, ]

    def _expected_weights(self):
        a, b, c = np.array([8, 4, 8 / 5])
        expected = np.array([a, b, b, c, c, c, c, c])
        expected /= expected.sum()
        return expected

    def test_weights(self):
        self.assertAllEqual(
            compute_metrics.weights(self.labels, self.y_true, self.y_pred),
            self._expected_weights())

    def test_weighted_accuracy(self):
        weights = self._expected_weights()
        self.assertAllEqual(
            compute_metrics.accuracy_score(self.y_true, self.y_pred, weights),
            weights[0] + weights[1] + 2 * weights[-1])



class AssembleComposite(tf.test.TestCase):
    
    results = compute_metrics.InferenceResults([1], ['A'], ['C'],
                         ['DATE'], [{'A': 0.4, 'B': 0.3, 'C': 0.3}], ['A', 'B', 'C'],
                         [1], ['A'], ['C'],
                         ['DATE'], [{'A': 0.4, 'B': 0.3, 'C': 0.3}])

    mapping = [
        ['F', ['A']],
        ['G', ['B', 'C']]
    ]


    def test_basic(self):
        new = compute_metrics.assemble_composite(self.results, self.mapping)
        self.assertAllEqual(new.all_inferred_labels, ['G'])
        self.assertAllEqual(new.all_true_labels, ['G'])
        self.assertAllEqual(new.all_scores, [{'F': 0.4, 'G': 0.6}])



class Consolidate(tf.test.TestCase):

    now = datetime.datetime.now()
    one_day = datetime.timedelta(days=1)

    results = compute_metrics.InferenceResults([1, 1], ['A', 'B'], ['C', 'C'],
                         [now - one_day, now], 
                         [{'A': 0.4, 'B': 0.25, 'C': 0.35}, {'A': 0.3, 'B': 0.4, 'C': 0.3}], 
                         ['A', 'B', 'C'],
                         [1, 1], ['A', 'B'], ['C', 'C'],
                         [now - one_day, now], 
                         [{'A': 0.4, 'B': 0.25, 'C': 0.35}, {'A': 0.3, 'B': 0.4, 'C': 0.3}])

    def test_base(self):
        new = compute_metrics.consolidate_across_dates(self.results)
        print(new.all_scores)


if __name__ == '__main__':
    tf.test.main()
