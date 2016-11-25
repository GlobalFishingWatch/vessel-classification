import os
import csv
import numpy as np
import tensorflow as tf
import compute_metrics

class BasicMetricTests(tf.test.TestCase):
    y_true = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 0, 1, 0, 1, 0, 0]

    def test_precision_score(self):
        self.assertEqual(compute_metrics.precision_score(self.y_true, self.y_pred), 0.66666666666666663)

    def test_recall_score(self):
        self.assertEqual(compute_metrics.recall_score(self.y_true, self.y_pred), 0.5)

    def test_f1_score(self):
        self.assertEqual(compute_metrics.f1_score(self.y_true, self.y_pred), 0.5714285714285714)

    def test_accuracy_score(self):
        self.assertEqual(compute_metrics.accuracy_score(self.y_true, self.y_pred), 0.7)


# class MultiClassMetrics(tf.test.TestCase):

#     labels = [0, 1, 2]
#     y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
#     y_pred = [0, 1, 2, 2, 0, 2, 1, 1, 0]

#     def test_precision_recall(self):
#         self.assertAllEqual(compute_metrics.precision_recall(self.labels, self.y_true, self.y_pred), 
#             [[0, 1, 1, 1],
#              [1, 1, 1, 1],
#              [2, 1, 1, 1]])

#  x: array([[ 0.      ,  0.333333,  0.333333,  0.555556],
#        [ 1.      ,  0.666667,  0.666667,  0.777778],
#        [ 2.      ,  0.666667,  0.666667,  0.777778]])



if __name__ == '__main__':
    tf.test.main()
