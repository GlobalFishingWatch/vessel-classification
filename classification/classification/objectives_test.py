import calendar
import dateutil.parser
import numpy as np
import objectives
import utility
import tensorflow as tf


def _dt(s):
    return dateutil.parser.parse(s)


class RegressionLossTest(tf.test.TestCase):
    def test_simple_loss(self):
        with self.test_session():
            prediction = np.array([1.0, 4.0, 5.0, 6.0, 3.0])
            mmsis = np.array([1, 2, 3, 6, 9])

            real_values = {1: 1.0, 2: 4.0, 3: 5.0, 6: 6.0, 9: 3.0}

            objective = objectives.RegressionObjective(
                'a label', 'A name', lambda mmsi: real_values.get(mmsi))

            objective.prediction = prediction
            loss, _ = objective.build_trainer(None, mmsis)

            self.assertAlmostEqual(0.0, loss.eval())

    def test_loss_missing_values(self):
        with self.test_session():
            prediction = np.array([1.0, 4.0, 5.0, 6.0, 3.0])
            mmsis = np.array([1, 2, 3, 4, 5])

            real_values = {1: 2.0, 2: 2.5, 3: 4.5}

            objective = objectives.RegressionObjective(
                'a label', 'A name', lambda mmsi: real_values.get(mmsi))
            objective.prediction = prediction

            loss, _ = objective.build_trainer(None, mmsis)

            self.assertAlmostEqual(1.0, loss.eval())


class FishingLocalisationLossTest(tf.test.TestCase):
    def test_simple_loss(self):
        with self.test_session():
            logits = np.array([[1, 0, 0, 1, 0], [1, 0, 1, 1, 0]], np.float32)
            targets = np.array([[1, 0, -1, 0, -1], [1, 0, 0, -1, -1]],
                               np.float32)

            loss = utility.fishing_localisation_loss(logits, targets)

            filtered_logits = logits[targets != -1]
            filtered_targets = targets[targets != -1]

            filtered_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(filtered_logits,
                                                        filtered_targets))

            self.assertAlmostEqual(filtered_loss.eval(), loss.eval(), places=5)

    def test_loss_scaling_floor(self):
        with self.test_session():
            logits = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.float32)
            targets = np.array([[0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
                               np.float32)

            loss = utility.fishing_localisation_loss(logits, targets)

            self.assertAlmostEqual(0.66164052, loss.eval())

    def test_loss_no_targets(self):
        with self.test_session():
            logits = np.array([[0, 0, 0, 0, 0]], np.float32)
            targets = np.array([[-1, -1, -1, -1, -1]], np.float32)

            loss = utility.fishing_localisation_loss(logits, targets)

            self.assertAlmostEqual(0.0, loss.eval())


class FishingLocalisationMseTest(tf.test.TestCase):
    def test_simple_mse(self):
        with self.test_session():
            predictions = np.array([[1, 0, 0, 1, 0], [1, 0, 1, 1, 0]],
                                   np.float32)
            targets = np.array([[1, 0, -1, 0, -1], [1, 0, 0, -1, -1]],
                               np.float32)

            mse = utility.fishing_localisation_mse(predictions, targets)

            self.assertAlmostEqual(0.33333333, mse.eval())


# Check we are actually getting vessels with fishing
# localisation info (check loading the metadata, and choosing the
# segments).
class ObjectiveFunctionsTest(tf.test.TestCase):
    vmd_dict = {'Test': {100001: ({'label': 'Longliner'}, 1.0)}}
    range1 = utility.FishingRange(
        _dt("2015-04-01T06:00:00Z"), _dt("2015-04-01T09:30:0Z"), 1.0)
    range2 = utility.FishingRange(
        _dt("2015-04-01T09:30:00Z"), _dt("2015-04-01T12:30:01Z"), 0.0)

    def _build_trainer(self, logits, objective):
        timestamps = [
            _dt("2015-04-01T08:30:00Z"),
            _dt("2015-04-01T09:00:00Z"),
            _dt("2015-04-01T09:29:00Z"),
            _dt("2015-04-01T10:00:00Z"),
            _dt("2015-04-01T10:30:00Z"),
            _dt("2015-04-01T11:00:00Z"),
        ]
        epoch_timestamps = [[calendar.timegm(t.utctimetuple())
                             for t in timestamps]]
        mmsis = [100001]
        objective.build(logits)
        return objective.build_trainer(epoch_timestamps, mmsis)

    def test_fishing_range_objective_no_ranges(self):
        logits = [[np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]]
        vmd = utility.VesselMetadata(self.vmd_dict, {}, 1.0)

        o = objectives.FishingLocalisationObjectiveMSE(
            'fishing_localisation', 'Fishing Localisation', vmd)

        with self.test_session() as sess:
            trainer = self._build_trainer(logits, o)
            self.assertAlmostEqual(0.0, trainer.loss.eval())

    def test_fishing_range_objective_half_specified(self):
        logits = [[np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]]
        fishing_range_dict = {100001: [self.range1]}
        vmd = utility.VesselMetadata(self.vmd_dict, fishing_range_dict, 1.0)

        o = objectives.FishingLocalisationObjectiveMSE(
            'fishing_localisation', 'Fishing Localisation', vmd)

        with self.test_session() as sess:
            trainer = self._build_trainer(logits, o)
            self.assertAlmostEqual(0.0, trainer.loss.eval())

    def test_fishing_range_objective_fully_specified(self):
        logits = [[np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]]
        fishing_range_dict = {100001: [self.range1, self.range2]}
        vmd = utility.VesselMetadata(self.vmd_dict, fishing_range_dict, 1.0)

        o = objectives.FishingLocalisationObjectiveMSE(
            'fishing_localisation', 'Fishing Localisation', vmd)

        with self.test_session() as sess:
            trainer = self._build_trainer(logits, o)
            self.assertAlmostEqual(0.0, trainer.loss.eval())

    def test_fishing_range_objective_fully_specified_mismatch(self):
        logits = [[np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf]]
        fishing_range_dict = {100001: [self.range1, self.range2]}
        vmd = utility.VesselMetadata(self.vmd_dict, fishing_range_dict, 1.0)

        o = objectives.FishingLocalisationObjectiveMSE(
            'fishing_localisation', 'Fishing Localisation', vmd)

        with self.test_session() as sess:
            trainer = self._build_trainer(logits, o)
            self.assertAlmostEqual(0.16666667, trainer.loss.eval())


if __name__ == '__main__':
    tf.test.main()
