import tensorflow as tf
import feature_generation
# TODO: rather than importing this, hang the shape off of the objective
from classification.models.objectives import MultiClassificationObjective
from classification import metadata
from . import feature_generation


def input_fn(vessel_metadata,
             filenames,
            num_features,
            max_time_delta,
            window_size,
            min_timeslice_size,
            objectives,
            num_parallel_reads=4):

    
    def _add_labels(mmsi, timestamps):
        labels = [o.create_label(mmsi, timestamps) for o in objectives]
        return labels

    def add_labels(features, timestamps, time_bounds, mmsi):
        labels =  tf.py_func(
            _add_labels, 
            [mmsi, timestamps],
            [tf.float32] * len(objectives))
        return ((features, timestamps, time_bounds, mmsi), tuple(labels))

    class_count = len(metadata.VESSEL_CLASS_DETAILED_NAMES)

    def set_label_shape(labels):
        results = []
        for i, obj in enumerate(objectives):
            t = labels[i]
            if isinstance(obj, MultiClassificationObjective):
                t.set_shape([class_count])
            else:
                t.set_shape([])
            results.append(t)
        return tuple(results)


    base = feature_generation.input_fn(
                vessel_metadata,
                filenames,
                num_features,
                max_time_delta,
                window_size,
                min_timeslice_size,
                num_parallel_reads=num_parallel_reads,
                add_labels_fn=add_labels,
                set_labels_shape_fn=set_label_shape
        )

    def lbls_as_dict(features, labels):
        d = {obj.name : labels[i] for (i, obj) in enumerate(objectives)}
        return features, d

    return base.map(lbls_as_dict)