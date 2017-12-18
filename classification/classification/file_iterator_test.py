
import gc
from file_iterator import *

def test_file_iterator():
    path = "gs://machine-learning-dev-ttl-30d/classification/timothyhochberg/features-old-old-school/pipeline/output/features/255805905.tfrecord"
    with tf.Session() as sess:
        for i, val in enumerate(all_fixed_window_feature_file_iterator([path], 15,
                        256, 64, datetime.datetime(2015,1,1), datetime.datetime(2017,7,31))):
            assert (val[0].shape, val[1].shape, val[2].shape, val[3].shape) == ((1, 256, 14), (256,), (2,), ())
        assert i == 1




def test_deserialize_file():
    path = "gs://machine-learning-dev-ttl-30d/classification/timothyhochberg/features-old-old-school/pipeline/output/features/255805905.tfrecord"

    with tf.Session() as sess:
        deserializer = Deserializer(num_features=15)
        with GCSExampleIter(path) as exmpliter:
            for i, exmp in enumerate(exmpliter):
                context_features, sequence_features = deserializer(exmp)
                mmsi = context_features['mmsi']
                movement_features = sequence_features['movement_features']

    assert mmsi == 255805905
    assert i == 0
    assert np.allclose(movement_features[0], [  1.48371507e+09,   1.48784456e+01,   4.23178339e+00,   0.00000000e+00,
   2.34343752e-05,   1.41666666e-01,   6.42787635e-01,   7.66044438e-01,
  -8.66025388e-01,  -5.00000000e-01,  -4.13155377e-01,   0.00000000e+00,
   0.00000000e+00,   0.00000000e+00,   5.00000000e+00])



def log_mem():
    import resource
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logging.info("Mem Usage: %s", mem)


def test_read_files_from_gcs():
    path = "gs://machine-learning-dev-ttl-30d/classification/timothyhochberg/features-old-old-school/pipeline/output/mmsis/part-00000-of-00001.txt"
    with GCSFile(path) as fp:
        assert fp.read().split()[::20000][:10] == ['1', '200009595', '211516550', '225023360', '235003666', '240552000', '244780789', '255805905', '265725430', '304010593']


def test_iterator_leak():
    path = "gs://machine-learning-dev-ttl-30d/classification/timothyhochberg/features-old-old-school/pipeline/output/features/255805905.tfrecord"
    with tf.Session() as sess:
        deserializer = Deserializer(num_features=15)
        for i in range(100):
            gc.collect()
            log_mem()
            for i, val in enumerate(all_fixed_window_feature_file_iterator([path], deserializer,
                            256, 64, datetime.datetime(2015,1,1), datetime.datetime(2017,7,31))):
                pass   



def test_deserialize_leak():
    path = "gs://machine-learning-dev-ttl-30d/classification/timothyhochberg/features-old-old-school/pipeline/output/features/255805905.tfrecord"
    with tf.Session() as sess:
        deserializer = Deserializer(num_features=15)
        for i in range(100):
            gc.collect()
            log_mem()
            with GCSExampleIter(path) as exmpliter:
                for i, exmp in enumerate(exmpliter):
                    context_features, sequence_features = deserializer(exmp)



def test_read_leak():
    path = "gs://machine-learning-dev-ttl-30d/classification/timothyhochberg/features-old-old-school/pipeline/output/features/255805905.tfrecord"
    with tf.Session() as sess:
        for i in range(100):
            gc.collect()
            log_mem()
            with GCSExampleIter(path) as exmpliter:
                for x in exmpliter:
                    pass


logging.basicConfig(level=logging.INFO)
test_read_files_from_gcs()
test_deserialize_file()
test_iterator_leak() # Fixed (at least mostly; now tops out at 185). Earlier leaks (316)
# test_deserialize_leak() # Doesn't leak
# test_read_leak() # doesn't leak
# test_file_iterator() # doesn't leaks
