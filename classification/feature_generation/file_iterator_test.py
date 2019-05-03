
import gc
import posixpath as pp
from file_iterator import *
import pytest

def test_file_iterator():
    path = "gs://world-fishing-827/data-production/classification/release-0.1.2/pipeline/output/features/251822362.tfrecord"
    with tf.Session() as sess:
        deserializer = Deserializer(num_features=13)
        for i, val in enumerate(all_fixed_window_feature_file_iterator([path], deserializer,
                        256, 64, datetime.datetime(2015,1,1), datetime.datetime(2017,7,31), 96, 160)):
            assert (val[0].shape, val[1].shape, val[2].shape, val[3].shape) == ((1, 256, 12), (256,), (2,), ()), (val[0].shape, val[1].shape, val[2].shape, val[3].shape)
        assert i == 0, (i, val)


epoch = datetime.datetime(1970,1,1)
def to_stamp(x):
    return (x - epoch).total_seconds()


def test_deserialize_file():
    path = "gs://world-fishing-827/data-production/classification/release-0.1.2/pipeline/output/features/251822362.tfrecord"
    with tf.Session() as sess:
        deserializer = Deserializer(num_features=13)
        with GCSExampleIter(path) as exmpliter:
            for i, exmp in enumerate(exmpliter):
                context_features, sequence_features = deserializer(exmp)
                id_ = context_features['id']
                movement_features = sequence_features['movement_features']

    assert id_ == 251822362
    assert i == 0
    assert np.allclose(movement_features[0], [  
       1.38347622e+09,   1.61996498e+01,   4.07687426e-02,   0.00000000e+00,
       3.83523524e-09,   5.63333273e-01,  -2.08333328e-01,   8.33333313e-01,
      -9.90370393e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
       0.00000000e+00])



def log_mem():
    import resource
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logging.info("Mem Usage: %s", mem)


def test_read_files_from_gcs():
    path = "gs://world-fishing-827/data-production/classification/release-0.1.2/pipeline/output/ids/part-00000-of-00001.txt"
    with GCSFile(path) as fp:
        read_id = fp.read().split()[::20000][:10]
        print(read_id)
        assert read_id == ['1', '200002405', '211151080', '220529000', '229268000', '235108004', '244650777', '246065069', '257222040', '269105470']

@pytest.mark.skip(reason="debug only")
def test_iterator_leak():
    path = "gs://world-fishing-827/data-production/classification/release-0.1.2/pipeline/output/features/251822362.tfrecord"
    with tf.Session() as sess:
        deserializer = Deserializer(num_features=13)
        for i in range(100):
            gc.collect()
            log_mem()
            for i, val in enumerate(all_fixed_window_feature_file_iterator([path], deserializer,
                            256, 64, datetime.datetime(2015,1,1), datetime.datetime(2017,7,31))):
                pass   


@pytest.mark.skip(reason="debug only")
def test_deserialize_leak():
    path = "gs://world-fishing-827/data-production/classification/release-0.1.2/pipeline/output/features/251822362.tfrecord"
    with tf.Session() as sess:
        deserializer = Deserializer(num_features=13)
        for i in range(100):
            gc.collect()
            log_mem()
            with GCSExampleIter(path) as exmpliter:
                for i, exmp in enumerate(exmpliter):
                    context_features, sequence_features = deserializer(exmp)


@pytest.mark.skip(reason="debug only")
def test_read_leak():
    path = "gs://world-fishing-827/data-production/classification/release-0.1.2/pipeline/output/features/251822362.tfrecord"
    with tf.Session() as sess:
        for i in range(100):
            gc.collect()
            log_mem()
            with GCSExampleIter(path) as exmpliter:
                for x in exmpliter:
                    pass

@pytest.mark.skip(reason="debug only")
def test_coverage(base_path, id_list, num_features, year):
    start = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)
    dates = set()
    with tf.Session() as sess:
        deserializer = Deserializer(num_features=num_features)
        for id_ in id_list:
            path = pp.join(base_path, 'features', str(id_) + '.tfrecord')
            counter = 0
            with GCSExampleIter(path) as exmpliter:
                for exmp in exmpliter:
                    context_features, sequence_features = deserializer(exmp)
                    movement_features = sequence_features['movement_features']
                    timestamps = movement_features[:, 0]
                    for t in timestamps:
                        counter += 1
                        date = datetime.datetime.utcfromtimestamp(t).date()
                        if start <= date <= end:
                            dates.add(date)
                        if counter % 10000 == 0:
                            print(len(dates))
    return dates


def read_id(base_path):
    path = pp.join(base_path, "ids/part-00000-of-00001.txt")
    with GCSFile(path) as fp:
        return fp.read().strip().split()


if __name__ == '__main__':
    tf.test.main()




