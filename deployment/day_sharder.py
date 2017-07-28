
import gzip




def shard_by_date(iterable, limit=None):
    assert (limit is None) or (limit > 0)
    for i, row in enumerate(iterable):
        print(row)
        # use Regex to pull date out
        # parse date
        # group by day
        # write out by day

        if limit and i > limit:
            break



if __name__ == "__main__":
    # TODO: get filename based on argparse
    path = "./update_fishing_detection.json.gz"
    with gzip.GzipFile(path) as f:
        shard_by_date(f.readlines(), 100)

