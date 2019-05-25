import subprocess
import numpy as np
import pandas as pd
import argparse
from classification import metadata


def read_ids(gcs_path):
    id_text = subprocess.check_output(['gsutil', 'cat', gcs_path])
    return set(id_text.strip().split())

def fishing_range_mmsi(fishdbname, dataset):
    return '''
        (
            select vessel_id, 
                min(start_time) as first_timestamp,
                max(end_time) as last_timestamp,
                sum(is_fishing) = 0 as transit_only
            from `{fishdbname}` a
            join `{dataset}.vessel_info` b
            on (a.mmsi = cast(ssvid as int64))
            group by vessel_id
        )
    '''.format(fishdbname=fishdbname, dataset=dataset)

def read_vessel_database_for_char_vessel_id(dbname, dataset):
    query = '''
        with 

        core as (
          select vessel_id as id, length_m as length, tonnage_gt as tonnage, 
                 engine_power_kw as engine_power, 
                 geartype as label, crew as crew_size,
                 confidence, -- 1==low, 2==typical, 3==high
                 pos_count
          from {dbname} a
          join `{dataset}.vessel_info` b
          on (a.mmsi = cast(ssvid as int64))
          where 
           (not ((a.last_timestamp is not null and 
                  a.last_timestamp < b.first_timestamp) or 
                 (a.first_timestamp is not null and
                  a.first_timestamp > b.last_timestamp)))
            and (length_m is not null or tonnage_gt is not null or 
                engine_power_kw is not null or geartype is not null
                or crew is not null)
        ),

        counted as (
          select * except(pos_count), sum(pos_count) as count
          from core 
          group by id, length, tonnage, 
                 engine_power, label, crew_size,
                 confidence
        ),

        ordered as (
          select *, 
                 row_number() over (partition by id 
                        order by count desc, label,
                        length, tonnage, engine_power, 
                        crew_size, confidence) as rk
          from counted
        )
            
        select * except(rk, count) from ordered
        where rk = 1
    '''.format(**locals())
    try:
        return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
    except:
        print query
        raise

def read_vessel_database_for_detect_vessel_id(dbname, fishdbname, dataset):
    fishing_range_query=fishing_range_mmsi(fishdbname, dataset)
    query = '''
        with 

        fishing_range_mmsi as {fishing_range_query},

        core as (
          select vessel_id as id, length_m as length, tonnage_gt as tonnage, 
                 engine_power_kw as engine_power, 
                 geartype as label, crew as crew_size,
                 confidence, -- 1==low, 2==typical, 3==high
                 pos_count, transit_only
          from {dbname} a
          join `{dataset}.vessel_info` b
          on (a.mmsi = cast(b.ssvid as int64))
          join `fishing_range_mmsi` c
          using (vessel_id)
          where -- valid times overlap with segments
           (not ((a.last_timestamp is not null and 
                 a.last_timestamp < b.first_timestamp) or 
                 (a.first_timestamp is not null and
                 a.first_timestamp > b.last_timestamp)))
            and -- valid times overlaps with fishing ranges
           (not ((a.last_timestamp is not null and 
                  a.last_timestamp < c.first_timestamp) or 
                 (a.first_timestamp is not null and 
                  a.first_timestamp > c.last_timestamp))) 
        ),

        counted as (
          select * except(pos_count, confidence, transit_only, length, tonnage, engine_power,
                          label, crew_size), 
                    sum(pos_count) as count,
                   avg(confidence) as confidence, min(transit_only) as transit_only,
                   avg(length) as length, avg(tonnage) as tonnage, avg(engine_power) as engine_power,
                   any_value(label) as label, avg(crew_size) as crew_size
          from core 
          group by id
        ),     

        ordered as (
          select *, 
                 row_number() over (partition by id 
                        order by count desc, label,
                        length, tonnage, engine_power, 
                        crew_size, confidence) as rk
          from counted
        )
            
        select * except(rk, count) from ordered
    '''.format(**locals())
    try:
        return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
    except:
        print query
        raise


def read_fishing_ranges_vessel_id(fishdbname, dataset):
    query = '''
        with 

        fishing_range_mmsi as {fishing_range_mmsi},

        core as (
          select mmsi as ssvid, vessel_id as id, 
                 start_time, end_time, is_fishing, pos_count
          from
          `{fishdbname}` a
          join
          `{dataset}.vessel_info` b
          on (a.mmsi = cast(ssvid as int64))
          join `fishing_range_mmsi` c
          using (vessel_id)
          where 
            (not (b.last_timestamp <  c.first_timestamp or 
                  b.first_timestamp > c.last_timestamp))
        ),

        counted as (
          select id, start_time, end_time, is_fishing, sum(pos_count) as count
          from core 
          group by id, start_time, end_time, is_fishing
        ),
            
        ordered as (
          select *, 
                 row_number() over (partition by id, start_time, end_time
                                     order by count desc) as rk
          from counted
        )
            
        select * except(rk, count) from ordered
        where rk = 1
    '''.format(fishdbname=fishdbname,
               dataset=dataset, 
               fishing_range_mmsi=fishing_range_mmsi(fishdbname, dataset))
    try:
        return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
    except:
        print(query)
        raise


def assign_split(df, seed=888, check_fishing=False):
    rnd = np.random.RandomState(seed)
    if check_fishing:
        # If we are looking at fishing any vessel can be
        # be fishing, but stratify by all listed types
        labels = sorted(set(df.label))
    else:
        # Otherwise, only allow atomic classes into test
        labels = metadata.VESSEL_CLASS_DETAILED_NAMES
    all_args = np.argsort(df.id.values)
    split = ['Training'] * len(df)
    for lbl in labels:
        mask = (df.label == lbl)
        if check_fishing:
            mask &= (df.transit_only == 0)
        candidates = all_args[mask]
        rnd.shuffle(candidates)
        for i in  candidates[:len(candidates)//2]:
          split[i] = '0'
        for i in candidates[len(candidates)//2:]:
          split[i] = '1'
    df['split'] = split


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create Training Info')

    parser.add_argument(
        '--vessel-database',
        required=True,
        help='The BQ table holding the vessel database')

    parser.add_argument(
        '--fishing-table',
        required=True,
        help='The BQ table holding fishing ranges')

    parser.add_argument(
        '--id-type',
        choices=['vessel-id'],
        required=True
        )

    parser.add_argument(
        '--id-list',
        help="GCS location of ids present in features"
        )

    parser.add_argument(
        '--dataset',
        help="Name of the dataset to draw uvi mapping from"
        )

    parser.add_argument(
        '--charinfo-file',
        )

    parser.add_argument(
        '--detinfo-file',
        )

    parser.add_argument(
        '--detranges-file',
        )

    parser.add_argument(
        '--charinfo-table',
        )

    parser.add_argument(
        '--detinfo-table',
        )

    parser.add_argument(
        '--detranges-table',
        )

    args = parser.parse_args()

    available_ids = read_ids(args.id_list)

    if args.id_type == 'vessel-id':
        charinfo_df = read_vessel_database_for_char_vessel_id(args.vessel_database, 
                                                              args.dataset)
        detinfo_df = read_vessel_database_for_detect_vessel_id(args.vessel_database, 
                                                 args.fishing_table, args.dataset)
        det_df = read_fishing_ranges_vessel_id(args.fishing_table, args.dataset)

    # Make ordering consistent across runs
    charinfo_df, detinfo_df, det_df = [x.sort_values(by=list(x.columns)) 
                                for x in (charinfo_df, detinfo_df, det_df)]

    # Remove unavailable ids
    def filter(df):
        mask = [(x in available_ids) for x in df.id]
        return df[mask]
    charinfo_df, detinfo_df, det_df = [filter(x) 
                                for x in (charinfo_df, detinfo_df, det_df)]

    assign_split(charinfo_df)
    assign_split(detinfo_df, check_fishing=True)

    if args.charinfo_file:
        charinfo_df.to_csv(args.charinfo_file, index=False)

    if args.detinfo_file:
        detinfo_df.to_csv(args.detinfo_file, index=False)

    if args.detranges_file:
        det_df.to_csv(args.detranges_file, index=False)

    if args.charinfo_table:
        charinfo_df.to_gbq(args.charinfo_table,  
                          'world-fishing-827', if_exists='fail')
    if args.detinfo_table:
        detinfo_df.to_gbq(args.detinfo_table,  
                          'world-fishing-827', if_exists='fail')
    if args.detranges_table:
        det_df.to_gbq(args.detranges_table,  
                          'world-fishing-827', if_exists='fail')
r'''
python -m train.create_train_info \
    --vessel-database vessel_database.all_vessels_20190102 \
    --fishing-table machine_learning_production.fishing_ranges_by_mmsi_v20190506 \
    --id-type vessel-id \
    --id-list gs://machine-learning-dev-ttl-120d/features/v3_vid_features_v20190503b/ids/part-00000-of-00001.txt \
    --dataset pipe_production_b \
    --charinfo-file classification/data/char_info_v20190520b.csv \
    --detinfo-file classification/data/det_info_v20190520b.csv \
    --detranges-file classification/data/det_ranges_v20190520b.csv \
    --charinfo-table machine_learning_dev_ttl_120d.char_info_v20190515 \
    --detinfo-table machine_learning_dev_ttl_120d.det_info_v20190515 \
    --detranges-table machine_learning_dev_ttl_120d.det_ranges_v20190515
'''
