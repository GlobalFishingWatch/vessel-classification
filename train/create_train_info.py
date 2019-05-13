import subprocess
import numpy as np
import pandas as pd
import argparse
from classification import metadata


def fishing_range_mmsi(fishdbname):
    return '''
        (
            select mmsi, first_timestamp, last_timestamp, transit_only from (
               select
               mmsi, 
               first_value(start_time) over(partition by mmsi order by start_time) as first_timestamp, 
               last_value(end_time) over(partition by mmsi order by end_time) as last_timestamp,
               last_value(is_fishing) over(partition by mmsi order by is_fishing) as transit_only
               from `{fishdbname}`
               ) group by mmsi, first_timestamp, last_timestamp, transit_only
        )
    '''.format(fishdbname=fishdbname)


def read_vessel_database_vessel_id(dbname, fishdbname, dataset):
    if fishdbname is None:
        fishing_subquery = ''

        core = '''
        (
          select ssvid, vessel_id as id, length_m as length, tonnage_gt as tonnage, 
                 engine_power_kw as engine_power, 
                 geartype as label, crew as crew_size,
                 confidence, -- 1==low, 2==typical, 3==high
                 pos_count
          from {dbname} a
          join `{dataset}.segment_identity_2*` b
          on (a.mmsi = cast(ssvid as int64))
          where 
           (not ((a.last_timestamp is not null and 
                  a.last_timestamp < b.first_timestamp) or 
                 (a.first_timestamp is not null and
                  a.first_timestamp > b.last_timestamp)))
            and (length_m is not null or tonnage_gt is not null or 
                engine_power_kw is not null or geartype is not null
                or crew is not null)
        )
        '''.format(dbname=dbname, dataset=dataset)
        extra_group = ''
    else:
        fishing_subquery = '''
        fishing_range_mmsi as {fishing_range_mmsi},
        '''.format(fishing_range_mmsi=fishing_range_mmsi(fishdbname))

        core = '''
        (
          select ssvid, vessel_id as id, length_m as length, tonnage_gt as tonnage, 
                 engine_power_kw as engine_power, 
                 geartype as label, crew as crew_size,
                 confidence, -- 1==low, 2==typical, 3==high
                 pos_count, transit_only
          from {dbname} a
          join `{dataset}.segment_identity_2*` b
          on (a.mmsi = cast(ssvid as int64))
          join `fishing_range_mmsi` c
          using (mmsi)
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
        )
        '''.format(dbname=dbname, dataset=dataset)
        extra_group = ', transit_only'
    query = '''
        with 

        {fishing_subquery}

        core as {core},

        counted as (
          select * except(pos_count), sum(pos_count) as count
          from core 
          group by ssvid, id, length, tonnage, 
                 engine_power, label, crew_size,
                 confidence {extra_group}
        ),
            
        ordered as (
          select *, 
                 row_number() over (partition by ssvid 
                        order by count, id desc, label,
                        length, tonnage, engine_power, 
                        crew_size, confidence) as rk
          from counted
        )
            
        select * except(rk, ssvid, count) from ordered
        where rk = 1
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
          `{dataset}.segment_identity_2*` b
          on (a.mmsi = cast(ssvid as int64))
          join `fishing_range_mmsi` c
          using (mmsi)
          where 
            (not (b.last_timestamp <  c.first_timestamp or 
                  b.first_timestamp > c.last_timestamp))
        ),

        counted as (
          select ssvid, id, start_time, end_time, is_fishing, sum(pos_count) as count
          from core 
          group by ssvid, id, start_time, end_time, is_fishing
        ),
            
        ordered as (
          select *, 
                 row_number() over (partition by ssvid, start_time, end_time
                                     order by count desc) as rk
          from counted
        )
            
        select * except(rk, ssvid, count) from ordered
        where rk = 1
    '''.format(fishdbname=fishdbname,
               dataset=dataset, 
               fishing_range_mmsi=fishing_range_mmsi(fishdbname))
    try:
        return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
    except:
        print(query)
        raise


def assign_split(df, seed=888, check_fishing=False):
    rnd = np.random.RandomState(seed)
    atomic = metadata.VESSEL_CLASS_DETAILED_NAMES
    all_args = np.argsort(df.id.values)
    is_test = np.zeros([len(df)], dtype=bool)
    for atm in atomic:
        mask = (df.label == atm)
        if check_fishing:
            mask &= (df.transit_only == 0)
        candidates = all_args[mask]
        rnd.shuffle(candidates)
        test_ndxs = candidates[:len(candidates)//2]
        is_test[test_ndxs] = True
    split = ['Test' if x else 'Training' for x in is_test]
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

    if args.id_type == 'vessel-id':
        charinfo_df = read_vessel_database_vessel_id(args.vessel_database, 
                                                 None, args.dataset)
        detinfo_df = read_vessel_database_vessel_id(args.vessel_database, 
                                                 args.fishing_table, args.dataset)
        det_df = read_fishing_ranges_vessel_id(args.fishing_table, args.dataset)
        # Make ordering consistent across runs
        charinfo_df, detinfo_df, det_df = [x.sort_values(by=list(x.columns)) 
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
            --dataset pipe_production_b \
            --charinfo-file classification/data/char_info_v20190510.csv \
            --detinfo-file classification/data/det_info_v20190510.csv \
            --detranges-file classification/data/det_ranges_v20190510.csv \
            --charinfo-table machine_learning_dev_ttl_120d.char_info_v20190510 \
            --detinfo-table machine_learning_dev_ttl_120d.det_info_v20190510 \
            --detranges-table machine_learning_dev_ttl_120d.det_ranges_v20190510
'''
