from __future__ import print_function, division, absolute_import
import subprocess
import numpy as np
import pandas as pd
import hashlib
import argparse
import sys
from classification import metadata
import six

remapping = {
  # 'seismic_vessel' : 'research'
}


def read_ids(gcs_path):
    id_text = subprocess.check_output(['gsutil', 'cat', gcs_path])
    return set([six.ensure_text(x).strip() for x in id_text.strip().split()])


def fishing_range_vessel_id(fishdbname, dataset):
    return '''
        (
            select vessel_id, 
                min(start_time) as first_timestamp,
                max(end_time) as last_timestamp,
                sum(is_fishing) = 0 as transit_only
            from `{fishdbname}` a
            join `{dataset}.segment_info` b
            on (a.mmsi = cast(ssvid as int64))
            group by vessel_id
        )
    '''.format(fishdbname=fishdbname, dataset=dataset)

def fishing_range_mmsi(fishdbname, dataset):
    return '''
        (
            select cast(mmsi as string) as mmsi, 
                min(start_time) as first_timestamp,
                max(end_time) as last_timestamp,
                sum(is_fishing) = 0 as transit_only
            from `{fishdbname}`
            group by mmsi
        )
    '''.format(fishdbname=fishdbname, dataset=dataset)

def read_vessel_database_for_char_vessel_id(dbname, dataset):
    query = '''
        with 

        core as (
          select vessel_id as id, 
                 feature.length_m as length, 
                 feature.tonnage_gt as tonnage, 
                 feature.engine_power_kw as engine_power, 
                 feature.geartype as label, 
                 feature.crew as crew_size,
                 array_to_string(feature.geartype, '|') as lbl,
                 pos_count
          from (select * from {dbname} cross join unnest (activity)) a
          join `{dataset}.vessel_info` b
          on (a.identity.ssvid = b.ssvid)
          where 
           (not ((a.last_timestamp is not null and 
                  a.last_timestamp < b.first_timestamp) or 
                 (a.first_timestamp is not null and
                  a.first_timestamp > b.last_timestamp)))
            and (feature.length_m is not null or feature.tonnage_gt is not null or 
                feature.engine_power_kw is not null or feature.geartype is not null
                --or feature.crew is not null
                )
        ),

        counted as (
          select * except(pos_count, label, lbl), lbl as label, sum(pos_count) as count
          from core 
          group by id, length, tonnage, 
                 engine_power, label, crew_size
        ),

        ordered as (
          select *, 
                 row_number() over (partition by id 
                        order by count desc, label,
                        length, tonnage, engine_power, 
                        crew_size) as rk
          from counted
        )
            
        select * except(rk, count) from ordered
        where rk = 1
    '''.format(**locals())
    try:
        return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
    except:
        print(query)
        raise


def read_vessel_database_for_char_mmsi(dbname, dataset):
    query = '''
      with multi_id as (
        select identity.ssvid as id
        from {dbname}
        group by id
        having count(identity.ssvid) > 1
      )
      
      select identity.ssvid as id, 
             feature.length_m as length, 
             feature.tonnage_gt as tonnage, 
             feature.engine_power_kw as engine_power, 
             feature.crew as crew_size,
             array_to_string(feature.geartype, '|') as label
      from {dbname} a
      where (feature.length_m is not null or 
            feature.tonnage_gt is not null or 
            feature.engine_power_kw is not null or 
            feature.crew is not null or
            (feature.geartype is not null and array_length(feature.geartype) > 0)) and
            identity.ssvid not in (select * from multi_id)
            order by id
    '''.format(**locals())
    try:
        return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
    except:
        print(query)
        raise


def read_vessel_database_for_detect_vessel_id(dbname, fishdbname, dataset):
    fishing_range_query=fishing_range_vessel_id(fishdbname, dataset)
    query = '''
        with 

        fishing_range_vessel_id as {fishing_range_query},

        core as (
          select vessel_id as id, feature.length_m as length, feature.tonnage_gt as tonnage, 
                 feature.engine_power_kw as engine_power, 
                 array_to_string(feature.geartype, '|') as label,
                 feature.crew as crew_size,
                 pos_count, transit_only
          from (select * from {dbname} cross join unnest (activity)) a
         join `{dataset}.segment_info` b
            on (cast(a.identity.ssvid as string) = ssvid)
          join `fishing_range_vessel_id` c
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
          select * except(pos_count, transit_only, length, tonnage, engine_power,
                          label, crew_size), 
                    sum(pos_count) as count,
                   min(transit_only) as transit_only,
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
                        crew_size) as rk
          from counted
        )
            
        select * except(rk, count) from ordered
    '''.format(**locals())
    try:
        return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
    except:
        print(query)
        raise




def read_vessel_database_for_detect_mmsi(dbname, fishdbname, dataset):
    fishing_range_query=fishing_range_mmsi(fishdbname, dataset)
    query = '''
        with 

        fishing_range_mmsi as {fishing_range_query},

        core as (
          select identity.ssvid as id, length_m as length, tonnage_gt as tonnage, 
                 engine_power_kw as engine_power, 
                 geartype as label, crew as crew_size,
                 confidence, -- 1==low, 2==typical, 3==high
                 (select sum(messages) from unnest(activity)) as pos_count, transit_only
          from {dbname} a
          cross join unnest (registry)
          join `fishing_range_mmsi` c
          on (cast(mmsi as string) = identity.ssvid)
          where  -- valid times overlaps with fishing ranges
          -- TODO: should do each activity period separately here.
           (not (((select min(last_timestamp) from unnest(activity)) < c.first_timestamp) or 
                 ((select min(first_timestamp) from unnest(activity)) > c.last_timestamp))) 
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
        print(query)
        raise


def read_fishing_ranges_vessel_id(fishdbname, dataset):
    query = '''
        with 

        fishing_ranges as {fishing_ranges},

        core as (
          select mmsi as ssvid, vessel_id as id, 
                 start_time, end_time, is_fishing, pos_count
          from
          `{fishdbname}` a
          join
          `{dataset}.vessel_info` b
          on (a.mmsi = cast(ssvid as int64))
          join `fishing_ranges` c
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
               fishing_ranges=fishing_range_vessel_id(fishdbname, dataset))
    try:
        return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
    except:
        print(query)
        raise




def read_fishing_ranges_mmsi(fishdbname, dataset):
    query = '''
        with 

        fishing_ranges as {fishing_ranges},

        core as (
          select c.mmsi as id, 
                 start_time, end_time, is_fishing
          from
          `{fishdbname}` a
          join `fishing_ranges` c
          on c.mmsi = cast(a.mmsi as string)
        )

        select * from core
    '''.format(fishdbname=fishdbname,
               dataset=dataset, 
               fishing_ranges=fishing_range_mmsi(fishdbname, dataset))
    try:
        return pd.read_gbq(query, dialect='standard', project_id='world-fishing-827')
    except:
        print(query)
        raise


category_map = {k: v for (k, v) in  metadata.VESSEL_CATEGORIES}
def disintegrate(label):
    parts = set()

    for sub in label.split('|'):
        for atomic in category_map[sub]:
            parts.add(atomic)
    return parts


def apply_remapping(df, map):
    new_labels = []
    for lbl in df.label:
        if lbl and lbl not in ["unknown", "fishing", "non_fishing"]:
            atoms = disintegrate(lbl)
            new_atoms = set([remapping.get(x, x) for x in atoms])
            if new_atoms == atoms:
                # If no remapping occurred, keep old label as it's likely more compact.
                new_labels.append(lbl)
            else:
                new_labels.append('|'.join(sorted(new_atoms)))
        else:
          new_labels.append(lbl)
    df['label'] = new_labels


def assign_split(df, max_examples, seed=888, check_fishing=False):
    rnd = np.random.RandomState(seed)
    if check_fishing:
        # If we are looking at fishing any vessel can be
        # be fishing, but stratify by all listed types
        labels = sorted(set(df.label))
    else:
        # Otherwise, only allow atomic classes into test
        labels = metadata.VESSEL_CLASS_DETAILED_NAMES
    all_args = np.arange(len(df))
    split = ['Training'] * len(df)
    if check_fishing:
        split_a, split_b = '0', '1'
    else:
        # Characterization doesn's support splits yet
        split_a, split_b = 'Test', 'Training'
    # Half for train half for test
    total_max_examples =  2 * max_examples 
    for lbl in labels:
        lbl = six.ensure_text(lbl)
        base_mask = np.array([six.ensure_text(x) == lbl for x in df.label.values], dtype=bool)
        mask = base_mask.copy()
        if check_fishing:
            mask &= (df.transit_only.values == 0)
        elif mask.sum() > total_max_examples: 
            trues = np.random.choice(np.nonzero(mask)[0], size=[total_max_examples], replace=False)
            mask.fill(False)
            mask[trues] = True
        for i in all_args[base_mask]:
          split[i] = None
        candidates = sorted(all_args[mask], 
                       key=lambda x: hashlib.sha256(six.ensure_binary(df.id.iloc[x])).hexdigest())
        for i in candidates[:len(candidates)//2]:
          split[i] = split_a
        for i in candidates[len(candidates)//2:]:
          split[i] = split_b
    df['split'] = split


if __name__ == '__main__':
    # assert sys.version_info[0] == 2, 'must generate with Python 2 until feature sharding is updated'


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
        choices=['vessel-id', 'mmsi'],
        required=True
        )

    parser.add_argument(
        '--id-list',
        help="GCS location of ids present in features"
        )

    parser.add_argument(
        '--dataset',
        help="Name of the dataset to draw vessel_id mapping from"
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

    parser.add_argument(
        '--gear_file',
        help='gear IDS to add to training mmsi'
        )

    parser.add_argument(
        '--max_examples', type=int,  default=999,
        help='Include at most this number of total examples each in train/test'
        )

    args = parser.parse_args()

    if args.id_type == 'vessel-id':
        charinfo_df = read_vessel_database_for_char_vessel_id(args.vessel_database, 
                                                              args.dataset)
        detinfo_df = read_vessel_database_for_detect_vessel_id(args.vessel_database, 
                                                 args.fishing_table, args.dataset)
        det_df = read_fishing_ranges_vessel_id(args.fishing_table, args.dataset)
    elif args.id_type == 'mmsi':
        charinfo_df = read_vessel_database_for_char_mmsi(args.vessel_database, 
                                                              args.dataset)
        detinfo_df = read_vessel_database_for_detect_mmsi(args.vessel_database, 
                                                 args.fishing_table, args.dataset)
        det_df = read_fishing_ranges_mmsi(args.fishing_table, args.dataset)


    # Make ordering consistent across runs
    charinfo_df, detinfo_df, det_df = [x.sort_values(by=list(x.columns)) 
                                for x in (charinfo_df, detinfo_df, det_df)]

    print(charinfo_df.head())
    print(detinfo_df.head())
    print(det_df.head())

    if args.id_list:
        # Remove unavailable ids
        available_ids = read_ids(args.id_list)
        def filter(df):
            mask = [(x in available_ids) for x in df.id]
            return df[mask]
        charinfo_df, detinfo_df, det_df = [filter(x) 
                                  for x in (charinfo_df, detinfo_df, det_df)]

    print(len(detinfo_df))
    print(available_ids)

    if args.gear_file:
        with open(args.gear_file) as f:
            gear_ids = [x for x in f.read().strip().split()]
        existing_ids = set(charinfo_df.id)
        new = []
        for id_ in gear_ids:
              if id_ in existing_ids: 
                  continue
              new.append({'id' : id_, 'length' : np.nan, 'tonnage' : np.nan, 
                     'engine_power' : np.nan, 'crew_size' : np.nan,
                     'label' : 'gear'})
        new_df = pd.DataFrame(new, 
          columns=[u'id', u'length', u'tonnage', u'engine_power', u'crew_size', u'label'])
        charinfo_df = pd.concat([charinfo_df, new_df])

    apply_remapping(charinfo_df, remapping)
    print()
    apply_remapping(detinfo_df, remapping)
    assign_split(charinfo_df, args.max_examples, check_fishing=False)
    assign_split(detinfo_df, args.max_examples, check_fishing=True)

    charinfo_df = charinfo_df[~charinfo_df.split.isnull()]
    detinfo_df = detinfo_df[~detinfo_df.split.isnull()]

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

r'''
python -m train.create_train_info \
    --vessel-database vessel_database.all_vessels_20190102 \
    --fishing-table machine_learning_production.fishing_ranges_by_mmsi_v20190506 \
    --id-type uvi \
    --id-list gs://machine-learning-dev-ttl-120d/features/uvi_features_v20190528/ids/part-00000-of-00001.txt \
    --dataset pipe_production_v20190502 \
    --charinfo-file classification/data/char_info_uvi_v20190502.csv \
    --detinfo-file classification/data/det_info_uvi_v20190502.csv \
    --detranges-file classification/data/det_ranges_uvi_v20190502.csv 
'''


r'''
python -m train.create_train_info \
    --vessel-database vessel_database.all_vessels_v20191101 \
    --fishing-table machine_learning_production.fishing_ranges_by_mmsi_v20190506 \
    --id-type mmsi \
    --dataset pipe_production_v20190502 \
    --charinfo-file classification/data/char_info_mmsi_v20191127.csv \
    --detinfo-file classification/data/det_info_mmsi_v20191127.csv \
    --detranges-file classification/data/det_ranges_mmsi_v20191127.csv \
    --gear_file classification/data/old_gear.txt
'''

r'''
    python -m train.create_train_info \
        --vessel-database vessel_database.matched_vessels_one_record_per_ssvid_v20200101 \
        --fishing-table machine_learning_production.fishing_ranges_by_mmsi_v20190506 \
        --id-type mmsi \
        --id-list gs://machine-learning-dev-ttl-120d/features/mmsi_features_v20191126/ids/part-00000-of-00001.txt \
        --dataset pipe_production_v20190502 \
        --charinfo-file classification/data/char_info_mmsi_v20200120.csv \
        --detinfo-file classification/data/det_info_mmsi_v20200120.csv \
        --detranges-file classification/data/det_ranges_mmsi_v20200120.csv
'''

