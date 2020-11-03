# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

query = """
with 

labels as (
  select cast(id as string) as ssvid, length as length_lbl
  from `machine_learning_dev_ttl_120d.char_info_mmsi_v20200124`
  where split = 'Test' and length is not null
    and cast(id as string) not in ('367661820') -- this has bogus length
),

inferred as (
  select ssvid, start_time, length
  from `world-fishing-827.gfw_research_precursors.vc_v20200124_results_*`
),

monthly_activity as (
  select ssvid, sum(positions) positions, sum(active_positions) active_positions,
         extract(month from date) month, 
         extract(year from date) year 
  from gfw_research.pipe_v20190502_segs_daily
  group by ssvid, month, year
),

semiyearly_activity as (
 select ssvid, sum(positions) positions, sum(active_positions) active_positions,
        timestamp(datetime(year, 1, 1, 0, 0, 0)) start_time, year
 from monthly_activity
 where month <= 6
 group by ssvid, year
 union all 
 select ssvid, sum(positions) positions, sum(active_positions) active_positions,
        timestamp(datetime(year, 7, 1, 0, 0, 0)) start_time, year
 from monthly_activity
 where month > 6
 group by ssvid, year
)


select * 
from labels
join inferred
using (ssvid)
join semiyearly_activity 
using (ssvid, start_time)
"""
length_df = pd.read_gbq(query, project_id='world-fishing-827', dialect='standard')

# ## By SSVID only

# +

df = length_df.groupby(by = ['ssvid']).mean()
plt.plot(df.length_lbl, df.length, '.')
r2 = np.corrcoef(length_df.length_lbl, length_df.length)[0,1] ** 2
r2avg = np.corrcoef(df.length_lbl, df.length)[0,1] ** 2

# +
lbls = []
lens = []
for key, group in length_df.groupby(by = ['ssvid']):
    lbls.append(group.length_lbl.mean())
    scale = 10 * np.log(group.active_positions + 1) + np.log(group.positions + 1)
    l = (group.length * scale).sum() / scale.sum()
    lens.append(l)

plt.plot(lbls, lens, '.')

r2avg2 = np.corrcoef(lbls, lens)[0,1] ** 2
print(f'{r2:.3f}, {r2avg:.3f}, {r2avg2:.3f}')
# -

# ## By SSVID and year

# +

df = length_df.groupby(by = ['ssvid', 'year']).mean()
plt.plot(df.length_lbl, df.length, '.')
r2 = np.corrcoef(length_df.length_lbl, length_df.length)[0,1] ** 2
r2avg = np.corrcoef(df.length_lbl, df.length)[0,1] ** 2

# +
lbls = []
lens = []
for key, group in length_df.groupby(by = ['ssvid', 'year']):
    lbls.append(group.length_lbl.mean())
    scale = 100 * np.log(group.active_positions + 1) + np.log(group.positions + 1)
    l = (group.length * scale).sum() / scale.sum()
    lens.append(l)

plt.plot(lbls, lens, '.')

r2avg2 = np.corrcoef(lbls, lens)[0,1] ** 2
print(f'{r2:.3f}, {r2avg:.3f}, {r2avg2:.3f}')
