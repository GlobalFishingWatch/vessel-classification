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

_lengths as (
  SELECT ssvid, date(timestamp_add(start_time, interval 182 day)) as change_date,
         LAG(length, 3) over (partition by ssvid order by start_time) as length_p4,
         LAG(length, 2) over (partition by ssvid order by start_time) as length_p3,
         LAG(length, 1) over (partition by ssvid order by start_time) as length_p2,
         length as length_p1, 
         LEAD(length, 1) over (partition by ssvid order by start_time) as length_n1,
         LEAD(length, 2) over (partition by ssvid order by start_time) as length_n2,
         LEAD(length, 3) over (partition by ssvid order by start_time) as length_n3,
         LEAD(length, 4) over (partition by ssvid order by start_time) as length_n4,
  FROM `world-fishing-827.gfw_research_precursors.vc_v20200124_results_*`
),

lengths as (
  select *
  from _lengths
  where length_p1 is not null
    and length_p2 is not null
    and length_p3 is not null
    and length_p4 is not null
    and length_n1 is not null
    and length_n2 is not null
    and length_n3 is not null
    and length_n4 is not null
)


select * from lengths
where least(length_p1, length_p2, length_p3, length_p4) > 2 * greatest(length_n1, length_n2, length_n3, length_n4)
   or 2 * greatest(length_p1, length_p2, length_p3, length_p4) < least(length_n1, length_n2, length_n3, length_n4)
order by ssvid, change_date
"""
length_df = pd.read_gbq(query, project_id="world-fishing-827", dialect="standard")

# ## By SSVID only

# +

df = length_df.groupby(by=["ssvid"]).mean()
plt.plot(df.length_lbl, df.length, ".")
r2 = np.corrcoef(length_df.length_lbl, length_df.length)[0, 1] ** 2
r2avg = np.corrcoef(df.length_lbl, df.length)[0, 1] ** 2

# +
lbls = []
lens = []
for key, group in length_df.groupby(by=["ssvid"]):
    lbls.append(group.length_lbl.mean())
    scale = 10 * np.log(group.active_positions + 1) + np.log(group.positions + 1)
    l = (group.length * scale).sum() / scale.sum()
    lens.append(l)

plt.plot(lbls, lens, ".")

r2avg2 = np.corrcoef(lbls, lens)[0, 1] ** 2
print(f"{r2:.3f}, {r2avg:.3f}, {r2avg2:.3f}")
# -

# ## By SSVID and year

# +

df = length_df.groupby(by=["ssvid", "year"]).mean()
plt.plot(df.length_lbl, df.length, ".")
r2 = np.corrcoef(length_df.length_lbl, length_df.length)[0, 1] ** 2
r2avg = np.corrcoef(df.length_lbl, df.length)[0, 1] ** 2

# +
lbls = []
lens = []
for key, group in length_df.groupby(by=["ssvid", "year"]):
    lbls.append(group.length_lbl.mean())
    scale = 100 * np.log(group.active_positions + 1) + np.log(group.positions + 1)
    l = (group.length * scale).sum() / scale.sum()
    lens.append(l)

plt.plot(lbls, lens, ".")

r2avg2 = np.corrcoef(lbls, lens)[0, 1] ** 2
print(f"{r2:.3f}, {r2avg:.3f}, {r2avg2:.3f}")
# -


"""
with

_lengths as (
  SELECT ssvid, start_time,
         LAG(length, 2) over (partition by ssvid order by start_time) as length_p3,
         LAG(length, 1) over (partition by ssvid order by start_time) as length_p2,
         length as length_p1, 
         LEAD(length, 1) over (partition by ssvid order by start_time) as length_n1,
         LEAD(length, 2) over (partition by ssvid order by start_time) as length_n2,
         LEAD(length, 3) over (partition by ssvid order by start_time) as length_n3,
  FROM `world-fishing-827.gfw_research_precursors.vc_v20200124_results_*`
),

lengths as (
  select *
  from _lengths
  where length_p1 is not null
    and length_p2 is not null
    and length_p3 is not null
    and length_n1 is not null
    and length_n2 is not null
    and length_n3 is not null
)


select * from lengths
where least(length_p1, length_p2, length_p3) > 1.5 * greatest(length_n1, length_n2, length_n3)
   or 1.5 * greatest(length_p1, length_p2, length_p3) < least(length_n1, length_n2, length_n3)
order by ssvid, start_time

"""
