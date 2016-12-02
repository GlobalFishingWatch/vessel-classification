## Overview

This directory contains the results of training models for fishing localisation
and fishing vessel classification, as well as metric report files showing the
accuracy of each.

## Full output data

Full output data, including features and vessel and fishing localisation
timeranges and annotations can be found on GCS here:

gs://world-fishing-827/data-production/classification/release-0.1.0/results

## Methods

* The anchorages pipeline was run only on 2015 data, due to an open issue with
  the pipeline OOM-ing for the full time range.
* The model feature pipeline was run on 2012-2016 with adjacency disabled
  (because this was added recently and we were running too low on time to try
  running with it enabled), but anchorage visits enabled.
* Both fishing localisation and vessel classification models were trained until
  there was no further improvement in test performance (currently they do not
  appear to overfit and start dropping performance within 500k training
  iterations).
* Inference and metrics computations were run on the Test dataset split.