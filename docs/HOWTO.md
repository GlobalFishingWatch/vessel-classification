# A set of examples of how to perform (prod) classification.

For dev runs, use `--env=dev`.

## Building features, training and model, running inference.

### Run the anchorages pipeline to generate a set of anchorages.

```
./sbt
> project anchorages
> run --env=prod --job-name=release-0.1.0_anchorages_full_timerange --zone=europe-west1-c  --maxNumWorkers=200
```

### Run the feature pipeline to generate a set of features.

(ensure `deploy_cloudml_config_template.txt` points to the latest feature path).

```
./sbt
> project features
> run --env=prod --job-name=release-0.1.0 --anchorages-root-path=gs://world-fishing-827/data-production/classification/release-0.1.0/pipeline/output --generate-encounters=false --zone=europe-west1-c  --maxNumWorkers=600 --diskSizeGb=150
```

### Run training

For fishing ranges:

```
./deploy_cloudml.py --model_name alex.fishing_range_classification --job_name release-0.1.0 --env prod
```


For vessel classification and length regression:

```
./deploy_cloudml.py --model_name alex.vessel_classification --job_name release-0.1.0 --env prod
```

### Run inference

For fishing ranges (all likely fishing vessels):

```
python -m classification.run_inference alex.fishing_range_classification --root_feature_path gs://world-fishing-827/data-production/classification/release-0.1.0/pipeline/output/features --inference_results_path=`pwd`/release-0.1.0-fishing-ranges-likely-fishing.json.gz --inference_parallelism 20  --feature_dimensions 12 --model_checkpoint_path fishing-range-model.ckpt-500007 --dataset_split likely_fishing_mmsis.txt
```

(for the Test set, for metric computation add `--dataset_split Test`).

For vessel classification and length regression (whole ocean):

```
python -m classification.run_inference alex.vessel_classification --root_feature_path gs://world-fishing-827/data-production/classification/release-0.1.0/pipeline/output/features --inference_results_path=`pwd`/release-0.1.0-vessel-classification-test.json.gz  --inference_parallelism 20  --feature_dimensions 12 --model_checkpoint_path vessel-classification-model.ckpt-159530
```

(for the Test set, for metric computation add `--dataset_split Test`).

### Compute metrics for the model runs

For fishing ranges:

```
python compute_metrics.py --inference-path release-0.1.0-fishing-ranges-test.json.gz --label-path ./classification/data/net_training_20161115.csv --fishing-ranges ./classification/data/combined_fishing_ranges.csv --dest-path release-0.1.0-fishing-ranges-test-report.html
```

For vessel classification and length regression:

```
python compute_metrics.py --inference-path release-0.1.0-vessel-classification-test.json.gz --label-path ./classification/data/net_training_20161115.csv --fishing-ranges ./classification/data/combined_fishing_ranges.csv --dest-path vessel-classification-test-report.html
```

### Annotate likely fishing vessel AIS data with fishing ranges

## Other functionality

* Run the feature pipeline to generate a set of encounters.
