# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v3.0.2 - 2020-03-29

### Added

  * [GlobalFishingWatch/gfw-eng-tasks#44](https://github.com/GlobalFishingWatch/gfw-eng-tasks/issues/44): Adds
    a fix bug when range didn't include features.
    Also adds missing test files

## v3.0.1 - 2020-03-20

### Added

  * [GlobalFishingWatch/gfw-eng-tasks#34](https://github.com/GlobalFishingWatch/gfw-eng-tasks/issues/34): Adds
    * Removes commented code
    * Pull out dependence on ujson and NewLineJson since no longer used
    * Reinstate padding; factor out new hash function so can be used from feature pipeline
    * Disable padding during fishing inference
    * Fix padding; remove approx_means
    * Fix pathname when generaring training paths
    * Switch to using blake2b for hashing
    * Fix padding bug that generated a lot of fishing regions with bad timestamps
    * Add missing input functions
    * Fix type causing fail when range was done
    * Remove debuggin logging
    * Bug fix
    * Fix ranges in vessel classification
    * More logging
    * More debugginf logging
    * Change logging to warning to make sure gets through
    * Add debugging logging for time range computation
    * Use research vessels again
    * Stop converting id to int when running inference
    * Force zip to return list under py3
    * Fix speed problems and memory leaks
    * Python 3 compatibility; much directed at working around change in builtin hash
    * Simple python 3 fixes; mostly in tests
    * Fix print statements
    * Separate cargo and tanker in coarse mapping
    * Improve metrics computation and add some support for auto generating docs from metadata
    * Add model that uses depths so we can test depth inference
    * Dont apply synonyms to top level classes
    * Automatically convert seismic_vessels to research
    * Tweak compute_metrics to keep fine classification table in defined order
    * Reinstate seismic vessel as unused class; improvements to metrics
    * Fix compute vessel metrics
    * Remove seismic vessel and update training and testing to use vessel database
    * Debug training data generation and update metric computation.
    * Silence yaml warning by switching to  safe_load
    * Tweak training invokation to try to improve speed
    * Switch to computing vessel parameters in terms of MMSI

## v3.0.0 - 2019-06-12

### Added

**BREAKING CHANGE, requires pipe-tools and pipe-features 3.0**
* [#41](https://github.com/GlobalFishingWatch/pipe-features/pull/41)
  * Refactor to use new Tensorflow Dataset and Estimator APIs
  * Support more recent tensorflow versions.
  * Go back to original random rolling of data during training since tests 
    showed slightly better accuracy.
  * Changes to support UVI and MMSI simultaneously.
  * Fix way vessel types are upsampled
  * Fix vessel metrics to work with vessel_id.
  * Correctly stratify data using new classes.
  * Generate training data directly from Vessel Database
  * Modify training invocation to make tracking runs easier.
