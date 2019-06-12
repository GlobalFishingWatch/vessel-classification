# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v3.0.0 - (2019-06-12)

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