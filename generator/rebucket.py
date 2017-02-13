#!/usr/bin/env python
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
...

"""
import argparse
import base64
import csv
import datetime
import json
import multiprocessing as mp
import pathlib
import random
import string
import os
import sys
import time

from apiclient import discovery
from dateutil.parser import parse
import httplib2
from oauth2client.client import GoogleCredentials

LINE_BATCHES = 100  # report periodic progress


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument("--filepath", help="shipping data filepath")
  args = parser.parse_args()

  filepath = args.filepath
  print("filepath: %s" % filepath)
  line_count = 0

  filelist = [p for p in pathlib.Path(args.filepath).glob('**/*' ) if p.is_file()]
  print("\n----filelist: %s" % filelist)
  dirname = 'bkts/%s' % args.filepath
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  time.sleep(5)

  for fname in filelist:
    buckets = [[] for i in range(24)]
    with fname.open() as f:
      print("\n----working on: %s" % fname)
      time.sleep(10)
      for line in f:
        jline = json.loads(line)
        # print("jsonified line is: %s" % jline)
        line_count += 1
        if (line_count % LINE_BATCHES) == 0:
          sys.stdout.write('.')
        # print("----got timestamp: %s" % jline['timestamp'])
        timeinfo = parse(jline['timestamp'])
        hour = timeinfo.hour
        buckets[hour].append(line)

    print("appending to buckets")
    for hour in range(24):
      bucket_fname = "%s/bucket%02d.json" % (dirname, hour)
      with open(bucket_fname, "a") as f2:
        f2.writelines(buckets[hour])


if __name__ == '__main__':
    main(sys.argv)
