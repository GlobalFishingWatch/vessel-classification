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

I believe requires py3.3+ for timestamp stuff.

Run the script like this to 'replay', with pauses in data publication
consistent with pauses in the series of data timestamps, which arrive every 5
minutes:
$ python shipping_data_generator.py --filename 'yourdatafile'
...
Run 'python shipping_data_generator.py -h' for more information.

"""
import argparse
import base64
import csv
import datetime
import json
import multiprocessing as mp
import random
import sys
import time

from apiclient import discovery
from dateutil.parser import parse
import httplib2
from oauth2client.client import GoogleCredentials

DEFAULT_POOL_SIZE = 4


def send_loop(client, q, pubsub_topic):  # pragma: NO COVER
    """Process loop for indefinitely sending logs to Cloud Pub/Sub.

    Args:
      client: Pub/Sub client.
      q: mp.JoinableQueue instance to get the message from.
      topic: Cloud Pub/Sub topic name to send the logs.
      retry: How many times to retry upon Cloud Pub/Sub API failure.
    """
    if client is None:
        # then exit
        print("in send_loop, no client defined")
        return
    while True:
        try:
            messages = q.get()
        except Empty:
            continue
        try:
            publish(client, pubsub_topic, messages)
        except errors.RecoverableError as e:
            # Records the exception, puts the messages list to the deque,
            # and prints the exception to stderr.
            q.put(messages)
            print( "error: %s" % e)
        except Exception as e:
            print( "There was a non recoverable error %s, exiting." % e)
            return
        q.task_done()

# default; set to your traffic topic. Can override on command line.
SHIPPING_TOPIC = 'projects/aju-vtests2/topics/shipping'

LINE_BATCHES = 10  # report periodic progress

PUBSUB_SCOPES = ['https://www.googleapis.com/auth/pubsub']
NUM_RETRIES = 3


def create_pubsub_client():
  """Build the pubsub client."""
  credentials = GoogleCredentials.get_application_default()
  if credentials.create_scoped_required():
    credentials = credentials.create_scoped(PUBSUB_SCOPES)
  http = httplib2.Http()
  credentials.authorize(http)
  return discovery.build('pubsub', 'v1beta2', http=http)


def publish(client, pubsub_topic, messages):
  """Publish to the given pubsub topic."""
  body = {'messages': messages}
  resp = client.projects().topics().publish(
    topic=pubsub_topic, body=body).execute(num_retries=NUM_RETRIES)
  return resp

def create_msg(data_line, msg_attributes=None):
  """..."""
  data = base64.b64encode(data_line.encode('utf-8'))
  msg_payload = {'data': data.decode('utf-8')}
  if msg_attributes:
    msg_payload['attributes'] = msg_attributes
  return msg_payload


def main(argv):
  parser = argparse.ArgumentParser()
  # TODO - process set of files/subdir
  parser.add_argument("--filename", help="shipping data filename")
  parser.add_argument("--num_lines", type=int, default=0,
            help="The number of lines to process. " +
            "0 indicates all.")
  parser.add_argument("--topic", default=SHIPPING_TOPIC,
            help="The pubsub 'shipping' topic to publish to. " +
            "Should already exist.")
  args = parser.parse_args()

  pubsub_topic = args.topic
  print("Publishing to pubsub 'shipping' topic: %s" % pubsub_topic)
  filename = args.filename
  print("filename: %s" % filename)
  num_lines = args.num_lines
  if num_lines:
    print("processing %s lines" % num_lines)

  client = create_pubsub_client()

  q = mp.JoinableQueue()
  for _ in range(DEFAULT_POOL_SIZE):
      p = mp.Process(target=send_loop,
           args=(client, q, pubsub_topic))
      p.daemon = True
      p.start()


  line_count = 0
  incident_count = 0
  messages = []

  with open(args.filename, "r") as f:
    for line in f:
      jline = json.loads(line)
      # print("jsonified line is: %s" % jline)
      line_count += 1
      if num_lines:  # if terminating after num_lines processed
        if line_count >= num_lines:
          print( "Have processed %s lines" % num_lines)
          break
      if (line_count % LINE_BATCHES) == 0:
        sys.stdout.write('.')
        print("----calling q.put at %s" % line_count)
        q.put(messages)
        # publish(client, pubsub_topic, messages)
        messages = []
      ts = parse(jline['timestamp']).timestamp()
      print("----got timestamp: %s" % ts)
      msg_attributes = {'timestamp': str(int(ts * 1000))}  # need to convert s -> ms
      msg_data = json.dumps(jline)
      msg = create_msg(msg_data, msg_attributes)
      messages.append(msg)
  # pick up the residue
  print("\n---calling final q.put")
  # publish(client, pubsub_topic, messages)
  if messages:
    q.put(messages)
  print( "sleeping...")
  time.sleep(60)
  print( "end sleep")





if __name__ == '__main__':
    main(sys.argv)
