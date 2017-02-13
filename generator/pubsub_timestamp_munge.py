#!/usr/bin/env python
# Copyright 2014 Google Inc. All Rights Reserved.
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


"""Cloud Pub/Sub sample application."""


import argparse
import base64
import json
from random import randint
import re
import socket
import sys
import time

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials


PUBSUB_SCOPES = ["https://www.googleapis.com/auth/pubsub"]

BOTNAME = 'pubsub-irc-bot/1.0'

PORT = 6667

NUM_RETRIES = 3

BATCH_SIZE = 30


def fqrn(resource_type, project, resource):
    """Return a fully qualified resource name for Cloud Pub/Sub."""
    return "projects/{}/{}/{}".format(project, resource_type, resource)


def get_full_topic_name(project, topic):
    """Return a fully qualified topic name."""
    return fqrn('topics', project, topic)


def get_full_subscription_name(project, subscription):
    """Return a fully qualified subscription name."""
    return fqrn('subscriptions', project, subscription)


def list_topics(client, args):
    """Show the list of current topics."""
    next_page_token = None
    while True:
        resp = client.projects().topics().list(
            project='projects/{}'.format(args.project_name),
            pageToken=next_page_token).execute(num_retries=NUM_RETRIES)
        if 'topics' in resp:
            for topic in resp['topics']:
                print topic['name']
        next_page_token = resp.get('nextPageToken')
        if not next_page_token:
            break


def list_subscriptions(client, args):
    """Show the list of current subscriptions.

    If a topic is specified, only subscriptions associated with the topic will
    be listed.
    """
    next_page_token = None
    while True:
        if args.topic is None:
            resp = client.projects().subscriptions().list(
                project='projects/{}'.format(args.project_name),
                pageToken=next_page_token).execute(num_retries=NUM_RETRIES)
        else:
            topic = get_full_topic_name(args.project_name, args.topic)
            resp = client.projects().topics().subscriptions().list(
                topic=topic,
                pageToken=next_page_token).execute(num_retries=NUM_RETRIES)
        for subscription in resp['subscriptions']:
            print json.dumps(subscription, indent=1)
        next_page_token = resp.get('nextPageToken')
        if not next_page_token:
            break


def create_topic(client, args):
    """Create a new topic."""
    topic = client.projects().topics().create(
        name=get_full_topic_name(args.project_name, args.topic),
        body={}).execute(num_retries=NUM_RETRIES)
    print 'Topic {} was created.'.format(topic['name'])


def delete_topic(client, args):
    """Delete a topic."""
    topic = get_full_topic_name(args.project_name, args.topic)
    client.projects().topics().delete(
        topic=topic).execute(num_retries=NUM_RETRIES)
    print 'Topic {} was deleted.'.format(topic)


def create_subscription(client, args):
    """Create a new subscription to a given topic.

    If an endpoint is specified, this function will attach to that
    endpoint.
    """
    name = get_full_subscription_name(args.project_name, args.subscription)
    if '/' in args.topic:
        topic_name = args.topic
    else:
        topic_name = get_full_topic_name(args.project_name, args.topic)
    body = {'topic': topic_name}
    if args.push_endpoint is not None:
        body['pushConfig'] = {'pushEndpoint': args.push_endpoint}
    subscription = client.projects().subscriptions().create(
        name=name, body=body).execute(num_retries=NUM_RETRIES)
    print 'Subscription {} was created.'.format(subscription['name'])


def delete_subscription(client, args):
    """Delete a subscription."""
    subscription = get_full_subscription_name(args.project_name,
                                              args.subscription)
    client.projects().subscriptions().delete(
        subscription=subscription).execute(num_retries=NUM_RETRIES)
    print 'Subscription {} was deleted.'.format(subscription)


def _check_connection(irc):
    """Check a connection to an IRC channel."""
    readbuffer = ''
    while True:
        readbuffer = readbuffer + irc.recv(1024)
        temp = readbuffer.split('\n')
        readbuffer = temp.pop()
        for line in temp:
            if "004" in line:
                return
            elif "433" in line:
                sys.err.write('Nickname is already in use.')
                sys.exit(1)


def connect_irc(client, args):
    """Connect to an IRC channel and publishe messages."""
    server = args.server
    channel = args.channel
    topic = get_full_topic_name(args.project_name, args.topic)
    nick = 'bot-{}'.format(args.project_name)
    irc = socket.socket()
    print 'Connecting to {}'.format(server)
    irc.connect((server, PORT))

    irc.send("NICK {}\r\n".format(nick))
    irc.send("USER {} 8 * : {}\r\n".format(nick, BOTNAME))
    readbuffer = ''
    _check_connection(irc)
    print 'Connected to {}.'.format(server)

    irc.send("JOIN {}\r\n".format(channel))
    priv_mark = "PRIVMSG {} :".format(channel)
    p = re.compile(
        r'\x0314\[\[\x0307(.*)\x0314\]\]\x03.*\x0302(http://[^\x03]*)\x03')
    while True:
        readbuffer = readbuffer + irc.recv(1024)
        temp = readbuffer.split('\n')
        readbuffer = temp.pop()
        for line in temp:
            line = line.rstrip()
            parts = line.split()
            if parts[0] == "PING":
                irc.send("PONG {}\r\n".format(parts[1]))
            else:
                i = line.find(priv_mark)
                if i == -1:
                    continue
                line = line[i + len(priv_mark):]
                m = p.match(line)
                if m:
                    line = "Title: {}, Diff: {}".format(m.group(1), m.group(2))
                body = {
                    'messages': [{'data': base64.b64encode(str(line))}]
                }
                client.projects().topics().publish(
                    topic=topic, body=body).execute(num_retries=NUM_RETRIES)


def publish_message(client, args):
    """Publish a message to a given topic."""
    topic = get_full_topic_name(args.project_name, args.topic)
    message = base64.b64encode(str(args.message))
    body = {'messages': [{'data': message}]}
    resp = client.projects().topics().publish(
        topic=topic, body=body).execute(num_retries=NUM_RETRIES)
    print ('Published a message "{}" to a topic {}. The message_id was {}.'
           .format(args.message, topic, resp.get('messageIds')[0]))


def pull_messages(client, args):
    """Pull messages from a given subscription."""
    subscription = get_full_subscription_name(
        args.project_name,
        args.subscription)
    # arghh
    # TODO -- if I need to keep this script, unhardwire.
    topic2 = get_full_topic_name(args.project_name, "gfwfeatures2")
    body = {
        'returnImmediately': False,
        'maxMessages': BATCH_SIZE
    }
    while True:
        try:
            resp = client.projects().subscriptions().pull(
                subscription=subscription, body=body).execute(
                    num_retries=NUM_RETRIES)
        except Exception as e:
            time.sleep(0.5)
            print e
            continue
        receivedMessages = resp.get('receivedMessages')
        if receivedMessages:
            ack_ids = []
            messages = []
            for receivedMessage in receivedMessages:
                message = receivedMessage.get('message')
                if message:
                    rint = randint(0,4999)
                    if not rint:  # print a sampling of what's getting output
                        print(base64.b64decode(str(message.get('data'))))
                    s_info = base64.b64decode(str(message.get('data')))
                    j_info = json.loads(s_info)
                    timestamp = j_info['firstTimestamp']
                    if not rint:
                        print("%s : got timestamp: %s" % (rint, timestamp))
                    message['attributes'] = {'timestamp': str(timestamp)}
                    messages.append(message)
                    ack_ids.append(receivedMessage.get('ackId'))
            body2 = {'messages': messages}
            resp = client.projects().topics().publish(
                topic=topic2, body=body2).execute(num_retries=NUM_RETRIES)
            ack_body = {'ackIds': ack_ids}
            client.projects().subscriptions().acknowledge(
                subscription=subscription, body=ack_body).execute(
                    num_retries=NUM_RETRIES)
        if args.no_loop:
            break


def main(argv):
    """Invoke a subcommand."""
    # Main parser setup
    parser = argparse.ArgumentParser(
        description='A sample command line interface for Pub/Sub')
    parser.add_argument('project_name', help='Project name in console')

    topic_parser = argparse.ArgumentParser(add_help=False)
    topic_parser.add_argument('topic', help='Topic name')
    subscription_parser = argparse.ArgumentParser(add_help=False)
    subscription_parser.add_argument('subscription', help='Subscription name')

    # Sub command parsers
    sub_parsers = parser.add_subparsers(
        title='List of possible commands', metavar='<command>')

    list_topics_str = 'List topics in project'
    parser_list_topics = sub_parsers.add_parser(
        'list_topics', description=list_topics_str, help=list_topics_str)
    parser_list_topics.set_defaults(func=list_topics)

    create_topic_str = 'Create a topic with specified name'
    parser_create_topic = sub_parsers.add_parser(
        'create_topic', parents=[topic_parser],
        description=create_topic_str, help=create_topic_str)
    parser_create_topic.set_defaults(func=create_topic)

    delete_topic_str = 'Delete a topic with specified name'
    parser_delete_topic = sub_parsers.add_parser(
        'delete_topic', parents=[topic_parser],
        description=delete_topic_str, help=delete_topic_str)
    parser_delete_topic.set_defaults(func=delete_topic)

    list_subscriptions_str = 'List subscriptions in project'
    parser_list_subscriptions = sub_parsers.add_parser(
        'list_subscriptions',
        description=list_subscriptions_str, help=list_subscriptions_str)
    parser_list_subscriptions.set_defaults(func=list_subscriptions)
    parser_list_subscriptions.add_argument(
        '-t', '--topic', help='Show only subscriptions for given topic')

    create_subscription_str = 'Create a subscription to the specified topic'
    parser_create_subscription = sub_parsers.add_parser(
        'create_subscription', parents=[subscription_parser, topic_parser],
        description=create_subscription_str, help=create_subscription_str)
    parser_create_subscription.set_defaults(func=create_subscription)
    parser_create_subscription.add_argument(
        '-p', '--push_endpoint',
        help='Push endpoint to which this method attaches')

    delete_subscription_str = 'Delete the specified subscription'
    parser_delete_subscription = sub_parsers.add_parser(
        'delete_subscription', parents=[subscription_parser],
        description=delete_subscription_str, help=delete_subscription_str)
    parser_delete_subscription.set_defaults(func=delete_subscription)

    connect_irc_str = 'Connect to the topic IRC channel'
    parser_connect_irc = sub_parsers.add_parser(
        'connect_irc', parents=[topic_parser],
        description=connect_irc_str, help=connect_irc_str)
    parser_connect_irc.set_defaults(func=connect_irc)
    parser_connect_irc.add_argument('server', help='Server name')
    parser_connect_irc.add_argument('channel', help='Channel name')

    publish_message_str = 'Publish a message to specified topic'
    parser_publish_message = sub_parsers.add_parser(
        'publish_message', parents=[topic_parser],
        description=publish_message_str, help=publish_message_str)
    parser_publish_message.set_defaults(func=publish_message)
    parser_publish_message.add_argument('message', help='Message to publish')

    pull_messages_str = ('Pull messages for given subscription. '
                         'Loops continuously unless otherwise specified')
    parser_pull_messages = sub_parsers.add_parser(
        'pull_messages', parents=[subscription_parser],
        description=pull_messages_str, help=pull_messages_str)
    parser_pull_messages.set_defaults(func=pull_messages)
    parser_pull_messages.add_argument(
        '-n', '--no_loop', action='store_true',
        help='Execute only once and do not loop')

    # Google API setup
    credentials = GoogleCredentials.get_application_default()
    if credentials.create_scoped_required():
        credentials = credentials.create_scoped(PUBSUB_SCOPES)
    client = discovery.build('pubsub', 'v1', credentials=credentials)

    args = parser.parse_args(argv[1:])
    args.func(client, args)


if __name__ == '__main__':
    main(sys.argv)
