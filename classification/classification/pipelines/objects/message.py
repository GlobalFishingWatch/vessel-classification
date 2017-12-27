from collections import namedtuple
from .namedtuples import NamedtupleCoder

Message = namedtuple("Message", 
    ["vessel_id", "timestamp", "lat", "lon", 
     "fishing_score"])


class MessageCoder(NamedtupleCoder):
    target = Message
    time_fields = ['timestamp']


MessageCoder.register()
