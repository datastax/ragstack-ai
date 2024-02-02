"""
Utilities to provide connection to Cassandra
"""
import os

from cassandra.cluster import (
    Cluster,
)
from cassandra.auth import PlainTextAuthProvider

LOCAL_KEYSPACE = os.environ.get('LOCAL_KEYSPACE', 'cassio_tutorials')
LOCAL_CONTACT_POINT_STRING = os.environ.get('LOCAL_CONTACT_POINTS', '')

def getCassandraCQLSession():
    contact_points = [
        node_addr.strip()
        for node_addr in LOCAL_CONTACT_POINT_STRING.split(",") 
        if node_addr.strip()
    ]
    if contact_points:
        cluster = Cluster(contact_points)
    else:
        cluster = Cluster()
    localSession = cluster.connect()
    return localSession

def getCassandraCQLKeyspace():
    return LOCAL_KEYSPACE