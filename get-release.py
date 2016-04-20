#!/usr/bin/env python
from __future__ import print_function

from clint.textui import progress
import requests
import sys

if len(sys.argv) != 2:
    print("Usage: {} release-tag".format(sys.argv[0]))
    sys.exit(1)

baseurl = 'https://api.github.com/repos/meta-toolkit/metapy/releases/tags'

r = requests.get('{}/{}'.format(baseurl, sys.argv[1]))

if r.status_code != 200:
    print("Error: {}".format(r.status_code))
    print(r.text)
    sys.exit(1)

json = r.json()

print("Found release {} tagged by {}".format(json['tag_name'],
    json['author']['login']))

for asset in json['assets']:
    url = asset['browser_download_url']
    name = asset['name']
    print("Fetching {}...".format(name))

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print("Error fetching {}: {}".format(name, r.status_code))
        print(r.text)
        sys.exit(1)

    with open('dist/{}'.format(name), 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size = 4096),
                expected_size = total_length / 4096 + 1):
            if chunk:
                f.write(chunk)
        f.flush()

print("Done!")
