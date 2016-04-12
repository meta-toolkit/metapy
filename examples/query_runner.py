"""
Mimics MeTA's query-runner program.
"""

import sys
import time

import metapy

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: {} config.toml queries.txt".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    idx = metapy.index.make_inverted_index(cfg)

    query_path = sys.argv[2]
    start_time = time.time()
    with open(query_path) as query_file:
        okapi = metapy.index.OkapiBM25()
        for line in query_file:
            query = metapy.index.Document()
            query.content(line.strip())
            res_num = 1
            for doc in okapi.score(idx, query, 1000):
                print("{}. {} (id={}, score={})".format(
                    res_num, idx.doc_name(doc[0]), doc[0], round(doc[1], 4)))
                res_num += 1

    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
