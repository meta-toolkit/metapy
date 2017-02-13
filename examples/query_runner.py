"""
Mimics MeTA's query-runner program.
"""

import math
import sys
import time

import metapy

class PL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA
    """
    def __init__(self, c_param=0.5):
        self.c = c_param
        super(PL2Ranker, self).__init__()

    def score_one(self, sd):
        lda = sd.num_docs / sd.corpus_term_count
        tfn = sd.doc_term_count * math.log2(1.0 + self.c * sd.avg_dl /
                sd.doc_size)
        if lda < 1 or tfn <= 0:
            return 0.0
        numerator = tfn * math.log2(tfn * lda) \
                        + math.log2(math.e) * (1.0 / lda - tfn) \
                        + 0.5 * math.log2(2.0 * math.pi * tfn)
        return sd.query_term_weight * numerator / (tfn + 1.0)

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: {} config.toml queries.txt start_query".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    idx = metapy.index.make_inverted_index(cfg)

    query_path = sys.argv[2]
    query_num = int(sys.argv[3])
    start_time = time.time()
    with open(query_path) as query_file:
        pl2 = PL2Ranker()
        for line in query_file:
            query = metapy.index.Document()
            query.content(line.strip())
            res_num = 1
            for doc in pl2.score(idx, query, 1000):
                docno = idx.metadata(doc[0]).get('name')
                print("{}\t_\t{}\t{}\t{}\tMeTA".format( query_num, docno,
                    res_num, doc[1]))
                res_num += 1
            query_num += 1

    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
