#python test-hmm-beam.py ../../data/wiki-en-test.norm model.txt my_answer.pos 0.95 1000000 5
#../../script/gradepos.pl ../../data/wiki-en-test.pos my_answer.pos

from collections import defaultdict
import math
import sys

def test_hmm_beam(fin_path, model_path, fout_path, lambda_1, vocabulary, beam_size):
    transition = defaultdict(lambda: 0.0)
    emission = defaultdict(lambda: 0.0)
    possible_tags = dict()

    with open(fin_path, "r") as fin, open(model_path, "r") as fm, open(fout_path, "w") as fout:
        for line in fm:
            typ, context, word, prob = line.strip("\n").split(" ")
            possible_tags[context] = 1
            if typ == "T":
                transition["{} {}".format(context, word)] = float(prob)
            else:
                emission["{} {}".format(context, word)] = float(prob)

        for line in fin:
            words = line.strip("\n").split(" ")
            best_score = dict()
            best_edge = dict()
            best_score["0 <s>"] = 0
            best_edge["0 <s>"] = "NULL"
            active_tags = [["<s>"]]

            for i in range(0, len(words)):
                my_best = dict()
                for prev in active_tags[i]:
                    for nex in possible_tags.keys():
                        if "{} {}".format(i, prev) in best_score and "{} {}".format(prev, nex) in transition:
                            score = best_score["{} {}".format(i, prev)] - math.log(transition["{} {}".format(prev, nex)], 2) - math.log(lambda_1 * emission["{} {}".format(nex, words[i])] + (1 - lambda_1) / vocabulary, 2)
                            if "{} {}".format(i + 1, nex) not in best_score or best_score["{} {}".format(i + 1, nex)] > score:
                                best_score["{} {}".format(i + 1, nex)] = score
                                best_edge["{} {}".format(i + 1, nex)] = "{} {}".format(i, prev)
                                my_best[nex] = score
                active_tags.append([element for rank, (element, score) in zip(range(beam_size), sorted(my_best.items(), key = lambda x: x[1]))])

            for prev in active_tags[i + 1]:
                if "{} {}".format(len(words), prev) in best_score and "{} </s>".format(prev) in transition:
                    score = best_score["{} {}".format(len(words), prev)] - math.log(transition["{} </s>".format(prev)], 2)
                    if "{} </s>".format(len(words) + 1) not in best_score or best_score["{} </s>".format(len(words) + 1)] > score:
                        best_score["{} </s>".format(len(words) + 1)] = score
                        best_edge["{} </s>".format(len(words) + 1)] = "{} {}".format(len(words), prev)

            tags = list()
            next_edge = best_edge["{} </s>".format(len(words) + 1)]
            while next_edge != "0 <s>":
                position, tag = next_edge.split()
                tags.append(tag)
                next_edge = best_edge[next_edge]
            tags.reverse()
            fout.write(" ".join(tags) + "\n")

if __name__ == "__main__":
    test_hmm_beam(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
