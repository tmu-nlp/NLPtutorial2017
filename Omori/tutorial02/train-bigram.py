import sys
from collections import defaultdict

def train_bigram(input_file, output_file):
    counts = defaultdict(int)
    context_counts = defaultdict(int)
    with open(input_file, 'r') as f:
        for line in f:
            word_list = line.strip().split()
            word_list.append("</s>")
            word_list.insert(0, "<s>")
            for i, word in enumerate(word_list[:-1]):
                counts[word+' '+word_list[i+1]] += 1
                context_counts[word] += 1
                counts[word_list[i+1]] += 1
                context_counts[''] += 1   

    with open(output_file, 'w') as f:
        for ngram, count in sorted(counts.items(), key=lambda x: -x[1]): 
            words = ngram.split(' ')
            del(words[-1])
            context = ''.join(words)
            p = counts[ngram] / context_counts[context]
            f.write(ngram+'\t'+str(p)+'\n')

if __name__ == "__main__":
    train_bigram(sys.argv[1], sys.argv[2])
