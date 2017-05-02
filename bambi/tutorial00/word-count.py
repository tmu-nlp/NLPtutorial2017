import sys
input_file = open(sys.argv[1],"r")

word_counts = {}
for line in input_file:
    line = line.strip()
    if len(line) != 0:
        words = line.split(" ")
        for word in words:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

for word in sorted(word_counts):
    print(word + " " + str(word_counts[word]))
