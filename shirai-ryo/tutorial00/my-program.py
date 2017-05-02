#!/usr/bin/python3
print("Hello World!")
my_int = 4
my_float = 2.5
my_string = "hello"

print("string: %s\tfloat: %f\tint: %d" % (my_string, my_float, my_int))


my_variable = 5

if my_variable == 4:
	print("my_variable is 4")
else:
	print("my_variable is not 4")

for i in range(1, my_variable):
	print("i == %d" % (i))

my_list = [1, 2, 4, 8, 16]
my_list.append(32)
print(len(my_list))
print("")	#改行を作ることができる
print(my_list[3])


my_dict = {"alan": 22, "bill": 45, "chris": 17, "dan": 27}
my_dict["eric"] = 33

for foo, bar in sorted(my_dict.items()):
	print("%s --> %r" % (foo, bar))


from collections import defaultdict
my_dict = defaultdict(lambda: 0)



sentence = "this is a pen"
words = sentence.split(" ")

for word in words:
	print(word)

print(" ||| ".join(words))

#!/usr/bin/python3	

import sys
my_file = open(sys.argv[1], "r")
for line in my_file:
	line = line.strip()
	if len(line) != 0:
		print(line)

