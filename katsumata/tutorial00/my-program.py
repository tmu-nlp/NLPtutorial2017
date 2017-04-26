import sys
from collections import defaultdict
my_int = 4
my_float = 2.5
my_string = "hello"
my_variable = 5
print ("Hello World!")
print ("string: %s\tfloat: %f\tint: %d" % (my_string, my_float, my_int))
if my_variable == 4:
    print("my_variable is 4")
else:
    print("my_variable is not 4")
for i in range(1, my_variable):
    print ("i == %d" % (i))
my_dict = {"alan": 22, "bill": 45, "chris": 17, "dan": 27}

my_dict["eric"] = 33

if "dan" in my_dict:
    print("dan exists in my_dict")
for foo, bar in sorted(my_dict.items()):
    print("%s --> %d" %(foo, bar))

my_dict = defaultdict(lambda: 0)
my_dict["eric"] = 33

print(my_dict["eric"])
print(my_dict["fred"])

for foo, bar in sorted(my_dict.items()):
    print("%s --> %d" %(foo, bar))

sentence = "this is a pen"
words = sentence.split(" ")

for word in words:
    print (word)
print(" ||| ".join(words))

my_file = open(sys.argv[1], "r")

for line in my_file:
    line = line
    #line = line.strip()

    if len(line) != 0:
        print(line)
my_file.close()
