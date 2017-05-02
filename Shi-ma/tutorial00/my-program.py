# print #
print('Hello World!!')

my_int = 4
my_float = 2.5
my_string = 'hello'
print('string: %s\tfloat: %f\tint: %d' % (my_string, my_float, my_int))

# if #
my_variable = 5
if my_variable == 4:
    print('my_variable is 4')
else:
    print('my_variable is not 4')

# for #
for i in range(1, my_variable):
    print('i == %d' % (i))

# dict #
my_dict = {'alan': 22, 'bill': 45, 'chris': 17, 'dan': 27}
my_dict['eric'] = 33
print(len(my_dict))
print(my_dict['chris'])
if 'dan' in my_dict:
    print('dan exsists in my_dict')
for foo, bar in sorted(my_dict.items()): # .items() キーと値の両方 .keys() キー .values() 値 #
    print('{0} --> {1}'.format(foo, bar))

# defaultdict #
from collections import defaultdict
my_dict = defaultdict(lambda: 0)
my_dict['eric'] = 33
print(my_dict['eric'])
print(my_dict['fred'])
for foo, bar in sorted(my_dict.items()):
    print('{0} --> {1}'.format(foo, bar))

# split $ join #
sentence = 'this is a pen'
words = sentence.split(' ')

for word in words:
    print(word)
print(' ||| '.join(words))

# def #
def add_and_abs(x, y):
    z = x + y
    if z >= 0:
        return z
    else:
        return z * -1
print(add_and_abs(-4, 1))

# comand line #
import sys
my_file = open(sys.argv[1], 'r')
for line in my_file:
    line = line.strip()
    if len(line) != 0:
        print(line)
