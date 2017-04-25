print('Hello World!')

my_int = 4
my_float = 2.5
my_string = 'Hello'

print('String: %s\tfloat: %f\tint: %d' % (my_string, my_float, my_int))

my_variable = 5
if my_variable == 4:
    print('my_variable is 4')
else:
    print('my_variable is not 4')

for i in range(1, my_variable):
    print("i == %d" % (i) )

d = {'a':22, 'b':1, 'c':11, 'd':27}
d['e']=3
print(len(d))
print(d['c'])

if 'd' in d:
    print("d in dict")

for foo, bar in sorted(d.items()):
    print("%s -> %r" % (foo, bar))
    
from collections import defaultdict
defdict = defaultdict(lambda:0)

defdict['eric'] = 33

print(defdict['eric'])
print(defdict['f'])


for foo, bar in sorted(defdict.items()):
    print("%s -> %r" % (foo, bar))
    
s = "this is a pen"
ws = s.split(' ')

for w in ws:
     print(w)

print(" ||| ".join(ws))

import sys
myfile = open(sys.argv[1], "r")

for l in myfile:
    l = l.strip()
    if len(l) != 0:
        print(l)

