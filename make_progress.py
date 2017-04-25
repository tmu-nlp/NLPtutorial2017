import os
import re
import numpy as np
import matplotlib

matplotlib.use('Agg')

import pylab as plt

ignore = ['.git', 'README', 'data', 'script', 'test']
maxcounts = [2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0]

user = list()
progress = list()
for name in [f for f in os.listdir() if os.path.isdir(f)]:
    if name in ignore:
        continue
    user.append(name)
    
    score = list()
    for num, maxcount in zip(range(14), maxcounts):
        count = 0
        chapter = 'tutorial{0:02d}'.format(num)
        if chapter in os.listdir(name):
            for script in os.listdir(os.path.join(name, chapter)):
                count += 1 if re.match(r'.+\.py', script) else 0
        score.append(min([count / maxcount, 1.0]))

    progress.append(np.array(score, dtype=np.float32))

npscore = np.vstack(progress)
colors = ['red', 'orange', 'yellow', 'lime', 'green', 'turquoise', 'blue', 'indigo', 'purple', 'pink', 'red', 'orange', 'yellow', 'lime']
labels = ['tutorial{0:02d}'.format(num) for num in range(14)]

offset = np.zeros(len(user))
for i in range(npscore.shape[1]):
    plt.bar(range(npscore.shape[0]), npscore[:, i], 0.6, offset, align='center', color=colors[i], label=labels[i])
    offset += npscore[:, i]

plt.xticks(range(npscore.shape[0]), user, fontsize=7)
plt.yticks(np.arange(0, 15, 1))
plt.legend(fontsize=8, bbox_to_anchor=(1.2, 1.0))
plt.subplots_adjust(right=0.8)
plt.grid(True)
plt.savefig('progress.png')
plt.show()
