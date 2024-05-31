"""Module"""

import re

full_name = "Федір Бургонов".split(" ")
res = []

for i in full_name[0]:
    if i in full_name[1]:
        res.append(i)
        res = sorted(res)

print(f"These are the same letters: [{''.join(res)}]")

res = []

for i in range(1, 1001):
    if i % 7 == 0:
        continue

    if i % 5 == 0:
        res.append(i)

    elif i % 3 == 0:
        res.append(i)

print(res)

data = re.sub("[ \u2019]", "", "Сім п’ятниць на тиждень")
unique_letters = set(data)
num_letters = len(set(unique_letters))
res = {}

for i in data:
    if i in res:
        res[i] += 1
    else:
        res[i] = 1

res = sorted(res.items(), key=lambda x: x[1], reverse=True)

for key, val in res[:3]:
    print(f"[{key}]: {val} times")
