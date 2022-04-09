import sqlite3
import os
import sys

root = sys.argv[1]
result_paths = [os.path.join(root, p) for p in os.listdir(root)]
result_paths.sort()
print(result_paths)
for p in result_paths:
    db = sqlite3.connect(p)
    c = db.execute("select * from CUPTI_ACTIVITY_KIND_MEMCPY2;")
    s = 0
    for row in c:
        s += row[5]
    print(p.split("/")[-1], s)
