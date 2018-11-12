from pathlib import Path
import sys
import os

# main
if len(sys.argv) < 2:
    print("Usage: python evalAll.py [p2/p3/...]")
    sys.exit()

task = sys.argv[1]

datasets = ["EN", "FR", "CN", "SG"]
output = "{}_results".format(task)

with open(output, "w") as f:
    f.write("{} Results\n\n\n".format(task.upper()))
    for ds in datasets:
        datafolder = Path(ds)
        predictFile = datafolder / "dev.{}.out".format(task)
        testFile = datafolder / "dev.out"
        result = os.popen("python evalResult.py {} {}".format(testFile, predictFile)).read()

        f.write(ds)
        f.write(result)
        f.write(40 * "_" + 3 * "\n")

print("Output: {}".format(output))
print("Done!")
