import sys

file = sys.argv[1]
with open(file) as f:
    for line in f:
        word = line.strip().split(" ")[0]
        if word == "_START" or word == "_STOP":
            print("NOT UNIQUE")
            break
