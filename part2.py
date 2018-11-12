import sys
from sharedFunctions import estEmissions


def predictSentiments(emissions, testfile):
    """
    Predicts sentiments using argmax(emission)
    Saves labelled file as dev.p2.out

    @param emissions: output from estEmissions function
    @param testfile: input file with unlabelled text
    """
    with open(testfile) as f, open("dev.p2.out", "w") as out:
        for line in f:
            if line == "\n":
                out.write(line)
            else:
                word = line.strip()

                # find most likely tag for word
                bestP = 0
                bestTag = ""
                for tag in emissions.keys():
                    if word in emissions[tag]:
                        if emissions[tag][word] > bestP:
                            bestP = emissions[tag][word]
                            bestTag = tag
                    else:
                        if emissions[tag]["#UNK#"] > bestP:
                            bestP = emissions[tag]["#UNK#"]
                            bestTag = tag

                out.write("{} {}\n".format(word, bestTag))


# main
if len(sys.argv) != 3:
    print("Usage: python3 part2.py [train file] [test file]")

_, trainfile, testfile = sys.argv
emissions = estEmissions(trainfile)
predictSentiments(emissions, testfile)
