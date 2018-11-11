import sys
from sharedFunctions import estEmissions


def predictSentiments(emissions, file):
    """
    Predicts sentiments using argmax(emission)
    Saves labelled file as dev.p2.out

    @param emissions: output from estEmissions function
    @param file: file with unlabelled text
    """
    # find best #UNK# for later use
    unkTag = "O"
    unkP = 0
    for tag in emissions.keys():
        if emissions[tag]["#UNK#"] > unkP:
            unkTag = tag

    with open(file) as f, open("dev.p2.out", "w") as out:
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

                if bestTag == "":
                    bestTag = unkTag

                out.write("{} {}\n".format(word, bestTag))


# main
if len(sys.argv) != 3:
    print("Usage: python3 part2.py [train file] [test file]")

_, train, test = sys.argv
emissions = estEmissions(train)
predictSentiments(emissions, test)
