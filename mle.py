import sys


def estEmissions(file, k=3):
    """
    Given training file, return emission parameters

    @param k: Words appearing less than k times will be
    replaced with #UNK#

    @return Dict: {tag: {word: emission}}
    """
    emissions = {}
    yCounts = {}

    with open(file) as f:
        for line in f:
            temp = line.strip().split(" ")

            # ignore empty lines
            if len(temp) == 1:
                continue
            else:
                x = temp[0]
                y = temp[1]

                # update count(y)
                if y in yCounts:
                    yCounts[y] += 1
                else:
                    yCounts[y] = 1

                # update count(y->x)
                if y in emissions:
                    if x in emissions[y]:
                        emissions[y][x] += 1
                    else:
                        emissions[y][x] = 1
                else:
                    emissions[y] = {x: 1}

    # convert counts to emissions
    for y, xDict in emissions.items():
        unkCount = 0
        toRemove = []
        for x, xCount in xDict.items():
            if xCount >= k:
                xDict[x] = xCount / float(yCounts[y])
            else:
                # Word is too rare
                toRemove.append(x)
                unkCount += xCount

        # Remove rare words and get emission of #UNK#
        for x in toRemove:
            xDict.pop(x)
        emissions[y]["#UNK#"] = unkCount / float(yCounts[y])

    return emissions


def predictSentiments(emissions, file):
    """
    Predicts sentiments using argmax(emission)
    Saves labelled file as dev.p2.out

    @param emiisions: output from estEmissions function
    @param file: file with unlabelled text
    """
    out = open("dev.p2.out", "w")

    # Find best #UNK# for later use
    unkTag = "O"
    unkP = 0
    for tag in emissions.keys():
        if emissions[tag]["#UNK#"] > unkP:
            unkTag = tag

    with open(file) as f:
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

    out.close()


# Main
if len(sys.argv) != 3:
    print("Usage: python3 mle.py [train file] [test file]")

_, train, test = sys.argv
emissions = estEmissions(train)
predictSentiments(emissions, test)
