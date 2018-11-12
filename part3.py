from pathlib import Path
from sharedFunctions import estEmissions, estTransitions
from math import log


def predictViterbiFile(emissions, transitions, file):
    """
    Predicts sentiments using the Viterbi algorithm
    Saves labelled file as dev.p3.out

    @param emissions: output from estEmissions function
    @param transitions: output from estTransitions function
    @param file: file with unlabelled text
    """
    with open(file) as f, open("dev.p3.out", "w") as out:
        sentence = []

        for line in f:
            # form sentence
            if line != "\n":
                word = line.strip()
                sentence.append(word)

            # predict tag sequence
            else:
                sequence = predictViterbiList(emissions, transitions, sentence)
                for i in range(len(sequence)):
                    out.write("{} {}\n".format(sentence[i], sequence[i]))
                out.write("\n")
                sentence = []


def predictViterbiList(emissions, transitions, textList):
    """
    Predicts sentiments for a list of words using the
    Viterbi algorithm

    @param emissions: output from estEmissions function
    @param transitions: output from estTransitions function
    @param textList: list of words

    @return: most probable y sequence for given textList as a list
    """
    # base case
    tags = emissions.keys()
    pies = {}
    pies[0] = {"_START": [0.0, None]}

    # forward iterations
    # Calculate log pie to combat underflow problem
    for i in range(1, len(textList) + 1):
        word = textList[i - 1]
        for curr in tags:
            bestPie = None
            parent = None

            # Check if word has been seen before
            if word in emissions[curr]:
                b = emissions[curr][word]
            else:
                b = emissions[curr]["#UNK#"]
            if b == 0.0:
                continue

            for prev, prevPie in pies[i - 1].items():

                # Check if transition pair and prevPie exist
                if curr not in transitions[prev] or \
                   prevPie[0] is None or \
                   transitions[prev][curr] == 0:
                    continue

                a = transitions[prev][curr]

                # Calculate pie
                tempPie = prevPie[0] + log(a) + log(b)

                if bestPie is None or tempPie > bestPie:
                    bestPie = tempPie
                    parent = prev

            # Update pies
            if i in pies:
                pies[i][curr] = [bestPie, parent]
            else:
                pies[i] = {curr: [bestPie, parent]}

    # stop case
    bestPie = None
    parent = None

    for prev, prevPie in pies[len(textList)].items():
        # Check prev can lead to a stop
        if "_STOP" in transitions[prev]:
            a = transitions[prev]["_STOP"]
            if a == 0 or prevPie[0] is None:
                continue

            tempPie = prevPie[0] + log(a)
            if bestPie is None or tempPie > bestPie:
                bestPie = tempPie
                parent = prev

    pies[len(textList) + 1] = {"_STOP": [bestPie, parent]}

    # backtracking to get sequence
    sequence = []
    curr = "_STOP"
    i = len(textList)
    while True:
        parent = pies[i + 1][curr][1]
        if parent == "_START":
            break
        else:
            sequence.append(parent)
            curr = parent
            i -= 1
    sequence.reverse()

    return sequence


# main
datasets = ["EN", "FR", "CN", "SG"]
for ds in datasets:
    datafolder = Path(ds)
    trainFile = datafolder / "train"
    testFile = datafolder / "dev.in"
    outputFile = datafolder / "dev.p3.out"

    emissions = estEmissions(trainFile)
    transitions = estTransitions(trainFile)
    predictViterbiFile(emissions, transitions, testFile)

    print("Output:", outputFile)

print("Done!")
