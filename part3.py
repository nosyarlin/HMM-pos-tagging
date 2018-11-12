from pathlib import Path
from sharedFunctions import estEmissions, estTransitions, getDictionary
from math import log


def predictViterbiFile(emissions, transitions, dictionary, file):
    """
    Predicts sentiments using the Viterbi algorithm
    Saves labelled file as dev.p3.out

    @param emissions: output from estEmissions function
    @param transitions: output from estTransitions function
    @param dictionary: output from getDictionary function
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
                sequence = predictViterbiList(emissions, transitions, dictionary, sentence)
                for i in range(len(sequence)):
                    out.write("{} {}\n".format(sentence[i], sequence[i]))
                out.write("\n")
                sentence = []


def predictViterbiList(emissions, transitions, dictionary, textList):
    """
    Predicts sentiments for a list of words using the
    Viterbi algorithm

    @param emissions: output from estEmissions function
    @param transitions: output from estTransitions function
    @param dictionary: output from getDictionary function
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
        word = textList[i - 1].lower()

        # Replace word with #UNK# if not in train
        if word not in dictionary:
            word = "#UNK#"

        for curr in tags:
            bestPie = None
            parent = None

            # Check if word can come from curr symbol
            if word not in emissions[curr] or \
               emissions[curr][word] == 0:
                continue

            b = emissions[curr][word]

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
            if i == 6:
                print(curr, bestPie, parent, word)
            if i in pies:
                pies[i][curr] = [bestPie, parent]
            else:
                pies[i] = {curr: [bestPie, parent]}

    print('\n\n\n')
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

    # print(emissions)
    # print("\n\n")
    # print(transitions)
    # print("\n\n")
    print(pies)
    print("\n\n")
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
    dictionary = getDictionary(trainFile)
    predictViterbiFile(emissions, transitions, dictionary, testFile)

    print("Output:", outputFile)

print("Done!")
