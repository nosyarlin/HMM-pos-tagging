import sys
from sharedFunctions import estEmissions, estTransitions


def predictViterbiFile(emissions, transitions, file):
    """
    Predicts sentiments using the Viterbi algorithm
    Saves labelled file as dev.p3.out

    @param emissions: output from estEmissions function
    @param transitions: output from estTransitions function
    @param file: file with unlabelled text
    """
    out = open("dev.p3.out")
    with open(file) as f:
        sentence = []

        for line in f:
            # form sentence
            if line != "\n":
                word = line.strip()
                sentence.append(word)

            # predict tag sequence
            else:
                pass


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
    pies[0] = {"_START": [1.0, None]}

    # forward iterations
    for i in range(1, len(textList) + 1):
        word = textList[i - 1]
        for curr in tags:
            bestPie = 0
            parent = None

            # Check if word has been seen before
            if word in emissions[curr]:
                b = emissions[curr][word]
            else:
                b = emissions[curr]["#UNK#"]

            for prev, prevPie in pies[i - 1].items():

                # Check if transition pair exists
                if curr in transitions[prev]:
                    a = transitions[prev][curr]
                else:
                    a = 0

                # Calculate pie
                tempPie = prevPie[0] * a * b
                if tempPie > bestPie:
                    bestPie = tempPie
                    parent = prev

            # Update pies
            if i in pies:
                pies[i][curr] = [bestPie, parent]
            else:
                pies[i] = {curr: [bestPie, parent]}

    # stop case
    bestPie = 0
    parent = None

    for prev, prevPie in pies[len(textList)].items():
        tempPie = prevPie[0] * transitions[prev]["_STOP"]
        if tempPie > bestPie:
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
if len(sys.argv) != 3:
    print("Usage: python3 part3.py [train file] [test file]")

_, train, test = sys.argv
emissions = estEmissions(train)
transitions = estTransitions(train)

with open(test) as t:
    sentence = []
    for line in t:
        line = line.strip()
        if line != '':
            sentence.append(line)

print(predictViterbiList(emissions, transitions, sentence))
