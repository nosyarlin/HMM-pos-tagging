import sys
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
                b = log(emissions[curr][word])
            else:
                b = log(emissions[curr]["#UNK#"])

            for prev, prevPie in pies[i - 1].items():

                # Check if transition pair and prevPie exist
                if curr not in transitions[prev] or prevPie[0] is None:
                    continue

                a = log(transitions[prev][curr])
                tempPie = prevPie[0] + a + b

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
            tempPie = prevPie[0] + log(transitions[prev]["_STOP"])
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
if len(sys.argv) != 3:
    print("Usage: python3 part3.py [train file] [test file]")

_, train, test = sys.argv
emissions = estEmissions(train)
print(emissions)
transitions = estTransitions(train)
predictViterbiFile(emissions, transitions, test)
