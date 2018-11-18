from pathlib import Path
from sharedFunctions import estEmissions, estTransitions2, getDictionary
from math import log


def predictViterbiFile(emissions, transitions, dictionary, inputFile, outputFile):
    """
    Predicts sentiments using the Viterbi algorithm
    If not outputFile given, saves labelled file as dev.p3.out

    @param emissions: output from estEmissions function
    @param transitions: output from estTransitions function
    @param dictionary: output from getDictionary function
    @param inputFile: name of file with unlabelled text
    @param outputFile: name of file to save output of unlabelled text to
    """
    with open(inputFile) as f, open(outputFile, "w") as out:
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


def isMissing(child, parent, d):
    """
    Returns whether child is not related to parent in dictionary d

    @return: True if child not found under parent in d
    """
    return (parent not in d) or (child not in d[parent]) or (d[parent][child] == 0)


def predictViterbiList(emissions, transitions, dictionary, textList):
    """
    Predicts sentiments for a list of words using the
    Viterbi algorithm

    @param emissions: output from estEmissions function
    @param transitions: output from estTransitions2 function
    @param dictionary: output from getDictionary function
    @param textList: list of words

    @return: most probable y sequence for given textList as a list
    """
    # base case
    tags = emissions.keys()
    pies = {}
    pies[0] = {"_START": [0.0, None, None]}
    pies[1] = {"_START": [0.0, None, None]}

    # forward iterations
    # Calculate log pie to combat underflow problem
    for i in range(2, len(textList) + 2):
        word = textList[i - 2].lower()

        # Replace word with #UNK# if not in train
        if word not in dictionary:
            word = "#UNK#"

        for currTag in tags:
            bestPie = None
            parent_jm1 = None
            parent_jm2 = None

            # Skip over words that can't come from currTag
            if isMissing(word, currTag, emissions):
                continue

            b = emissions[currTag][word]

            for y_jm2, jm2_Pie in pies[i - 2].items():

                for y_jm1, jm1_Pie in pies[i - 1].items():

                    # Check if transition pair and prevPie exist
                    if isMissing(currTag, (y_jm2, y_jm1), transitions) or \
                       jm1_Pie[0] is None:
                        continue

                    a = transitions[(y_jm2, y_jm1)][currTag]

                    # Calculate pie
                    tempPie = jm1_Pie[0] + log(a) + log(b)

                    if bestPie is None or tempPie > bestPie:
                        bestPie = tempPie
                        parent_jm1 = y_jm1
                        parent_jm2 = y_jm2

            # Update pies
            if i in pies:
                pies[i][currTag] = [bestPie, parent_jm1, parent_jm2]
            else:
                pies[i] = {currTag: [bestPie, parent_jm1, parent_jm2]}

    # stop case
    bestPie = None
    parent_jm1 = None
    parent_jm2 = None

    for y_jm2, jm2_Pie in pies[len(textList)].items():

        for y_jm1, jm1_Pie in pies[len(textList) + 1].items():

            # Check prev can lead to a stop
            if isMissing("_STOP", (y_jm2, y_jm1), transitions) or \
               jm1_Pie[0] is None:
                continue

            a = transitions[(y_jm2, y_jm1)]["_STOP"]
            tempPie = jm1_Pie[0] + log(a)

            if bestPie is None or tempPie > bestPie:
                parent_jm1 = y_jm1
                parent_jm2 = y_jm2

    pies[len(textList) + 2] = {"_STOP": [bestPie, parent_jm1, parent_jm2]}

    # backtracking to get sequence
    sequence = []
    curr = "_STOP"
    i = len(textList) + 1

    while True:
        parent = pies[i + 1][curr][1]
        if parent is None:
            parent = list(pies[i].keys())[0]

        if parent == "_START":
            break

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
    outputFile = datafolder / "dev.p4.out"

    emissions = estEmissions(trainFile)
    transitions = estTransitions2(trainFile)
    dictionary = getDictionary(trainFile)
    predictViterbiFile(emissions, transitions, dictionary, testFile, outputFile)

    print("Output:", outputFile)

print("Done!")

# trainFile = "train"
# testFile = "test"
# outputFile = "test.p4.out"

# emissions = estEmissions(trainFile)
# transitions = estTransitions2(trainFile)
# dictionary = getDictionary(trainFile)
# predictViterbiFile(emissions, transitions, dictionary, testFile, outputFile)

# print("Output:", outputFile)
# print("Done!")
