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
    tags = list(emissions.keys())
    tags.append('_START')
    tags.append('_STOP')
    d = {}
    c = {}

    # b[i] = {X: {parent: b_i(parent, X)}}
    d[0] = {"_START": {"_None": 1.0}}
    d[1] = {"_START": {"_START": 1.0}}
    d[2] = {}

    for tag in tags:
        if (isMissing(tag, ("_START", "_START"), transitions)) or \
           (isMissing(textList[0], tag, emissions)):
            tempPie = 0
        else:
            tempPie = transitions[("_START", "_START")][tag] * emissions[tag][textList[0]]
        d[2][tag] = {"_START": tempPie}

    # forward iterations
    # Calculate log pie to combat underflow problem
    for i in range(3, len(textList) + 2):
        word = textList[i - 2].lower()

        # Replace word with #UNK# if not in train
        if word not in dictionary:
            word = "#UNK#"
        for n in tags:
            for m in tags:
                bestPie = 0.0
                grandparent = None

                for l in tags:
                    # Skip if probability of child given parent and grandparent is 0
                    if isMissing(l, m, d[i - 1]):
                        continue

                    # Skip over words that can't come from n and if transition pair doesnt exist
                    if(isMissing(word, n, emissions) or
                       isMissing(n, (l, m), transitions)):
                        continue

                    # Calculate pie
                    a = transitions[(l, m)][n]
                    b = emissions[n][word]
                    tempPie = d[i - 1][m][l] * a * b
                    if tempPie > bestPie:
                        bestPie = tempPie
                        grandparent = l

                # Update d
                if i in d and n in d[i]:
                    d[i][n][m] = bestPie
                elif i in d:
                    d[i][n] = {m: bestPie}
                else:
                    d[i] = {n: {m: bestPie}}

                # Update c
                if i in c:
                    c[i][(m, n)] = grandparent
                else:
                    c[i] = {(m, n): grandparent}

    # stop case
    bestPie = 0.0
    grandparent = None
    parent = None
    for m in tags:
        for l in tags:
            if isMissing(l, m, d[len(textList) + 1]):
                continue

            if d[len(textList) + 1][m][l] > bestPie:
                bestPie = d[len(textList) + 1][m][l]
                grandparent = l
                parent = m

    # backtracking to get sequence
    sequence = [parent, grandparent]
    i = len(textList) + 1

    while True:
        l = c[i][(grandparent, parent)]
        if l == "_START":
            break

        sequence.append(l)
        parent = grandparent
        grandparent = l
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
