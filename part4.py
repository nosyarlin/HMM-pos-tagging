# -*- coding: utf-8 -*-
from pathlib import Path
from math import log
from sharedFunctions import estEmissions, estTransitions2, getDictionary


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
    with open(inputFile, encoding="utf-8") as f, open(outputFile, "w", encoding="utf-8") as out:
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
    return (parent not in d) or (child not in d[parent]) or (d[parent][child] is None)


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
    d[0] = {"_START": {"_None": 0.0}}
    d[1] = {"_START": {"_START": 0.0}}

    # forward iterations
    # Calculate log pie to combat underflow problem
    for i in range(2, len(textList) + 2):
        word = textList[i - 2].lower()

        # Replace word with #UNK# if not in train
        if word not in dictionary:
            word = "#UNK#"
        for n in tags:

            # Skip if emission is 0
            if isMissing(word, n, emissions):
                continue
            b = emissions[n][word]

            for m in tags:
                bestPie = None
                grandparent = None

                for l in tags:
                    # Skip if probability of child given parent and grandparent is 0
                    if isMissing(l, m, d[i - 1]):
                        continue

                    # Skip if transition pair doesnt exist
                    if isMissing(n, (l, m), transitions):
                        continue

                    # Calculate pie
                    a = transitions[(l, m)][n]
                    tempPie = d[i - 1][m][l] + log(a) + log(b)
                    if bestPie is None or tempPie > bestPie:
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
    bestPie = None
    grandparent = None
    parent = None
    for m in tags:
        for l in tags:
            if (isMissing(l, m, d[len(textList) + 1]) or
               d[len(textList) + 1][m][l] is None):
                continue

            if bestPie is None or d[len(textList) + 1][m][l] > bestPie:
                bestPie = d[len(textList) + 1][m][l]
                grandparent = l
                parent = m

    # backtracking to get sequence
    i = len(textList) + 1
    if parent is None:
        parent = list(d[i].keys())[0]
    if grandparent is None:
        grandparent = list(d[i][parent].keys())[0]
    sequence = [parent, grandparent]

    while True:
        l = c[i][(grandparent, parent)]
        if l is None:
            l = list(d[i - 1][grandparent].keys())[0]

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
