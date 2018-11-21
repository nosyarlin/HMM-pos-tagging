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
    tags.append('_START')
    tags.append('_STOP')
    pies = {}
    c = {}
    # pies[i] = {X : [Probabiity, grandparent, parent]}
    pies[0] = {"_START": { "_None": 1.0}}
    pies[1] = {"_START": { "_START": 1.0}}
    pies[2] = {}

    for l in tags:
        if (isMissing(l,("_START","_START"),transitions)) or \
           (isMissing(textList[0],l,emissions)) :
            tempPie = 0
        else: 
            tempPie = transitions[("_START","_START")][l] * emissions[l][textList[0]]
        pies[2][l] = {"_START": tempPie}

    # forward iterations
    # Calculate log pie to combat underflow problem
    for i in range(3,len(textList)+2):
        word = textList[i - 2].lower()

        # Replace word with #UNK# if not in train
        if word not in dictionary:
            word = "#UNK#"
        for parent in tags:
            for child in tags:
                bestPie = 0.0
                l = None               

                for grandparent in tags:
                    #get probability of child given parent and grandparent 
                    if parent not in pies[i-1]:
                        continue
                    if grandparent not in pies[i-1][parent]:
                        continue

                    # Skip over words that can't come from currTag and if transition pair doesnt exist
                    if(isMissing(word,child,emissions) or \
                       isMissing(child,(grandparent,parent),transitions)):
                        continue

                    # Calculate pie
                    a = transitions[(grandparent,parent)][child]
                    b = emissions[child][word]
                    tempPie = pies[i-1][parent][grandparent] * a * b 
                    if tempPie > bestPie:
                        bestPie = tempPie
                        l = grandparent


                # Update pies
                if i in pies and child in pies[i]:
                    pies[i][child][parent] = bestPie
                elif i in pies:
                    pies[i][child] = {parent: bestPie}
                else:
                    pies[i] = {child: {parent: bestPie}}

                # Update c
                if i in c:
                    c[i][(parent, child)] = l
                else:
                    c[i] = {(parent, child): l}

    # stop case
    bestPie = 0.0
    l = None
    m = None
    for parent in tags:
        for grandparent in tags:
            if isMissing(grandparent, parent, pies[len(textList)+1]) or \
               isMissing(parent, len(textList)+1, pies):
                continue

            if pies[len(textList)+1][parent][grandparent] > bestPie:
                bestPie = pies[len(textList)+1][parent][grandparent]
                l = grandparent
                m = parent

    # backtracking to get sequence
    sequence = [m, l]
    i = len(textList) + 1

    while True:
        grandparent = c[i][(l, m)]
        if grandparent == "_START":
            break

        sequence.append(grandparent)
        m = l
        l = grandparent
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
