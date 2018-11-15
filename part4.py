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
    return (child not in d[parent]) or (d[parent][child] == 0)


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
    initialise = False
    # forward iterations
    # Calculate log pie to combat underflow problem
    for i in range(1, len(textList) + 1):
        word = textList[i - 1].lower()

        # Replace word with #UNK# if not in train
        if word not in dictionary:
            word = "#UNK#"

        for currTag in tags:
            bestPie = None
            parent = None

            # Skip over words that can't come from currTag
            if isMissing(word, currTag, emissions):
                continue

            b = emissions[currTag][word]
            
            i = len(pies)
            
            for prevTag, prevPie in pies[i - 1].items():
                #Check if prevTag is _Start, add pair
                if(prevTag == "_START"):
                    initialise = True
                    continue

                # Check if transition pair and prevPie exist
                elif isMissing(currTag, prevTag, transitions) or \
                   prevPie[0] is None:
                    continue

                
                a = transitions[prevTag][currTag]

                # Calculate pie
                tempPie = prevPie[0] + log(a) + log(b)

                if bestPie is None or tempPie > bestPie:
                    bestPie = tempPie
                    parent = prevTag

            if (initialise):
                # Update pies
                newTag = ('_START',currTag)
                jm1 = currTag
                pies[0] = {newTag:[0.0, None]}
                initialise = False
            else:
                # Update pies
                newTag = (jm1,currTag)
                jm1 = currTag
                if i in pies:
                    pies[i][newTag] = [bestPie, parent]
                else:
                    pies[i] = {newTag: [bestPie, parent]}

    # stop case
    bestPie = None
    parent = None

    for prevTag, prevPie in pies[len(textList)-1].items():
        # Check prev can lead to a stop
        if "_STOP" in transitions[prevTag]:
            a = transitions[prevTag]["_STOP"]
            if a == 0 or prevPie[0] is None:
                continue

            tempPie = prevPie[0] + log(a)
            if bestPie is None or tempPie > bestPie:
                bestPie = tempPie
                parent = prevTag

    newTag = (jm1,"_STOP")
    pies[len(textList)] = {newTag: [bestPie, parent]}

    # backtracking to get sequence
    sequence = []
    curr = newTag
    i = len(textList)

    while True:
        parent = pies[i][curr][1]
        if parent is None:
            parent = list(pies[i-1].keys())[0]

        if "_START" in parent:
            break

        sequence.append(parent)
        curr = parent
        i -= 1
    sequence.reverse()

    return sequence


# main
# datasets = ["EN", "FR", "CN", "SG"]
# for ds in datasets:
#     datafolder = Path(ds)
#     trainFile = datafolder / "train"
#     testFile = datafolder / "dev.in"
#     outputFile = datafolder / "dev.p3.out"

#     emissions = estEmissions(trainFile)
#     transitions = estTransitions2(trainFile)
#     dictionary = getDictionary(trainFile)
#     predictViterbiFile(emissions, transitions, dictionary, testFile, outputFile)

#     print("Output:", outputFile)

# print("Done!")
trainFile = "train"
testFile = "test"
outputFile = "test.p4.out"

emissions = estEmissions(trainFile)
transitions = estTransitions2(trainFile)
dictionary = getDictionary(trainFile)
predictViterbiFile(emissions, transitions, dictionary, testFile, outputFile)

print("Output:", outputFile)

print("Done!")