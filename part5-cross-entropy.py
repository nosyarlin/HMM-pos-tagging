from pathlib import Path
from math import exp, log
import matplotlib.pyplot as plt
import os
import copy


# Perceptron with tags sort by frequency

def predict(transitions, emissions, words, tokens, parent, word):
    """
    Given features about a word, return the most likely POS tag

    @param transitions: dict of transition scores
    @param emissions: dict of emission scores
    @return: Most likely tag
    """
    bestScore = 0
    bestTag = None
    scores = []

    if word not in words:
        word = "#UNK#"

    for tag in tokens:
        if tag not in transitions[parent]:
            continue

        if word not in emissions[tag]:
            continue

        score = transitions[parent][tag] + emissions[tag][word]
        scores.append(score)
        if score > bestScore or bestTag is None:
            bestScore = score
            bestTag = tag

    probability = exp(bestScore) / sum([exp(x) for x in scores])
    return bestTag, probability


def initParams(file):
    """
    Get all possible tokens and words
    and initialize transitions and emissions
    """
    tokens = set()
    words = {'#UNK#'}
    with open(file) as f:
        for line in f:
            temp = line.strip()

            # ignore empty lines
            if len(temp) == 0:
                continue

            last_space_index = temp.rfind(" ")
            x = temp[:last_space_index].lower()
            y = temp[last_space_index + 1:]

            tokens.add(y)
            words.add(x)

    transitions = {}
    emissions = {}
    for u in tokens.union({'_START'}):
        for v in tokens.union({'_STOP'}):
            if u not in transitions:
                transitions[u] = {v: 0}
            else:
                transitions[u][v] = 0

    for u in tokens:
        for word in words:
            if u not in emissions:
                emissions[u] = {word: 0}
            else:
                emissions[u][word] = 0

    tokens = {token: 0 for token in tokens}

    return transitions, emissions, words, tokens


def train(file, threshold):
    """
    Given training file, return transitions and emissions
    trained using perceptron algorithm
    """
    transitions, emissions, words, tokens = initParams(file)

    with open(file) as f:
        # sort tokens
        for line in f:
            temp = line.strip()

            if len(temp) == 0:
                continue

            last_space_index = temp.rfind(" ")
            y = temp[last_space_index + 1:]
            tokens[y] += 1
        f.seek(0)

        sortedTokens = list(tokens.keys())
        sortedTokens.sort(key=lambda x: tokens[x], reverse=True)

        # train weights
        prevCrossEntropy = 9999999999

        i = 0
        while True:
            prev = "_START"
            learnrate = 1 / (i + 1)
            currCrossEntropy = 0

            prevTransitions = copy.deepcopy(transitions)
            prevEmissions = copy.deepcopy(emissions)

            for line in f:
                temp = line.strip()

                # Sentence has ended
                if len(temp) == 0:
                    transitions[prev]['_STOP'] += 1
                    prev = '_START'
                    continue

                # Sentence has not ended
                else:
                    last_space_index = temp.rfind(" ")
                    x = temp[:last_space_index].lower()
                    y = temp[last_space_index + 1:]
                    tokens[y] += 1

                # Predict and update
                prediction, probability = predict(transitions, emissions, words,
                                                  sortedTokens, prev, x)
                currCrossEntropy -= log(probability)
                if prediction != y:
                    transitions[prev][y] += learnrate
                    emissions[y][x] += learnrate

                    transitions[prev][prediction] -= learnrate
                    emissions[prediction][x] -= learnrate

                prev = y

            print("Epoch {}, Cross Entropy: {}".format(i + 1, currCrossEntropy))
            if currCrossEntropy > prevCrossEntropy:
                transitions = prevTransitions
                emissions = prevEmissions
                break
            else:
                f.seek(0)
                i += 1
                prevCrossEntropy = currCrossEntropy

    return transitions, emissions, words, sortedTokens


def predictAll(trainFile, testFile, outputFile, threshold):
    """
    Given a file of sentences, predict POS tag sequences
    for each sentence using Viterbi Algorithm
    """
    transitions, emissions, words, tokens = train(trainFile, threshold)

    with open(testFile) as f,\
         open(outputFile, "w") as out:

        prev = "_START"
        for line in f:
            temp = line.strip()

            # Sentence has ended
            if len(temp) == 0:
                out.write("\n")
                prev = "_START"

            # Sentence has not ended
            else:
                word = temp.lower()

                # find most likely tag for word
                prediction, _ = predict(transitions, emissions, words, tokens, prev, word)
                out.write("{} {}\n".format(word, prediction))
                prev = prediction


# main
datasets = ["EN", "FR", "CN", "SG"]
for ds in datasets:
    datafolder = Path(ds)
    trainFile = datafolder / "train"
    testFile = datafolder / "dev.in"
    outputFile = datafolder / "dev.p9.out"

    threshold = 0.01

    predictAll(trainFile, testFile, outputFile, threshold)
    print("Output:", outputFile)

print("Done!")


# # Code for tuning number of epochs
# # main
# datasets = ["EN", "FR"]
# for ds in datasets:
#     datafolder = Path(ds)
#     trainFile = datafolder / "train"
#     testFile = datafolder / "dev.in"
#     outputFile = datafolder / "dev.p8.out"

#     entity_F = []
#     entity_type_F = []
#     scores = []
#     for i in range(1, 51):
#         predictAll(trainFile, testFile, outputFile, i)
#         results = os.popen("python3 evalResult.py ./{}/dev.out {}".format(ds, outputFile)).read()
#         results = results.split("\n")
#         score = float(results[7][-6:]) + float(results[-2][-6:])
#         scores.append(score)
#         entity_F.append(float(results[7][-6:]))
#         entity_type_F.append(float(results[-2][-6:]))

#     print(scores, entity_F, entity_type_F)
#     plt.plot(scores)
#     print("Output:", outputFile)

# plt.show(block=True)
# print("Done!")
