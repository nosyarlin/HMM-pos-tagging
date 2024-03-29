from pathlib import Path
import matplotlib.pyplot as plt
import os


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

    if word not in words:
        word = "#UNK#"

    for tag in tokens:
        if tag not in transitions[parent]:
            continue

        if word not in emissions[tag]:
            continue

        score = transitions[parent][tag] + emissions[tag][word]
        if score > bestScore or bestTag is None:
            bestScore = score
            bestTag = tag

    return bestTag


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


def train(file, epoch):
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
        for i in range(epoch):
            prev = "_START"
            learnrate = 1 / (i + 1)
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
                prediction = predict(transitions, emissions, words,
                                     sortedTokens, prev, x)
                if prediction != y:
                    transitions[prev][y] += learnrate
                    emissions[y][x] += learnrate

                    transitions[prev][prediction] -= learnrate
                    emissions[prediction][x] -= learnrate

                prev = y

            f.seek(0)

    return transitions, emissions, words, sortedTokens


def predictAll(trainFile, testFile, outputFile, epoch):
    """
    Given a file of sentences, predict POS tag sequences
    for each sentence using Viterbi Algorithm
    """
    transitions, emissions, words, tokens = train(trainFile, epoch)

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
                prediction = predict(transitions, emissions, words, tokens, prev, word)
                out.write("{} {}\n".format(word, prediction))
                prev = prediction


# main
folder = input("Which language do you wish to use? (EN/FR/CN/SG) \n")
dataset = input("dev, test or test2? \n")

if folder.upper() not in ["EN", "FR", "CN", "SG"]:
    print("Please choose either EN, FR, CN or SG")

elif dataset.lower() not in ["dev", "test", "test2"]:
    print("Please choose either dev, test or test2")

else:
    # Prepare file paths
    datafolder = Path(folder.upper())
    trainFile = datafolder / "train"
    testFile = datafolder / "{}.in".format(dataset.lower())
    outputFile = datafolder / "{}.p5.out".format(dataset.lower())

    # Train and predict
    epochs = 20
    if folder == "FR":
        epochs = 23

    predictAll(trainFile, testFile, outputFile, epochs)

    print("Output:", outputFile)
    print("Done!")
