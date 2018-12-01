from pathlib import Path
import os
import matplotlib.pyplot as plt


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


def isMissing(child, parent, d):
    """
    Returns whether child is not related to parent in dictionary d

    @return: True if child not found under parent in d
    """
    return (parent not in d) or (child not in d[parent]) or (d[parent][child] is None)


def predictViterbi(transitions, emissions, dictionary, tags, sentence):
    """
    Predicts sentiments for a list of words using the
    Viterbi algorithm
    """
    # base case
    pies = {}
    pies[0] = {'_START': [0.0, None]}

    # forward iterations
    for i in range(1, len(sentence) + 1):
        word = sentence[i - 1].lower()

        # Replace word with #UNK# if not in train
        if word not in dictionary:
            word = "#UNK#"

        for currTag in tags:
            bestPie = None
            parent = None

            # Check that word can be emitted from currTag
            if isMissing(word, currTag, emissions):
                continue

            b = emissions[currTag][word]

            for prevTag, prevPie in pies[i - 1].items():

                # Check that currTag can transit from prevTag and prevPie exist
                if isMissing(currTag, prevTag, transitions) or \
                   prevPie[0] is None:
                    continue

                a = transitions[prevTag][currTag]

                # Calculate pie
                tempPie = prevPie[0] + a + b

                if bestPie is None or tempPie > bestPie:
                    bestPie = tempPie
                    parent = prevTag

            # Update pies
            if i in pies:
                pies[i][currTag] = [bestPie, parent]
            else:
                pies[i] = {currTag: [bestPie, parent]}

    # stop case
    bestPie = None
    parent = None

    for prevTag, prevPie in pies[len(sentence)].items():

        if prevPie[0] is None:
            continue

        # Check prev can lead to a stop
        if "_STOP" in transitions[prevTag]:
            a = transitions[prevTag]["_STOP"]

            tempPie = prevPie[0] + a
            if bestPie is None or tempPie > bestPie:
                bestPie = tempPie
                parent = prevTag

    pies[len(sentence) + 1] = {"_STOP": [bestPie, parent]}

    # backtracking to get sequence
    sequence = []
    curr = "_STOP"
    i = len(sentence)

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


def predictAll(trainFile, testFile, outputFile, epoch):
    """
    Given a file of sentences, predict POS tag sequences
    for each sentence using Viterbi Algorithm
    """
    transitions, emissions, words, tokens = train(trainFile, epoch)

    with open(testFile, encoding="utf-8") as f,\
         open(outputFile, "w", encoding="utf-8") as out:

        sentence = []

        for line in f:
            # form sentence
            if line != "\n":
                word = line.strip()
                sentence.append(word)

            # predict tag sequence
            else:
                sequence = predictViterbi(transitions, emissions, words, tokens, sentence)
                for i in range(len(sequence)):
                    out.write("{} {}\n".format(sentence[i], sequence[i]))
                out.write("\n")
                sentence = []


# main
datasets = ["EN", "FR"]
for ds in datasets:
    datafolder = Path(ds)
    trainFile = datafolder / "train"
    testFile = datafolder / "dev.in"
    outputFile = datafolder / "dev.p5.out"

    epochs = 6
    if ds == "FR":
        epochs = 1

    predictAll(trainFile, testFile, outputFile, epochs)
    print("Output:", outputFile)

print("Done!")


# # Code for tuning number of epochs
# # main
# datasets = ["EN", "FR"]
# for ds in datasets:
#     datafolder = Path(ds)
#     trainFile = datafolder / "train"
#     testFile = datafolder / "dev.in"
#     outputFile = datafolder / "dev.p6.out"

#     entity_F = []
#     entity_type_F = []
#     scores = []
#     for i in range(1, 30):
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
