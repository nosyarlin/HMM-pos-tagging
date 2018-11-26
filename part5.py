from pathlib import Path
import os


def predict(transitions, emissions, words, parent, word):
    """
    Given features about a word, return the most likely POS tag

    @param transitions: dict of transition scores
    @param emissions: dict of emission scores
    @return: Most likely tag
    """
    tags = list(emissions.keys())
    bestScore = 0
    bestTag = None

    if word not in words:
        word = "#UNK#"

    for tag in tags:
        if tag not in transitions[parent]:
            continue

        if word not in emissions[tag]:
            continue

        score = transitions[parent][tag] + emissions[tag][word]
        if score >= bestScore:
            bestScore = score
            bestTag = tag

    return bestTag


def initParams(file):
    """
    Get all possible tokens and words
    and initialize transitions and emissions
    """
    tokens = {'_START'}
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
    for u in tokens:
        for v in tokens:
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

    return transitions, emissions, words


def train(file, epoch):
    """
    Given training file, return transitions and emissions
    trained using perceptron algorithm
    """
    transitions, emissions, words = initParams(file)

    with open(file) as f:
        for i in range(epoch):
            prev = "_START"
            for line in f:
                temp = line.strip()

                # ignore empty lines
                if len(temp) == 0:
                    prev = "_START"
                    continue

                last_space_index = temp.rfind(" ")
                x = temp[:last_space_index].lower()
                y = temp[last_space_index + 1:]

                # Predict and update
                prediction = predict(transitions, emissions, words, prev, x)
                if prediction != y:
                    transitions[prev][y] += 1
                    emissions[y][x] += 1

                    transitions[prev][prediction] -= 1
                    emissions[prediction][x] -= 1

                prev = y

            f.seek(0)

    return transitions, emissions, words


def predictAll(trainFile, testFile, outputFile, epoch):
    transitions, emissions, words = train(trainFile, epoch)

    with open(testFile, encoding="utf-8") as f,\
         open(outputFile, "w", encoding="utf-8") as out:

        prev = "_START"
        for line in f:
            temp = line.strip()

            # ignore empty lines
            if len(temp) == 0:
                out.write("\n")

            else:
                word = temp.lower()

                # find most likely tag for word
                prediction = predict(transitions, emissions, words, prev, word)
                out.write("{} {}\n".format(word, prediction))


# main
datasets = ["FR"]
for ds in datasets:
    datafolder = Path(ds)
    trainFile = datafolder / "train"
    testFile = datafolder / "dev.in"
    outputFile = datafolder / "dev.p5.out"

    predictAll(trainFile, testFile, outputFile, 10)
    print("Output:", outputFile)

print("Done!")
