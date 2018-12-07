from pathlib import Path
from sharedFunctions import estEmissions


def predictSentiments(emissions, testfile, outputfile="dev.p2.out"):
    """
    Predicts sentiments using argmax(emission)
    If no outputfile given, saves labelled file as dev.p2.out

    @param emissions: output from estEmissions function
    @param testfile: input file with unlabelled text
    @param outputfile: name of file to save the output of labelled text
    """
    # find best #UNK# for later use
    unkTag = "O"
    unkP = 0
    for tag in emissions.keys():
        if emissions[tag]["#UNK#"] > unkP:
            unkTag = tag

    with open(testfile) as f, open(outputfile, "w") as out:
        for line in f:
            if line == "\n":
                out.write(line)
            else:
                word = line.strip().lower()

                # find most likely tag for word
                bestP = 0
                bestTag = ""
                for tag in emissions.keys():
                    if word in emissions[tag]:
                        if emissions[tag][word] > bestP:
                            bestP = emissions[tag][word]
                            bestTag = tag

                if bestTag == "":
                    bestTag = unkTag

                out.write("{} {}\n".format(word, bestTag))


# main
datasets = ["EN", "FR", "CN", "SG"]
for ds in datasets:
    datafolder = Path(ds)
    trainFile = datafolder / "train"
    testFile = datafolder / "dev.in"
    outputFile = datafolder / "dev.p2.out"

    emissions = estEmissions(trainFile)
    predictSentiments(emissions, testFile, outputFile)

    print("Output:", outputFile)

print("Done!")
