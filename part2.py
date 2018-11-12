from pathlib import Path
from sharedFunctions import estEmissions


def predictSentiments(emissions, testfile, outputfile="dev.p2.out"):
    """
    Predicts sentiments using argmax(emission)
    Saves labelled file as dev.p2.out

    @param emissions: output from estEmissions function
    @param testfile: input file with unlabelled text
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
                word = line.strip()

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
runAll = input("Want to run for all datasets EN, FR, CN, SG? (y/n)\n")
if (runAll.strip()).lower() == "y":
    datasets = ["EN", "FR", "CN", "SG"]
    for ds in datasets:
        datafolder = Path(ds)
        trainfile = datafolder / "train"
        testfile = datafolder / "dev.in"
        outputfile = datafolder / "dev.p2.out"
        emissions = estEmissions(trainfile)
        predictSentiments(emissions, testfile, outputfile)
        print("Output:", outputfile)
    print("Done!")

else:
    trainfile = input("Please give me the file path for the training set: \n")
    testfile = input("Please give me the file path for the testing set: \n")
    print("Training:", trainfile, "\nTesting:", testfile)
    emissions = estEmissions(trainfile)
    predictSentiments(emissions, testfile)
    print("Output: dev.p2.out \nDone!")
