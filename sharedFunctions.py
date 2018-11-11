def estEmissions(file, k=3):
    """
    Given training file, return emission parameters

    @param k: Words appearing less than k times will be
    replaced with #UNK#

    @return Dict: {tag: {word: emission}}
    """
    emissions = {}
    yCounts = {}

    with open(file) as f:
        for line in f:
            temp = line.strip().split(" ")

            # ignore empty lines
            if len(temp) == 1:
                continue
            else:
                x = temp[0]
                y = temp[1]

                # update count(y)
                if y in yCounts:
                    yCounts[y] += 1
                else:
                    yCounts[y] = 1

                # update count(y->x)
                if y in emissions:
                    if x in emissions[y]:
                        emissions[y][x] += 1
                    else:
                        emissions[y][x] = 1
                else:
                    emissions[y] = {x: 1}

    # convert counts to emissions
    for y, xDict in emissions.items():
        unkCount = 0
        toRemove = []
        for x, xCount in xDict.items():
            if xCount >= k:
                xDict[x] = xCount / float(yCounts[y])
            else:
                # Word is too rare
                toRemove.append(x)
                unkCount += xCount

        # Remove rare words and get emission of #UNK#
        for x in toRemove:
            xDict.pop(x)
        emissions[y]["#UNK#"] = unkCount / float(yCounts[y])

    return emissions


def estTransitions(file):
    """
    Given training file, return transition parameters

    @param k: Words appearing less than k times will be
    replaced with #UNK#

    @return Dict: {y_prev: {y_curr: transition}}
    """
    pass
