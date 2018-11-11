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
            temp = line.strip()

            # ignore empty lines
            if len(temp) == 0 :
                continue
            else:
                last_space_index = temp.rfind(" ")
                x = temp[:last_space_index]
                y = temp[last_space_index+1:]

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
                # word is too rare
                toRemove.append(x)
                unkCount += xCount

        # remove rare words and get emission of #UNK#
        for x in toRemove:
            xDict.pop(x)
        emissions[y]["#UNK#"] = unkCount / float(yCounts[y])

    return emissions  

def estTransitions(file):
    """
    Given training file, return transition parameters

    @return Dict: {y_prev: {y_curr: transition}}
    """
    start = "_START"
    stop = "_STOP"
    transitions = {}
    yCounts = {start: 0}
    prev = start
    with open(file) as f:
        for line in f:
            temp = line.strip().split(" ")

            # sentence has ended
            if len(temp) == 1:
                if prev in transitions:
                    if stop in transitions[prev]:
                        transitions[prev][stop] += 1
                    else:
                        transitions[prev][stop] = 1
                else:
                    transitions[prev] = {stop: 1}
                prev = start

            # part of a sentence
            else:
                curr = temp[1]

                # update count(start) if new sentence
                if prev == start:
                    yCounts[start] += 1

                # update count(y)
                if curr in yCounts:
                    yCounts[curr] += 1
                else:
                    yCounts[curr] = 1

                # update count(prev, curr)
                if prev in transitions:
                    if curr in transitions[prev]:
                        transitions[prev][curr] += 1
                    else:
                        transitions[prev][curr] = 1
                else:
                    transitions[prev] = {curr: 1}

                prev = curr

        # add count(prev, stop) if no blank lines at EOF
        if prev != start:
            if prev in transitions:
                if stop in transitions[prev]:
                    transitions[prev][stop] += 1
                else:
                    transitions[prev][stop] = 1
            else:
                transitions[prev] = {stop: 1}
            prev = start

    # convert counts to transitions
    for prev, currDict in transitions.items():
        for curr, currCount in currDict.items():
            currDict[curr] = currCount / float(yCounts[prev])

    return transitions
