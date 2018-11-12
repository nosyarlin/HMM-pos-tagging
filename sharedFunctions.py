def estEmissions(file, k=1):
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
            if len(temp) == 0:
                continue
            else:
                last_space_index = temp.rfind(" ")
                x = temp[:last_space_index].lower()
                y = temp[last_space_index + 1:]

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
        for x, xCount in xDict.items():
            xDict[x] = xCount / float(yCounts[y] + k)

        emissions[y]["#UNK#"] = k / float(yCounts[y] + k)

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
            temp = line.strip()

            # sentence has ended
            if len(temp) == 0:
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
                last_space_index = temp.rfind(" ")
                curr = temp[last_space_index + 1:]

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


def getDictionary(file):
    """
    Given training file, return set of all words

    @return Set: set of all words in file
    """
    out = set()
    with open(file) as f:
        for line in f:
            temp = line.strip()

            # ignore empty lines
            if len(temp) == 0:
                continue
            else:
                last_space_index = temp.rfind(" ")
                word = temp[:last_space_index].lower()
                out.add(word)

    return out
