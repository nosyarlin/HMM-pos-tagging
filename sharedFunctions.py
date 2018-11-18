def incrementCount(parent, child, d):
    """
    Increment the count of [parent][child] in dictionary d
    """
    if parent in d:
        if child in d[parent]:
            d[parent][child] += 1
        else:
            d[parent][child] = 1
    else:
        d[parent] = {child: 1}


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
                incrementCount(y, x, emissions)

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
                incrementCount(prev, stop, transitions)
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
                incrementCount(prev, curr, transitions)

                prev = curr

        # add count(prev, stop) if no blank lines at EOF
        if prev != start:
            incrementCount(prev, stop, transitions)
            prev = start

    # convert counts to transitions
    for prev, currDict in transitions.items():
        for curr, currCount in currDict.items():
            currDict[curr] = currCount / float(yCounts[prev])

    return transitions


def estTransitions2(file):
    """
    Given training file, return transition parameters

    @return Dict: {(y_jm2,y_jm1): {y_j: transition}}
    """
    start = "_START"
    stop = "_STOP"
    transitions = {}
    yCounts = {(start, start): 0}
    y_jm2 = start
    y_jm1 = start
    with open(file) as f:
        for line in f:
            temp = line.strip()

            # sentence has ended
            if len(temp) == 0:
                incrementCount((y_jm2, y_jm1), stop, transitions)
                y_jm2 = start
                y_jm1 = start

            # part of a sentence
            else:
                last_space_index = temp.rfind(" ")
                y_j = temp[last_space_index + 1:]

                # update count(start) if new sentence
                if (y_jm2, y_jm1) == (start, start):
                    yCounts[(y_jm2, y_jm1)] += 1

                # update count(y)
                if (y_jm1, y_j) in yCounts:
                    yCounts[(y_jm1, y_j)] += 1
                else:
                    yCounts[(y_jm1, y_j)] = 1

                # update count(prev, curr)
                incrementCount((y_jm2, y_jm1), y_j, transitions)

                y_jm2 = y_jm1
                y_jm1 = y_j

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
