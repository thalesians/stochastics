import numpy as np

def problessthan(sample, bw, value, count, random_state):
    sample = sample.flatten()
    idxs = random_state.randint(0, len(sample), size=count)
    epsilons = random_state.normal(size=count)
    lessthanflags = sample[idxs] + bw * epsilons < value
    return float(np.sum(lessthanflags))/float(count)

def isoutlier(sample, bw, value, threshold, count, random_state):
    plt = problessthan(sample, bw, value, count, random_state)
    pgt = 1. - plt
    return plt <= threshold or pgt <= threshold

def _test():
    import doctest
    doctest.testmod(verbose=False)

if __name__ == '__main__':
    _test()
