from collections import Counter, OrderedDict

def calculate_sampler_weights(dataset):
    targets = [i for i in dataset.targets]
    class_counts = Counter(targets)
    counts_ord = OrderedDict(sorted(class_counts.items()))
    length = 0
    for value in counts_ord.values():
        length += int(value)
    assert length == len(dataset)
    # calcuate weight of each class
    class_weights = [1.0/n for n in counts_ord.values()]
    # assign weight to each sample
    sample_weights = [class_weights[n] for n in dataset.targets]

    return sample_weights
