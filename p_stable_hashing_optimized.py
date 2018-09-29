"""Experiment similarity based on locality sensitive hashing
1. Perform locality sensitive hashing on sampled subset of data (subspaces are sampled)
2. Objects that usually end up in the same bucket are more likely to be similar
3. Construct the similarity matrix by inspecting the hash tables

The technique of locality sensitive hashing is described here:
http://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p253-datar.pdf

https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134
https://medium.com/engineering-brainly/locality-sensitive-hashing-explained-304eb39291e4
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

NO_OF_TRIALS = 5
PRIME_1 = 31
PRIME_2 = 37
PRIME_3 = 41
PRIME_4 = 43
PRIME_5 = 47
PRIME_6 = 53
PRIME_7 = 59

orig_data = np.array(pd.read_csv('data/input_500x20.csv'))
no_row, no_col = orig_data.shape


def perform_hash(data, r, b, factor):
    """Perform the actual hashing of a group of points by simply applying the hash function floor((dot(a,v)+b)/r)
    http://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p253-datar.pdf
    """
    no_p, no_dimen = data.shape
    vector_a = np.random.uniform(0, 1, size=(no_dimen, 1)) * factor
    hash_buckets = np.floor((np.dot(data, vector_a) + b) / r)

    return hash_buckets


def sample_and_hash(data, r, factor, subspace_dimen, iteration=100, no_of_band=10, band_width=1):
    """Sample the dimensions (with replacement) to form subspace. Then perform hashing on the sampled data."""
    results = np.zeros(shape=(data.shape[0], iteration))
    distances = np.zeros(shape=(no_row, no_row))
    hash_all_iterations = np.array([])
    subspaces = []

    for i in range(iteration):
        # generate a random combination of dimensions to form the sampling subspace
        subspace = []
        while subspace == [] or subspace in subspaces:
            # subspace=sorted(np.random.randint(0,data.shape[1],size=subspace_dimen))
            subspace = sorted(np.random.choice(data.shape[1], 3, replace=False))

        subspaces.append(subspace)
        subspace_data = data[:, subspace]

        primes = [PRIME_1, PRIME_2, PRIME_3, PRIME_4, PRIME_5, PRIME_6, PRIME_6]
        bs = np.random.uniform(0, r, (no_of_bands, 1))
        hash_subspace = np.zeros(shape=(no_of_bands, data.shape[0]))
        for j in range(no_of_bands):
            hash_subspace[j, :] = np.transpose(perform_hash(subspace_data, r, bs[j], factor))
        # t=(sum(hash_subspace)*10+7)%31
        # results[:,i]=hash_subspace

        # if any pair of the hash values match between two points, they belong to the same bucket
        hash_of_this_subspace = np.zeros(shape=(1, no_row))
        # for m in range(no_row-1):
        #    for n in range(m+1,no_row):
        #        for k in range(0,15,3):
        #            if all(hash_subspace[k:k+3,m]==hash_subspace[k:k+3,n]):
        #                distances[m,n]+=1
        #                break

        # take the hash of the bands of values. Each band has 3 values
        band_hash = np.array([])

        for k in range(0, no_of_bands, band_width):
            if len(band_hash) == 0:
                band_hash = np.sum(
                    np.transpose(hash_subspace[k:k + band_width, :]) * np.repeat([primes[:band_width]], no_row, axis=0),
                    axis=1)
            else:
                hash_of_this_band = np.sum(
                    np.transpose(hash_subspace[k:k + band_width, :]) * np.repeat([primes[:band_width]], no_row, axis=0),
                    axis=1)
                band_hash = np.vstack((band_hash, hash_of_this_band))
        # hash each bucket of groups of 3 items each

        if len(hash_all_iterations) == 0:
            hash_all_iterations = band_hash
        else:
            hash_all_iterations = np.vstack((hash_all_iterations, band_hash))

    return hash_all_iterations, subspaces


def process_hash_table(hash_all_iterations, hash_batch_size=5):
    """aggregate the hash tables to see how frequently the items end up in the same buckets, subsequently build the
    similarity matrix"""

    # get the frequency of hash values of each hash
    freqs_of_hash = []
    distances = np.zeros(shape=(no_row, no_row))

    for i in range(hash_all_iterations.shape[0]):
        freqs_of_hash.append(Counter(hash_all_iterations[i]))

    # fill the distances matrix
    for i in range(no_row - 1):
        # get the weights (the frequency of the hash values in each hash)
        weights = []
        for w in range(len(freqs_of_hash)):
            weights.append(freqs_of_hash[w][hash_all_iterations[w, i]])

        for j in range(i + 1, no_row):
            distances[i, j] += sum((hash_all_iterations[:, i] == hash_all_iterations[:, j]) * weights)

    return distances


no_of_bands = 15
band_width = 3
r = 500
iteration = 500

for i in range(NO_OF_TRIALS):
    hash_all_iterations, subspaces = sample_and_hash(orig_data, r, 100, 3, iteration=iteration, no_of_band=no_of_bands,
                                                     band_width=band_width)
    distances = process_hash_table(hash_all_iterations)
    # distances=cdist(results,results,metric='cityblock')

    plt.figure(i)
    plt.imshow(distances, cmap='gray')
    plt.title(f"r={r} band_width={band_width} no_of_bands={no_of_bands} iteration={iteration}")
    print("DONE")

plt.show()
