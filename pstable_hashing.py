"""Initial implementation of LSH

The technique of locality sensitive hashing is described here:
http://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p253-datar.pdf

https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134
https://medium.com/engineering-brainly/locality-sensitive-hashing-explained-304eb39291e4

"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

orig_data = np.array(pd.read_csv('data/input_500x20.csv'))
no_row, no_col = orig_data.shape

PRIME_1 = 1069
PRIME_2 = 1091
PRIME_3 = 1109


def perform_hash(data, r, b, factor):
    """Perform the actual hashing of a group of points by simply applying the hash function floor((dot(a,v)+b)/r)
    http://www.cs.princeton.edu/courses/archive/spring05/cos598E/bib/p253-datar.pdf
    """

    no_p, no_dimen = data.shape
    vector_a = np.random.uniform(0, 1, size=(no_dimen, 1)) * factor
    hash_buckets = np.floor((np.dot(data, vector_a) + b) / r)

    return hash_buckets


def sample_and_hash(data, r, factor, subspace_dimen, iteration=100):
    """Sample the dimensions (with replacement) to form subspace. Then perform hashing on the sampled data."""

    results = np.zeros(shape=(data.shape[0], iteration))
    distances = np.zeros(shape=(no_row, no_row))
    subspaces = []

    for i in range(iteration):
        print(i)
        # generate a random combination of dimensions to form the sampling subspace
        subspace = []
        while subspace == [] or subspace in subspaces:
            # subspace=sorted(np.random.randint(0,data.shape[1],size=subspace_dimen))
            subspace = sorted(np.random.choice(data.shape[1], 3, replace=False))

        subspaces.append(subspace)
        subspace_data = data[:, subspace]

        bs = np.random.uniform(0, r, (15, 1))
        hash_subspace = np.zeros(shape=(15, data.shape[0]))
        for j in range(15):
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
        for k in range(0, 15, 3):
            if len(band_hash) == 0:
                band_hash = np.sum(
                    np.transpose(hash_subspace[k:k + 3, :]) * np.repeat([[PRIME_1, PRIME_2, PRIME_3]], no_row, axis=0),
                    axis=1)
            else:
                hash_of_this_band = np.sum(
                    np.transpose(hash_subspace[k:k + 3, :]) * np.repeat([[PRIME_1, PRIME_2, PRIME_3]], no_row, axis=0),
                    axis=1)
                band_hash = np.vstack((band_hash, hash_of_this_band))
        # hash each bucket of groups of 3 items each

        freqs_of_hash = []
        for k in range(band_hash.shape[0]):
            freqs_of_hash.append(Counter(band_hash[k, :]))

        # hashes_of_points=np.transpose(band_hash)
        # distances+=cdist(hashes_of_points,hashes_of_points,lambda u,v:sum(u==v))
        for m in range(no_row - 1):
            for n in range(m + 1, no_row):

                weights = []
                for w in range(len(freqs_of_hash)):
                    weights.append(freqs_of_hash[w][band_hash[w, m]])

                distances[m, n] += sum((band_hash[:, m] == band_hash[:, n]) * weights)

    return results, distances, subspaces


results, distances, subspaces = sample_and_hash(orig_data, 997, 100, 3, iteration=250)
# distances=cdist(results,results,metric='cityblock')

fi = plt.figure()
plt.imshow(distances, cmap='gray', interpolation='nearest')
plt.show()
print(results)
