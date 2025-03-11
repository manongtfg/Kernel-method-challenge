import itertools
import numpy as np
from collections import Counter
from itertools import product


class LinearKernel():
    
    def _init__(self):
        """
        Initialize the linear kernel.
        """

    def compute_kernel_matrix(self, X):

        if isinstance(X, np.ndarray):
            X_np = X  
        else:
            X_np = X.to_numpy()  

        # dot product between each vector             
        cross_term = X_np @ X_np.T  

        K = cross_term

        return K
        
    
    def compute_kernel_vector(self, X1, x2):

        if isinstance(X1, np.ndarray):
            X1_np = X1
            X2_np = x2
        else:
            X1_np = X1.to_numpy() 
            X2_np = x2.to_numpy()

        cross_term = X1_np @ X2_np.T 

        K = cross_term

        return K
    
class GaussianKernel():
    
    def __init__(self, sigma):
        """
        Initialize the Gaussian kernel with the value sigma.
        Params
        ------
        sigma: value of the variance in the exponential 
        """
        self.sigma = sigma

    def compute_kernel_matrix(self, X):
        
        X_np = X.to_numpy() 

        # Computation of the square euclidean distance between each pair (i, j)
        X_sq = np.sum(X_np**2, axis=1)[:, np.newaxis]  

        # dot product between each vector
        cross_term = X_np @ X_np.T  

        dist_sq = X_sq + X_sq - 2 * cross_term 

        K = np.exp(-dist_sq / (2 * self.sigma ** 2))

        return K
    
    def compute_kernel_vector(self, X1, x2):

        n1, _ = X1.shape
        K = np.zeros((n1, 1))

        return np.exp(-np.linalg.norm((X1-x2), axis=1) /(2 * self.sigma ** 2))

class SpectrumKernel():

    def __init__(self, k, alphabet="ACGT"):
        """
        Initialize the Spectrum Kernel with k-mer length k and a given alphabet.
        Params
        ------
        k: Length of k-mers
        alphabet: Set of valid characters in the sequences
        """
        self.k = k
        self.alphabet = alphabet
        self.kmers = ["".join(p) for p in product(alphabet, repeat=k)]  # Generate all possible k-mers
        self.kmer_to_index = {kmer: i for i, kmer in enumerate(self.kmers)}  # Map k-mers to indices
        self.features_vectors = None # List of different strings existing to optimize 

    def extract_kmer_counts_test(self, sequence):
        """
        Convert a sequence into a k-mer frequency vector.
        Params
        ------
        sequence: Input sequence (string)

        Returns 
        -------
        A numpy array representing k-mer frequencies
        """

        counts = Counter([sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)])
        feature_vector = np.zeros(len(self.kmers))

        for kmer, count in counts.items():
            if kmer in self.kmer_to_index: 

                feature_vector[self.kmer_to_index[kmer]] = count
        
        return feature_vector


    def extract_kmer_counts(self, seq):
        """ Count the number os subsequences for each subsequence in the sequence seq
        Params
        ------
        seq: Input sequence (string)
        """
        kmers = [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]
        return Counter(kmers)

    def compute_kernel_matrix(self, sequences):
        """ Compute the kernel matrix K with the sequences in order to fit the model """

        seqs = sequences.loc[:, 'seq'].values
        n = len(seqs)

        # Creation of a dictionnary with all possible kmers
        kmer_set = set()
        kmer_vectors = []
        self.features_vectors = list()

        for seq in seqs:

            kmer_counts = self.extract_kmer_counts(seq)
            kmer_counts_test = self.extract_kmer_counts_test(seq)
            self.features_vectors.append(kmer_counts_test)

            kmer_set.update(kmer_counts.keys())
            kmer_vectors.append(kmer_counts)

        self.features_vectors = np.array(self.features_vectors)
        # Mapping k-mers with vector 
        kmer_list = list(kmer_set)
        kmer_dict = {kmer: i for i, kmer in enumerate(kmer_list)}
        d = len(kmer_list)  

        # Construction of characteristic matrix 
        X = np.zeros((n, d))
        for i, kmer_count in enumerate(kmer_vectors):
            for kmer, count in kmer_count.items():
                X[i, kmer_dict[kmer]] = count

        # kernel computation 
        K = X @ X.T

        return K

    def compute_kernel_vector(self, train_sequences, test_sequence):
        """
        Compute the spectrum kernel vector between a test sequence and a set of training sequences.
        Params
        ------
        train_sequences: List of training sequences
        test_sequence: A single test sequence

        Returns
        -------
        A numpy array containing kernel values
        """

        train_features = self.features_vectors # Already calculated during the computation of the kernel matrix 

        test_features = self.extract_kmer_counts_test(test_sequence)

        return train_features @ test_features  # Compute kernel values




