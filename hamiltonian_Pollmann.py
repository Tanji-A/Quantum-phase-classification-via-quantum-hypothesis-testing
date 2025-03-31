import os
import time
import tqdm
import itertools
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg


def read_eigenvectors(file):
    """
    Takes a dataset and returns the J1J2 values that
    are associated for the eigenvector for each line
    :param file: str - file location
    :return: tuple of list & np.array
    """
    with open(file, 'r+') as f:
        text_data = f.readlines()

        h_vals = []
        for i in range(len(text_data)):
            J1J2, eigenvector = text_data[i].split("_")

            h_vals.append(tuple(map(float, J1J2[1: -1].split(', '))))
            text_data[i] = eigenvector

        return h_vals, np.loadtxt(text_data, dtype=complex)


def find_kron(array_li, index_li, q_bits):
    """

    :param array: sparse.dia_matrix - Tensor (X or Z)
    :param index: int - location of X or Z in identities
    :param q_bits: int - number of qbits
    :return:
    """
    order = np.ones(q_bits)
    for index in index_li:
        order[index-1] = 0  # Sets index as 0 to represent the array parameter given
        assert index <= q_bits  # n elements should always be larger than index for array

    idx = 0
    for i in range(1, len(order)):
        # Sets next element to Identity if next element is a 1, else array (Z or X)
        if order[i] == 0:
            current = array_li[idx]
            idx += 1
        else:
            current = II

        if i == 1:  # First time - compute kron(j-1, j)
            if order[i-1] == 0:
                t = array_li[idx]
                idx += 1
            else:
                t = II
        t = sparse.kron(t, current)

    return t


class Hamiltonian:
    def __init__(self, qbits=4, J1_metadata=(-4, 4), J2_metadata=(-4, 4), v=1):
        self.qbits = qbits
        self.verbose = v
        self.J1_min, self.J1_max = J1_metadata
        self.J2_min, self.J2_max = J2_metadata

        self.size = pow(2, self.qbits)

    def get_first_term(self):
        self.first_term = 0

        for i in range(self.qbits - 2):
            elem = i + 1  # math element is indexes at 1
            if self.verbose: print(f"first term {elem}/{self.qbits - 2}")
            s = time.time()
            self.first_term -= find_kron([Z,X,Z], [elem,elem+1,elem+2], self.qbits)
            print(time.time()-s)
            # s = time.time()
            # a = find_kron(Z, elem, self.qbits)
            # b = find_kron(X, elem + 1, self.qbits)
            # c = find_kron(Z, elem + 2, self.qbits)
            # self.first_term -= (a.dot(b).dot(c))
            # print(time.time()-s)
            print(type(self.first_term))
        self.first_term -= find_kron([Z,X,Z], [elem+1,elem+2,1], self.qbits)
        self.first_term -= find_kron([Z,X,Z], [elem+2,1,2], self.qbits)
        # a = find_kron(X, elem + 1, self.qbits)
        # b = find_kron(Z, elem + 2, self.qbits)
        # c = find_kron(X, 1, self.qbits) 
        # self.first_term -= (a.dot(b).dot(c)).toarray()
        # a = find_kron(X, elem + 2, self.qbits)
        # b = find_kron(Z, 1, self.qbits)
        # c = find_kron(X, 2, self.qbits) 
        # self.first_term -= (a.dot(b).dot(c)).toarray() # periodic

    def get_second_term(self):
        self.second_term = 0
        for i in range(self.qbits):
            elem = i + 1  # math element is indexes at 1
            if self.verbose: print(f"second term {elem}/{self.qbits}")
            self.second_term += find_kron([X], [elem], self.qbits)

    def get_third_term(self):
        self.third_term = 0
        for i in range(self.qbits - 1):  # This is actually 1 to N-2, python indexing has self.n-1
            elem = i + 1  # math element is indexes at 1
            if self.verbose: print(f"third term {elem}/{self.qbits-1}")
            
            self.third_term -= find_kron([Z,Z], [elem,elem+1], self.qbits)
        self.third_term -= find_kron([Z,Z], [elem+1,1], self.qbits)
        # b1 = find_kron(X, elem + 1, self.qbits)
        # b2 = find_kron(X, 1, self.qbits)
        # self.third_term -= (b1.dot(b2)).toarray() # periodic
        

    def generate_data(self, J1_range, J2_range, name, n_min=0, label=None):
        """
        Given a filename, and J1 + J2 ranges, calculate the three terms used to
        construct the hamiltonian in advance (unchanging between values of J1, J2)
        and start to iterate through J1 and J2, appending to a text file each time
        to avoid storing the huge dataset and saving once.

        :param J1_range: float - # of steps to get from self.J1_min to self.J1_max
        :param J2_range: float - # of steps to get from self.J2_min to self.J2_max
        :param filename: filename start appending outputs to in streaming mode
        :return:
        """
        t0 = time.time()
        self.get_first_term()
        self.get_second_term()
        self.get_third_term()
        print(f"{round(time.time() - t0, 4)}s elapsed to calculate term")

        # Delete the output file if exists so we can append to a fresh ones.
        if label == None:
            filename = f'dataset_Pollmann_n={self.qbits}_' + name + ".txt"
        else:
            filename = f'dataset_Pollmann_n={self.qbits}_' + name + f'_{label}' + ".txt"
        if os.path.isfile(filename): os.remove(filename)

        # Create a list of J1 and J2 values to loop over
        J1J2 = [[round(J1,3), round(J2,3)] for J1 in np.linspace(self.J1_min, self.J1_max, J1_range)
                for J2 in np.linspace(self.J2_min, self.J2_max, J2_range)]
        if name=="test1":
            J1J2 = []
            #for J2 in np.linspace(-0.5, 0.76, 10):
            for J2 in np.linspace(0.06, 0.76, 6):
                J1_li = np.linspace(J2+1-0.45, J2+1+0.45, 10)
                J1J2_li = [[round(J1,3), round(J2,2)] for J1 in J1_li]
                J1J2 += J1J2_li
        elif name=="test2":
            #J1_li = np.linspace(-0.9, 0.9, 10)
            J1_li = np.linspace(-0.5, 0.5, 6)
            J2_li = np.linspace(1-0.45, 1+0.45, 10)
            J1J2 = [[round(J1,1), round(J2,3)] for J1 in J1_li for J2 in J2_li]
        for J1, J2 in tqdm.tqdm(J1J2):

            if name == "train1": J2 = 0  # If in training mode, J2 should be 0!
            if name == "train2": J1 = 0

            h = (self.first_term * J2) + self.second_term + (self.third_term * J1)
            if n_min == 0:
                eigenvalue, eigenvector = self.find_eigval_with_sparse(h)
            else:
                eigenvalue, eigenvector = self.find_eigval_with_np2(h, n_min=n_min)

            # self.test_dataset(h, eigenvalue)  # SLOW! Compares np.eig with sparse.eig

            # Write to file each time to avoid saving to ram
            with open(filename, 'a+') as f:
                f.write(f"{J1, J2}_")  # Append J1, J2 for reference
                for line in eigenvector:
                    f.write(str(line) + " ")
                f.write("\n")

    @staticmethod
    def find_eigval_with_sparse(h):
        """
        Uses an approximation to find minimum eigenvalue and corresponding
        eigenvector, works well for sparse Hamiltonians (Valid for this class)
        :param h: np.array - 2D hamiltonian
        :return: np.ndarray, np.ndarray - Minimum EigVal and EigVec
        """
        b, c = sparse.linalg.eigsh(h, k=1, which='SA')
        return b, c.flatten()

    @staticmethod
    def find_eigval_with_np(h):
        """
        Uses the much slower way to find the minimized eigenvalue and corresponding
        eigenvector of a hamiltonian. MUCH slower and very inefficient for large H
        :param h: np.array - 2D hamiltonian
        :return: float64, np.array - EigVal and EigVec
        """
        ww, vv = np.linalg.eig(h)  # Old method with linalg
        index = np.where(ww == np.amin(ww))  # Find lowest eigenvalue
        np_eig_vec, np_eig_val = vv[:, index], ww[index]  # Use index to find lowest eigenvector

        """
        np.linalg.eig returns the eigenvalues and vectors of a matrix
        BUT, it returns a list of lists of lists, where the elements of
        each triple nested list is the first element of each eigenvector,
        not a list of eigenvectors like any sensical person would return.
        """  # np.linalg.eig is grade A stupid, change my mind...
        eig_vect_list = []
        for eigVal in range(len(np_eig_val)):
            temp_vec = []

            for eigVec in range(len(np_eig_vec)):
                temp_vec.append(np_eig_vec[eigVec][0][eigVal])
            eig_vect_list.append(np.array(temp_vec))

        sum_vec = np.sum(eig_vect_list, axis=0)
        return np_eig_val[0], sum_vec / np.linalg.norm(sum_vec)
    
    @staticmethod
    def find_eigval_with_np2(h, n_min):
        ww, vv = np.linalg.eigh(h)  # Old method with linalg
        index = np.where(ww == np.amin(ww))  # Find lowest eigenvalue
        ww_copy = np.copy(ww)
        if n_min == 'rand':
            n_min = np.random.randint(0, len(ww))
        for _ in range(n_min):
            ww_copy = np.delete(ww_copy, index)
            index = np.where(ww_copy == np.amin(ww_copy))
        index = np.where(ww == np.amin(ww_copy))
        np_eig_vec, np_eig_val = vv[:, index], ww[index]  # Use index to find lowest eigenvector

        eig_vect_list = []
        for eigVal in range(len(np_eig_val)):
            temp_vec = []

            for eigVec in range(len(np_eig_vec)):
                temp_vec.append(np_eig_vec[eigVec][0][eigVal])
            eig_vect_list.append(np.array(temp_vec))

        sum_vec = np.sum(eig_vect_list, axis=0)
        return np_eig_val[0], sum_vec / np.linalg.norm(sum_vec)

    def test_dataset(self, h, possible_eigenvalue):
        """
        Computes the explicit eigenvector for a given Hamiltonian
        to check if the eigenvalue is a valid eigenvalue
        :param h: np.array - Hamiltonian
        :param possible_eigenvalue: np.array
        :return: N/A
        """
        _, np_eig_vec = self.find_eigval_with_np(h)
        magnitude = (h @ np_eig_vec) / possible_eigenvalue

        assert np.allclose(magnitude, np.array(np_eig_vec, dtype=complex), 1e-9)

    # デバッグ用
    def get_hamilton(self, J1, J2):
        t0 = time.time()
        self.get_first_term()
        self.get_second_term()
        self.get_third_term()
        print(f"{round(time.time() - t0, 4)}s elapsed to calculate term") 

        h = (self.first_term * J2) + self.second_term + (self.third_term * J1)
        return h

# Define the various pauli operators required to compute the hamiltonian.
II = sparse.dia_matrix((np.ones(2), np.array([0])), dtype=int, shape=(2, 2))
Z = sparse.dia_matrix((np.array([1, -1]), np.array([0])), dtype=int, shape=(2, 2))
X = sparse.dia_matrix((np.array([np.ones(1)]), np.array([-1])), dtype=int, shape=(2, 2))
X.setdiag(np.ones(1), 1)

if __name__ == '__main__':
    s = time.time()

    # n represents the number of qbits in the system. The larger the value,
    # the more complicated and slower the calculations. Note that computation
    # scales by 2^n, so anything larger than 9 or 10 starts to become
    # exponentially long
    n = 15

    # Create the hamiltonian and generate both train and test data sets
    H = Hamiltonian(n, J1_metadata=(0.05, 1.95))

    # Train data only requires 40x1 resolution
    H.generate_data(20, 1, "train1")
    #H.generate_data(40, 1, "train", n_min=2, label='n_min=2')
    # Train2 data only requires 1x40 resolution
    #H.generate_data(1, 20, "train2")

    # Testing data is a 64x64 grid, as defined in the paper
    #H.generate_data(64, 64, "test1")

    # デバッグ
    #eigenvalue, eigenvector = np.linalg.eig(H.get_hamilton(0.5, 0.7))
    #eigenvalue, eigenvector = H.find_eigval_with_np2(H.get_hamilton(0.5, 0.7), n_min=3)
    #print(eigenvalue.shape, eigenvector.shape)
    #print(eigenvalue, eigenvector)

    print(f"Time for creating dataset was {time.time() - s} seconds")