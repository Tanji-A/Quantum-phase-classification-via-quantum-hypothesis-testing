from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np

class MyNdarray:
    def __init__(self, arr:np.ndarray):
        self.arr = arr
    # we can use A^B instead of np.kron(A,B)
    def __xor__(self, other):
        return MyNdarray(np.kron(self.arr, other.arr))
    def __add__(self, other):
        return MyNdarray(self.arr+other.arr)
    
class ExactQCNN:
    def __init__(self, n_qubits, operator=False):
        self.n_qubits = n_qubits
        self.operator = operator

    """
        ExactQCNN Z (Ferromagnetic)
    """
    # convolutional- and pooling- layer 
    def _Ucp_Z(self, depth, n_dim=None):
        if n_dim is None:
            assert 1<=depth and depth<=self.n_depth
            n_dim = 4 * 2**(self.n_depth-depth)
        else:
            assert 1<=n_dim and n_dim<=self.n_qubits
            depth = None

        qc = QuantumCircuit(n_dim, name='Ucp')
        # convolutional-layer
        for i in range(1, n_dim, 2):
            if i+1<n_dim:
                qc.cx(i, i+1)
            else:
                qc.cx(i, 0) # periodic
        for i in range(0, n_dim, 2):
            if i+2<n_dim:
                qc.ccx(i, i+2, i+1)
            elif i+2==n_dim:
                qc.ccx(i, 0, i+1) # periodic
            elif i+1==n_dim:
                qc.ccx(i, 1, 0) # periodic
        qc.barrier()
        
        # pooling-layer
        for i in range(0, n_dim, 2):
            if i+1<n_dim:
                qc.cz(i, i+1)
            else:
                qc.cz(i, 0) # periodic

        return qc
    
    def _create_qcnn_Z(self):
        qc = QuantumCircuit(self.n_qubits, name='qcnn')
        
        if self.mode == "efficient":
            # Ucpを作用させるqubitsのindex_list
            index_list = [[i for i in range(self.n_qubits)]]
            for depth in range(2, self.n_depth+1):
                index_list.append(index_list[-1][1::2])
            # Ucp
            for depth in range(1, self.n_depth+1):
                qc.append(self._Ucp_Z(depth).to_instruction(), index_list[depth-1])
                qc.barrier()
            # 縮約をとるqubits
            measurement_idx = index_list[-1][1::2]
            reduced_qubits = [x for x in range(self.n_qubits) if not x in measurement_idx]
            
        elif self.mode == "not_efficient":
            # Ucpを作用させるqubitsのindex_list
            index_list = [[i for i in range(self.n_qubits)]]
            while len(index_list[-1])>3: # 残りが2or3qubitsになるまで
                if len(index_list[-1])%2 == 1: # odd-qubitsなら1番上のqubitは無視
                    index_list[-1].pop(0)
                index_list.append(index_list[-1][1::2])
            # Ucp
            for indices in index_list[:-1]:
                qc.append(self._Ucp_Z(None, n_dim=len(indices)).to_instruction(), indices)
                qc.barrier()
            # 縮約をとるqubits
            measurement_idx = [index_list[-1][-2], index_list[-1][-1]]
            reduced_qubits = [x for x in range(self.n_qubits) if not x in measurement_idx]

        if self.operator:
            return Operator(qc.reverse_bits()).data, reduced_qubits
        else:
            return qc.reverse_bits(), reduced_qubits
        
    # n_qubits = 2**(depth+1): efficient
    # the others : not_efficient
    def prepare_Z(self):
        self.n_depth = round(np.log(self.n_qubits) / np.log(2)) - 1
        # mode
        if (2 ** (self.n_depth+1)) == self.n_qubits:
            self.mode = "efficient"
        else:
            self.mode = "not_efficient"

        # projectionを定義(0:{ZZ>0}, 1:{ZZ<0})
        zero = MyNdarray(np.array([[1,0],[0,0]]))
        one = MyNdarray(np.array([[0,0],[0,1]]))
        pro_zero = ((zero^zero)+(one^one)).arr
        pro_one = ((zero^one)+(one^zero)).arr
        self.projection_list = [pro_zero, pro_one]
        
        # ansatz, 縮約をとるqubits
        self.qc, self.reduced_qubits = self._create_qcnn_Z()


    """
        ExactQCNN SOP (SPT)
    """        
    # convolutional- and pooling- layer 
    def _Ucp_SOP(self, depth, n_dim=None):
        if n_dim is None:
            assert 1<=depth and depth<=self.n_depth
            n_dim = 9 * 3**(self.n_depth-depth)
        else:
            assert 1<=n_dim and n_dim<=self.n_qubits
            depth = None
        
        qc = QuantumCircuit(n_dim, name='Ucp')
        # convolutional-layer
        for i in range(0, n_dim-1, 2):
            qc.cz(i, i+1)
        for i in range(1, n_dim-1, 2):
            qc.cz(i, i+1)
        qc.cz(0, n_dim-1) # periodic
        qc.barrier()
        for i in range(1, n_dim-3, 6):
            qc.cz(i, i+3)
        for i in range(4, n_dim-3, 6):
            qc.cz(i, i+3)
        qc.cz(1, n_dim-2) # periodic
        qc.barrier()
        for i in range(1, n_dim-1, 3):
            qc.h([i-1, i+1])
            qc.ccx(i-1, i+1, i)
            qc.h([i-1, i+1])
        for i in range(2, n_dim-1, 3):
            qc.swap(i, i+1)
        qc.swap(0, n_dim-1) # periodic
        qc.barrier()
        
        # pooling-layer
        for i in range(0, n_dim-1, 3):
            qc.h(i)
            qc.cz(i, i+1)
        for i in range(2, n_dim,   3):
            qc.h(i)
            qc.cz(i, i-1)
                
        return qc

    def _create_qcnn_SOP(self):
        qc = QuantumCircuit(self.n_qubits, name='qcnn')
        
        if self.mode == "efficient":
            # Ucpを作用させるqubitsのindex_list
            index_list = [[i for i in range(self.n_qubits)]]
            for depth in range(2, self.n_depth+1):
                index_list.append(index_list[-1][1::3])
            # Ucp
            for depth in range(1, self.n_depth+1):
                qc.append(self._Ucp_SOP(depth).to_instruction(), index_list[depth-1])
                qc.barrier()
            # 縮約をとるqubits
            measurement_idx = index_list[-1][1::3]
            reduced_qubits = [x for x in range(self.n_qubits) if not x in measurement_idx]
        
        elif self.mode == "semi_efficient":
            # Ucpを作用させるqubitsのindex_list
            index_list = [[i for i in range(self.n_qubits)]]
            for depth in range(2, self.n_depth+1):
                index_list.append(index_list[-1][1::3])
            # Ucp
            for depth in range(1, self.n_depth+1):
                qc.append(self._Ucp_SOP(None, n_dim=len(index_list[depth-1])).to_instruction(), index_list[depth-1])
                qc.barrier()
            # 縮約をとるqubits
            measurement_idx = index_list[-1][1::3]
            measurement_idx = [measurement_idx[len(measurement_idx)//2+i] for i in (-1,0,1)]
            reduced_qubits = [x for x in range(self.n_qubits) if not x in measurement_idx]   

        if self.operator:
            return Operator(qc.reverse_bits()).data, reduced_qubits
        else:
            return qc.reverse_bits(), reduced_qubits

    # n_qubits = 3**(n_depth+1): efficient
    # n_qubits = 3**n_depth * m, m\in(4,5,6,7,8) : semi_efficient
    def prepare_SOP(self):
        self.n_depth = round(np.log(self.n_qubits) / np.log(3)) - 1
        # mode
        if (3 ** (self.n_depth+1)) == self.n_qubits:
            self.mode = "efficient"
        else:
            for i in range(4, 9):
                self.n_depth = round(np.log(self.n_qubits/i) / np.log(3))
                if (3 ** self.n_depth * i) == self.n_qubits:
                    self.mode = "semi_efficient"
                    break
                self.mode = None
            if self.mode is None:
                raise Exception("It has not supported yet.")

        # projectionを定義(0:{ZXZ>0}, 1:{ZXZ<0})
        zero = MyNdarray(np.array([[1,0],[0,0]]))
        one = MyNdarray(np.array([[0,0],[0,1]]))
        plus = MyNdarray(0.5*np.array([[1,1],[1,1]]))
        minus = MyNdarray(0.5*np.array([[1,-1],[-1,1]]))
        pro_zero = ((zero^plus^zero)+(zero^minus^one)+(one^minus^zero)+(one^plus^one)).arr
        pro_one = ((zero^plus^one)+(zero^minus^zero)+(one^plus^zero)+(one^minus^one)).arr
        self.projection_list = [pro_zero, pro_one]

        # ansatz, 縮約をとるqubits
        self.qc, self.reduced_qubits = self._create_qcnn_SOP()      


