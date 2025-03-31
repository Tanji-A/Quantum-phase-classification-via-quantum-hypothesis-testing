from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_statevector, partial_trace
import numpy as np

from my_tensornetwork import MyFiniteMPS


class MyNdarray:
    def __init__(self, arr:np.ndarray):
        self.arr = arr
    # we can use A^B instead of np.kron(A,B)
    def __xor__(self, other):
        return MyNdarray(np.kron(self.arr, other.arr))
    def __add__(self, other):
        return MyNdarray(self.arr+other.arr)

class ExactQcnnMPS:
    def __init__(self, n_qubits:int,
                 max_singular_values:int=None,
                 max_truncation_err:int=None,
                 max_truncation_err2:int=None): # n_qubits = 3**(n_depth+1)
        self.n_qubits = n_qubits
        self.n_depth = round(np.log(n_qubits) / np.log(3)) - 1
        if (3 ** (self.n_depth+1)) == n_qubits:
            assert (3 ** (self.n_depth+1)) == n_qubits
        else:
            self.n_depth = round(np.log(n_qubits) / np.log(2)) - 1
            if (2 ** (self.n_depth+1)) == n_qubits:
                assert (2 ** (self.n_depth+1)) == n_qubits
            else:
                print("Warning: ExactQcnnMPS's n_qubits")


        # projectionを定義(0:{ZXZ>0}, 1:{ZXZ<0})
        zero = MyNdarray(np.array([[1,0],[0,0]]))
        one = MyNdarray(np.array([[0,0],[0,1]]))
        plus = MyNdarray(0.5*np.array([[1,1],[1,1]]))
        minus = MyNdarray(0.5*np.array([[1,-1],[-1,1]]))
        pro_zero = ((zero^plus^zero)+(zero^minus^one)+(one^minus^zero)+(one^plus^one)).arr
        pro_one = ((zero^plus^one)+(zero^minus^zero)+(one^plus^zero)+(one^minus^one)).arr
        self.projection_list = [pro_zero, pro_one]
        # 量子ゲートをndarrayで定義
        self._def_gates()
        # 縮約をとるqubits
        meas_qubits = [self.n_qubits//2-1,self.n_qubits//2,self.n_qubits//2+1]
        self.reduced_qubits = [i for i in range(self.n_qubits) if i not in meas_qubits]

        # 最大の特異値数と切り捨て誤差
        self.max_singular_values = max_singular_values
        self.max_truncation_err = max_truncation_err
        self.max_truncation_err2 = max_truncation_err2

    def _def_gates(self):
        # I,X,Y,Z
        self.I = np.eye(2)
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.Z = np.diag([1,-1])
        # hadamard
        self.H = 1/(2**0.5) * np.array([[1,1],[1,-1]])
        # controlled-X
        self.CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        # controlled-Z
        self.CZ = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
        # toffoli (target:center, control:Z)
        qc = QuantumCircuit(3)
        qc.ccx(0,2,1)
        self.CXC_z = Operator(qc.reverse_bits()).data.real
        # toffoli (target:center, control:X)
        qc = QuantumCircuit(3)
        qc.h([0,2]); qc.ccx(0,2,1); qc.h([0,2])
        self.CXC_x = Operator(qc.reverse_bits()).data.real
        # swap
        self.swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

    def meas_Z(self, mps:MyFiniteMPS)->float:
        # gate to tensor
        CX = self.CX.reshape(2,2,2,2)
        CZ = self.CZ.reshape(2,2,2,2)
        swap = self.swap.reshape(2,2,2,2)
        CXC = self.CXC_z.reshape(2,2,2,2,2,2)
        # poolingで捨てられていない残りのqubits
        remained_qubits = list(range(self.n_qubits))
        # # 2**nのqubitsにする
        # i = 0
        # while 2**(i+1) <= self.n_qubits:
        #     i += 1
        # del_qubits = (self.n_qubits-2**i)//2
        # del remained_qubits[:del_qubits]
        # del_qubits = self.n_qubits-2**i-del_qubits
        # del remained_qubits[:-(del_qubits+1):-1]
        if len(remained_qubits)%2 == 1: # odd-qubitsなら1番上のqubitは無視
            remained_qubits.pop(0)
        #print("start_qubits", remained_qubits)
        # 残りが2or3qubitsになるまでpoolingする
        while len(remained_qubits)>3:
            # first conv layer
            for i in remained_qubits[1:-1:2]:
                mps.position(i,False)
                #print("CX", i,i+1)
                mps.apply_two_site_gate(CX, i,i+1,
                                        self.max_singular_values,self.max_truncation_err,
                                        i+1)
            mps.position(remained_qubits[-1],False)
            #print("CX", remained_qubits[-1],remained_qubits[0])
            mps.apply_distant_two_site_gate(CX.swapaxes(0,1).swapaxes(2,3), remained_qubits[0],remained_qubits[-1],
                                            self.max_singular_values,self.max_truncation_err,
                                            remained_qubits[-1]) # periodic
            # second conv layer
            for i in remained_qubits[0:-2:2]:
                mps.position(i,False)
                #print("CXC", i,i+1,i+2)
                mps.apply_three_site_gate(CXC, i,i+1,i+2,
                                          self.max_singular_values,self.max_truncation_err,
                                          i+2)
            mps.position(remained_qubits[0],False)
            for i in remained_qubits[:-1]:
                #print("swap", i,i+1)
                mps.apply_two_site_gate(swap, i,i+1,
                                        self.max_singular_values,self.max_truncation_err,
                                        i+1)
            #print("CXC", i-1,i,i+1)
            mps.apply_three_site_gate(CXC, i-1,i,i+1,
                                      self.max_singular_values,self.max_truncation_err,
                                      i+1) # periodic
            for i in remained_qubits[-2::-1]:
                #print("swap", i,i+1)
                mps.apply_two_site_gate(swap, i,i+1,
                                        self.max_singular_values,self.max_truncation_err,
                                        i)  
            # first pool layer
            for i in remained_qubits[:-1:2]:
                mps.position(i,False)
                #print("CZ", i,i+1)
                mps.apply_two_site_gate(CZ, i,i+1,
                                        self.max_singular_values,self.max_truncation_err,
                                        i+1)
            # update remained_qubits
            remained_qubits = remained_qubits[1::2]
            if len(remained_qubits)%2 == 1:
                remained_qubits.pop(0)
            #print("remained_qubits", remained_qubits)
            # gather in the center
            center_qubit = remained_qubits[len(remained_qubits)//2-1]
            idx = 1
            for i in reversed(remained_qubits[:len(remained_qubits)//2-1]):
                mps.position(i,False)
                #print("swap", i,center_qubit-idx)
                mps.apply_distant_two_site_gate(swap, i,center_qubit-idx,
                                                self.max_singular_values,self.max_truncation_err,
                                                i)
                idx += 1
            idx = 1
            for i in remained_qubits[len(remained_qubits)//2:]:
                mps.position(i,False)
                #print("swap", center_qubit+idx,i)
                mps.apply_distant_two_site_gate(swap, center_qubit+idx,i,
                                                self.max_singular_values,self.max_truncation_err,
                                                i)
                idx += 1
            # update gathered remained_qubits
            remained_qubits = (list(range(center_qubit-len(remained_qubits[:len(remained_qubits)//2-1]), center_qubit)) 
                               + list(range(center_qubit, center_qubit+1+len(remained_qubits[len(remained_qubits)//2:]))))
            #print("remained_qubits", remained_qubits)
        # normalization
        mps.canonicalize()
        # meas
        op = np.kron(self.Z,self.Z)
        expectation = mps.measure_two_site_observable(op.reshape(2,2,2,2), remained_qubits[-2],remained_qubits[-1])
        return expectation

    def meas_SOP(self, mps:MyFiniteMPS)->float:
        # gate to tensor
        H = self.H
        CZ = self.CZ.reshape(2,2,2,2)
        swap = self.swap.reshape(2,2,2,2)
        CXC = self.CXC_x.reshape(2,2,2,2,2,2)
        # poolingで捨てられていない残りのqubits
        remained_qubits = list(range(self.n_qubits))
        # n_depth回だけpoolingする
        for depth in reversed(range(1,self.n_depth+1)):
            # first conv layer
            for i in remained_qubits[:-1:2]:
                mps.position(i,False)
                try:
                    #print("CZ", i,i+1)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_two_site_gate(CZ, i,i+1,
                                            self.max_singular_values,self.max_truncation_err,
                                            i+1)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_two_site_gate(CZ, i,i+1,
                                            self.max_singular_values,self.max_truncation_err2,
                                            i+1)                
            # second conv layer
            for i in remained_qubits[:0:-2]:
                mps.position(i,False)
                try:
                    #print("CZ", i-1,i)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_two_site_gate(CZ, i-1,i,
                                            self.max_singular_values,self.max_truncation_err,
                                            i-1)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_two_site_gate(CZ, i-1,i,
                                            self.max_singular_values,self.max_truncation_err2,
                                            i-1)      
            # mps.position(remained_qubits[0],False)
            # try:
            #     #print("CZ", remained_qubits[0],remained_qubits[-1])
            #     mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
            #     mps.apply_distant_two_site_gate(CZ, remained_qubits[0],remained_qubits[-1],
            #                                     self.max_singular_values,self.max_truncation_err,
            #                                     remained_qubits[0]) # periodic
            # except np.linalg.LinAlgError:
            #     print("LinAlgError!")
            #     mps = mps_copy
            #     mps.apply_distant_two_site_gate(CZ, remained_qubits[0],remained_qubits[-1],
            #                                     self.max_singular_values,self.max_truncation_err2,
            #                                     remained_qubits[0]) # periodic                     
            # third conv layer
            for i in remained_qubits[1:-3:6]:
                mps.position(i,False)
                try:
                    #print("CZ", i,i+3)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_distant_two_site_gate(CZ, i,i+3,
                                                    self.max_singular_values,self.max_truncation_err,
                                                    i)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_distant_two_site_gate(CZ, i,i+3,
                                                    self.max_singular_values,self.max_truncation_err2,
                                                    i)           
            # fourth conv layer
            for i in remained_qubits[-2:3:-6]:
                mps.position(i,False)
                try:
                    #print("CZ", i-3,i)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_distant_two_site_gate(CZ, i-3,i,
                                                    self.max_singular_values,self.max_truncation_err,
                                                    i)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_distant_two_site_gate(CZ, i-3,i,
                                                    self.max_singular_values,self.max_truncation_err2,
                                                    i)                
            # mps.position(remained_qubits[1],False)
            # try:
            #     #print("CZ", remained_qubits[1],remained_qubits[-2])
            #     mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
            #     mps.apply_distant_two_site_gate(CZ, remained_qubits[1],remained_qubits[-2],
            #                                     self.max_singular_values,self.max_truncation_err,
            #                                     remained_qubits[1]) # periodic
            # except np.linalg.LinAlgError:
            #     print("LinAlgError!")
            #     mps = mps_copy
            #     mps.apply_distant_two_site_gate(CZ, remained_qubits[1],remained_qubits[-2],
            #                                     self.max_singular_values,self.max_truncation_err2,
            #                                     remained_qubits[1]) # periodic         
            # fifth conv layer
            for i in remained_qubits[1:-1:3]:
                mps.position(i,False)
                try:
                    #print("CXC", i-1,i,i+1)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_three_site_gate(CXC, i-1,i,i+1,
                                            self.max_singular_values,self.max_truncation_err,
                                            i+1)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_three_site_gate(CXC, i-1,i,i+1,
                                            self.max_singular_values,self.max_truncation_err2,
                                            i+1) 
            # sixth conv layer
            for i in remained_qubits[-3:0:-3]:
                mps.position(i,False)
                try:
                    #print("swap", i-1,i)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_two_site_gate(swap, i-1,i,
                                            self.max_singular_values,self.max_truncation_err,
                                            i-1)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_two_site_gate(swap, i-1,i,
                                            self.max_singular_values,self.max_truncation_err2,
                                            i-1)              
            # mps.position(remained_qubits[0],False)
            # try:
            #     #print("swap", remained_qubits[0],remained_qubits[-1])
            #     mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
            #     mps.apply_distant_two_site_gate(swap, remained_qubits[0],remained_qubits[-1],
            #                                     self.max_singular_values,self.max_truncation_err,
            #                                     remained_qubits[0]) # periodic
            # except np.linalg.LinAlgError:    
            #     print("LinAlgError!")
            #     mps = mps_copy
            #     mps.apply_distant_two_site_gate(swap, remained_qubits[0],remained_qubits[-1],
            #                                     self.max_singular_values,self.max_truncation_err2,
            #                                     remained_qubits[0]) # periodic
            # first pool layer
            for i in remained_qubits[:-1:3]:
                mps.position(i,False)
                mps.apply_one_site_gate(H, i)
                #print("H", i)
                try:
                    #print("CZ", i,i+1)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_two_site_gate(CZ, i,i+1,
                                            self.max_singular_values,self.max_truncation_err,
                                            i+1)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_two_site_gate(CZ, i,i+1,
                                            self.max_singular_values,self.max_truncation_err2,
                                            i+1)
            # second pool layer
            for i in remained_qubits[:0:-3]:
                mps.position(i,False)
                mps.apply_one_site_gate(H, i)
                #print("H", i)
                try:
                    #print("CZ", i-1,i)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_two_site_gate(CZ, i-1,i,
                                            self.max_singular_values,self.max_truncation_err,
                                            i-1)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_two_site_gate(CZ, i-1,i,
                                            self.max_singular_values,self.max_truncation_err2,
                                            i-1)
            # update remained_qubits
            remained_qubits = remained_qubits[1:-1:3]
            #print("remained_qubits", remained_qubits)
            # gather in the center
            center_qubit = remained_qubits[len(remained_qubits)//2]
            idx = 1
            for i in reversed(remained_qubits[:len(remained_qubits)//2]):
                mps.position(i,False)
                try:
                    #print("swap", i,center_qubit-idx)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_distant_two_site_gate(swap, i,center_qubit-idx,
                                                    self.max_singular_values,self.max_truncation_err,
                                                    i)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_distant_two_site_gate(swap, i,center_qubit-idx,
                                                    self.max_singular_values,self.max_truncation_err2,
                                                    i)
                idx += 1
            idx = 1
            for i in remained_qubits[len(remained_qubits)//2+1:]:
                mps.position(i,False)
                try:
                    #print("swap", center_qubit+idx,i)
                    mps_copy = MyFiniteMPS(mps.tensors, mps.center_position, True, mps.backend)
                    mps.apply_distant_two_site_gate(swap, center_qubit+idx,i,
                                                    self.max_singular_values,self.max_truncation_err,
                                                    i)
                except np.linalg.LinAlgError:
                    print("LinAlgError!")
                    mps = mps_copy
                    mps.apply_distant_two_site_gate(swap, center_qubit+idx,i,
                                                    self.max_singular_values,self.max_truncation_err2,
                                                    i)
                idx += 1
            # update gathered remained_qubits
            remained_qubits = (list(range(center_qubit-len(remained_qubits[:len(remained_qubits)//2]), center_qubit)) 
                               + list(range(center_qubit, center_qubit+1+len(remained_qubits[len(remained_qubits)//2+1:]))))
            #print("remained_qubits", remained_qubits)
        # normalization
        mps.canonicalize()
        # meas
        op = np.kron(np.kron(self.Z,self.X),self.Z)
        expectation = mps.measure_three_site_observable(op.reshape(2,2,2,2,2,2), *remained_qubits)
        return expectation

        
        
if __name__=='__main__':
    from exactqcnn import ExactQCNN
    N = 9
    state = random_statevector(2**N).data
    mps = MyFiniteMPS.from_statevector(state)
    ins1 = ExactQcnnMPS(N)
    print(ins1.meas_SOP(mps))
    ins2 = ExactQCNN(N, operator=True); ins2.prepare_SOP()
    print(np.trace(partial_trace(ins2.qc@state, [0,2,3,5,6,8])@np.kron(np.kron(ins1.Z,ins1.X),ins1.Z)))

    N = 8
    state = random_statevector(2**N).data
    mps = MyFiniteMPS.from_statevector(state)
    ins1 = ExactQcnnMPS(N)
    print(ins1.meas_Z(mps))
    ins2 = ExactQCNN(N, operator=True); ins2.prepare_Z()
    print(np.trace(partial_trace(ins2.qc@state, [0,1,2,4,5,6])@np.kron(ins1.Z,ins1.Z)))
