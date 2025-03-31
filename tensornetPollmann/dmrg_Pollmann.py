import tensornetwork as tn
from my_tensornetwork import FinitePollmann

import numpy as np
import time



class DMRGPollmann:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

        # train1
        read_data = self._dataset_Pollmann("train1")
        self.h1h2_i_train1 = read_data[0]
        self.label_i_train1 = read_data[1] 
        # train2
        read_data = self._dataset_Pollmann("train2")
        self.h1h2_i_train2 = read_data[0]
        self.label_i_train2 = read_data[1] 
        # test1
        read_data = self._dataset_Pollmann("test1")
        self.h1h2_i_test1 = read_data[0]
        self.label_i_test1 = read_data[1]
        # test2
        read_data = self._dataset_Pollmann("test2")
        self.h1h2_i_test2 = read_data[0]
        self.label_i_test2 = read_data[1]
        # other
        read_data = self._dataset_Pollmann("other")
        self.h1h2_i_other = read_data[0]
        self.label_i_other = read_data[1]

    def _dataset_Pollmann(self, mode="test1",
                          label_dict = {"trivial":0, "SPT": 1, "FM": 2, "AFM": 3}):
        # get J1J2_i
        if mode == "train1":
            J1_li = np.linspace(0.05, 1.95, 20)
            J1J2_i = [[round(J1,2), 0] for J1 in J1_li]
        elif mode == "train2":
            J2_li = np.linspace(0.05, 1.95, 20)
            J1J2_i = [[0, round(J2,2)] for J2 in J2_li]
        elif mode == "test1":
            J1J2_i = []
            for J2 in np.linspace(-0.5, 0.76, 10):
                J1_li = np.linspace(J2+1-0.45, J2+1+0.45, 10)
                J1J2_li = [[round(J1,3), round(J2,2)] for J1 in J1_li]
                J1J2_i += J1J2_li
        elif mode == "test2":
            J1_li = np.linspace(-0.9, 0.9, 10)
            J2_li = np.linspace(1-0.45, 1+0.45, 10)
            J1J2_i = [[round(J1,1), round(J2,3)] for J1 in J1_li for J2 in J2_li]
        elif mode == "other":
            J2 = -0.5
            J1_li = np.linspace(J2+1-0.725, J2+1+0.725, 30)
            J1J2_i = [[round(J1,3), round(J2,2)] for J1 in J1_li]
        # labeling phase
        label_i_str = []
        for J1, J2 in J1J2_i:
            if (J2>J1-1) and (J2>-J1-1) and (J2>1):
                label_i_str.append("SPT")
            elif (J2<J1-1) and (J2<-J1-1):
                label_i_str.append("SPT")
            elif (J2<=J1-1) and (J2>=-J1-1):
                label_i_str.append("FM")
            elif (J2>=J1-1) and (J2<=-J1-1):
                label_i_str.append("AFM")
            else:
                label_i_str.append("trivial")
        label_i = [label_dict[s] for s in label_i_str]

        return np.array(J1J2_i), np.array(label_i, dtype=int)
    
    def _select_testdata(self, s:int=5, label_i=None):
        """
            相境界の前後s個のtestdataだけを選ぶ
        """
        if label_i is None: label_i = self.label_i

        indices = np.where(np.diff(label_i) != 0)[0]
        bool_arr = np.full(len(label_i), False)
        assert indices[0]-(s-1)>=0 and indices[-1]+(s+1)<=len(label_i)
        assert (np.diff(indices)>=2*s).all()
        for idx in indices:
            bool_arr[idx-(s-1):idx+(s+1)] = True
        return bool_arr
    
    def run_dmrg(self, data="test1", max_bond=100, num_sweeps=4):
        if data=="train1":
            J1J2_i = self.h1h2_i_train1
        elif data=="train2":
            J1J2_i = self.h1h2_i_train2
        elif data=="test1":
            J1J2_i =self.h1h2_i_test1
        elif data=="test2":
            J1J2_i =self.h1h2_i_test2
        elif data=="other":
            J1J2_i =self.h1h2_i_other
        
        for idx, (J1, J2) in enumerate(J1J2_i):
            st = time.time()
            print(f"### J1={J1}, J2={J2} ###")
            s=time.time()
            mpo = FinitePollmann(np.full(self.n_qubits, J1), np.full(self.n_qubits, J2),
                            dtype=np.float64)
            print("prepare mpo:", time.time()-s)
            s=time.time()
            mps = tn.FiniteMPS.random([2]*self.n_qubits, [max_bond]*(self.n_qubits-1), dtype=np.float64)
            print("prepare mps:", time.time()-s)
            s=time.time()
            dmrg = tn.FiniteDMRG(mps, mpo)
            energy = dmrg.run_two_site(max_bond, num_sweeps)
            print("\ndmrg:", time.time()-s)
            print(f"dmrg energy is {energy}")
            s=time.time()
            np.savez_compressed(f"./results/dmrg_Pollmann_{data}/L={self.n_qubits}/L={self.n_qubits},J1={J1},J2={J2}.npz", *mps.tensors)
            print("save npz:", time.time()-s)
            print(f"idx is {idx}(/{len(J1J2_i)})", time.time()-st)

if __name__ == '__main__':
    ins = DMRGPollmann(27)
    ins.run_dmrg("test1", max_bond=200, num_sweeps=16)
    ins.run_dmrg("test2", max_bond=200, num_sweeps=16)
