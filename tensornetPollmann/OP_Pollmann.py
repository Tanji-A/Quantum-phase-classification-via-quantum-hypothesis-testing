import tensornetwork as tn
from my_tensornetwork import MyFiniteMPS, FinitePollmann

import numpy as np
import json
import os
import re
from multiprocessing import Pool
from tqdm import tqdm
import time



class OPPollmann:
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
    
    def ferromagnetic(self, data="test1"):
        if data=="train1":
            J1J2_i = self.h1h2_i_train1
        elif data=="train2":
            J1J2_i = self.h1h2_i_train1
        elif data=="test1":
            J1J2_i =self.h1h2_i_test1
        elif data=="test2":
            J1J2_i =self.h1h2_i_test2
        elif data=="other":
            J1J2_i =self.h1h2_i_other

        folder_dmrg = "./results"
        ave_Z_li = [None for _ in range(len(J1J2_i))]
        for npz_path in tqdm(os.listdir(folder_dmrg+f"/dmrg_Pollmann_{data}/L={self.n_qubits}"), leave=False):
            match = re.search(r'J1=(-?\d+\.?\d*e?-?\d*),J2=(-?\d+\.?\d*e?-?\d*)', npz_path)
            J1J2 = [float(match.group(1)), float(match.group(2))]
            match = np.isclose(J1J2_i,J1J2)
            match = np.where(match[:,0]&match[:,1])[0]
            if match.shape[0] == 0:
                continue
            else:
                pass
            mps = MyFiniteMPS(list(np.load(folder_dmrg+f"/dmrg_Pollmann_{data}/L={self.n_qubits}/{npz_path}").values()))
            Z = np.array([[1,0],[0,-1]], dtype=np.float64) # OPにcomplexが無いので,mpsもfloatで良い
            ave_Z = 1/self.n_qubits * sum(mps.measure_local_operator([Z for _ in range(self.n_qubits)], list(range(self.n_qubits))))
            ave_Z_li[match[0]] = ave_Z
        with open(f"./OP(L={self.n_qubits},{data})_ave_Z.json", "w") as f:
            json.dump({"J1J2":np.array(J1J2_i).tolist(), "ave_Z":ave_Z_li}, f, indent=2)

    def SOP(self, data="test2"):
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

        folder_dmrg = "./results"
        ave_SOP_li = [None for _ in range(len(J1J2_i))]
        for npz_path in tqdm(os.listdir(folder_dmrg+f"/dmrg_Pollmann_{data}/L={self.n_qubits}"), leave=False):
            match = re.search(r'J1=(-?\d+\.?\d*e?-?\d*),J2=(-?\d+\.?\d*e?-?\d*)', npz_path)
            J1J2 = [float(match.group(1)), float(match.group(2))]
            match = np.isclose(J1J2_i,J1J2)
            match = np.where(match[:,0]&match[:,1])[0]
            if match.shape[0] == 0:
                continue
            else:
                pass
            mps = MyFiniteMPS(list(np.load(folder_dmrg+f"/dmrg_Pollmann_{data}/L={self.n_qubits}/{npz_path}").values()))
            X = np.array([[0,1],[1,0]], dtype=np.float64) # OPにcomplexが無いので,mpsもfloatで良い
            Z = np.array([[1,0],[0,-1]], dtype=np.float64)
            mps_nodes = [tn.Node(tensor) for tensor in mps.tensors]
            mps_conj_nodes = [tn.Node(tn.conj(tensor)) for tensor in mps.tensors]
            for node1, node2 in zip(mps_nodes[:-1], mps_nodes[1:]):
                node1[2] ^ node2[0]
            mps_nodes[-1][2] ^ mps_nodes[0][0]
            for node1, node2 in zip(mps_conj_nodes[:-1], mps_conj_nodes[1:]):
                node1[2] ^ node2[0]
            mps_conj_nodes[-1][2] ^ mps_conj_nodes[0][0]
            SOP_nodes = []
            SOP_nodes.append(tn.Node(Z))
            SOP_nodes[-1][0] ^ mps_nodes[0][1]; SOP_nodes[-1][1] ^ mps_conj_nodes[0][1]
            for i in range(1, self.n_qubits-1):
                if i%2 == 0:
                    mps_nodes[i][1] ^ mps_conj_nodes[i][1]
                else:
                    SOP_nodes.append(tn.Node(X))
                    SOP_nodes[-1][0] ^ mps_nodes[i][1]; SOP_nodes[-1][1] ^ mps_conj_nodes[i][1]
            SOP_nodes.append(tn.Node(Z))
            SOP_nodes[-1][0] ^ mps_nodes[-1][1]; SOP_nodes[-1][1] ^ mps_conj_nodes[-1][1]
            ave_SOP = tn.contractors.auto(mps_nodes+mps_conj_nodes+SOP_nodes).tensor.item()
            ave_SOP_li[match[0]] = ave_SOP
        with open(f"./OP(L={self.n_qubits},{data})_ave_SOP.json", "w") as f:
            json.dump({"J1J2":np.array(J1J2_i).tolist(), "ave_SOP":ave_SOP_li}, f, indent=2)

if __name__ == '__main__':
    ins = OPPollmann(27)
    ins.ferromagnetic("test1")
    ins.SOP("test2")
