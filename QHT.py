from qiskit.quantum_info import Statevector, partial_trace
from exactqcnn import ExactQCNN
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import math
import os
import itertools
import json
import time
from tqdm import tqdm


class QHT:
    def __init__(self, qcnn, exactqcnn:ExactQCNN, fname_json:str):
        self.qcnn = qcnn
        self.exactqcnn = exactqcnn
        self.fname_json = fname_json
        self.n_qubits = self.exactqcnn.n_qubits

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
            #for J2 in np.linspace(-0.5, 0.76, 10):
            for J2 in np.linspace(0.06, 0.76, 6):
                J1_li = np.linspace(J2+1-0.45, J2+1+0.45, 10)
                J1J2_li = [[round(J1,3), round(J2,2)] for J1 in J1_li]
                J1J2_i += J1J2_li
        elif mode == "test2":
            #J1_li = np.linspace(-0.9, 0.9, 10)
            J1_li = np.linspace(-0.5, 0.5, 6)
            J2_li = np.linspace(1-0.45, 1+0.45, 10)
            J1J2_i = [[round(J1,1), round(J2,3)] for J1 in J1_li for J2 in J2_li]
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
    
    def _read_eigenvectors(self, file, save_path='./dataset/'):
        with open(save_path+file, 'r') as f:
            textdata = f.readlines()

            h_vals = []
            for i in range(len(textdata)):
                h1h2, eigenvector = textdata[i].split("_")

                h_vals.append(tuple(map(float, h1h2[1: -1].split(', '))))
                textdata[i] = eigenvector

        return np.array(h_vals), np.loadtxt(textdata, dtype=float)

    def _get_SOP(self, n_qubits=None):
        if n_qubits is None: n_qubits = self.n_qubits

        I = np.eye(2,dtype=int)
        X = np.array([[0,1],[1,0]])
        Z = np.array([[1,0],[0,-1]])
        
        Sab = Z
        Sab_string = ['Z']
        for i in range(n_qubits-2):
            if i%2 == 0:
                Sab = np.kron(Sab, X)
                Sab_string.append('X')
            else:
                Sab = np.kron(Sab, I)
                Sab_string.append('I')
        Sab = np.kron(Sab, Z)
        Sab_string.append('Z')
        #print(Sab_string)
        return Sab
    
    def _get_ZOP(self, n_qubits=None):
        if n_qubits is None: n_qubits = self.n_qubits

        I = np.eye(2,dtype=int)
        Z = np.array([[1,0],[0,-1]])

        OP = 0
        for i in range(n_qubits):
            OP += np.kron(np.eye(2**i), np.kron(Z, np.eye(2**(n_qubits-i-1))))
        return OP
    
    def _get_qcnn_ansatz(self, epoch=-1, fname_json=None, save_path="./json_data/"):
        if fname_json is None: fname_json = self.fname_json
        
        with open(save_path+fname_json) as f:
            json_dict = json.load(f)

        # 学習後のlearning_params
        learning_params = np.array(json_dict["learning_params"][epoch])            
        # lossが最小となるlearning_paramsでansatz作る
        qc = self.qcnn.create_ansatz(learning_params)
        return qc
    
    def calc_error_separable_OP_FMz(self, n, alpha_min_li=[0.25], read_json=True, data="test1", mode="bayes"):
        if data=="test1":
            J1J2_i = self.h1h2_i_test1
            label_i = self.label_i_test1

        n_qubits = self.n_qubits
        n_is_int = type(n) == int
        if n_is_int:
            n_li = [n]
        else:
            n_li = n
        result_li = []
        rng = np.random.default_rng(seed=2)

        with open(f"./json_data/OP(L={self.n_qubits},{data})_ave_Z.json") as f:
            json_dict = json.load(f)
        J1J2_json = json_dict["J1J2"]
        assert np.allclose(np.array(J1J2_json), J1J2_i)
        # distributions of projective meas. described by O_{FM} = \sum_{i=1}^{L} Z_i
        dists_prob = np.array(json_dict["dist_Z"])
        #dist_eigen = [1-2*(m/n_qubits) for m in range(n_qubits+1)]
        dist_prob_tri = np.mean(dists_prob[label_i==0], axis=0)
        dist_prob_nt = np.mean(dists_prob[label_i!=0], axis=0)

        # for dist_prob in dists_prob:
        #     if not np.isclose(sum(dist_prob), 1.0):
        #         raise ValueError("The probabilities in dist_prob must sum to 1.")

        if mode == "naive":
            def compute_exact_probability(dist_prob, n, L, delta_li):
                """
                サンプル平均が指定した範囲に収まる確率を厳密に計算します。

                Args:
                    dist_prob (list): 各整数値 x に対する確率分布 (p(X=x))。
                    n (int): サンプル数。
                    L (int): 量子ビット数。
                    delta_li (list[float]): 許容範囲の半径のリスト。

                Returns:
                    total_probability (list[float]): サンプル平均が範囲に収まる確率。
                """
                # サンプル平均の範囲
                lower_bound_li = [L / 2 - delta for delta in delta_li]
                upper_bound_li = [L / 2 + delta for delta in delta_li]
                # 全ての可能なサンプルの組み合わせ
                outcomes = range(len(dist_prob))  # x = 0, 1, ..., L
                all_samples = itertools.product(outcomes, repeat=n)
                # 確率を合計
                total_probability = [0.0 for _ in range(len(delta_li))]
                for sample in all_samples:
                    sample_mean = np.mean(sample)  # サンプル平均を計算
                    prob = np.prod([dist_prob[x] for x in sample]) # サンプルの確率を計算
                    for idx, (lower_bound, upper_bound) in enumerate(zip(lower_bound_li, upper_bound_li)):
                        if lower_bound <= sample_mean <= upper_bound:
                            total_probability[idx] += prob
                return total_probability
            
            # 許容範囲を決める
            delta_li = np.linspace(0, n_qubits/2, len(alpha_min_li))
            for n in n_li:
                # n=1は厳密に計算
                if n == 1:
                    alpha_li = 1 - np.array(compute_exact_probability(dist_prob_tri, n, n_qubits, delta_li))
                    beta_li = np.array(compute_exact_probability(dist_prob_nt, n, n_qubits, delta_li))
                # n>1はMC
                else:
                    N_mc = int(1e+6)
                    # trivial (期待値が許容範囲内)
                    samples_multi = rng.multinomial(n, dist_prob_tri, size=N_mc)
                    means_multi = (samples_multi@np.arange(n_qubits+1)[:,None]).flatten() / np.sum(samples_multi, axis=1)
                    print(means_multi)
                    alpha_li = np.mean(np.abs(means_multi-(n_qubits/2))>delta_li.reshape(-1,1), axis=1)
                    # non-trivial (期待値が許容範囲外)
                    samples_multi = rng.multinomial(n, dist_prob_nt, size=N_mc)
                    means_multi = (samples_multi@np.arange(n_qubits+1)[:,None]).flatten() / np.sum(samples_multi, axis=1)
                    print(means_multi)
                    beta_li = np.mean(np.abs(means_multi-(n_qubits/2))<=delta_li.reshape(-1,1), axis=1)
                # 数値誤差で[0,1]を超えるかも
                alpha_li[alpha_li<=0] = 0
                beta_li[beta_li>=1] = 1
                result_li.append((alpha_li.tolist(), beta_li.tolist()))
        
        elif mode == "bayes":
            from scipy.special import loggamma
            def log_BF_10(x, L, n=None):
                """
                ベイズファクター10(周辺尤度比 H_1/H_0)の対数

                Args:
                    x (list[int]): 多項分布からn回サンプルした実現値、サイズL+1の配列
                    L (int): 量子ビット
                    n (int): サンプル数

                Returns:
                    result (float): log BF10 
                """
                if n is None:
                    n = sum(x)
                result_li = []
                # H_1で分布が右に寄っていると仮定 (+Z方向のFM相)
                result = 0
                for m, xm in enumerate(x):
                    result += (loggamma_di[(m+1)/((L+1)*(L+2)//2)+xm] + loggamma_di[LCm_li[m]/2**L] - loggamma_di[(m+1)/((L+1)*(L+2)//2)] - loggamma_di[LCm_li[m]/2**L+xm])
                result_li.append(result)
                # H_1で分布が左に寄っていると仮定 (-Z方向のFM相)
                result = 0
                for m, xm in enumerate(x):
                    result += (loggamma_di[(L-m+1)/((L+1)*(L+2)//2)+xm] + loggamma_di[LCm_li[m]/2**L] - loggamma_di[(m+1)/((L+1)*(L+2)//2)] - loggamma_di[LCm_li[m]/2**L+xm])
                result_li.append(result)          
                # 0からより離れている方を採用
                result = result_li[0] if abs(result_li[0])>=abs(result_li[1]) else result_li[1]
                return result
            
            # combinationを計算しておく
            nCr = math.comb
            LCm_li = [nCr(n_qubits, m) for m in range(n_qubits+1)]
            # log gammaを計算しておく
            st = time.time()
            loggamma_di = {}
            L = n_qubits
            for m in range(L+1):
                if (m+1)/((L+1)*(L+2)//2) not in loggamma_di:
                    loggamma_di[(m+1)/((L+1)*(L+2)//2)] = loggamma((m+1)/((L+1)*(L+2)//2))
                if LCm_li[m]/2**L not in loggamma_di:
                    loggamma_di[LCm_li[m]/2**L] = loggamma(LCm_li[m]/2**L)
                for xm in range(max(n_li)+1):
                    if (m+1)/((L+1)*(L+2)//2)+xm not in loggamma_di:
                        loggamma_di[(m+1)/((L+1)*(L+2)//2)+xm] = loggamma((m+1)/((L+1)*(L+2)//2)+xm)
                    if (L-m+1)/((L+1)*(L+2)//2)+xm not in loggamma_di:
                        loggamma_di[(L-m+1)/((L+1)*(L+2)//2)+xm] = loggamma((L-m+1)/((L+1)*(L+2)//2)+xm)
                    if LCm_li[m]/2**L+xm not in loggamma_di:
                        loggamma_di[LCm_li[m]/2**L+xm] = loggamma(LCm_li[m]/2**L+xm)
            print("clac loggamma:", time.time()-st)
            print(len(loggamma_di), loggamma_di)
            for n in n_li:
                if n == 1:
                    log_BF_10_li = np.array([log_BF_10(x=[1 if i==m else 0 for i in range(n_qubits+1)], L=n_qubits, n=1) for m in range(n_qubits+1)])
                    threshold_li = np.linspace(min(log_BF_10_li), max(log_BF_10_li)+0.001, len(alpha_min_li)) # log BF10の閾値を決める
                    alpha_li = np.array([np.sum(dist_prob_tri[i]) for i in log_BF_10_li>=threshold_li.reshape(-1,1)])
                    beta_li = np.array([np.sum(dist_prob_nt[i]) for i in log_BF_10_li<threshold_li.reshape(-1,1)])
                else:
                    N_mc = int(1e+4)
                    # trivial (log_BF10が小さい)
                    samples_multi = rng.multinomial(n, dist_prob_tri, size=N_mc)
                    log_BF_10_li = []
                    from tqdm.notebook import tqdm
                    for samples in tqdm(samples_multi):
                        log_BF_10_li.append(log_BF_10(x=samples, L=n_qubits, n=n))
                    threshold_li = np.linspace(min(log_BF_10_li), max(log_BF_10_li)+0.001, len(alpha_min_li)) # log BF10の閾値を決める
                    alpha_li = np.mean(np.array(log_BF_10_li)>=threshold_li.reshape(-1,1), axis=1)
                    # non-trivial (log_BF10が大きい)
                    samples_multi = rng.multinomial(n, dist_prob_nt, size=N_mc)
                    log_BF_10_li = []
                    for samples in samples_multi:
                        log_BF_10_li.append(log_BF_10(x=samples, L=n_qubits, n=n))
                    threshold_li = np.linspace(min(log_BF_10_li), max(log_BF_10_li)+0.001, len(alpha_min_li)) # log BF10の閾値を決める
                    beta_li = np.mean(np.array(log_BF_10_li)<threshold_li.reshape(-1,1), axis=1)            
                # 数値誤差で[0,1]を超えるかも
                alpha_li[alpha_li<=0] = 0
                beta_li[beta_li>=1] = 1
                result_li.append((alpha_li.tolist(), beta_li.tolist()))
        if n_is_int:
            return result_li[0]
        else:
            return result_li
        
    def calc_error_separable_OP(self, n, alpha_min_li=[0.25], read_json=True, data="test1", mode="bayes"):
        if data=="test1":
            J1J2_i = self.h1h2_i_test1
            label_i = self.label_i_test1
        elif data=="test2":
            J1J2_i = self.h1h2_i_test2
            label_i = self.label_i_test2

        if not read_json:
            if data=="test1":
                read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test1.txt")
                OP = self._get_ZOP()
                ave_OP = np.diag(read_data[1].conj()@OP@read_data[1].T)
                with open(f"./OP(L={self.n_qubits},{data})_ave_Z.json", "w") as f:
                    json.dump({"J1J2":read_data[0].tolist(), "ave_Z":ave_OP.flatten().tolist()}, f, indent=2)
            elif data=="test2":
                read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test2.txt")
                OP = self._get_SOP()
                ave_OP = np.diag(read_data[1].conj()@OP@read_data[1].T)
                with open(f"./OP(L={self.n_qubits},{data})_ave_SOP.json", "w") as f:
                    json.dump({"J1J2":read_data[0].tolist(), "ave_SOP":ave_OP.flatten().tolist()}, f, indent=2)
        else:
            if data=="test1":
                # with open(f"./json_data/OP(L={self.n_qubits},{data})_ave_Z.json") as f:
                #     json_dict = json.load(f)
                # J1J2_json = json_dict["J1J2"]
                # ave_OP_json = json_dict["ave_Z"]
                # ave_OP_nt = np.mean(np.abs(np.array(ave_OP_json))[label_i!=0])
                # ave_OP_tri = np.mean(np.abs(np.array(ave_OP_json))[label_i==0])
                return self.calc_error_separable_OP_FMz(n, alpha_min_li, read_json, data, mode)
            elif data=="test2":
                with open(f"./json_data/OP(L={self.n_qubits},{data})_ave_SOP.json") as f:
                    json_dict = json.load(f)
                J1J2_json = json_dict["J1J2"]
                ave_OP_json = json_dict["ave_SOP"]
                ave_OP_nt = np.mean(np.array(ave_OP_json)[label_i!=0])
                ave_OP_tri = np.mean(np.array(ave_OP_json)[label_i==0])
            assert np.allclose(np.array(J1J2_json), J1J2_i)

        bernoulli_p_nt = (ave_OP_nt+1)/2
        bernoulli_p_tri = (ave_OP_tri+1)/2

        # 尤度比検定
        nCr = math.comb
        n_is_int = type(n) == int
        if n_is_int:
            n_li = [n]
        else:
            n_li = n
        result_li = []
        for n in n_li:
            # k_alphaを見つけ、gammaも求める
            k_alpha_li = []
            gamma_li = []
            for alpha_min in alpha_min_li:
                F = 0
                for k in range(n+1):
                    f = nCr(n, k) * 0.5**n
                    F += f
                    if alpha_min >= 1-F:
                        k_alpha_li.append(k)
                        gamma_li.append((alpha_min-1+F)/f)
                        break
                    elif k==n:
                        k_alpha_li.append(k)
                        gamma_li.append((alpha_min-1+F)/f)
            # alphaとbetaを求める
            alpha_li = []
            beta_li = []
            for i, k_alpha in enumerate(k_alpha_li): 
                alpha = 1- sum([nCr(n,j)*bernoulli_p_tri**j*(1-bernoulli_p_tri)**(n-j) for j in range(k_alpha+1)])
                alpha += gamma_li[i] * nCr(n,k_alpha)*bernoulli_p_tri**k_alpha*(1-bernoulli_p_tri)**(n-k_alpha)
                alpha_li.append(alpha)
                beta = sum([nCr(n,j)*bernoulli_p_nt**j*(1-bernoulli_p_nt)**(n-j) for j in range(k_alpha)])
                beta += (1-gamma_li[i]) * nCr(n,k_alpha)*bernoulli_p_nt**k_alpha*(1-bernoulli_p_nt)**(n-k_alpha)
                beta_li.append(beta)
            result_li.append((alpha_li, beta_li))
        if n_is_int:
            return result_li[0]
        else:
            return result_li
    
    def gen_qNeyman(self, seed=None, a=0, a_li=None, shots=20*300*2, n_ent=3, data="train1", haar="haar"):
        if data=="train1":
            J1J2_i = self.h1h2_i_train1
            label_i = self.label_i_train1
            read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_train1.txt")
            vec_li = read_data[1][self.label_i_train1==0]
            rho_in_tri = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
            vec_li = read_data[1][self.label_i_train1!=0]
            rho_in_nt = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
        elif data=="train2":
            J1J2_i = self.h1h2_i_train2
            label_i = self.label_i_train2
            read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_train2.txt")
            vec_li = read_data[1][self.label_i_train2==0]
            rho_in_tri = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
            vec_li = read_data[1][self.label_i_train2!=0]
            rho_in_nt = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
        if seed != None: np.random.seed(seed)
        self.shots = shots

        folder = f"./random_unitary/random_{haar}/"
        file_idx = 1
        remained_random = np.load(folder+f"{n_ent}qubits_40x2x10000num({file_idx}).npy")
        rho_hat_li_nt = []
        rho_hat_li_tri = []
        for qidx in tqdm(range(0,self.n_qubits,n_ent), leave=False):
            if qidx+n_ent <= self.n_qubits:
                _n = n_ent
                while remained_random.shape[0] < self.shots:
                    file_idx += 1
                    remained_random = np.concatenate([remained_random, np.load(folder+f"{n_ent}qubits_40x2x10000num({file_idx}).npy")])
            else:
                _n = self.n_qubits-qidx
                file_idx = 1
                remained_random = np.load(folder+f"{_n}qubits_40x2x10000num({file_idx}).npy")
                while remained_random.shape[0] < self.shots:
                    file_idx += 1
                    remained_random = np.concatenate([remained_random, np.load(folder+f"{_n}qubits_40x2x10000num({file_idx}).npy")])                    
            choice_haar_nt = remained_random[:self.shots//2]
            choice_haar_tri = remained_random[self.shots//2:self.shots]
            remained_random = remained_random[self.shots:]
            trace_systems = [i for i in range(self.n_qubits) if i not in [qidx+j for j in range(_n)]]
            partial_rho_nt = partial_trace(rho_in_nt, trace_systems)
            partial_rho_tri = partial_trace(rho_in_tri, trace_systems)
            choice_rho_nt = choice_haar_nt@partial_rho_nt[None,:,:]@choice_haar_nt.conj().swapaxes(1,2)
            choice_rho_tri = choice_haar_tri@partial_rho_tri[None,:,:]@choice_haar_tri.conj().swapaxes(1,2)
            meas_nt = [np.random.choice(list(range(2**_n)), p=np.maximum(np.diag(rho_nt).real,0)) for rho_nt in choice_rho_nt]
            meas_tri = [np.random.choice(list(range(2**_n)), p=np.maximum(np.diag(rho_tri).real,0)) for rho_tri in choice_rho_tri]
            snapshot_nt = np.zeros((2**_n,2**_n), dtype=np.complex128)
            snapshot_tri = np.zeros((2**_n,2**_n), dtype=np.complex128)
            for i, (nt, tri) in enumerate(zip(meas_nt, meas_tri)):
                snapshot = choice_haar_nt[i].T.conj()@Statevector.from_int(nt,2**_n).data[:,None]
                snapshot = (2**_n+1) * snapshot@snapshot.T.conj() - np.eye(2**_n)
                snapshot_nt += snapshot
                snapshot = choice_haar_tri[i].T.conj()@Statevector.from_int(tri,2**_n).data[:,None]
                snapshot = (2**_n+1) * snapshot@snapshot.T.conj() - np.eye(2**_n)
                snapshot_tri += snapshot
            rho_hat_li_nt.append(snapshot_nt/len(meas_nt))
            rho_hat_li_tri.append(snapshot_tri/len(meas_tri))

        # vv, uu = np.linalg.eigh(self.rho_in_nspt_train - np.exp(a)*self.rho_in_spt_train)
        # qNeyman = uu[:,vv>0]@uu.T[vv>0].conj()
        # self.qNeyman_train = qNeyman

        if a_li is None:
            vv_li = []
            uu_li = []
            for rho_nt, rho_tri in zip(rho_hat_li_nt, rho_hat_li_tri):
                vv, uu = np.linalg.eigh(rho_tri - np.exp(a)*rho_nt)
                vv_li.append(vv)
                uu_li.append(uu)
            self.qNeyman_hat = [[vv_li, uu_li]]
        else:
            self.qNeyman_hat = []
            for a in tqdm(a_li, leave=False):
                vv_li = []
                uu_li = []
                for rho_nt, rho_tri in zip(rho_hat_li_nt, rho_hat_li_tri):
                    vv, uu = np.linalg.eigh(rho_tri - np.exp(a)*rho_nt)
                    vv_li.append(vv)
                    uu_li.append(uu)
                self.qNeyman_hat.append([vv_li, uu_li])
        return 
    
    def calc_error_separable_CSqNeyman(self, n, seed=None, a_li=[0], shots=20*300*2, 
                                       n_ent=3, data="test1", haar="haar",
                                       read_json=True, read_npz=True):
        if data=="test1":
            J1J2_i = self.h1h2_i_test1
            label_i = self.label_i_test1
            if not read_json:
                read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test1.txt")
                vec_li = read_data[1][self.label_i_test1==0]
                rho_in_tri = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                vec_li = read_data[1][self.label_i_test1!=0]
                rho_in_nt = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                data_train = "train1"
        elif data=="test2":
            J1J2_i = self.h1h2_i_test2
            label_i = self.label_i_test2
            if not read_json:
                read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test2.txt")
                vec_li = read_data[1][self.label_i_test2==0]
                rho_in_tri = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                vec_li = read_data[1][self.label_i_test2!=0]
                rho_in_nt = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                data_train = "train2"

        if not read_json:
            if read_npz:
                folder_npz = f"./npz_data/CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent})/"
                assert (np.array(a_li) == np.load(folder_npz+f"CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent},{data_train}{','+haar if haar!='haar' else ''})_a_li.npy")).all()
                qNeyman_hat = []
                vv_li_li = np.load(folder_npz+f"CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent},{data_train}{','+haar if haar!='haar' else ''})_vv_li.npz").values()
                uu_li_li = np.load(folder_npz+f"CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent},{data_train}{','+haar if haar!='haar' else ''})_uu_li.npz").values()
                num_dev = math.ceil(self.n_qubits/n_ent)
                vv_li_li = [x for x in vv_li_li]
                vv_li_li = [vv_li_li[i:i+num_dev] for i in range(0,len(vv_li_li),num_dev)]
                uu_li_li = [x for x in uu_li_li]
                uu_li_li = [uu_li_li[i:i+num_dev] for i in range(0,len(uu_li_li),num_dev)]
                for vv_li, uu_li in zip(vv_li_li, uu_li_li):
                    qNeyman_hat.append([vv_li, uu_li])
                self.qNeyman_hat = qNeyman_hat
                qNeyman_hat=None; vv_li_li=None; uu_li_li=None
            else:
                self.gen_qNeyman(seed, a_li=a_li, shots=shots, n_ent=n_ent, data=data_train, haar=haar)
                folder_npz = f"./CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent})/"
                os.makedirs(folder_npz)
                np.save(folder_npz+f"CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent},{data_train}{','+haar if haar!='haar' else ''})_a_li.npy", a_li)
                np.savez_compressed(folder_npz+f"CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent},{data_train}{','+haar if haar!='haar' else ''})_vv_li.npz", *sum([x[0] for x in self.qNeyman_hat],[]))
                np.savez_compressed(folder_npz+f"CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent},{data_train}{','+haar if haar!='haar' else ''})_uu_li.npz", *sum([x[1] for x in self.qNeyman_hat],[]))          

            rng = np.random.default_rng()
            alpha_li = [0 for _ in range(len(a_li))]
            beta_li = [0 for _ in range(len(a_li))]
            for i_a, a in tqdm(enumerate(a_li)):
                meas_tri_li = []
                meas_nt_li = []
                qidx = 0
                for i, uu in enumerate(self.qNeyman_hat[i_a][1]):
                    _n = int(np.log2(uu.shape[0]))
                    trace_systems = [i for i in range(self.n_qubits) if i not in [qidx+j for j in range(_n)]]
                    qidx += _n
                    partial_rho_tri = uu.T.conj()@partial_trace(rho_in_tri, trace_systems)@uu
                    partial_rho_nt = uu.T.conj()@partial_trace(rho_in_nt, trace_systems)@uu
                    meas_tri_li.append(np.diag(partial_rho_tri).real.tolist())
                    meas_nt_li.append(np.diag(partial_rho_nt).real.tolist())
                
                max_length = max(len(sublist) for sublist in meas_tri_li) # zero padding (for transforming list to array)
                meas_tri_li = np.array([sublist + [0] * (max_length - len(sublist)) for sublist in meas_tri_li])
                max_length = max(len(sublist) for sublist in meas_nt_li)
                meas_nt_li = np.array([sublist + [0] * (max_length - len(sublist)) for sublist in meas_nt_li])            
                max_length = max(len(sublist) for sublist in self.qNeyman_hat[i_a][0])
                vv_li = np.array([sublist.tolist() + [0] * (max_length - len(sublist)) for sublist in self.qNeyman_hat[i_a][0]])
                K = int(1e+7)
                sample_tri_li = np.array([rng.choice(meas_tri_li.shape[1], size=K, p=probabilities) for probabilities in meas_tri_li])
                sample_nt_li = np.array([rng.choice(meas_nt_li.shape[1], size=K, p=probabilities) for probabilities in meas_nt_li])
                selected_tri_values = vv_li[np.arange(meas_tri_li.shape[0])[:, None], sample_tri_li]
                selected_nt_values = vv_li[np.arange(meas_nt_li.shape[0])[:, None], sample_nt_li]
                # product_tri_values = np.prod(selected_tri_values, axis=0)
                # product_nt_values = np.prod(selected_nt_values, axis=0)
                # positive_tri_prob = np.count_nonzero(product_tri_values > 0)/K
                # positive_nt_prob = np.count_nonzero(product_nt_values > 0)/K
                positive_tri_count = np.sum(selected_tri_values>0, axis=0)
                positive_nt_count = np.sum(selected_nt_values>0, axis=0)
                positive_tri_prob = np.count_nonzero(positive_tri_count > meas_tri_li.shape[0]//2)/K
                positive_nt_prob = np.count_nonzero(positive_nt_count > meas_nt_li.shape[0]//2)/K
                alpha_li[i_a] += positive_tri_prob
                beta_li[i_a] += positive_nt_prob
            alpha_li = 1 - np.array(alpha_li)
            beta_li = np.array(beta_li)

            self.alpha_li = alpha_li; self.beta_li = beta_li
            with open(f"./CSqNeyman(L={self.n_qubits}_shots={shots}_nent={n_ent},{data}{','+haar if haar!='haar' else ''})_alphabeta.json", "w") as f:
                json.dump({"a_li":a_li, "alpha":alpha_li.tolist(), "beta":beta_li.tolist()}, f, indent=2)

        else:
            with open(f"./json_data/CSqNeyman(L={self.n_qubits}_shots={shots}_nent={n_ent},{data}{','+haar if haar!='haar' else ''})_alphabeta.json") as f:
                json_dict = json.load(f)
            assert (np.array(a_li) == np.array(json_dict["a_li"])).all()
            alpha_li = np.array(json_dict["alpha"])
            beta_li = np.array(json_dict["beta"])

        n_is_int = type(n) == int
        if n_is_int:
            n_li = [n]
        else:
            n_li = n
        result_li = []
        for n in n_li:
            if n>1:
                nCr = math.comb
                # (1-alpha)よりalphaの方を多く得る確率=alpha_n
                alpha_n_li = np.sum([nCr(n, r)*alpha_li**(n-r)*(1-alpha_li)**r for r in range(n//2+1)], axis=0)
                beta_n_li = np.sum([nCr(n, r)*beta_li**(n-r)*(1-beta_li)**r for r in range(n//2+1)], axis=0)
                # 古典Neyman-Pearsonでいうgamma=0.5なので、足し過ぎた分引く
                if n%2 == 0:
                    alpha_n_li -= 0.5 * nCr(n,n//2)*alpha_li**(n//2)*(1-alpha_li)**(n//2)
                    beta_n_li -= 0.5 * nCr(n,n//2)*beta_li**(n//2)*(1-beta_li)**(n//2)
            elif n==1:
                alpha_n_li = alpha_li; beta_n_li = beta_li
            result_li.append((alpha_n_li.tolist(), beta_n_li.tolist()))
        if n_is_int:
            return result_li[0]
        else:
            return result_li 
    
    def calc_error_separable_PartialCSqNeyman(self, n, a_li=[0], 
                                              n_ent=3, data="test1", 
                                              read_json=True, read_npz=True):
        if data=="test1":
            J1J2_i = self.h1h2_i_test1
            label_i = self.label_i_test1
            if not read_json:
                read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test1.txt")
                vec_li = read_data[1][self.label_i_test1==0]
                rho_in_tri = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                vec_li = read_data[1][self.label_i_test1!=0]
                rho_in_nt = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                data_train = "train1"
                if not read_npz:
                    read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_train1.txt")
                    vec_li = read_data[1][self.label_i_train1==0]
                    rho_in_tri_train = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                    vec_li = read_data[1][self.label_i_train1!=0]
                    rho_in_nt_train = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj() 
        elif data=="test2":
            J1J2_i = self.h1h2_i_test2
            label_i = self.label_i_test2
            if not read_json:
                read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test2.txt")
                vec_li = read_data[1][self.label_i_test2==0]
                rho_in_tri = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                vec_li = read_data[1][self.label_i_test2!=0]
                rho_in_nt = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                data_train = "train2"
                if not read_npz:
                    read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_train2.txt")
                    vec_li = read_data[1][self.label_i_train2==0]
                    rho_in_tri_train = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj()
                    vec_li = read_data[1][self.label_i_train2!=0]
                    rho_in_nt_train = (1/vec_li.shape[0] * vec_li.T)@vec_li.conj() 

        if not read_json:
            if not read_npz:
                partial_rho_tri_li = []
                partial_rho_nt_li = []
                for qidx in range(0,self.n_qubits,n_ent):
                    if qidx+n_ent <= self.n_qubits:
                        _n = n_ent
                    else:
                        _n = self.n_qubits-qidx                 
                    trace_systems = [i for i in range(self.n_qubits) if i not in [qidx+j for j in range(_n)]]
                    partial_rho_tri_li.append(partial_trace(rho_in_tri_train, trace_systems))
                    partial_rho_nt_li.append(partial_trace(rho_in_nt_train, trace_systems))
                print(rho_in_tri_train.dtype, partial_rho_tri_li[-1].dtype)
                self.qNeyman_hat = []
                for a in tqdm(a_li, leave=False):
                    vv_li = []
                    uu_li = []
                    for rho_nt, rho_tri in zip(partial_rho_nt_li, partial_rho_tri_li):
                        vv, uu = np.linalg.eigh(rho_tri - np.exp(a)*rho_nt)
                        vv_li.append(vv)
                        uu_li.append(uu)
                    self.qNeyman_hat.append([vv_li, uu_li])
                folder_npz = f"./PartialCSqNeyman(L={self.n_qubits}_nent={n_ent})/"
                os.makedirs(folder_npz)
                np.save(folder_npz+f"PartialCSqNeyman(L={self.n_qubits}_nent={n_ent},{data_train})_a_li.npy", a_li)
                np.savez_compressed(folder_npz+f"PartialCSqNeyman(L={self.n_qubits}_nent={n_ent},{data_train})_vv_li.npz", *sum([x[0] for x in self.qNeyman_hat],[]))
                np.savez_compressed(folder_npz+f"PartialCSqNeyman(L={self.n_qubits}_nent={n_ent},{data_train})_uu_li.npz", *sum([x[1] for x in self.qNeyman_hat],[]))
            else:
                folder_npz = f"./npz_data/PartialCSqNeyman(L={self.n_qubits}_nent={n_ent})/"
                assert (np.array(a_li) == np.load(folder_npz+f"PartialCSqNeyman(L={self.n_qubits}_nent={n_ent},{data_train})_a_li.npy")).all()
                qNeyman_hat = []
                vv_li_li = np.load(folder_npz+f"PartialCSqNeyman(L={self.n_qubits}_nent={n_ent},{data_train})_vv_li.npz").values()
                uu_li_li = np.load(folder_npz+f"PartialCSqNeyman(L={self.n_qubits}_nent={n_ent},{data_train})_uu_li.npz").values()
                # num_dev = math.ceil(self.n_qubits/n_ent)
                # vv_li_li = [x for x in vv_li_li]
                # vv_li_li = [vv_li_li[i:i+num_dev] for i in range(0,len(vv_li_li,num_dev))]
                # uu_li_li = [x for x in uu_li_li]
                # uu_li_li = [uu_li_li[i:i+num_dev] for i in range(0,len(uu_li_li,num_dev))]
                for vv_li, uu_li in zip(vv_li_li, uu_li_li):
                    qNeyman_hat.append([vv_li, uu_li])
                self.qNeyman_hat = qNeyman_hat
                qNeyman_hat=None; vv_li_li=None; uu_li_li=None

            rng = np.random.default_rng()
            alpha_li = [0 for _ in range(len(a_li))]
            beta_li = [0 for _ in range(len(a_li))]
            for i_a, a in tqdm(enumerate(a_li)):
                meas_tri_li = []
                meas_nt_li = []
                qidx = 0
                for i, uu in enumerate(self.qNeyman_hat[i_a][1]):
                    _n = int(np.log2(uu.shape[0]))
                    trace_systems = [i for i in range(self.n_qubits) if i not in [qidx+j for j in range(_n)]]
                    qidx += _n
                    partial_rho_tri = uu.T.conj()@partial_trace(rho_in_tri, trace_systems)@uu
                    partial_rho_nt = uu.T.conj()@partial_trace(rho_in_nt, trace_systems)@uu
                    meas_tri_li.append(np.diag(partial_rho_tri).real.tolist())
                    meas_nt_li.append(np.diag(partial_rho_nt).real.tolist())
                
                max_length = max(len(sublist) for sublist in meas_tri_li) # zero padding (for transforming list to array)
                meas_tri_li = np.array([sublist + [0] * (max_length - len(sublist)) for sublist in meas_tri_li])
                max_length = max(len(sublist) for sublist in meas_nt_li)
                meas_nt_li = np.array([sublist + [0] * (max_length - len(sublist)) for sublist in meas_nt_li])            
                max_length = max(len(sublist) for sublist in self.qNeyman_hat[i_a][0])
                vv_li = np.array([sublist.tolist() + [0] * (max_length - len(sublist)) for sublist in self.qNeyman_hat[i_a][0]])
                K = int(1e+7)
                sample_tri_li = np.array([rng.choice(meas_tri_li.shape[1], size=K, p=probabilities) for probabilities in meas_tri_li])
                sample_nt_li = np.array([rng.choice(meas_nt_li.shape[1], size=K, p=probabilities) for probabilities in meas_nt_li])
                selected_tri_values = vv_li[np.arange(meas_tri_li.shape[0])[:, None], sample_tri_li]
                selected_nt_values = vv_li[np.arange(meas_nt_li.shape[0])[:, None], sample_nt_li]
                # product_tri_values = np.prod(selected_tri_values, axis=0)
                # product_nt_values = np.prod(selected_nt_values, axis=0)
                # positive_tri_prob = np.count_nonzero(product_tri_values > 0)/K
                # positive_nt_prob = np.count_nonzero(product_nt_values > 0)/K
                positive_tri_count = np.sum(selected_tri_values>0, axis=0)
                positive_nt_count = np.sum(selected_nt_values>0, axis=0)
                positive_tri_prob = np.count_nonzero(positive_tri_count > meas_tri_li.shape[0]//2)/K
                positive_nt_prob = np.count_nonzero(positive_nt_count > meas_nt_li.shape[0]//2)/K
                alpha_li[i_a] += positive_tri_prob
                beta_li[i_a] += positive_nt_prob
            alpha_li = 1 - np.array(alpha_li)
            beta_li = np.array(beta_li)

            self.alpha_li = alpha_li; self.beta_li = beta_li
            with open(f"./PartialCSqNeyman(L={self.n_qubits}_nent={n_ent},{data})_alphabeta.json", "w") as f:
                json.dump({"a_li":a_li, "alpha":alpha_li.tolist(), "beta":beta_li.tolist()}, f, indent=2)
            
        else:
            with open(f"./json_data/PartialCSqNeyman(L={self.n_qubits}_nent={n_ent},{data})_alphabeta.json") as f:
                json_dict = json.load(f)
            assert (np.array(a_li) == np.array(json_dict["a_li"])).all()
            alpha_li = np.array(json_dict["alpha"])
            beta_li = np.array(json_dict["beta"])

        n_is_int = type(n) == int
        if n_is_int:
            n_li = [n]
        else:
            n_li = n
        result_li = []
        for n in n_li:
            if n>1:
                nCr = math.comb
                # (1-alpha)よりalphaの方を多く得る確率=alpha_n
                alpha_n_li = np.sum([nCr(n, r)*alpha_li**(n-r)*(1-alpha_li)**r for r in range(n//2+1)], axis=0)
                beta_n_li = np.sum([nCr(n, r)*beta_li**(n-r)*(1-beta_li)**r for r in range(n//2+1)], axis=0)
                # 古典Neyman-Pearsonでいうgamma=0.5なので、足し過ぎた分引く
                if n%2 == 0:
                    alpha_n_li -= 0.5 * nCr(n,n//2)*alpha_li**(n//2)*(1-alpha_li)**(n//2)
                    beta_n_li -= 0.5 * nCr(n,n//2)*beta_li**(n//2)*(1-beta_li)**(n//2)
            elif n==1:
                alpha_n_li = alpha_li; beta_n_li = beta_li
            result_li.append((alpha_n_li.tolist(), beta_n_li.tolist()))
        if n_is_int:
            return result_li[0]
        else:
            return result_li

    def calc_error_separable_qcnn(self, n, shots_per_rho=None, index=150,
                                  alpha_min_li=[0.25],
                                  read_json=True, data="test1",
                                  qcnn_mode="qcnn1"):
        if data=="test1":
            J1J2_i = self.h1h2_i_test1
            label_i = self.label_i_test1
        elif data=="test2":
            J1J2_i = self.h1h2_i_test2
            label_i = self.label_i_test2

        if not read_json:
            raise Exception(f".")
        else:
            with open(f"./json_data/{qcnn_mode}(L={self.n_qubits}_shots={shots_per_rho},{data}).json") as f:
                json_dict = json.load(f)
            ave_qcnn_json = json_dict["expectation_list"][f"index={index}"]
            ave_qcnn_nt = np.mean(np.array(ave_qcnn_json)[label_i!=0])
            ave_qcnn_tri = np.mean(np.array(ave_qcnn_json)[label_i==0])

        if "qcnn1" in qcnn_mode:
            p0 = 3/4
        elif "qcnn2" in qcnn_mode:
            p0 = 1/2

        bernoulli_p_nt = (ave_qcnn_nt+1)/2
        bernoulli_p_tri = (ave_qcnn_tri+1)/2

        # 尤度比検定
        nCr = math.comb
        n_is_int = type(n) == int
        if n_is_int:
            n_li = [n]
        else:
            n_li = n
        result_li = []
        for n in n_li:
            # k_alphaを見つけ、gammaも求める
            k_alpha_li = []
            gamma_li = []
            for alpha_min in alpha_min_li:
                F = 0
                for k in range(n+1):
                    f = nCr(n, k) * p0**k*(1-p0)**(n-k)
                    F += f
                    if alpha_min >= 1-F:
                        k_alpha_li.append(k)
                        gamma_li.append((alpha_min-1+F)/f)
                        break
                    elif k==n:
                        k_alpha_li.append(k)
                        gamma_li.append((alpha_min-1+F)/f)
            # alphaとbetaを求める
            alpha_li = []
            beta_li = []
            for i, k_alpha in enumerate(k_alpha_li): 
                alpha = 1- sum([nCr(n,j)*bernoulli_p_tri**j*(1-bernoulli_p_tri)**(n-j) for j in range(k_alpha+1)])
                alpha += gamma_li[i] * nCr(n,k_alpha)*bernoulli_p_tri**k_alpha*(1-bernoulli_p_tri)**(n-k_alpha)
                alpha_li.append(alpha)
                beta = sum([nCr(n,j)*bernoulli_p_nt**j*(1-bernoulli_p_nt)**(n-j) for j in range(k_alpha)])
                beta += (1-gamma_li[i]) * nCr(n,k_alpha)*bernoulli_p_nt**k_alpha*(1-bernoulli_p_nt)**(n-k_alpha)
                beta_li.append(beta)
            result_li.append((alpha_li, beta_li))
        if n_is_int:
            return result_li[0]
        else:
            return result_li

    def calc_error_separable_exactqcnn(self, n, alpha_min_li=[0.25], read_json=True, data="test1"):
        if data=="test1":
            J1J2_i = self.h1h2_i_test1
            label_i = self.label_i_test1
            if not read_json:
                vec_li = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test1.txt")[1]
                self.exactqcnn.prepare_Z()
                op = np.kron(np.array([[1,0],[0,-1]]),np.array([[1,0],[0,-1]])) # ZZ
        elif data=="test2":
            J1J2_i = self.h1h2_i_test2
            label_i = self.label_i_test2
            if not read_json:
                vec_li = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test2.txt")[1]
                self.exactqcnn.prepare_SOP()
                op = np.kron(np.kron(np.array([[1,0],[0,-1]]),np.array([[0,1],[1,0]])),np.array([[1,0],[0,-1]])) # ZXZ

        if not read_json:
            reduced_qubits = self.exactqcnn.reduced_qubits

            qc = self.exactqcnn.qc
            vec_li = [Statevector(vec) for vec in vec_li]
            vec_li = [vec.evolve(qc) for vec in vec_li]
            vec_li = [partial_trace(vec.data, reduced_qubits) for vec in vec_li]
            meas_li = [np.trace(vec@op).real for vec in vec_li]

            if data=="test1" or data=="train1":
                with open(f"./exactqcnn(L={self.n_qubits},{data})_ave_Z.json", "w") as f:
                    json.dump({"J1J2":np.array(J1J2_i).tolist(), "ave_Z":meas_li}, f, indent=2)
            elif data=="test2" or data=="train2":
                with open(f"./exactqcnn(L={self.n_qubits},{data})_ave_SOP.json", "w") as f:
                    json.dump({"J1J2":np.array(J1J2_i).tolist(), "ave_SOP":meas_li}, f, indent=2)
            ave_exactqcnn_nt = np.mean(np.array(meas_li)[label_i!=0])
            ave_exactqcnn_tri = np.mean(np.array(meas_li)[label_i==0])
        else:
            if data=="test1" or data=="train1":
                with open(f"./json_data/exactqcnn(L={self.n_qubits},{data})_ave_Z.json") as f:
                    json_dict = json.load(f)
                J1J2_json = json_dict["J1J2"]
                ave_exactqcnn_json = json_dict["ave_Z"]
                ave_exactqcnn_nt = np.mean(np.array(ave_exactqcnn_json)[label_i!=0])
                ave_exactqcnn_tri = np.mean(np.array(ave_exactqcnn_json)[label_i==0])
            elif data=="test2" or data=="train2":
                with open(f"./json_data/exactqcnn(L={self.n_qubits},{data})_ave_SOP.json") as f:
                    json_dict = json.load(f)
                J1J2_json = json_dict["J1J2"]
                ave_exactqcnn_json = json_dict["ave_SOP"]
                ave_exactqcnn_nt = np.mean(np.array(ave_exactqcnn_json)[label_i!=0])
                ave_exactqcnn_tri = np.mean(np.array(ave_exactqcnn_json)[label_i==0])
            assert np.allclose(np.array(J1J2_json), J1J2_i)

        p0 = 3/4

        bernoulli_p_nt = (ave_exactqcnn_nt+1)/2
        bernoulli_p_tri = (ave_exactqcnn_tri+1)/2

        # 尤度比検定
        nCr = math.comb
        n_is_int = type(n) == int
        if n_is_int:
            n_li = [n]
        else:
            n_li = n
        result_li = []
        for n in n_li:
            # k_alphaを見つけ、gammaも求める
            k_alpha_li = []
            gamma_li = []
            for alpha_min in alpha_min_li:
                F = 0
                for k in range(n+1):
                    f = nCr(n, k) * p0**k*(1-p0)**(n-k)
                    F += f
                    if alpha_min >= 1-F:
                        k_alpha_li.append(k)
                        gamma_li.append((alpha_min-1+F)/f)
                        break
                    elif k==n:
                        k_alpha_li.append(k)
                        gamma_li.append((alpha_min-1+F)/f)
            # alphaとbetaを求める
            alpha_li = []
            beta_li = []
            for i, k_alpha in enumerate(k_alpha_li): 
                alpha = 1- sum([nCr(n,j)*bernoulli_p_tri**j*(1-bernoulli_p_tri)**(n-j) for j in range(k_alpha+1)])
                alpha += gamma_li[i] * nCr(n,k_alpha)*bernoulli_p_tri**k_alpha*(1-bernoulli_p_tri)**(n-k_alpha)
                alpha_li.append(alpha)
                beta = sum([nCr(n,j)*bernoulli_p_nt**j*(1-bernoulli_p_nt)**(n-j) for j in range(k_alpha)])
                beta += (1-gamma_li[i]) * nCr(n,k_alpha)*bernoulli_p_nt**k_alpha*(1-bernoulli_p_nt)**(n-k_alpha)
                beta_li.append(beta)
            result_li.append((alpha_li, beta_li))
        if n_is_int:
            return result_li[0]
        else:
            return result_li
        
    def calc_probnt_separable_CSqNeyman(self, seed=None, a_=0, shots=20*300*2, 
                                        n_ent=3, data="test1", haar="haar",
                                        read_json=True):
        if data=="test1":
            J1J2_i = self.h1h2_i_test1
            label_i = self.label_i_test1
            if not read_json:
                read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test1.txt")
                vec_li = read_data[1]
                data_train = "train1"
        elif data=="test2":
            J1J2_i = self.h1h2_i_test2
            label_i = self.label_i_test2
            if not read_json:
                read_data = self._read_eigenvectors(f"dataset_Pollmann_n={self.n_qubits}_test2.txt")
                vec_li = read_data[1]
                data_train = "train2"

        if not read_json:
            with open(f"./json_data/CSqNeyman(L={self.n_qubits}_shots={shots}_nent={n_ent},{data}{','+haar if haar!='haar' else ''})_alphabeta.json") as f:
                json_dict = json.load(f)
            folder_npz = f"./npz_data/CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent})/"
            a_li = [x for x in np.load(folder_npz+f"CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent},{data_train}{','+haar if haar!='haar' else ''})_a_li.npy")]
            assert a_ in a_li
            assert (np.array(a_li) == np.array(json_dict["a_li"])).all()
            qNeyman_hat = []
            vv_li_li = np.load(folder_npz+f"CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent},{data_train}{','+haar if haar!='haar' else ''})_vv_li.npz").values()
            uu_li_li = np.load(folder_npz+f"CSqNeyman(L={self.n_qubits}_shots={shots}_seed={seed}_nent={n_ent},{data_train}{','+haar if haar!='haar' else ''})_uu_li.npz").values()
            num_dev = math.ceil(self.n_qubits/n_ent)
            vv_li_li = [x for x in vv_li_li]
            vv_li_li = [vv_li_li[i:i+num_dev] for i in range(0,len(vv_li_li),num_dev)]
            uu_li_li = [x for x in uu_li_li]
            uu_li_li = [uu_li_li[i:i+num_dev] for i in range(0,len(uu_li_li),num_dev)]
            for vv_li, uu_li in zip(vv_li_li, uu_li_li):
                qNeyman_hat.append([vv_li, uu_li])
            self.qNeyman_hat = qNeyman_hat
            qNeyman_hat=None; vv_li_li=None; uu_li_li=None
      
            rng = np.random.default_rng()
            probnt_li = []
            for i_a, a in enumerate(a_li):
                if a != a_: continue
                for vec in tqdm(vec_li):
                    meas_li = []
                    qidx = 0
                    for i, uu in enumerate(self.qNeyman_hat[i_a][1]):
                        _n = int(np.log2(uu.shape[0]))
                        trace_systems = [i for i in range(self.n_qubits) if i not in [qidx+j for j in range(_n)]]
                        qidx += _n
                        partial_rho = uu.T.conj()@partial_trace(vec, trace_systems)@uu
                        meas_li.append(np.diag(partial_rho).real.tolist())
                    
                    max_length = max(len(sublist) for sublist in meas_li) # zero padding (for transforming list to array)
                    meas_li = np.array([sublist + [0] * (max_length - len(sublist)) for sublist in meas_li])          
                    max_length = max(len(sublist) for sublist in self.qNeyman_hat[i_a][0])
                    vv_li = np.array([sublist.tolist() + [0] * (max_length - len(sublist)) for sublist in self.qNeyman_hat[i_a][0]])
                    K = int(1e+7)
                    sample_li = np.array([rng.choice(meas_li.shape[1], size=K, p=probabilities) for probabilities in meas_li])
                    selected_values = vv_li[np.arange(meas_li.shape[0])[:, None], sample_li]
                    positive_count = np.sum(selected_values>0, axis=0)
                    positive_prob = np.count_nonzero(positive_count > meas_li.shape[0]//2)/K
                    probnt_li.append(1-positive_prob)

            json_dict[f"probnt_a={a_}"] = probnt_li
            with open(f"./CSqNeyman(L={self.n_qubits}_shots={shots}_nent={n_ent},{data}{','+haar if haar!='haar' else ''})_alphabeta.json", "w") as f:
                json.dump(json_dict, f, indent=2)

        else:
            with open(f"./json_data/CSqNeyman(L={self.n_qubits}_shots={shots}_nent={n_ent},{data}{','+haar if haar!='haar' else ''})_alphabeta.json") as f:
                json_dict = json.load(f)
            probnt_li = json_dict[f"probnt_a={a_}"]
        return np.array(probnt_li)


    def plot_alpha_vs_beta(self, n,
                           results_error_li=[],
                           results_error_point=[],
                           settings_dict=[],
                           order=[],
                           scale="linear"):
        """
            --n -> int
            --results_error_li -> type:(alpha_li, beta_li)
            --results_error_point -> type:(alpha, beta)
            --settings_dict -> [{"color":"red", "linestyle":"--", "marker":"o", "label":"qcnn"}, ...]
            --order -> [0,1,4,2,3,...]
            --result_error_separable_SOP -> (alpha_li, beta_li)
            --result_error_separable_qcnn -> (alpha, beta)
            --result_error_separable_exactqcnn -> (alph(a, beta)
            --result_error_separable_qcnnWqNeyman -> (alpha_li, beta_li)
            --result_error_separable_exactqcnnWqNeyman -> (alpha_li, beta_li)
            --result_error_separable_qNeyman -> (alpha_li, beta_li)
            --result_error_entangled_qNeyman -> (alpha_li, beta_li)
            --result_error_separable_CSqNeyman -> (alpha_li, beta_li)
        """
        fig = plt.figure(figsize=(4.8,4.8))
        # 設定
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["font.size"] = 20
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # type: (alpha_li, beta_li)
        for i, result_error_li in enumerate(results_error_li):
            plt.plot(*result_error_li, **settings_dict[i])
        # type: (alpha, beta)
        for i, result_error_point in enumerate(results_error_point):
            plt.plot(*result_error_point, **settings_dict[i+len(results_error_li)])    

        plt.xscale(scale)
        plt.yscale(scale)
        plt.xlim(-0.05,1.05)
        plt.ylim(-0.05,1.05)
        plt.xlabel(fr"$\alpha_{n}$")
        plt.ylabel(fr"$\beta_{n}$")
        plt.xticks([0, 0.5, 1.0])
        plt.yticks([0, 0.5, 1.0])

        # plt.title(fr"$\beta_n$ vs. $\alpha_n$ $(n={n})$")
        # ordering legend)
        # if order:
        #     handles, labels = plt.gca().get_legend_handles_labels()
        #     plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right')
        # else:
        #     plt.legend()

        plt.show(fig)
        plt.close(fig)

    def plot_n_vs_logbeta(self, alpha_max, n_li,
                          results_error_li=[],
                          #results_error_point=[],
                          settings_dict=[],
                          order=[],
                          scale="linear"):
        """
            --alpha_max -> float: alpha<=alpha_maxの条件下でbetaが最小のものを選ぶ
            --n_li -> [int,...]
            --results_error_li -> type:(alpha_li, beta_li)
            --results_error_point -> type:(alpha, beta)
            --settings_dict -> [{"color":"red", "linestyle":"--", "marker":"o", "label":"qcnn"}, ...]
            --order -> [0,1,4,2,3,...]
            --result_error_separable_SOP -> [(alpha_li, beta_li),...]
            --result_error_separable_qcnn -> [(alpha, beta),...]
            --result_error_separable_exactqcnn -> [(alpha, beta),...]
            --result_error_separable_qcnnWqNeyman -> [(alpha_li, beta_li),...]
            --result_error_separable_exactqcnnWqNeyman -> [(alpha_li, beta_li),...]
            --result_error_separable_qNeyman -> [(alpha_li, beta_li),...]
            --result_error_entangled_qNeyman -> [(alpha_li, beta_li),...]
        """
        fig = plt.figure()
        # 設定
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["font.size"] = 14
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # データ整形用関数
        def data_processing_li(result_error_li):
            logbeta_li = []
            for i, n in enumerate(n_li):
                result_arr = np.array(result_error_li[i])
                result_arr = result_arr[:, result_arr[0]<=alpha_max]
                # beta = result_arr[1, np.argmax(result_arr[0])]
                beta = np.min(result_arr[1])
                logbeta_li.append(-np.log(beta)/n)
            return logbeta_li       
        # type: (alpha_li, beta_li)
        for i, result_error_li in enumerate(results_error_li):
            logbeta_li = data_processing_li(result_error_li)
            plt.plot(n_li,logbeta_li, **settings_dict[i])
        # # type: (alpha, beta)
        # for i, result_error_point in enumerate(results_error_point):
        #     logbeta_li = data_processing_point(result_error_point)
        #     plt.plot(n_li,logbeta_li, **settings_dict[i+len(results_error_li)])

        # rel_entropy
        plt.hlines(self.SOP_rel_entropy_list[1], min(n_li)-0.3,max(n_li)+0.3, color="k",linestyle='--')
        #plt.axhline(self.q_rel_entropy_list[1], n_li[0],n_li[-1], color="k",linestyle='--')

        plt.yscale(scale)
        plt.xticks(list(range(min(n_li), max(n_li)+1)))
        plt.xlabel(r"$n$")
        plt.ylabel(r"$-\frac{1}{n}\log(\beta_n)$")
        plt.title(r"$-\frac{1}{n}\log(\beta_n)$ vs. $n$"+fr" $(\alpha_n\leq{alpha_max})$")
        # ordering legend
        if order:
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right')
        else:
            plt.legend()

        plt.show(fig)
        plt.close(fig)

    def plot_alpha_vs_n(self, beta_max, 
                        results_error_li=[],
                        results_error_point=[],
                        settings_dict=[],
                        order=[],
                        scale="linear"):
        """
            --beta_max -> float: beta<=beta_maxの条件下でalpha<=alpha_maxとなるのに必要な最小nを探す
            --results_error_li -> type:(alpha_li, beta_li)
            --results_error_point -> type:(alpha, beta)
            --settings_dict -> [{"color":"red", "linestyle":"--", "marker":"o", "label":"qcnn"}, ...]
            --order -> [0,1,4,2,3,...]
            --result_error_separable_SOP -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_separable_qcnn -> [(alpha, beta),...]: n=1,2,...
            --result_error_separable_exactqcnn -> [(alpha, beta),...]: n=1,2,...
            --result_error_separable_qcnnWqNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_separable_exactqcnnWqNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_separable_qNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_entangled_qNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_separable_CSqNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
        """
        fig = plt.figure()
        # 設定
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["font.size"] = 14
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # データ整形用関数
        def data_processing_li(result_error_li):
            alpha_max = 1
            alpha_li = []
            n_li = []
            for n in range(1,len(result_error_li)+1):
                result_arr = np.array(result_error_li[n-1])
                result_arr = result_arr[:, np.logical_and(result_arr[0]<=alpha_max,result_arr[1]<beta_max)]
                if result_arr.shape[1] != 0: #条件を満たす要素がある場合
                    alpha_li += result_arr[0].tolist()
                    n_li += [n]*result_arr.shape[1]
                    alpha_max = min(alpha_li)
            return alpha_li, n_li
        def data_processing_point(result_error_point):
            alpha_max = 1
            alpha_li = []
            n_li = []
            for n in range(1,len(result_error_point)+1):
                result_arr = result_error_point[n-1]
                if result_arr[0]<=alpha_max and result_arr[1]<beta_max: #条件を満たす要素がある場合
                    alpha_li.append(result_arr[0])
                    n_li.append(n)
                    alpha_max = min(alpha_li)
            return alpha_li, n_li
        # type: (alpha_li, beta_li)
        for i, result_error_li in enumerate(results_error_li):
            alpha_li,n_li = data_processing_li(result_error_li)
            plt.plot(alpha_li,n_li, **settings_dict[i])
        # type: (alpha, beta)
        for i, result_error_point in enumerate(results_error_point):
            alpha_li,n_li = data_processing_point(result_error_point)
            plt.plot(alpha_li,n_li, **settings_dict[i+len(results_error_li)])

        plt.xscale(scale)
        #plt.yticks(list(range(min(n_li), max(n_li)+1)))
        #plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel(r"$\alpha_n$")
        plt.ylabel(r"$n$")
        plt.title(r"$n$ vs. $\alpha_n$"+fr" $(\beta_n\leq{beta_max})$")
        # ordering legend
        if order:
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right')
        else:
            plt.legend()

        plt.show(fig)
        plt.close(fig)

    def plot_beta_vs_n(self, alpha_max, 
                       results_error_li=[],
                       results_error_point=[],
                       settings_dict=[],
                       order=[],
                       scale="linear"):
        """
            --alpha_max -> float: alpha<=alpha_maxの条件下でbeta<=beta_maxとなるのに必要な最小nを探す
            --results_error_li -> type:(alpha_li, beta_li)
            --results_error_point -> type:(alpha, beta)
            --settings_dict -> [{"color":"red", "linestyle":"--", "marker":"o", "label":"qcnn"}, ...]
            --order -> [0,1,4,2,3,...]
            --result_error_separable_SOP -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_separable_qcnn -> [(alpha, beta),...]: n=1,2,...
            --result_error_separable_exactqcnn -> [(alpha, beta),...]: n=1,2,...
            --result_error_separable_qcnnWqNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_separable_exactqcnnWqNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_separable_qNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_entangled_qNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
            --result_error_separable_CSqNeyman -> [(alpha_li, beta_li),...]: n=1,2,...
        """
        fig = plt.figure(figsize=(4.8,4.8))
        # 設定
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["font.size"] = 20
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # データ整形用関数
        def data_processing_li(result_error_li):
            beta_max = 1
            beta_li = []
            n_li = []
            for n in range(1,len(result_error_li)+1):
                result_arr = np.array(result_error_li[n-1])
                result_arr = result_arr[:, np.logical_and(result_arr[0]<=alpha_max,result_arr[1]<beta_max)]
                if result_arr.shape[1] != 0: #条件を満たす要素がある場合
                    beta_li += result_arr[1].tolist()
                    n_li += [n]*result_arr.shape[1]
                    beta_max = min(beta_li)
            return beta_li, n_li
        def data_processing_point(result_error_point):
            beta_max = 1
            beta_li = []
            n_li = []
            for n in range(1,len(result_error_point)+1):
                result_arr = result_error_point[n-1]
                if result_arr[0]<=alpha_max and result_arr[1]<beta_max: #条件を満たす要素がある場合
                    beta_li.append(result_arr[1])
                    n_li.append(n)
                    beta_max = min(beta_li)
            return beta_li, n_li
        # type: (alpha_li, beta_li)
        for i, result_error_li in enumerate(results_error_li):
            beta_li,n_li = data_processing_li(result_error_li)
            plt.plot(beta_li,n_li, **settings_dict[i])
        # type: (alpha, beta)
        for i, result_error_point in enumerate(results_error_point):
            beta_li,n_li = data_processing_point(result_error_point)
            plt.plot(beta_li,n_li, **settings_dict[i+len(results_error_li)])

        plt.xscale(scale)
        plt.xticks([0, 0.5, 1.0])
        plt.yticks(list(range(0, max(n_li)+1, 5)))
        #plt.yticks(list(range(min(n_li), max(n_li)+1)))
        #plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel(r"$\beta_n$")
        plt.ylabel(r"$n$")
        # plt.title(r"$n$ vs. $\beta_n$"+fr" $(\alpha_n\leq{alpha_max})$")
        # ordering legend
        # if order:
        #     handles, labels = plt.gca().get_legend_handles_labels()
        #     plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right')
        # else:
        #     plt.legend()

        plt.show(fig)
        plt.close(fig)

    def plot_shots_vs_n(self, alpha_max, beta_max,
                        shots_li=[],
                        results_error_li=[],
                        results_error_point=[],
                        settings_dict=[],
                        order=[],
                        scale="linear"):
        """
            --alpha_max -> float: alpha<=alpha_maxかつbeta<=beta_maxの条件を満たすために
            --beta_max  -> float: 必要な最小nを各shotごとに探す
            --shots_li -> [[320,600,...],[0,500,...], ...]
            --results_error_li -> type:(alpha_li, beta_li)
            --results_error_point -> type:(alpha, beta)
            --settings_dict -> [{"color":"red", "linestyle":"--", "marker":"o", "label":"qcnn"}, ...]
            --order -> [0,1,4,2,3,...]
            --result_error_separable_SOP -> [[(alpha_li, beta_li),...], ...]: n=1,2,...: shot=320,600,...
            --result_error_separable_qcnn -> [[(alpha, beta),...], ...]: n=1,2,...: shot=320,600,...
            --result_error_separable_exactqcnn -> [[(alpha, beta),...], ...]: n=1,2,...: shot=320,600,...
            --result_error_separable_qcnnWqNeyman -> [[(alpha_li, beta_li),...], ...]: n=1,2,...: shot=320,600,...
            --result_error_separable_exactqcnnWqNeyman -> [[(alpha_li, beta_li),...], ...]: n=1,2,...: shot=320,600,...
            --result_error_separable_qNeyman -> [[(alpha_li, beta_li),...], ...]: n=1,2,...: shot=320,600,...
            --result_error_entangled_qNeyman -> [[(alpha_li, beta_li),...], ...]: n=1,2,...: shot=320,600,...
            --result_error_separable_CSqNeyman -> [[(alpha_li, beta_li),...], ...]: n=1,2,...: shot=320,600,...
        """
        fig = plt.figure()
        # 設定
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["font.size"] = 14
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # データ整形用関数
        def data_processing_li(shots,results_error_li_shots):
            shots_new = []
            n_li = []
            for shot, result_error_li in zip(shots,results_error_li_shots):
                for n in range(1,len(result_error_li)+1):
                    result_arr = np.array(result_error_li[n-1])
                    result_arr = result_arr[:, np.logical_and(result_arr[0]<=alpha_max,result_arr[1]<=beta_max)]
                    if result_arr.shape[1] != 0: #条件を満たす要素がある場合
                        shots_new.append(shot)
                        n_li.append(n)
                        break
            return shots_new, n_li
        def data_processing_point(shots,results_error_point_shots):
            shots_new = []
            n_li = []
            for shot, result_error_point in zip(shots,results_error_point_shots):
                for n in range(1,len(result_error_point)+1):
                    result_arr = result_error_point[n-1]
                    if result_arr[0]<=alpha_max and result_arr[1]<=beta_max: #条件を満たす要素がある場合
                        shots_new.append(shot)
                        n_li.append(n)
                        break
            return shots_new, n_li
        # type: (alpha_li, beta_li)
        for i, results_error_li_shots in enumerate(results_error_li):
            shots, n_li = data_processing_li(shots_li[i], results_error_li_shots)
            plt.plot(shots,n_li, **settings_dict[i])
        # type: (alpha, beta)
        for i, results_error_point_shots in enumerate(results_error_point):
            shots, n_li = data_processing_point(shots_li[i+len(results_error_li)], results_error_point_shots)
            plt.plot(shots,n_li, **settings_dict[i+len(results_error_li)])

        plt.xscale(scale)
        #plt.yticks(list(range(min(n_li), max(n_li)+1)))
        #plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel("training shots")
        plt.ylabel(r"$n$")
        plt.title(r"$n$ vs. training shots"+fr" $(\alpha_n\leq{alpha_max}, \beta_n\leq{beta_max})$")
        # ordering legend
        if order:
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        else:
            plt.legend()

        plt.show(fig)
        plt.close(fig)

        