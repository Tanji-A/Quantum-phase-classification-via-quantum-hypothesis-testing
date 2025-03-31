import tensornetwork as tn
import tensornetwork.ncon_interface as ncon
from tensornetwork.linalg.node_linalg import conj
from tensornetwork.backends.abstract_backend import AbstractBackend

import numpy as np
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence, Set, Tuple, Iterable
Tensor = Any

__all__ = ['MyFiniteMPS']

class MyFiniteMPS(tn.FiniteMPS):
    def __init__(self,
                 tensors: List[Tensor],
                 center_position: Optional[int] = None,
                 canonicalize: Optional[bool] = True,
                 backend: Optional[Union[AbstractBackend, Text]] = None) -> None:
        """Initialize a `FiniteMPS`. If `canonicalize` is `True` the state
        is brought into canonical form, with `BaseMPS.center_position`
        at `center_position`. if `center_position` is `None` and
        `canonicalize = True`, `BaseMPS.center_position` is set to 0.

        Args:
            tensors: A list of `Tensor` objects.
            center_position: The initial position of the center site.
            canonicalize: If `True` the mps is canonicalized at initialization.
            backend: The name of the backend that should be used to perform
                contractions. Available backends are currently 'numpy', 'tensorflow',
                'pytorch', 'jax'
        """
        super().__init__(tensors=tensors,
                         center_position=center_position,
                         canonicalize=canonicalize,
                         backend=backend)
        
    @classmethod
    def from_statevector(cls,
                         statevector: np.ndarray,
                         max_singular_values: int = None,
                         max_truncation_err: int = None,
                         center_position: Optional[int] = None,
                         canonicalize: Optional[bool] = True,
                         backend: Optional[Union[AbstractBackend, Text]] = None):
        """Generate `MyFiniteMPS` from a statevector ndarray 
        that has a shape like (2**N,), (1,2**N), (2 for _ in range(N)). 

        Args:
            statevector: A statevector ndarray.
            max_singular_values: The maximum number of singular values to keep.
            max_truncation_err: The maximum allowed truncation error.
            center_position: The initial position of the center site.
            canonicalize: If `True` the mps is canonicalized at initialization.
            backend: The name of the backend that should be used to perform
                contractions. Available backends are currently 'numpy', 'tensorflow',
                'pytorch', 'jax'
        Returns:
            `MyFiniteMPS`
        """    
        statevector = statevector.flatten()
        N = int(np.log2(statevector.shape[0]))
        assert 2**N == statevector.shape[0]

        tensors = []
        for i in range(N-1):
            if i == 0: 
                vh = tn.Node(statevector.reshape(1,*[2 for _ in range(N)],1))
            else:
                vh = vh_prime
            u_prime, vh_prime, truncation_err = tn.split_node(vh, left_edges=vh.edges[:2], right_edges=vh.edges[2:],
                                                              max_singular_values=max_singular_values,
                                                              max_truncation_err=max_truncation_err)
            tensors.append(u_prime)
        tensors.append(vh_prime)
        tensors = [node.tensor for node in tensors]

        return cls(tensors=tensors,
                   center_position=center_position,
                   canonicalize=canonicalize,
                   backend=backend)
    
    def to_statevector(self) -> np.ndarray:
        """From `MyFiniteMPS` to a statevector ndarray that shape is (2**N,).

        Returns:
            a statevector ndarray (2**N,)
        """    
        N = len(self)
        # connect the edges in the mps and contract over bond dimensions
        nodes = [tn.Node(tensor) for tensor in self.tensors]
        connected_bonds = [nodes[k].edges[2] ^ nodes[k+1].edges[0] for k in reversed(range(-1,N-1))]
        for x in connected_bonds:
            contracted_node = tn.contract(x) # update for each contracted bond
        return contracted_node.tensor.flatten()
    
    def apply_distant_two_site_gate(self,
                                    gate: Tensor,
                                    site1: int,
                                    site2: int,
                                    max_singular_values: Optional[int] = None,
                                    max_truncation_err: Optional[float] = None,
                                    center_position: Optional[int] = None,
                                    relative: bool = False) -> List[Tensor]:
        """Apply a 'distant' two-site gate to an MPS. This routine will in general destroy
        any canonical form of the state. If a canonical form is needed, the user
        can restore it using `FiniteMPS.position`.

        Args:
            gate: A two-body gate.
            site1: The first site where the gate acts.
            site2: The second site where the gate acts.
            max_singular_values: The maximum number of singular values to keep.
            max_truncation_err: The maximum allowed truncation error.
            center_position: An optional value to choose the MPS tensor at
                `center_position` to be isometric after the application of the gate.
                Defaults to `site1`. If the MPS is canonical (i.e.
                `BaseMPS.center_position != None`), and if the orthogonality center
                coincides with either `site1` or `site2`,  the orthogonality center will
                be shifted to `center_position` (`site1` by default). If the
                orthogonality center does not coincide with `(site1, site2)` then
                `MPS.center_position` is set to `None`.
            relative: Multiply `max_truncation_err` with the largest singular value.

        Warnings:
            center_position != self.center_position: It's recommended that center_position is set equal to self.center_position.
        Returns:
            List[`Tensor`]: A list of tensors containing the truncated weight of the truncation.
        """
        assert site1<site2 and site1>=0 and site2<=len(self)-1
        assert center_position is None or center_position in (site1, site2)
        use_svd = (max_truncation_err is not None) or (max_singular_values is not None)
        if use_svd:
            assert self.center_position in (site1, site2)
            if center_position is None:
                center_position = self.center_position
            if center_position != self.center_position:
                print("Warning: recommend center_position==self.center_position! {}!={}".format(center_position,self.center_position))
        else:
            if center_position is None:
                center_position = site1  

        swap_tensor = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]).reshape(2,2,2,2)
        tw_li = []
        if (use_svd and self.center_position == site1) or (not use_svd and center_position == site1):
            # swapでsite1をsite2-1まで移動
            for i in range(site1, site2-1):
                tw = self.apply_two_site_gate(swap_tensor,i,i+1,None,None,i+1,relative)
                tw_li.append(tw)
            # gateを演算
            tw = self.apply_two_site_gate(gate,site2-1,site2,max_singular_values,max_truncation_err,site2-1,relative)
            tw_li.append(tw)
            # swapでsite2-1をsite1まで戻す
            for i in reversed(range(site1, site2-1)):
                tw = self.apply_two_site_gate(swap_tensor,i,i+1,None,None,i,relative)   
                tw_li.append(tw)
            # このifに引っかかる場合, Warningが出るはず
            if center_position != site1:
                self.position(center_position, False)
        else:
            # swapでsite2をsite1+1まで移動
            for i in reversed(range(site1+1, site2)):
                tw = self.apply_two_site_gate(swap_tensor,i,i+1,None,None,i,relative)
                tw_li.append(tw)
            # gateを演算
            tw = self.apply_two_site_gate(gate,site1,site1+1,max_singular_values,max_truncation_err,site1+1,relative)
            tw_li.append(tw)
            # swapでsite1+1をsite2まで戻す
            for i in range(site1+1, site2):
                tw = self.apply_two_site_gate(swap_tensor,i,i+1,None,None,i+1,relative)   
                tw_li.append(tw)
            # このifに引っかかる場合, Warningが出るはず
            if center_position != site2:
                self.position(center_position, False)
        return tw_li
    
    def apply_three_site_gate(self,
                              gate: Tensor,
                              site1: int,
                              site2: int,
                              site3: int,
                              max_singular_values: Optional[int] = None,
                              max_truncation_err: Optional[float] = None,
                              center_position: Optional[int] = None,
                              relative: bool = False) -> List[Tensor]:
        """Apply a (nearest neighbor) three-site gate to an MPS. This routine will in general destroy
        any canonical form of the state. If a canonical form is needed, the user
        can restore it using `FiniteMPS.position`.

        Args:
            gate: A three-body gate.
            site1: The first site where the gate acts.
            site2: The second site where the gate acts.
            site3: The third site where the gate acts.
            max_singular_values: The maximum number of singular values to keep.
            max_truncation_err: The maximum allowed truncation error.
            center_position: An optional value to choose the MPS tensor at
                `center_position` to be isometric after the application of the gate.
                Defaults to `site1`. If the MPS is canonical (i.e.
                `BaseMPS.center_position != None`), and if the orthogonality center
                coincides with either `site1` or `site2`,  the orthogonality center will
                be shifted to `center_position` (`site1` by default). If the
                orthogonality center does not coincide with `(site1, site2)` then
                `MPS.center_position` is set to `None`.
            relative: Multiply `max_truncation_err` with the largest singular value.

        Returns:
            List[`Tensor`]: A list of tensors containing the truncated weight of the truncation.
        """

        if len(gate.shape) != 6:
            raise ValueError('rank of gate is {} but has to be 4'.format(len(gate.shape)))
        if site1 < 0 or site1 >= len(self) - 2:
            raise ValueError('site1 = {} is not between 0 <= site < N - 2 = {}'.format(site1, len(self)-2))
        if site2 < 1 or site2 >= len(self) - 1:
            raise ValueError('site2 = {} is not between 1 <= site < N - 1 = {}'.format(site2, len(self)-1))
        if site3 < 2 or site3 >= len(self):
            raise ValueError('site3 = {} is not between 2 <= site < N = {}'.format(site3, len(self)))
        if site2 <= site1 or site3 <= site2:
            raise ValueError('you have to set site1 = {} < site2 = {} < site3 = {}'.format(site1, site2, site3))
        if site2 != site1 + 1 or site3 != site2 + 1:
            raise ValueError("Found site3={}, site2 ={}, site1={}. Only nearest neighbor gates are currently supported".format(site3, site2, site1))

        if center_position is not None and center_position not in (site1, site2, site3):
            raise ValueError(f"center_position = {center_position} not in {(site1, site2, site3)} ")

        if (max_singular_values or max_truncation_err) and self.center_position not in (site1, site2, site3):
            raise ValueError('''center_position = {}, but gate is applied at sites {}, {}, {}. 
                                Truncation should only be done if the gate 
                                is applied at the center position of the MPS'''.format(self.center_position, site1, site2, site3))

        use_svd = (max_truncation_err is not None) or (max_singular_values is not None)
        gate = self.backend.convert_to_tensor(gate)
        tensor = ncon.ncon([self.tensors[site1], self.tensors[site2], self.tensors[site3], gate],
                        [[-1, 1, 2], [2, 3, 4], [4, 5, -5], [-2, -3, -4, 1, 3, 5]],
                        backend=self.backend)

        def set_center_position(site):
            if self.center_position is not None:
                    if self.center_position in (site1, site2, site3):
                        assert site in (site1, site2, site3)
                        self.center_position = site
                    else:
                        self.center_position = None

        if center_position is None:
            center_position = site1

        if use_svd:
            tw_li = []
            if center_position == site3 or center_position == site2:
                U, S, V, tw = self.backend.svd(tensor,
                                            pivot_axis=2,
                                            max_singular_values=max_singular_values,
                                            max_truncation_error=max_truncation_err,
                                            relative=relative)
                tw_li.append(tw)
                left_tensor = U
                right_tensor = ncon.ncon([self.backend.diagflat(S), V],
                                        [[-1, 1], [1, -2, -3, -4]],
                                        backend=self.backend)
                U, S, V, tw = self.backend.svd(right_tensor,
                                            pivot_axis=2,
                                            max_singular_values=max_singular_values,
                                            max_truncation_error=max_truncation_err,
                                            relative=relative)
                tw_li.append(tw)
                if center_position == site3:
                    center_tensor = U
                    right_tensor = ncon.ncon([self.backend.diagflat(S), V],
                                            [[-1, 1], [1, -2, -3]],
                                            backend=self.backend)
                    set_center_position(site3)
                else:
                    center_tensor = ncon.ncon([U, self.backend.diagflat(S)],
                                            [[-1, -2, 1], [1, -3]],
                                            backend=self.backend)
                    right_tensor = V
                    set_center_position(site2)
            else:
                U, S, V, tw = self.backend.svd(tensor,
                                            pivot_axis=3,
                                            max_singular_values=max_singular_values,
                                            max_truncation_error=max_truncation_err,
                                            relative=relative)
                tw_li.append(tw)
                left_tensor = ncon.ncon([U, self.backend.diagflat(S)],
                                        [[-1, -2, -3, 1], [1, -4]],
                                        backend=self.backend)
                right_tensor = V
                U, S, V, tw = self.backend.svd(left_tensor,
                                            pivot_axis=2,
                                            max_singular_values=max_singular_values,
                                            max_truncation_error=max_truncation_err,
                                            relative=relative)
                tw_li.append(tw)
                left_tensor = ncon.ncon([U, self.backend.diagflat(S)],
                                        [[-1, -2, 1], [1, -3]],
                                        backend=self.backend)
                center_tensor = V
                set_center_position(site1)

        else:
            tw = self.backend.zeros(1, dtype=self.dtype)
            tw_li = [tw, tw]
            if center_position == site3 or center_position == site2:
                R, Q = self.backend.rq(tensor, pivot_axis=2)
                left_tensor = R
                right_tensor = Q
                if center_position == site3:
                    R, Q = self.backend.rq(right_tensor, pivot_axis=2)
                    center_tensor = R
                    right_tensor = Q
                    set_center_position(site3)
                else:
                    Q, R = self.backend.qr(right_tensor, pivot_axis=2)
                    center_tensor = Q
                    right_tensor = R
                    set_center_position(site2)
            else:
                Q, R = self.backend.qr(tensor, pivot_axis=3)
                left_tensor = Q
                right_tensor = R
                Q, R = self.backend.qr(left_tensor, pivot_axis=2)
                left_tensor = Q
                center_tensor = R            
                set_center_position(site1)

        self.tensors[site1] = left_tensor
        self.tensors[site2] = center_tensor
        self.tensors[site3] = right_tensor
        return tw_li
    

    def apply_four_site_gate(self,
                             gate: Tensor,
                             site1: int,
                             site2: int,
                             site3: int,
                             site4: int,
                             max_singular_values: Optional[int] = None,
                             max_truncation_err: Optional[float] = None,
                             center_position: Optional[int] = None,
                             relative: bool = False) -> List[Tensor]:
        """Apply a (nearest neighbor) four-site gate to an MPS. This routine will in general destroy
        any canonical form of the state. If a canonical form is needed, the user
        can restore it using `FiniteMPS.position`.

        Args:
            gate: A four-body gate.
            site1: The first site where the gate acts.
            site2: The second site where the gate acts.
            site3: The third site where the gate acts.
            site4: The fourth site where the gate acts.
            max_singular_values: The maximum number of singular values to keep.
            max_truncation_err: The maximum allowed truncation error.
            center_position: An optional value to choose the MPS tensor at
                `center_position` to be isometric after the application of the gate.
                Defaults to `site1`. If the MPS is canonical (i.e.
                `BaseMPS.center_position != None`), and if the orthogonality center
                coincides with either `site1` or `site2`,  the orthogonality center will
                be shifted to `center_position` (`site1` by default). If the
                orthogonality center does not coincide with `(site1, site2)` then
                `MPS.center_position` is set to `None`.
            relative: Multiply `max_truncation_err` with the largest singular value.

        Returns:
            List[`Tensor`]: A list of tensors containing the truncated weight of the truncation.
        """

        if len(gate.shape) != 8:
            raise ValueError('rank of gate is {} but has to be 4'.format(len(gate.shape)))
        if site1 < 0 or site1 >= len(self) - 3:
            raise ValueError('site1 = {} is not between 0 <= site < N - 3 = {}'.format(site1, len(self)-3))
        if site2 < 1 or site2 >= len(self) - 2:
            raise ValueError('site2 = {} is not between 1 <= site < N - 2 = {}'.format(site2, len(self)-2))
        if site3 < 2 or site3 >= len(self) - 1:
            raise ValueError('site3 = {} is not between 2 <= site < N - 1 = {}'.format(site3, len(self)-1))
        if site4 < 3 or site4 >= len(self):
            raise ValueError('site4 = {} is not between 3 <= site < N = {}'.format(site4, len(self)))
        if site2 <= site1 or site3 <= site2 or site4 <= site3:
            raise ValueError('you have to set site1 = {} < site2 = {} < site3 = {} < site4 = {}'.format(site1, site2, site3, site4))
        if site2 != site1 + 1 or site3 != site2 + 1 or site4 != site3 + 1:
            raise ValueError("Found site4={}, site3={}, site2 ={}, site1={}. Only nearest neighbor gates are currently supported".format(site4, site3, site2, site1))

        if center_position is not None and center_position not in (site1, site2, site3, site4):
            raise ValueError(f"center_position = {center_position} not in {(site1, site2, site3, site4)} ")

        if (max_singular_values or max_truncation_err) and self.center_position not in (site1, site2, site3, site4):
            raise ValueError('''center_position = {}, but gate is applied at sites {}, {}, {}. 
                                Truncation should only be done if the gate 
                                is applied at the center position of the MPS'''.format(self.center_position, site1, site2, site3, site4))

        use_svd = (max_truncation_err is not None) or (max_singular_values is not None)
        gate = self.backend.convert_to_tensor(gate)
        tensor = ncon.ncon([self.tensors[site1], self.tensors[site2], self.tensors[site3], self.tensors[site4], gate],
                        [[-1, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, -6], [-2, -3, -4, -5, 1, 3, 5, 7]],
                        backend=self.backend)

        def set_center_position(site):
            if self.center_position is not None:
                    if self.center_position in (site1, site2, site3, site4):
                        assert site in (site1, site2, site3, site4)
                        self.center_position = site
                    else:
                        self.center_position = None

        if center_position is None:
            center_position = site1

        if use_svd:
            tw_li = []
            if center_position == site4 or center_position == site3:
                U, S, V, tw = self.backend.svd(tensor,
                                            pivot_axis=2,
                                            max_singular_values=max_singular_values,
                                            max_truncation_error=max_truncation_err,
                                            relative=relative)
                tw_li.append(tw)
                lleft_tensor = U
                right_tensor = ncon.ncon([self.backend.diagflat(S), V],
                                        [[-1, 1], [1, -2, -3, -4, -5]],
                                        backend=self.backend)
                U, S, V, tw = self.backend.svd(right_tensor,
                                            pivot_axis=2,
                                            max_singular_values=max_singular_values,
                                            max_truncation_error=max_truncation_err,
                                            relative=relative)
                tw_li.append(tw)
                left_tensor = U
                right_tensor = ncon.ncon([self.backend.diagflat(S), V],
                                        [[-1, 1], [1, -2, -3, -4]],
                                        backend=self.backend)
                U, S, V, tw = self.backend.svd(right_tensor,
                                                pivot_axis=2,
                                                max_singular_values=max_singular_values,
                                                max_truncation_error=max_truncation_err,
                                                relative=relative)
                tw_li.append(tw)
                if center_position == site4:
                    right_tensor = U
                    rright_tensor = ncon.ncon([self.backend.diagflat(S), V],
                                            [[-1, 1], [1, -2, -3]],
                                            backend=self.backend)
                    set_center_position(site4)
                else:
                    right_tensor = ncon.ncon([U, self.backend.diagflat(S)],
                                            [[-1, -2, 1], [1, -3]],
                                            backend=self.backend)
                    rright_tensor = V
                    set_center_position(site3)
            else:
                U, S, V, tw = self.backend.svd(tensor,
                                            pivot_axis=4,
                                            max_singular_values=max_singular_values,
                                            max_truncation_error=max_truncation_err,
                                            relative=relative)
                tw_li.append(tw)
                left_tensor = ncon.ncon([U, self.backend.diagflat(S)],
                                        [[-1, -2, -3, -4, 1], [1, -5]],
                                        backend=self.backend)
                rright_tensor = V
                U, S, V, tw = self.backend.svd(left_tensor,
                                            pivot_axis=3,
                                            max_singular_values=max_singular_values,
                                            max_truncation_error=max_truncation_err,
                                            relative=relative)
                tw_li.append(tw)
                left_tensor = ncon.ncon([U, self.backend.diagflat(S)],
                                        [[-1, -2, -3, 1], [1, -4]],
                                        backend=self.backend)
                right_tensor = V
                U, S, V, tw = self.backend.svd(left_tensor,
                                            pivot_axis=2,
                                            max_singular_values=max_singular_values,
                                            max_truncation_error=max_truncation_err,
                                            relative=relative) 
                tw_li.append(tw)           
                if center_position == site2:
                    lleft_tensor = U
                    left_tensor = ncon.ncon([self.backend.diagflat(S), V],
                                            [[-1, 1], [1, -2,-3]],
                                            backend=self.backend)
                    set_center_position(site2)
                else:
                    lleft_tensor = ncon.ncon([U, self.backend.diagflat(S)],
                                            [[-1, -2, 1], [1, -3]],
                                            backend=self.backend)
                    left_tensor = V
                    set_center_position(site1)

        else:
            tw = self.backend.zeros(1, dtype=self.dtype)
            tw_li = [tw, tw, tw]
            if center_position == site4 or center_position == site3:
                R, Q = self.backend.rq(tensor, pivot_axis=2)
                lleft_tensor = R
                right_tensor = Q
                R, Q = self.backend.rq(right_tensor, pivot_axis=2)
                left_tensor = R
                right_tensor = Q
                if center_position == site4:
                    R, Q = self.backend.rq(right_tensor, pivot_axis=2)
                    right_tensor = R
                    rright_tensor = Q
                    set_center_position(site4)
                else:
                    Q, R = self.backend.qr(right_tensor, pivot_axis=2)
                    right_tensor = Q
                    rright_tensor = R
                    set_center_position(site3)
            else:
                Q, R = self.backend.qr(tensor, pivot_axis=4)
                left_tensor = Q
                rright_tensor = R
                Q, R = self.backend.qr(left_tensor, pivot_axis=3)
                left_tensor = Q
                right_tensor = R
                if center_position == site2:
                    R, Q = self.backend.rq(left_tensor, pivot_axis=2)
                    lleft_tensor = R
                    left_tensor = Q
                    set_center_position(site2)
                else:
                    Q, R = self.backend.qr(left_tensor, pivot_axis=2)
                    lleft_tensor = Q
                    left_tensor = R
                    set_center_position(site1)

        self.tensors[site1] = lleft_tensor
        self.tensors[site2] = left_tensor
        self.tensors[site3] = right_tensor
        self.tensors[site4] = rright_tensor
        return tw_li

    def measure_two_site_observable(self, op: Tensor, site1: int, site2: int) -> float:
        if len(op.shape) != 4:
            raise ValueError('rank of gate is {} but has to be 4'.format(len(op.shape)))
        if site2 <= site1:
            raise ValueError('you have to set site1 = {} < site2 = {}'.format(site1, site2))
        if site1 < 0 or site2 >= len(self):
            raise ValueError('you have to set 0 <= all sites < N = {}'.format(len(self)))

        op = tn.Node(op)
        mps_nodes = [tn.Node(tensor) for tensor in self.tensors]
        mps_conj_nodes = [tn.Node(tn.conj(tensor)) for tensor in self.tensors]
        for node1, node2 in zip(mps_nodes[:-1], mps_nodes[1:]):
            node1[2] ^ node2[0]
        for node1, node2 in zip(mps_conj_nodes[:-1], mps_conj_nodes[1:]):
            node1[2] ^ node2[0]
        mps_nodes[-1][2] ^ mps_conj_nodes[-1][2]
        mps_nodes[0][0] ^ mps_conj_nodes[0][0]
        for i, (node1, node2) in enumerate(zip(mps_nodes, mps_conj_nodes)):
            if i not in (site1, site2):
                node1[1] ^ node2[1]
        op[0] ^ mps_conj_nodes[site1][1]
        op[1] ^ mps_conj_nodes[site2][1]
        op[2] ^ mps_nodes[site1][1]
        op[3] ^ mps_nodes[site2][1]
        result = tn.contractors.auto(mps_nodes+mps_conj_nodes+[op])
        return self.backend.item(result.tensor)
        
    def measure_three_site_observable(self, op: Tensor, site1: int, site2: int, site3: int) -> float:
        if len(op.shape) != 6:
            raise ValueError('rank of gate is {} but has to be 6'.format(len(op.shape)))
        if site2 <= site1 or site3 <= site2:
            raise ValueError('you have to set site1 = {} < site2 = {} < site3 = {}'.format(site1, site2, site3))
        if site1 < 0 or site3 >= len(self):
            raise ValueError('you have to set 0 <= all sites < N = {}'.format(len(self)))

        op = tn.Node(np.array(op, dtype=self.dtype))
        mps_nodes = [tn.Node(tensor) for tensor in self.tensors]
        mps_conj_nodes = [tn.Node(tn.conj(tensor)) for tensor in self.tensors]
        for node1, node2 in zip(mps_nodes[:-1], mps_nodes[1:]):
            node1[2] ^ node2[0]
        for node1, node2 in zip(mps_conj_nodes[:-1], mps_conj_nodes[1:]):
            node1[2] ^ node2[0]
        mps_nodes[-1][2] ^ mps_conj_nodes[-1][2]
        mps_nodes[0][0] ^ mps_conj_nodes[0][0]
        for i, (node1, node2) in enumerate(zip(mps_nodes, mps_conj_nodes)):
            if i not in (site1, site2, site3):
                node1[1] ^ node2[1]
        op[0] ^ mps_conj_nodes[site1][1]
        op[1] ^ mps_conj_nodes[site2][1]
        op[2] ^ mps_conj_nodes[site3][1]
        op[3] ^ mps_nodes[site1][1]
        op[4] ^ mps_nodes[site2][1]
        op[5] ^ mps_nodes[site3][1]
        result = tn.contractors.auto(mps_nodes+mps_conj_nodes+[op])
        return self.backend.item(result.tensor)
    
    def measure_four_site_observable(self, op: Tensor, site1: int, site2: int, site3: int, site4: int) -> float:
        if len(op.shape) != 8:
            raise ValueError('rank of gate is {} but has to be 8'.format(len(op.shape)))
        if site2 <= site1 or site3 <= site2 or site4 <= site3:
            raise ValueError('you have to set site1 = {} < site2 = {} < site3 = {} < site4 = {}'.format(site1, site2, site3, site4))
        if site1 < 0 or site4 >= len(self):
            raise ValueError('you have to set 0 <= all sites < N = {}'.format(len(self)))

        op = tn.Node(np.array(op, dtype=self.dtype))
        mps_nodes = [tn.Node(tensor) for tensor in self.tensors]
        mps_conj_nodes = [tn.Node(tn.conj(tensor)) for tensor in self.tensors]
        for node1, node2 in zip(mps_nodes[:-1], mps_nodes[1:]):
            node1[2] ^ node2[0]
        for node1, node2 in zip(mps_conj_nodes[:-1], mps_conj_nodes[1:]):
            node1[2] ^ node2[0]
        mps_nodes[-1][2] ^ mps_conj_nodes[-1][2]
        mps_nodes[0][0] ^ mps_conj_nodes[0][0]
        for i, (node1, node2) in enumerate(zip(mps_nodes, mps_conj_nodes)):
            if i not in (site1, site2, site3, site4):
                node1[1] ^ node2[1]
        op[0] ^ mps_conj_nodes[site1][1]
        op[1] ^ mps_conj_nodes[site2][1]
        op[2] ^ mps_conj_nodes[site3][1]
        op[3] ^ mps_conj_nodes[site4][1]
        op[4] ^ mps_nodes[site1][1]
        op[5] ^ mps_nodes[site2][1]
        op[6] ^ mps_nodes[site3][1]
        op[7] ^ mps_nodes[site4][1]
        result = tn.contractors.auto(mps_nodes+mps_conj_nodes+[op])
        return self.backend.item(result.tensor)
    
    def measure_six_site_observable(self, op: Tensor, site1: int, site2: int, site3: int, site4: int, site5: int, site6: int) -> float:
        if len(op.shape) != 12:
            raise ValueError('rank of gate is {} but has to be 12'.format(len(op.shape)))
        if site2 <= site1 or site3 <= site2 or site4 <= site3 or site5 <= site4 or site6 <= site5:
            raise ValueError('you have to set site1 = {} < site2 = {} < site3 = {} < site4 = {} < site5 = {} < site6 = {}'.format(site1, site2, site3, site4, site5, site6))
        if site1 < 0 or site6 >= len(self):
            raise ValueError('you have to set 0 <= all sites < N = {}'.format(len(self)))
        
        op = tn.Node(np.array(op, dtype=self.dtype))
        mps_nodes = [tn.Node(tensor) for tensor in self.tensors]
        mps_conj_nodes = [tn.Node(tn.conj(tensor)) for tensor in self.tensors]
        for node1, node2 in zip(mps_nodes[:-1], mps_nodes[1:]):
            node1[2] ^ node2[0]
        for node1, node2 in zip(mps_conj_nodes[:-1], mps_conj_nodes[1:]):
            node1[2] ^ node2[0]
        mps_nodes[-1][2] ^ mps_conj_nodes[-1][2]
        mps_nodes[0][0] ^ mps_conj_nodes[0][0]
        for i, (node1, node2) in enumerate(zip(mps_nodes, mps_conj_nodes)):
            if i not in (site1, site2, site3, site4, site5, site6):
                node1[1] ^ node2[1]
        op[0] ^ mps_conj_nodes[site1][1]
        op[1] ^ mps_conj_nodes[site2][1]
        op[2] ^ mps_conj_nodes[site3][1]
        op[3] ^ mps_conj_nodes[site4][1]
        op[4] ^ mps_conj_nodes[site5][1]
        op[5] ^ mps_conj_nodes[site6][1]
        op[6] ^ mps_nodes[site1][1]
        op[7] ^ mps_nodes[site2][1]
        op[8] ^ mps_nodes[site3][1]
        op[9] ^ mps_nodes[site4][1]
        op[10] ^ mps_nodes[site5][1]
        op[11] ^ mps_nodes[site6][1]
        result = tn.contractors.auto(mps_nodes+mps_conj_nodes+[op])
        return self.backend.item(result.tensor)