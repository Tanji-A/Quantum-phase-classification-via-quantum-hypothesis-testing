import tensornetwork as tn
from tensornetwork.matrixproductstates.mpo import FiniteMPO
from tensornetwork.backends.abstract_backend import AbstractBackend

import numpy as np
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence, Set, Tuple, Iterable
Tensor = Any

__all__ = ['FiniteZXZ', 'FinitePollmann', 'FiniteTLFI', 'FiniteSu']

class FiniteZXZ(FiniteMPO):
  """
    H = sum - J ZXZ - h1 X - h2 XX
  """
  def __init__(self,
               J: np.ndarray,
               h1: np.ndarray,
               h2: np.ndarray,
               dtype: Type[np.number],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = 'ZXZ_MPO') -> None:
    """
    Returns the MPO of the finite ZXZ model (Lukin's QCNN paper).
    Args:
      J:   Z*X*Z
      h1:  X Magnetic field on each lattice site.
      h2:  The X*X coupling strength between nearest neighbor lattice sites.
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.
    Returns:
      FiniteZXZ: The mpo of the finite ZXZ model.
    """
    self.J = J.astype(dtype)
    self.h1 = h1.astype(dtype)
    self.h2 = h2.astype(dtype)
    N = len(h1)
    sigma_i = np.diag([1, 1]).astype(dtype)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([1, -1]).astype(dtype)
    mpo = []
    temp = np.zeros(shape=[1, 5, 2, 2], dtype=dtype)
    #sigma_x
    temp[0, 0, :, :] = -1 * self.h1[0] * sigma_x
    #sigma_x
    temp[0, 1, :, :] = -1 * self.h2[0] * sigma_x
    #sigma_z
    temp[0, 3, :, :] = sigma_z
    #11
    temp[0, 4, :, :] =sigma_i
    mpo.append(temp)
    for n in range(1, N - 1):
      temp = np.zeros(shape=[5, 5, 2, 2], dtype=dtype)
      #11
      temp[0, 0, :, :] = sigma_i
      #sigma_x
      temp[1, 0, :, :] = sigma_x
      #Bsigma_z
      temp[2, 0, :, :] = sigma_z
      #sigma_x
      temp[4, 0, :, :] = -1 * self.h1[n] * sigma_x
      #sigma_x
      temp[4, 1, :, :] = -1 * self.h2[n] * sigma_x
      #sigma_x
      temp[3, 2, :, :] = -1 * self.J[n]  * sigma_x
      #sigma_z
      temp[4, 3, :, :] = sigma_z
      #11
      temp[4, 4, :, :] = sigma_i
      mpo.append(temp)

    temp = np.zeros([5, 1, 2, 2], dtype=dtype)
    #11
    temp[0, 0, :, :] = sigma_i
    #sigma_x
    temp[1, 0, :, :] = sigma_x
    #Bsigma_z
    temp[2, 0, :, :] = sigma_z
    #sigma_x
    temp[4, 0, :, :] = -1 * self.h1[n+1] * sigma_x
    mpo.append(temp)
    super().__init__(tensors=mpo, backend=backend, name=name)


class FinitePollmann(FiniteMPO):
  """
    H = sum J X - J1 ZZ - J2 ZXZ
  """
  def __init__(self,
               J: np.ndarray,
               J1: np.ndarray,
               J2: np.ndarray,
               dtype: Type[np.number],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = 'Pollmann_MPO') -> None:
    """
    Returns the MPO of the finite cluster model (Pollmann's paper).
    Args:
      J:   X
      J1:  The Z*Z coupling strength between nearest neighbor lattice sites.
      J2:  Z*X*Z coupling
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.
    Returns:
      FinitePollmann: The mpo of the finite cluster model.
    """
    self.J = J.astype(dtype)
    self.J1 = J1.astype(dtype)
    self.J2 = J2.astype(dtype)
    N = len(J1)
    sigma_i = np.diag([1, 1]).astype(dtype)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([1, -1]).astype(dtype)
    mpo = []
    temp = np.zeros(shape=[1, 4, 2, 2], dtype=dtype)
    #sigma_x
    temp[0, 0, :, :] = self.J[0] * sigma_x
    #sigma_z
    temp[0, 1, :, :] = -1 * self.J1[0] * sigma_z
    #sigma_z
    temp[0, 2, :, :] = sigma_z
    #11
    temp[0, 3, :, :] =sigma_i
    mpo.append(temp)
    for n in range(1, N - 1):
      temp = np.zeros(shape=[4, 4, 2, 2], dtype=dtype)
      #11
      temp[0, 0, :, :] = sigma_i
      #sigma_z
      temp[1, 0, :, :] = sigma_z
      #Bsigma_x
      temp[3, 0, :, :] = self.J[n] * sigma_x
      #sigma_x
      temp[2, 1, :, :] = -1 * self.J2[n] * sigma_x
      #sigma_z
      temp[3, 1, :, :] = -1 * self.J1[n] * sigma_z
      #sigma_z
      temp[3, 2, :, :] = sigma_z
      #11
      temp[3, 3, :, :] = sigma_i
      mpo.append(temp)

    temp = np.zeros([4, 1, 2, 2], dtype=dtype)
    #11
    temp[0, 0, :, :] = sigma_i
    #sigma_z
    temp[1, 0, :, :] = sigma_z
    #sigma_x
    temp[3, 0, :, :] = self.J[n+1] * sigma_x
    mpo.append(temp)
    super().__init__(tensors=mpo, backend=backend, name=name)


class FiniteTLFI(FiniteMPO):
  """
    H = sum - J1 ZZ - J2 X - J3 Z
  """

  def __init__(self,
               J1: np.ndarray,
               J2: np.ndarray,
               J3: np.ndarray,
               dtype: Type[np.number],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = 'TLFI_MPO') -> None:
    """
    Returns the MPO of the finite ising model (Luo's paper).
    Args:
      J1:  Z*Z
      J2:  X
      J3:  Z
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.
    Returns:
      FiniteTLFI: The mpo of the finite transverse ising model with a longitudial field.
    """
    self.J1 = J1.astype(dtype)
    self.J2 = J2.astype(dtype)
    self.J3 = J3.astype(dtype)
    N = len(J2)
    sigma_i = np.diag([1, 1]).astype(dtype)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_z = np.diag([1, -1]).astype(dtype)
    mpo = []
    temp = np.zeros(shape=[1, 3, 2, 2], dtype=dtype)
    #sigma_x + sigma_z
    temp[0, 0, :, :] = (-1*self.J2[0]*sigma_x) + (-1*self.J3[0]*sigma_z) 
    #sigma_x
    temp[0, 1, :, :] = -1*self.J1[0]*sigma_z
    #11
    temp[0, 2, :, :] = sigma_i
    mpo.append(temp)
    for n in range(1, N - 1):
      temp = np.zeros(shape=[3, 3, 2, 2], dtype=dtype)
      #11
      temp[0, 0, :, :] = sigma_i
      #sigma_z
      temp[1, 0, :, :] = sigma_z
      #sigma_x + sigma_z
      temp[2, 0, :, :] = (-1*self.J2[n]*sigma_x) + (-1*self.J3[n]*sigma_z) 
      #sigma_x
      temp[2, 1, :, :] = -1*self.J1[n]*sigma_z
      #11
      temp[2, 2, :, :] = sigma_i
      mpo.append(temp)

    temp = np.zeros([3, 1, 2, 2], dtype=dtype)
    #11
    temp[0, 0, :, :] = sigma_i
    #sigma_z
    temp[1, 0, :, :] = sigma_z
    #sigma_x + sigma_z
    temp[2, 0, :, :] = (-1*self.J2[n+1]*sigma_x) + (-1*self.J3[n+1]*sigma_z) 
    mpo.append(temp)
    super().__init__(tensors=mpo, backend=backend, name=name)

class FiniteSu(FiniteMPO):
  """
    H = sum_{i}^{N/2} J (XX+YY+ZZ)_{2i-1,2i} + J1 ZZ_{2i,2i+1} + J2 (XY-YX)_{2i-1,2i} + J2 (XY-YX)_{2i,2i+1}
  """

  def __init__(self,
               J: np.ndarray,
               J1: np.ndarray,
               J2: np.ndarray,
               dtype: Type[np.number],
               backend: Optional[Union[AbstractBackend, Text]] = None,
               name: Text = 'Su_MPO') -> None:
    """
    Returns the MPO of the spin-1/2 HIAC with an uniform DM interaction (Su's paper).
    Args:
      J:   XX+YY+ZZ
      J1:  ZZ
      J2:  XY-YX
      dtype: The dtype of the MPO.
      backend: An optional backend.
      name: A name for the MPO.
    Returns:
      FinitePollmann: The mpo of the finite HIAC model.
    """
    self.J = J.astype(dtype)
    self.J1 = J1.astype(dtype)
    self.J2 = J2.astype(dtype)
    N = len(J1)
    assert N%2 == 1
    sigma_i = np.diag([1, 1]).astype(dtype)
    sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
    sigma_y = np.array([[0, -1j], [1j, 0]]).astype(dtype)
    sigma_z = np.diag([1, -1]).astype(dtype)
    mpo = []
    temp = np.zeros(shape=[1, 5, 2, 2], dtype=dtype)
    #sigma_x - sigma_y
    temp[0, 1, :, :] = self.J[0]*sigma_x - self.J2[0]*sigma_y
    #sigma_x + sigma_y
    temp[0, 2, :, :] = self.J2[0]*sigma_x + self.J[0]*sigma_y
    #sigma_z
    temp[0, 3, :, :] = self.J[0]*sigma_z
    #11
    temp[0, 4, :, :] = sigma_i
    mpo.append(temp)
    odd = False
    for n in range(1, N - 1):
      temp = np.zeros(shape=[5, 5, 2, 2], dtype=dtype)
      #11
      temp[0, 0, :, :] = sigma_i
      #sigma_x
      temp[1, 0, :, :] = sigma_x
      #sigma_y
      temp[2, 0, :, :] = sigma_y
      #sigma_z
      temp[3, 0, :, :] = sigma_z
      if odd:
        #sigma_x - sigma_y
        temp[4, 1, :, :] = self.J[n]*sigma_x - self.J2[n]*sigma_y
        #sigma_z + sigma_y
        temp[4, 2, :, :] = self.J2[n]*sigma_x + self.J[n]*sigma_y
        #sigma_z
        temp[4, 3, :, :] = self.J[n]*sigma_z
        #11
        temp[4, 4, :, :] = sigma_i
        odd = False
      else:
        #-sigma_y
        temp[4, 1, :, :] = -self.J2[n]*sigma_y
        #sigma_x
        temp[4, 2, :, :] = self.J2[n]*sigma_x
        #sigma_z
        temp[4, 3, :, :] = self.J1[n]*sigma_z
        #11
        temp[4, 4, :, :] = sigma_i
        odd = True
      mpo.append(temp)

    temp = np.zeros([5, 1, 2, 2], dtype=dtype)
    #11
    temp[0, 0, :, :] = sigma_i
    #sigma_x
    temp[1, 0, :, :] = sigma_x
    #sigma_y
    temp[2, 0, :, :] = sigma_y
    #sigma_z
    temp[3, 0, :, :] = sigma_z
    mpo.append(temp)
    super().__init__(tensors=mpo, backend=backend, name=name)