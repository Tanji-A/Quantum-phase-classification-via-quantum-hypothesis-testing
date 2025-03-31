# Quantum Phase Classification via Quantum Hypothesis Testing

This repository contains the source code and datasets used in the research article:

> **"Quantum Phase Classification via Quantum Hypothesis Testing"**\
> *Akira Tanji, Hiroshi Yano, Naoki Yamamoto*\
> *(Submitted)*

---

## Repository Structure

| Path                                     | Description                                                                                              |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `QHT.py`, `QHT.ipynb`                    | Implementation of the proposed method and comparison methods.           |
| `QHT_tensornet.ipynb`                    | Implementation of our method using Matrix Product States (MPS).                                      |
| `exactqcnn.py`, `exactqcnn_tensornet.py` | Implementation of the ExactQCNN as described in the paper.                                        |
| `hamiltonian_Pollmann.py`                | Computes the ground state of the Hamiltonian used in the paper. Generated data are stored in `dataset/`. |
| `dataset/`                               | Contains datasets of ground states for quantum many-body systems used in training/testing.               |
| `tensornetPollmann/`                     | Contains MPS-based code and the corresponding generated ground states.                                   |
| `json_data/`, `npz_data/`                | Contain the numerical simulation results.       |

---

## Notes

- The `tensornetPollmann/` directory contains MPS-generated ground states and related code.\
  **This folder includes large files and may take time to clone or download.**

- Some simulations (e.g., those using `QHT.py`, `QHT_tensornet.ipynb`, etc.)\
  **require random unitaries (e.g., random Clifford or Haar circuits), which must be generated in advance using the code in `QHT.ipynb`.**\
  The generated unitaries should be stored in the `random_unitary/` directory.

---

## Memo

- **The code will be cleaned up and made more readable soon...** Thank you for your understanding!

