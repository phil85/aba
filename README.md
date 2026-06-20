<img src='figs/aba_logo.svg'>

[![License](https://img.shields.io/badge/License-MIT_License-blue)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-green)](https://arxiv.org/abs/2601.06351)

# **ABA - The Assignment-based Anticlustering Algorithm**

Anticlustering is a combinatorial optimization problem that consists of partitioning a set of objects into equally sized groups (anticlusters) such that objects within the same group are maximally diverse. Achieving diversity within groups is important in various applications, including recent large-scale applications in machine learning. The Assignment-based Anticlustering Algorithm (ABA) is designed for large-scale anticlustering in Euclidean spaces. It scales to millions of objects and hundreds of thousands of anticlusters. A detailed description of the algorithm can be found in our paper https://arxiv.org/abs/2601.06351. The paper also presents the main results of a computational study in which we compare the ABA algorithm with leading algorithms. The detailed results from the study are available [here](https://github.com/phil85/aba-results).       

Here we provide a Python implementation of our algorithm. The experiments in the paper were conducted using the C++ implementation of the ABA algorithm, which is available [here](https://github.com/j-yang-932/aba_Cpp). 

A variant of the ABA algorithm for maximum weight matching on Euclidean graphs is available [here](https://github.com/phil85/aba-for-maximum-weight-matching).

---

## Installation

1. Clone this repository  
2. Create and activate a virtual environment
3. Install the required dependencies (see requirements.txt)

## Usage

- To run the base version of ABA, look at the example in running_aba_base_version.py.
- To run ABA with categories, look at the example in running_aba_with_categories.py.
- To run ABA with hierarchical decomposition, look at the example in running_aba_with_hierarchical_decomposition.py

## Reference

If you use this algorithm in your research, please cite the following paper:

Baumann, P., Goldschmidt O., Hochbaum D.S., Yang J. (2026). A Fast and Effective Method for Euclidean Anticlustering: The Assignment-Based-Anticlustering Algorithm. arXiv paper.

Bibtex:
```
@misc{baumann2026fasteffectivemethodeuclidean,
      title={A Fast and Effective Method for Euclidean Anticlustering: The Assignment-Based-Anticlustering Algorithm}, 
      author={Philipp Baumann and Olivier Goldschmidt and Dorit S. Hochbaum and Jason Yang},
      year={2026},
      eprint={2601.06351},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.06351}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.