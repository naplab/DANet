# Deep Attractor Network (DANet) for single-channel speech separation

This repository provides the implementation of the Deep Attractor Network (DANet) for single-channel speech separation in Jupyter Notebook (.ipynb) format. DANet was introduced in the following papers:

Zhuo Chen, Yi Luo, and Nima Mesgarani, [Deep attractor network for single-microphone speaker separation](https://ieeexplore.ieee.org/abstract/document/7952155)

Yi Luo, Zhuo Chen, and Nima Mesgarani, [Speaker-independent speech separation with deep attractor network](https://ieeexplore.ieee.org/abstract/document/8264702)

Informations about the papers can also be found in [our lab website](http://naplab.ee.columbia.edu/danet.html).

## Citation

If you find the scripts helpful in your research, please consider citing:

    @inproceedings{chen2017deep,
    title={Deep attractor network for single-microphone speaker separation},
    author={Chen, Zhuo and Luo, Yi and Mesgarani, Nima},
    booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on},
    pages={246--250},
    year={2017},
    organization={IEEE}
    }

    @article{luo2018speaker,
    title={Speaker-independent speech separation with deep attractor network},
    author={Luo, Yi and Chen, Zhuo and Mesgarani, Nima},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    volume={26},
    number={4},
    pages={787--796},
    year={2018},
    publisher={IEEE}
    }
    
### Requirements
- Python 3.6.4
- Pytorch 0.4.1
- h5py 2.7.1
- sklearn 0.19.1
- numpy 1.15.0
- librosa 0.6.0
- jupyter 1.0.0 or above
- notebook 5.4.0 or above
