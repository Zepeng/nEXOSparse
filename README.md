# nEXOSparse
A new signal/background classifier based on deep neural network using [PyTorch](https://pytorch.org/docs/stable/index.html). The new classifier uses [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) to solve the problem of size of current waveform in the other classifier.

The input of the network is composed with the waveform on a "hit" channel and its position on the anode. Due to the high efficiency of SparseConvNet in dealing with sparse matrix, the input current waveform requires no downsampling. The length of the waveform is set to be 1700. In principle, the input could have multiple planes, similar to general deep neural network. SparseConvNet has its own input interface that does differently with general pytorch data loader. Currently, the X
and Y channel are stitched into one plane, with each having 500 indices. An example input is shown in ![Fig. 1](./images/example_wf.pdf)
