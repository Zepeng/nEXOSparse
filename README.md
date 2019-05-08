# nEXOSparse
A new signal/background classifier based on deep neural network. The new classifier uses SparseConvNet (https://github.com/facebookresearch/SparseConvNet) to solve the problem of size of current waveform in the other classifier.\
The input of the network is composed with the waveform on a "hit" channel and its position on the anode. Due to the high efficiency of SparseConvNet in dealing with sparse matrix, the input current waveform requires no downsampling. The length of the waveform is set to be 1700.
