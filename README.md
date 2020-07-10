# An Efficient and Effective Second-Order Training Algorithm For LSTM-based Adaptive Learning

This software accompanies the paper Vural, N.M., Ergut, S., Kozat S.S. (2020), "An Efficient and Effective Second-Order Training Algorithm For LSTM-based Adaptive Learning", IEEE Transaction on Signal Processing(under review). It is intended to share experiment codes for future works. For the details of the experiment, please see the paper. 

Hyperparameters of the compared algorithms can be found in **hyperparameters.txt**. Datasets can be found in **../data** files.

Required packages:
* [mtimesx](https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support)

The software is written in MATLAB. The software consists of three parts:
* **LSTM**: One-layered LSTM module
* **Optimizer**: Optimizer class-- includes SGD, RMSprop, Adam, DEKF, and EKF.
* **LSTM_IEKF**: Implementation of the proposed algorithm.
