This software accompanies the paper Vural, N.M., Ergut, S., Kozat S.S. (2020), "An Efficient and Effective Second-Order Training
Algorithm For LSTM-based Adaptive Learning", IEEE Transaction on Signal Processing(under review). Please cite it when using the code.

The software is intended to provide experiment codes for reproducibility of the proposed algorithm in future work. For the details of the experiment, please see the paper. 

Hyperparameters of the compared algorithms can be found in hyperparameters.txt.

The software is written in MATLAB. The software consists of three parts:
* LSTM: One-layered LSTM module
* Optimizer: Optimizer class, includes SGD, RMSprop, Adam, DEKF, EKF.
* LSTM_IEKF: Implementation of the proposed algorithm.
