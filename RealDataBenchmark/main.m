clc; clear all; close all;
load('data/elevators.mat')

h_size = 12;
repeat = 20;
%% LSTM - SGD, RMSprop, Adam
lstm1 = LSTM('Hidden', h_size);
lstm1.setTrainParameters(input_set,target_set, 'Repeat',repeat, 'Init', 0.1, 'Bias', 0.5);
lstm1.setGradientCalculator('BPTT', 'Backward', 10)

lstm1.setOptimizer('SGD', 'LRate', 0.35);
[err_sgd, k_err_sgd] = lstm1.train();

lstm1.setOptimizer('RmsProp', 'LRate', 0.005);
[err_rms, k_err_rms] = lstm1.train();

lstm1.setOptimizer('Adam', 'LRate', 0.005);
[err_adam, k_err_adam] =  lstm1.train();

%% LSTM - EKF, DEKF
lstm1.setOptimizer('EKF',  'P', 100, 'Rinit', 50, 'Rlast', 3, 'Qinit', 1e-2, 'Qlast', 1e-6);
[err_ekf, k_err_ekf] = lstm1.train();

lstm1.setOptimizer('DEKF', 'P', 100, 'Rinit', 50, 'Rlast', 3, 'Qinit', 1e-2, 'Qlast', 1e-6);
[err_dekf, k_err_dekf] = lstm1.train();
%

%% LSTM - Alg2
lstm2 = LSTM_IEKF('Hidden', h_size);
lstm2.setTrainParameters(input_set,target_set, 'Repeat',repeat, 'Init', 0.1, 'Bias', 5);
lstm2.setGradientCalculator('BPTT', 'Backward', 10)

lstm2.setOptimizer('IEKF', 'P', 10, 'Qinit', 1e-4, 'Qlast', 1e-8, 'ChiMin', 1e-2)
[err_iekf, k_err_iekf] = lstm2.train();
