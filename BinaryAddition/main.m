clear all; close all;
load('data/add5seq1.mat')
T = 100000;

input_set =  nngc_data(1:T,1:end-1)';
target_set = nngc_data(1:T,end)';

target_set(target_set == 0) = -0.5;
target_set(target_set == 1) = 0.5;

h_size = 12;
repeat = 1;
%% LSTM - RMSprop
lstm1 = LSTM('Hidden', h_size);
lstm1.setGradientCalculator('BPTT', 'Backward', 10)
lstm1.setOptimizer('RMSprop', 'LRate', 0.02)
lstm1.setTrainParameters(input_set,target_set,'Init',0.1);
lstm_rms =  lstm1.train();
%% LSTM - DEKF
lstm2 = LSTM('Hidden', h_size);
lstm2.setGradientCalculator('BPTT', 'Backward', 10)
lstm2.setOptimizer('DEKF', 'P', 100, 'Rinit', 3, 'Rlast', 3, 'Qinit', 1e-3, 'Qlast', 1e-6)
lstm2.setTrainParameters(input_set,target_set,'Init',0.1);
lstm_dekf =  lstm2.train();
%% LSTM - EKF
lstm3 = LSTM('Hidden', h_size);
lstm3.setGradientCalculator('BPTT', 'Backward', 10)
lstm3.setOptimizer('EKF', 'P', 100, 'Rinit', 3, 'Rlast', 3, 'Qinit', 1e-3, 'Qlast', 1e-6)
lstm3.setTrainParameters(input_set,target_set,'Init',0.1);
lstm_ekf =  lstm3.train();
%% LSTM - IEKF
lstm4 = LSTM_IEKF('Hidden', h_size);
lstm4.setGradientCalculator('BPTT', 'Backward', 10)
lstm4.setOptimizer('IEKF', 'P', 10, 'Qinit', 1e-7, 'Qlast', 1e-7, 'ChiMin', 1e-2)
lstm4.setTrainParameters(input_set,target_set,'Init',0.1);
lstm_iekf =  lstm4.train();


