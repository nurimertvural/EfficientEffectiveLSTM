clc; clear all; close all;
load('data/elevators.mat')
T= 1000;

input_set  = input_set(:,1:T);
target_set = target_set(:,1:T);

h_size = 12;
repeat = 10;

lr     = 0.001:0.001:0.015;

%% LSTM - RmsProp
lstm1 = LSTM('Hidden', h_size);
lstm1.setGradientCalculator('BPTT', 'Backward', 10)
lstm1.setOptimizer('RmsProp', 'LRate', lr(1));
lstm1.setTrainParameters(input_set,target_set, 'Repeat',repeat,'Bias',0.5,'Search',true);

lstm_rms_lr = 0;
err_mean_rms = 999;
for i = 1:length(lr)
    lstm1.setOptimizer('RmsProp', 'LRate', lr(i));
    mean_err = lstm1.train();
    if( mean(mean_err(:)) < err_mean_rms)
        err_mean_rms = mean(mean_err(:));
        lstm_rms_lr  = lr(i);
    end
end


%% LSTM - Adam
lstm_adam_lr = 0;
err_mean_adam    = 999;
for i = 1:length(lr)
    lstm1.setOptimizer('Adam', 'LRate', lr(i));
    mean_err = lstm1.train();
    if( mean(mean_err(:)) < err_mean_adam)
        err_mean_adam = mean(mean_err(:));
        lstm_adam_lr  = lr(i);
    end
end

%% LSTM - SGD
lr     = 0.1:0.05:0.5;


lstm_sgd_lr = 0;
err_mean_sgd    = 999;
for i = 1:length(lr)
    lstm1.setOptimizer('SGD', 'LRate', lr(i));
    mean_err = lstm1.train();
    if( mean(mean_err(:)) < err_mean_sgd)
        err_mean_sgd = mean(mean_err(:));
        lstm_sgd_lr  = lr(i);
    end
end

%% LSTM - DEKF
p_init  = [25,50,100];
r_init  = [10,30,50,100];
q_init  = [1e-2,1e-3,1e-4];
q_last  = q_init - T*(q_init- 1e-6)/2550;
r_last  = r_init - T*(r_init- 3)/2550;


lstm_dekf_rinit = 0;
lstm_dekf_qinit = 0;
lstm_dekf_pinit = 0;
err_mean_dekf   = 999;
for i = 1:length(r_init)
    for j = 1:length(q_init)
        for k = 1:length(p_init)
        
            lstm1.setOptimizer('DEKF',  'P', p_init(k), 'Rinit', r_init(i), 'Rlast', r_last(i), ...
                'Qinit', q_init(j), 'Qlast', q_last(j));
            
            mean_err = lstm1.train();
            
            if( mean(mean_err(:)) < err_mean_dekf)
                err_mean_dekf  = mean(mean_err(:));
                lstm_dekf_rinit = r_init(i);
                lstm_dekf_qinit = q_init(j);
                lstm_dekf_pinit = p_init(k);
            end
        
        end
    end
end
    
%% LSTM - Alg2
p_init  = [0.5,10,50,100];
q_init  = [1e-3,1e-4,1e-6,1e-7];
q_last  = q_init - T*(q_init- 1e-8)/2550;

lstm2 = LSTM_IEKF('Hidden', h_size);

lstm_iekf_qinit = 0;
lstm_iekf_pinit = 0;
err_mean_iekf   = 999;

for i = 1:length(p_init)
    for j = 1:length(q_init)
        
        lstm2.setTrainParameters(input_set,target_set, 'Repeat',repeat, 'Init', 0.1, 'Bias', 5, ...
            'Search', true);
        lstm2.setGradientCalculator('BPTT', 'Backward', 10)
        lstm2.setOptimizer('IEKF', 'P', p_init(i), 'Qinit', q_init(j), 'Qlast', q_last(j), ...
            'ChiMin', 1e-2)

        mean_err = lstm2.train();

        if( mean(mean_err(:)) < err_mean_iekf)
            err_mean_iekf  = mean(mean_err(:));
            lstm_iekf_pinit = p_init(i);
            lstm_iekf_qinit = q_init(j);
        end

    end
end





