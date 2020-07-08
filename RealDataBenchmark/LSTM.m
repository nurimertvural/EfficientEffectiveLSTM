
classdef LSTM < handle
   properties
       
       input_set   % Input dataset, X
       target_set  % Target dataset, y
       
       n_i     % Augmented input vector dimension
       n_h     % Hidden state dimension
       T       % Total number of outputs
       
       
       batch_size % BPTT window (or batch) size
       h_t_pre    % hidden state at the beginning of the batch 
       c_t_pre    % cell   state at the beginning of the batch 


       optimizer % Optimizer objects
       optimizer_name
       
       update

       is_optimizer_set   % Is optimizer set?           - Logical
       is_gcalculator_set % Is gradient calculator set? - Logical
       is_train_set       % Are trainor parameters set? - Logical
       
       temp_Wi
       temp_Wf
       temp_Wo
       temp_Wz
       temp_Wd

       
       d_t         % Target value at step t
       d_hat_t     % Estmation    at time step t 
       
       repeat
       init
       bias
       
       P
   end
   
   methods
       function obj=LSTM(varargin)
           
           p = inputParser;

           p.addParameter('Hidden', 5, @(x) x>0 & isscalar(x));
                      
           p.parse(varargin{:});
           
           %Initialize properties.
           obj.n_h  = p.Results.Hidden;
           
           % Initialize pre-forward variables.
           obj.h_t_pre    = zeros(obj.n_h,1);
           obj.c_t_pre    = zeros(obj.n_h,1);

            
           obj.is_optimizer_set   = false;
           obj.is_gcalculator_set = false;
           obj.is_train_set       = false;
       end
       
       function setTrainParameters(obj,input_set, target_set, varargin)
            p = inputParser;
            
            p.addRequired('input_set',  @(x) size(x,1) > 0 & size(x,2) == size(target_set,2));
            p.addRequired('target_set', @(x) size(x,1) == 1);
            p.addParameter('Init', 0.1, @(x) x>0);
            p.addParameter('Bias', 0.5, @(x) x>0);
            p.addParameter('Repeat', 1, @(x) x>0 & isscalar(x));
            
            p.parse(input_set, target_set, varargin{:});
           
            obj.input_set   = p.Results.input_set; %Input set is augmented.
            obj.target_set  = p.Results.target_set; 
            
            % Minmax target set normalization.
            
            %obj.target_set = 2 * obj.target_set / max(obj.target_set) - 1;
           
            
            obj.init   = p.Results.Init; 
            obj.bias   = p.Results.Bias; 
            obj.repeat = p.Results.Repeat;
   
            obj.n_i  = size( input_set , 1 ); %All inputs are column vectors.
            obj.T    = size( input_set , 2 );            
            

            
            obj.is_train_set = true;
            
       end
     
       function setGradientCalculator(obj, gcalculator, varargin)
           % TODO: Add RTRL as well.
           
           p = inputParser;
           
           expectedGradientCalculator = {'BPTT'};
           
           p.addRequired('GCalculator', @(x) any(validatestring(x,expectedGradientCalculator)) );
           p.addParameter('Backward', 10, @(x) x>0 & isscalar(x));
           
           p.parse(gcalculator,varargin{:});
           obj.batch_size = p.Results.Backward;
           
           obj.is_gcalculator_set = true;
       end
       
       function setOptimizer(obj, optimizer, varargin)
            
            p = inputParser;
            
            expectedOptimizer = {'SGD','Adam','RmsProp', 'EKF', 'DEKF'};
            
            p.addRequired('Optimizer' , @(x) any(validatestring(x,expectedOptimizer)) );
            
            p.addParameter('LRate', 0.005 , @(x) x>0  & 1>x);
            p.addParameter('Alpha', 0     , @(x) x>=0 & 1>x);
            p.addParameter('Beta' , 0.99  , @(x) x>0  & 1>x);
            p.addParameter('Beta1', 0.9   , @(x) x>0  & 1>x);
            p.addParameter('Beta2', 0.999 , @(x) x>0  & 1>x);
            p.addParameter('eps'  , 1e-8  , @(x) x>0  & 1>x);
            
            p.addParameter('P'     , 100  , @(x) x>0 );
            p.addParameter('Rinit' , 100   , @(x) x>0 );
            p.addParameter('Rlast' , 3   , @(x) x>0 );
            p.addParameter('Qinit' , 1e-2 , @(x) x>0 );
            p.addParameter('Qlast' , 1e-6 , @(x) x>0 );
            
               
            p.parse(optimizer,varargin{:});
            
            lr       = p.Results.LRate;
            alpha    = p.Results.Alpha;
            beta     = p.Results.Beta;
            beta_1   = p.Results.Beta1;
            beta_2   = p.Results.Beta2;
            eps      = p.Results.eps;
            
            obj.P  = p.Results.P;
            Q_init = p.Results.Qinit;
            Q_last = p.Results.Qlast;
            R_init = p.Results.Rinit;
            R_last = p.Results.Rlast;

                   
            name = p.Results.Optimizer;
            
            obj.optimizer  = Optimizer(name,lr,alpha,beta,beta_1,beta_2,eps,R_init,R_last,Q_init,Q_last);
            
            if(name == "DEKF")
                obj.update = @obj.update_DEKF;
            else
                obj.update = @obj.update1;
            end

            obj.is_optimizer_set = true;
            
       end       

       function [error_vec, kErr_vec] = train(obj)
            
            if(~obj.is_optimizer_set || ~obj.is_gcalculator_set || ~obj.is_train_set)
                error('Please set the optimizer and/or gradient calculator and/or training parameters!')
            end

            %Train RNN weights.
            Tx         = (obj.T-obj.batch_size+1) - 70;
           
            
            time_vec  = zeros(1,obj.repeat);
            error_vec = zeros(obj.repeat, Tx);
            kErr_vec  = zeros(obj.repeat, Tx);
            k = 50;
            
            rng('default');
            
            for r = 1:obj.repeat
                
                % Initialize the weights
                rng(r);     
                Wox = obj.init*randn( obj.n_h , obj.n_h + obj.n_i );
                Wix = obj.init*randn( obj.n_h , obj.n_h + obj.n_i );
                Wfx = obj.init*randn( obj.n_h , obj.n_h + obj.n_i );
                Wox(:,end) = - obj.bias; Wix(:,end) = - obj.bias; Wfx(:,end) = obj.bias;
                Wzx = obj.init*randn( obj.n_h , obj.n_h + obj.n_i );
                Wdx = obj.init*randn(    1    , obj.n_h + 1       );

                V1 = 0; S1 = 0;

                if( obj.optimizer.optimizer_name == "EKF")  
                    obj.optimizer.Tx =  Tx;
                    obj.optimizer.idx = 0;
                    
                    n_theta = 4 * obj.n_h * (obj.n_h + obj.n_i) + (obj.n_h + 1);
                    V1 = obj.P * eye(n_theta);
                elseif( obj.optimizer.optimizer_name == "DEKF") 
                    obj.optimizer.Tx =  Tx;
                    obj.optimizer.idx = 0;
                    
                    V1 = obj.P * repmat( eye(obj.n_h + obj.n_i), 1, 1, 4 * obj.n_h);
                    S1 = obj.P * eye(obj.n_h + 1);
                end

            
                
                tic
                
                for t=1:Tx
                    
                    % Update weights.
                    obj.optimizer.Vh = V1; clear V1;
                    obj.optimizer.Sh = S1; clear S1;
                    
                    obj.temp_Wd = Wdx; clear Wdx;
                    obj.temp_Wo = Wox; clear Wox;
                    obj.temp_Wf = Wfx; clear Wfx;
                    obj.temp_Wi = Wix; clear Wix;
                    obj.temp_Wz = Wzx; clear Wzx;
                    
                    
                    
                    
                    end_idx      = t + obj.batch_size - 1;
                    input_batch  = obj.input_set(:,t:end_idx + k - 1);
                    target_batch = obj.target_set(:,end_idx + k - 1);
                    
                    kErr = obj.kStepError(input_batch, target_batch, k);
                    kErr_vec(r, t) = kErr;
                    
                    % Choose input batch/target value.
                    end_idx = t + obj.batch_size-1;
                    input_batch = obj.input_set(:,t:end_idx);
                    
                    obj.d_t = obj.target_set(end_idx);
                    
                    % obj.grad_check(input_batch);
                    [Wdx, Wox, Wfx, Wix, Wzx, V1, S1] = obj.update(input_batch);
                    
                    error_vec(r, t) = (obj.d_t - obj.d_hat_t)^2;


                end
                
                time_vec(r) = toc;
            end
            
            mean_perf =   ( mean(prctile(error_vec,5))  + mean(prctile(error_vec,95))  ) / (2*var(obj.target_set));
            var_perf  =   ( mean(prctile(error_vec,95)) - mean(prctile(error_vec,5))   ) / (2*var(obj.target_set));
            k_mean_perf = ( mean(prctile(kErr_vec,5))   + mean(prctile(kErr_vec,95)) ) / (2*var(obj.target_set));
            k_var_perf  = ( mean(prctile(kErr_vec,95))  - mean( prctile(kErr_vec,5)) ) / (2*var(obj.target_set));


            fprintf('-LSTM_');
            fprintf(obj.optimizer.optimizer_name);
            
            if(obj.optimizer.optimizer_name ~= "EKF" || obj.optimizer.optimizer_name ~= "DEKF")
                fprintf(' Lr: %.3f \n', obj.optimizer.lr);
            else
                fprintf('\n');
            end
            
            fprintf('MSE: %.3f \x00B1  %.3f -- kMSE: %.3f \x00B1  %.3f, \n', mean_perf, var_perf, k_mean_perf, k_var_perf);
            fprintf('Run Time: %.2f! \n', mean(time_vec));

       end
       
   end
   
   methods(Access = private)
       
       function kErr = kStepError(obj, input_batch, target_batch, k)
           h_t  = obj.h_t_pre;
           c_t  = obj.c_t_pre;
           
           H_batchx   = zeros( obj.n_h , obj.batch_size + k );
           
           H_batchx(:,1) = h_t;
           
           for t= 1:obj.batch_size + k - 1
                
                concat = [h_t;input_batch(:,t)];
                
                z_t = tanh(obj.temp_Wz*concat);
                i_t = 0.5*(tanh(0.5*obj.temp_Wi*concat) + 1); % Fast implementation of sigmoid.
                f_t = 0.5*(tanh(0.5*obj.temp_Wf*concat) + 1);
                o_t = 0.5*(tanh(0.5*obj.temp_Wo*concat) + 1);
                
                c_t = z_t .* i_t + f_t .* c_t;
                h_t = o_t .* tanh(c_t);
                                
                H_batchx(:,t+1) = h_t;

                
           end
            
            ht    = H_batchx(:, end); 
            kEst  = tanh( obj.temp_Wd * [ ht ; 1 ] );
            
            kErr =  (target_batch - kEst).^2;
           
           
       end
       
       function [dWdx, dWox, dWfx, dWix, dWzx]= forward_backward(obj,input_batch)
            
            % Forward part begins.
            h_t           = obj.h_t_pre;
            c_t           = obj.c_t_pre;
            
            H_batchx   = zeros( obj.n_h , obj.batch_size + 1 );
            C_batchx   = zeros( obj.n_h , obj.batch_size + 1 );
            
            O_batchx   = zeros( obj.n_h , obj.batch_size );
            F_batchx   = zeros( obj.n_h , obj.batch_size );
            I_batchx   = zeros( obj.n_h , obj.batch_size );
            Z_batchx   = zeros( obj.n_h , obj.batch_size );

            
            H_batchx(:,1) = h_t;
            C_batchx(:,1) = c_t;
            
            
            for t= 1:obj.batch_size
                
                concat = [h_t;input_batch(:,t)];
                
                z_t = tanh(obj.temp_Wz*concat);
                i_t = 0.5*(tanh(0.5*obj.temp_Wi*concat) + 1); % Fast implementation of sigmoid.
                f_t = 0.5*(tanh(0.5*obj.temp_Wf*concat) + 1);
                o_t = 0.5*(tanh(0.5*obj.temp_Wo*concat) + 1);
                
                c_t = z_t .* i_t + f_t .* c_t;
                h_t = o_t .* tanh(c_t);
                
                O_batchx(:,t) = o_t;
                F_batchx(:,t) = f_t;  
                I_batchx(:,t) = i_t;  
                Z_batchx(:,t) = z_t;  
                
                H_batchx(:,t+1) = h_t;
                C_batchx(:,t+1) = c_t;
                
            end
            
            obj.d_hat_t     =  tanh( obj.temp_Wd * [h_t ; 1] );

            obj.h_t_pre = H_batchx(:,2); % Save the next hidden state
            obj.c_t_pre = C_batchx(:,2); % Save the next cell   state
            
            % Forward part eds.
            
            % Backward part begins.            
            sigma_d = ( 1 - obj.d_hat_t.^2 );
            
            h_t = H_batchx(:,end); % h_t
            c_t = C_batchx(:,end); % c_t
                        
            dWdx = sigma_d * [h_t ; 1]';
            
            sigma_h  = sigma_d * obj.temp_Wd(1,1:obj.n_h)';
            sigma_c  = zeros(obj.n_h,1); % Additional gradient due to constant error flow.
            
            sigma_o_mat = zeros(size(O_batchx));
            sigma_i_mat = zeros(size(I_batchx));
            sigma_f_mat = zeros(size(F_batchx));
            sigma_z_mat = zeros(size(Z_batchx));
            
            upd_Wox = obj.temp_Wo(:,1:obj.n_h)';
            upd_Wfx = obj.temp_Wf(:,1:obj.n_h)';
            upd_Wix = obj.temp_Wi(:,1:obj.n_h)';
            upd_Wzx = obj.temp_Wz(:,1:obj.n_h)';
                    
            for t=obj.batch_size:-1:1
                
                o_t = O_batchx(:,t); % o_t
                
                sigma_c = sigma_h .* o_t .* (1 - tanh(c_t).^2) + sigma_c;
                sigma_o = sigma_h .* tanh(c_t).* o_t .* (1 - o_t);
                
                f_t = F_batchx(:,t); % f_t 
                i_t = I_batchx(:,t); % i_t 
                z_t = Z_batchx(:,t); % z_t      
                
                sigma_i = sigma_c .* z_t .* i_t .* (1 - i_t);
                sigma_z = sigma_c .* i_t .* (1 - z_t.^2);
                
                c_t = C_batchx(:,t); %c_{t-1}
            
                sigma_f = sigma_c .* c_t .* f_t .* (1 - f_t);             

                sigma_o_mat( :, t, :) = sigma_o;
                sigma_f_mat( :, t, :) = sigma_f;
                sigma_i_mat( :, t, :) = sigma_i;
                sigma_z_mat( :, t, :) = sigma_z;
                
                % sigma_h_{t-1}
                sigma_h =  upd_Wox * sigma_o ...
                         + upd_Wfx * sigma_f ...
                         + upd_Wix * sigma_i ...
                         + upd_Wzx * sigma_z;
                
                sigma_c = sigma_c .* f_t; % sigma_c_t .* f_t
            end
            
            concat_mat = [ H_batchx(:, 1:end-1,:); input_batch ];
            
            dWox = sigma_o_mat * concat_mat';
            dWfx = sigma_f_mat * concat_mat';
            dWix = sigma_i_mat * concat_mat';
            dWzx = sigma_z_mat * concat_mat';
            % Backward part ends.
       end
                   
       function [Wdx, Wox, Wfx, Wix, Wzx, V1, S1] = update1(obj,input_batch)

            [dWdx, dWox, dWfx, dWix, dWzx]= obj.forward_backward(input_batch);  
            
            theta1  = [ obj.temp_Wo ,obj.temp_Wf, obj.temp_Wi, obj.temp_Wz ];
            dtheta1 = [ dWox, dWfx, dWix, dWzx];
            
            theta2  = obj.temp_Wd;
            dtheta2 = dWdx;
            
            theta  = [theta1(:) ;  theta2(:)]; 
            dtheta = [dtheta1(:); dtheta2(:)]; % H_t^T
            
            obj.temp_Wo = []; obj.temp_Wf = [];
            obj.temp_Wi = []; obj.temp_Wz = [];
            
            
            [theta, V1, S1] = obj.optimizer.optimizer(theta, dtheta, obj.d_t - obj.d_hat_t);
            
            n_w = obj.n_h * (obj.n_h + obj.n_i);
            Wox = reshape( theta( 1       : n_w   ) , obj.n_h, (obj.n_h + obj.n_i) ); 
            Wfx = reshape( theta( n_w+1   : 2*n_w ) , obj.n_h, (obj.n_h + obj.n_i) );
            Wix = reshape( theta( 2*n_w+1 : 3*n_w ) , obj.n_h, (obj.n_h + obj.n_i) );
            Wzx = reshape( theta( 3*n_w+1 : 4*n_w ) , obj.n_h, (obj.n_h + obj.n_i) );
            Wdx = theta( 4*n_w+1 : end )';
            
       end  
       
       function [Wdx, Wox, Wfx, Wix, Wzx, V1, S1] = update_DEKF(obj, input_batch)

            [dWdx, dWox, dWfx, dWix, dWzx]= obj.forward_backward(input_batch);
            
            % DEKF begins
            % Reshape weight matrices
            theta_hid  = reshape( [ obj.temp_Wo; obj.temp_Wf; obj.temp_Wi; obj.temp_Wz ]', obj.n_h + obj.n_i, 1, 4 * obj.n_h);
            dtheta_hid = reshape( [ dWox; dWfx; dWix; dWzx ]', obj.n_h + obj.n_i, 1, 4 * obj.n_h);
            
            theta_out  = obj.temp_Wd(:);
            dtheta_out = dWdx(:);
            
            obj.temp_Wo = []; obj.temp_Wf = [];
            obj.temp_Wi = []; obj.temp_Wz = [];
            
            
            [theta_hid, theta_out, V1, S1] = ...
                obj.optimizer.optimizer(theta_hid, theta_out, dtheta_hid, dtheta_out, obj.d_t - obj.d_hat_t);
            
            % Re-assign weights
            Wox = reshape( theta_hid( :, :, 1             : obj.n_h   ) , (obj.n_h + obj.n_i), obj.n_h ); 
            Wox = Wox';
            %
            Wfx = reshape( theta_hid( :, :, obj.n_h + 1   : 2*obj.n_h ) , (obj.n_h + obj.n_i), obj.n_h ); 
            Wfx = Wfx';
            %
            Wix = reshape( theta_hid( :, :, 2*obj.n_h + 1 : 3*obj.n_h ) , (obj.n_h + obj.n_i), obj.n_h ); 
            Wix = Wix';
            %
            Wzx = reshape( theta_hid( :, :, 3*obj.n_h + 1 : 4*obj.n_h ) , (obj.n_h + obj.n_i), obj.n_h ); 
            Wzx = Wzx';
            %
            Wdx       = theta_out';
            
       end  
      
       function grad_check(obj,input_batch)
           rh_pre = obj.h_t_pre;
           rc_pre = obj.c_t_pre;

           [dWdx, dWox, dWfx, dWix, dWzx]= obj.forward_backward(input_batch);  

           
           obj.h_t_pre = rh_pre;
           obj.c_t_pre = rc_pre;
           
           [num_dWo, num_dWi, num_dWf, num_dWz, num_dWd]= obj.numeric_grad(input_batch);
           
           grad_err = 0;
           grad_err = grad_err + sum( sum( (dWox - num_dWo).^2 ) );
           grad_err = grad_err + sum( sum( (dWix - num_dWi).^2 ) );
           grad_err = grad_err + sum( sum( (dWfx - num_dWf).^2 ) );
           grad_err = grad_err + sum( sum( (dWzx - num_dWz).^2 ) );
           grad_err = grad_err + sum( sum( (dWdx - num_dWd).^2 ) );
           
           fprintf('Grad err: %.6f\n',grad_err);
           
       end
        
       function [dWo, dWi,dWf,dWz,dWd] = numeric_grad(obj,input_batch)
           
            dWo = zeros(size(obj.temp_Wo));
            dWi = zeros(size(obj.temp_Wi));
            dWf = zeros(size(obj.temp_Wf));
            dWz = zeros(size(obj.temp_Wz));
            dWd = zeros(size(obj.temp_Wd));
                    
            eps = 1e-7;
            
            rWo = obj.temp_Wo; % real Wo
            rWi = obj.temp_Wi; % real Wi
            rWf = obj.temp_Wf; % real Wf
            rWz = obj.temp_Wz; % real Wz
            rWd = obj.temp_Wd; % real Wd
            
            rh_pre = obj.h_t_pre;
            rc_pre = obj.c_t_pre;
            
            % Calculate numeric gradient wrt Wd
            for i = 1: size(obj.temp_Wd,1)
                for j = 1: size(obj.temp_Wd,2)
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wd(i,j) = obj.temp_Wd(i,j) + eps;
                    obj.forward_backward(input_batch);
                    err1 = obj.d_hat_t;
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wd(i,j) = obj.temp_Wd(i,j) - 2*eps;
                    obj.forward_backward(input_batch);
                    err2 = obj.d_hat_t;
                    
                    dWd(i,j) = (err1-err2)/(2*eps);
                    
                    obj.temp_Wd = rWd;
                end
            end
            
            % Calculate numeric gradient wrt Wo
            for i = 1: size(obj.temp_Wo,1)
                for j = 1: size(obj.temp_Wo,2)
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wo(i,j) = obj.temp_Wo(i,j) + eps;
                    obj.forward_backward(input_batch);
                    err1 = obj.d_hat_t;
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wo(i,j) = obj.temp_Wo(i,j) - 2*eps;
                    obj.forward_backward(input_batch);
                    err2 = obj.d_hat_t;
                    
                    dWo(i,j) = (err1-err2)/(2*eps);
                    
                    obj.temp_Wo = rWo;
                end
            end
            
            % Calculate numeric gradient wrt Wi
            for i = 1: size(obj.temp_Wi,1)
                for j = 1: size(obj.temp_Wi,2)
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wi(i,j) = obj.temp_Wi(i,j) + eps;
                    obj.forward_backward(input_batch);
                    err1 = obj.d_hat_t;
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wi(i,j) = obj.temp_Wi(i,j) - 2*eps;
                    obj.forward_backward(input_batch);
                    err2 =  obj.d_hat_t;
                    
                    dWi(i,j) = (err1-err2)/(2*eps);
                    
                    obj.temp_Wi = rWi;
                end
            end
            
            % Calculate numeric gradient wrt Wf
            for i = 1: size(obj.temp_Wf,1)
                for j = 1: size(obj.temp_Wf,2)
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wf(i,j) = obj.temp_Wf(i,j) + eps;
                    obj.forward_backward(input_batch);
                    err1 =  obj.d_hat_t;
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wf(i,j) = obj.temp_Wf(i,j) - 2*eps;
                    obj.forward_backward(input_batch);
                    err2 =  obj.d_hat_t;
                    
                    dWf(i,j) = (err1-err2)/(2*eps);
                
                    obj.temp_Wf = rWf;
                end
            end
            
            % Calculate numeric gradient wrt Wz
            for i = 1: size(obj.temp_Wz,1)
                for j = 1: size(obj.temp_Wz,2)
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wz(i,j) = obj.temp_Wz(i,j) + eps;
                    obj.forward_backward(input_batch);
                    err1 = obj.d_hat_t;
                    
                    obj.h_t_pre = rh_pre;
                    obj.c_t_pre = rc_pre;
                    obj.temp_Wz(i,j) = obj.temp_Wz(i,j) - 2*eps;
                    obj.forward_backward(input_batch);
                    err2 = obj.d_hat_t;
                    
                    dWz(i,j) = (err1-err2)/(2*eps);
                    
                    obj.temp_Wz = rWz;
                end
            end

            
            
       end
       

       
   end
   
end



