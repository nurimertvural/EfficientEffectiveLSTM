
classdef LSTM_IEKF < handle
   properties
       
       input_set   % Input dataset, X
       target_set  % Target dataset, y
       
       n_i     % Augmented input vector dimension
       n_h     % Hidden state dimension
       T       % Total number of outputs
       Tx
       
       
       batch_size % BPTT window (or batch) size
       h_t_pre    % hidden state at the beginning of the batch 
       c_t_pre    % cell   state at the beginning of the batch 
       
       is_optimizer_set   % Is optimizer set?           - Logical
       is_gcalculator_set % Is gradient calculator set? - Logical
       is_train_set       % Are trainor parameters set? - Logical
        
       
       d_t         % Target value at step t
       d_hat_t     % Estmation    at time step t
       d_hat_t_vec % Estimation of instances at time step t

       repeat
       init
       bias
       
       optimizer_name
       P
       
       temp_Phid
       temp_Pout
       
       temp_Wi
       temp_Wf
       temp_Wo
       temp_Wz
       temp_Wd
       
       n_theta
       Q_init
       Q_last
       Q
       R_init
       R_last
       R
       
       inst_num
       chi_min
       chi_vec
       
       weights
       
       
   end
   
   methods
       function obj=LSTM_IEKF(varargin)
           
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
            
            expectedOptimizer = {'IEKF'};
            
            p.addRequired('Optimizer' , @(x) any(validatestring(x,expectedOptimizer)) );
            
            p.addParameter('P'      , 15  , @(x) x>0 );
            p.addParameter('Rinit'  , 0.5  , @(x) x>0 );
            p.addParameter('Rlast'  , 0.5   , @(x) x>0 );
            p.addParameter('Qinit'  , 1e-6 , @(x) x>0 );
            p.addParameter('Qlast'  , 1e-6 , @(x) x>0 );
            p.addParameter('ChiMin' , 1e-2 , @(x) x>0 );
            
               
            p.parse(optimizer,varargin{:});
            
            obj.P       = p.Results.P;
            obj.Q_init  = p.Results.Qinit;
            obj.Q_last  = p.Results.Qlast;
            obj.R_init  = p.Results.Rinit;
            obj.R_last  = p.Results.Rlast;
            obj.chi_min = p.Results.ChiMin;
            
            obj.optimizer_name = p.Results.Optimizer;
            

            obj.is_optimizer_set = true;
            
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
            
            obj.init   = p.Results.Init; 
            obj.bias   = p.Results.Bias; 
            obj.repeat = p.Results.Repeat;
   
            obj.n_i  = size( input_set , 1 ); %All inputs are column vectors.
            obj.T    = size( input_set , 2 );
            
            obj.n_theta = 4 * obj.n_h * (obj.n_h + obj.n_i) + (obj.n_h + 1);
            
            
            obj.is_train_set = true;
            
       end
     
       function last_step = train(obj)
            
            if(~obj.is_optimizer_set || ~obj.is_gcalculator_set || ~obj.is_train_set)
                error('Please set the optimizer and/or gradient calculator and/or training parameters!')
            end

            %Train RNN weights.
            obj.Tx         = (obj.T-obj.batch_size+1) - 50;
            
            time_vec  = zeros(1,obj.repeat);
            last_step = [];
            
            obj.inst_num = ceil( log2( 1/ obj.chi_min) + 1);  
            obj.chi_vec  = linspace( log2(obj.chi_min), log2(1) , obj.inst_num);
            obj.chi_vec  = 2.^obj.chi_vec';    
            
            obj.input_set = repmat( obj.input_set, 1, 1, obj.inst_num);
            
            rng('default');
            
            for r = 1:obj.repeat
                
                % Initialize the vector that contains the error values of
                % epoch
                counter = 0;
                max_counter = 0;
                max_counter_step = 0;
                
                % Initialize weights, covariance matrices and state vectors.
                obj.weights = linspace( 1, 0.005, obj.inst_num);
                obj.weights = obj.weights'/sum(obj.weights);

                rng(r); 
                Wox = repmat( obj.init*randn( obj.n_h , obj.n_h + obj.n_i ), 1, 1, obj.inst_num);
                Wix = repmat( obj.init*randn( obj.n_h , obj.n_h + obj.n_i ), 1, 1, obj.inst_num);
                Wfx = repmat( obj.init*randn( obj.n_h , obj.n_h + obj.n_i ), 1, 1, obj.inst_num);
                Wix(:, end, :) = obj.bias; Wfx(:, end, :) = obj.bias; 
                Wzx = repmat( obj.init*randn( obj.n_h , obj.n_h + obj.n_i ), 1, 1, obj.inst_num);
                Wdx = repmat( obj.init*randn(    1    , obj.n_h + 1       ), 1, 1, obj.inst_num);

                Phid = repmat( obj.P * eye(obj.n_h + obj.n_i), 1, 1, 4 * obj.n_h, obj.inst_num);
                Pout = repmat( obj.P * eye(obj.n_h + 1), 1, 1, obj.inst_num);

                obj.h_t_pre = zeros(obj.n_h, 1, obj.inst_num);
                obj.c_t_pre = zeros(obj.n_h, 1, obj.inst_num);

                tic
                
                for t=1:obj.Tx

                   
                    
                    % Update weights.
                    
                    obj.temp_Phid = Phid; clear Phid;
                    obj.temp_Pout = Pout; clear Pout;
                    
                    obj.temp_Wd = Wdx; clear Wdx;
                    obj.temp_Wo = Wox; clear Wox;
                    obj.temp_Wf = Wfx; clear Wfx;
                    obj.temp_Wi = Wix; clear Wix;
                    obj.temp_Wz = Wzx; clear Wzx;

                    % Choose input batch/target value.
                    end_idx = t + obj.batch_size-1;
                    input_batch = obj.input_set(:,t:end_idx,:);
                    
                    obj.d_t = obj.target_set(end_idx);
                    
                    % obj.grad_check(input_batch);
                    
                    % Calculate nose parameters.
                    obj.Q = obj.Q_init + (t-1)/(obj.Tx-1) *  (obj.Q_last - obj.Q_init);
                    obj.R = obj.R_init + (t-1)/(obj.Tx-1) *  (obj.R_last - obj.R_init);
            
                    
                    [Wdx, Wox, Wfx, Wix, Wzx, Phid, Pout] = obj.update(input_batch);
                    
                    
                    if(obj.d_t *obj.d_hat_t > 0)
                        counter = counter + 1;
                    else
                        if(counter > max_counter)
                            max_counter = counter;
                            max_counter_step = t;
                            % fprintf('Max Counter updated: %d, step: %d\n',max_counter,max_counter_step);
                        end
                        counter = 0;
                    end
                    
                    if( counter > 500)
                        last_step = t;
                        fprintf("- LSTM Alg2");
                        fprintf(' Last Step: %d, ', last_step);
                        
                        break;
                    end


                end
                
                time_vec(r) = toc;
            end
            
            if(isempty(last_step))
                fprintf("- LSTM Alg2");
                fprintf(', Could not learn! ');
                fprintf('- Max Counter %d -- Time step: %d\n', max_counter, max_counter_step);
            else
                fprintf('Run Time: %.2f! \n', mean(time_vec));
            end
            
            

       end
       
   end
   
   methods(Access = private)

       
       function [dWdx, dWox, dWfx, dWix, dWzx, pass_idx]= forward_backward(obj,input_batch)
            

            % Forward part begins.
            h_t           = obj.h_t_pre;
            c_t           = obj.c_t_pre;
            
            H_batchx   = zeros( obj.n_h , obj.batch_size + 1, obj.inst_num );
            C_batchx   = zeros( obj.n_h , obj.batch_size + 1, obj.inst_num );
            
            O_batchx   = zeros( obj.n_h , obj.batch_size, obj.inst_num );
            F_batchx   = zeros( obj.n_h , obj.batch_size, obj.inst_num );
            I_batchx   = zeros( obj.n_h , obj.batch_size, obj.inst_num );
            Z_batchx   = zeros( obj.n_h , obj.batch_size, obj.inst_num );

            
            H_batchx(:,1, :) = h_t;
            C_batchx(:,1, :) = c_t;
            
            
            for t= 1:obj.batch_size
                
                x_t = input_batch(:, t, :);
                concat = [ h_t; x_t];
                
                net_zt = mtimesx( obj.temp_Wz, concat);
                net_it = mtimesx( obj.temp_Wi, concat); 
                net_ft = mtimesx( obj.temp_Wf, concat); 
                net_ot = mtimesx( obj.temp_Wo, concat); 
                
                z_t = tanh( net_zt );
                i_t = 0.5*( tanh(0.5*net_it) + 1 ); % Fast implementation of sigmoid.
                f_t = 0.5*( tanh(0.5*net_ft) + 1 );
                o_t = 0.5*( tanh(0.5*net_ot) + 1 );
                
                c_t = z_t .* i_t + f_t .* c_t;
                h_t = o_t .* tanh(c_t);
                
                O_batchx(:, t, :) = o_t;
                F_batchx(:, t, :) = f_t;  
                I_batchx(:, t, :) = i_t;  
                Z_batchx(:, t, :) = z_t;  
                
                H_batchx(:, t+1, :) = h_t;
                C_batchx(:, t+1, :) = c_t;
                
            end
            
            aug_ht          = [h_t ; ones( 1, 1, obj.inst_num)];
            obj.d_hat_t_vec = tanh( mtimesx( obj.temp_Wd , aug_ht) );
            
            obj.d_hat_t     = obj.weights' * obj.d_hat_t_vec(:);
            

            obj.h_t_pre = H_batchx(:, 2, :); % Save the next hidden state
            obj.c_t_pre = C_batchx(:, 2, :); % Save the next cell   state
            
            % Forward part ends.
            
            % Backward part begins.
            loss_vec = (obj.d_t - obj.d_hat_t_vec(:)).^2; 

            % Update weights
            obj.weights = obj.weights .* exp( - loss_vec);
            obj.weights = obj.weights / sum(obj.weights);
            
            
            % Back-propagation
            sigma_d = ( 1 - obj.d_hat_t_vec.^2 );
            
            h_t    = H_batchx(:, end, :); % h_t 
            aug_ht = [h_t ; ones( 1, 1, obj.inst_num )];
            c_t    = C_batchx(:, end, :); % c_t
            
            dWdx = mtimesx( sigma_d, aug_ht, 'C' );
            
            sigma_h  = mtimesx( sigma_d, obj.temp_Wd( 1, 1:obj.n_h, :), 'C' );
            sigma_c  = zeros(  size(sigma_h) ); % Additional gradient due to constant error flow.
            
            sigma_o_mat = zeros(size(O_batchx));
            sigma_i_mat = zeros(size(I_batchx));
            sigma_f_mat = zeros(size(F_batchx));
            sigma_z_mat = zeros(size(Z_batchx));
            
            upd_Wo = permute(obj.temp_Wo(:, 1:obj.n_h, :), [ 2 1 3]);
            upd_Wi = permute(obj.temp_Wi(:, 1:obj.n_h, :), [ 2 1 3]);
            upd_Wf = permute(obj.temp_Wf(:, 1:obj.n_h, :), [ 2 1 3]);
            upd_Wz = permute(obj.temp_Wz(:, 1:obj.n_h, :), [ 2 1 3]);
                    
            for t=obj.batch_size:-1:1
                
                o_t = O_batchx(:, t, :); % o_t
                
                sigma_c = sigma_h .* o_t .* (1 - tanh(c_t).^2) + sigma_c;
                sigma_o = sigma_h .* tanh(c_t).* o_t .* (1 - o_t);
                
                f_t = F_batchx(:, t, :); % f_t 
                i_t = I_batchx(:, t, :); % i_t 
                z_t = Z_batchx(:, t, :); % z_t      
                
                sigma_i = sigma_c .* z_t .* i_t .* (1 - i_t);
                sigma_z = sigma_c .* i_t .* (1 - z_t.^2);
                
                c_t = C_batchx(:, t, :); %c_{t-1}
            
                sigma_f = sigma_c .* c_t .* f_t .* (1 - f_t);             
                
                sigma_o_mat( :, t, :) = sigma_o;
                sigma_f_mat( :, t, :) = sigma_f;
                sigma_i_mat( :, t, :) = sigma_i;
                sigma_z_mat( :, t, :) = sigma_z;
                
                % sigma_h_{t-1}
                sigma_h =   mtimesx( upd_Wo, sigma_o ) ...
                          + mtimesx( upd_Wf, sigma_f ) ...
                          + mtimesx( upd_Wi, sigma_i ) ...
                          + mtimesx( upd_Wz, sigma_z );
                
                sigma_c = sigma_c .* f_t; % sigma_c_t .* f_t
            end
               
            concat_mat = [ H_batchx(:, 1:end-1,:); input_batch ];
            
            dWox = mtimesx( sigma_o_mat , concat_mat, 'C' );
            dWfx = mtimesx( sigma_f_mat , concat_mat, 'C' );
            dWix = mtimesx( sigma_i_mat , concat_mat, 'C' );
            dWzx = mtimesx( sigma_z_mat , concat_mat, 'C' );
            
            % Find which instances and how many instances will not be updated.
            pass_idx  = loss_vec < 4 * obj.chi_vec; 
            
            dWox(:, :, pass_idx) = 0;
            dWix(:, :, pass_idx) = 0;
            dWfx(:, :, pass_idx) = 0;
            dWzx(:, :, pass_idx) = 0;
            dWdx(:, :, pass_idx) = 0;
            
            % Backward part ends.
       end
     
       function [Wdx, Wox, Wfx, Wix, Wzx, Phid, Pout] = update(obj, input_batch)
            
            
            [dWdx, dWox, dWfx, dWix, dWzx, pass_idx] = obj.forward_backward(input_batch);  
            
            Phid = obj.temp_Phid; obj.temp_Phid = [];
            Pout = obj.temp_Pout; obj.temp_Pout = [];
            
            W  = permute( [ obj.temp_Wo; obj.temp_Wf; obj.temp_Wi; obj.temp_Wz ], [ 2 1 3]);
            dW = permute( [ dWox; dWfx; dWix; dWzx ], [ 2 1 3]);
            
            obj.temp_Wo = []; obj.temp_Wf = [];
            obj.temp_Wi = []; obj.temp_Wz = [];
            
            
            % IEKF begins     
            % Reshape weight matrices
            theta_hid  = reshape( W , obj.n_h + obj.n_i, 1, 4 * obj.n_h, obj.inst_num );
            dtheta_hid = reshape( dW, obj.n_h + obj.n_i, 1, 4 * obj.n_h, obj.inst_num );
            
            theta_out  = permute( obj.temp_Wd, [ 2 1 3]); obj.temp_Wd = [];
            dtheta_out = permute( dWdx, [ 2 1 3]); 
            
            % Calculate intermediate values
            intm_hid  = mtimesx( Phid, dtheta_hid );
            intm2_hid = mtimesx( dtheta_hid, 'C' , intm_hid );
            
            intm_out  = mtimesx( Pout, dtheta_out);
            intm2_out = mtimesx( dtheta_out, 'C' , intm_out); 
            

            % 1d desired data olduðu için trace'e gerek yok su anlýk.
            R_hid    = obj.R + 3*intm2_hid; 
            R_out    = obj.R + 3*intm2_out;
            
            % Calculate Gain Matrices
            G_hid = intm_hid ./ (intm2_hid + R_hid);
            G_out = intm_out ./ (intm2_out + R_out);
            
            theta_hid = theta_hid + G_hid .* (obj.d_t - obj.d_hat_t);
            theta_out = theta_out + G_out .* (obj.d_t - obj.d_hat_t);
             
            % Re-assign weights
            Wox = reshape( theta_hid( :, :, 1             : obj.n_h ,  : ) , (obj.n_h + obj.n_i), obj.n_h, obj.inst_num  ); 
            Wox = permute( Wox, [ 2 1 3]);
            %
            Wfx = reshape( theta_hid( :, :, obj.n_h + 1   : 2*obj.n_h, : ) , (obj.n_h + obj.n_i), obj.n_h, obj.inst_num  ); 
            Wfx = permute( Wfx, [ 2 1 3]);
            %
            Wix = reshape( theta_hid( :, :, 2*obj.n_h + 1 : 3*obj.n_h, : ) , (obj.n_h + obj.n_i), obj.n_h, obj.inst_num  ); 
            Wix = permute( Wix, [ 2 1 3]);
            %
            Wzx = reshape( theta_hid( :, :, 3*obj.n_h + 1 : 4*obj.n_h, : ) , (obj.n_h + obj.n_i), obj.n_h, obj.inst_num  ); 
            Wzx = permute( Wzx, [ 2 1 3]);
            %
            Wdx = permute( theta_out, [ 2 1 3]);
            
            
            % Update the covariance matrices
            G_hid( G_hid == 0) = 1e-15;
            Phid = (Phid./G_hid - permute(intm_hid, [ 2 1 3 4])) .* G_hid ...
                    + obj.Q * eye(obj.n_h + obj.n_i) .* ones(1, 1, 4*obj.n_h, obj.inst_num ) .* reshape(~pass_idx, 1, 1, 1, obj.inst_num);


            G_out( G_out == 0) = 1e-15;
            Pout = (Pout./G_out - permute(intm_out, [ 2 1 3]) ).*G_out ...
                   +  obj.Q  * eye(obj.n_h + 1).* ones( 1, 1, obj.inst_num ) .* reshape( ~ pass_idx, 1, 1, obj.inst_num);
            
            
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
           
           fprintf('Grad err: %.6f\n', sum(grad_err(:)) );
           
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
                    for k = 1: size(obj.temp_Wd,3)
                    
                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wd(i,j,k) = obj.temp_Wd(i,j,k) + eps;
                        obj.forward_backward(input_batch);
                        err1 = obj.d_hat_t_vec(:, :, k);

                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wd(i,j,k) = obj.temp_Wd(i,j,k) - 2*eps;
                        obj.forward_backward(input_batch);
                        err2 = obj.d_hat_t_vec(:, :, k);

                        dWd(i,j,k) = (err1-err2)/(2*eps);
                        
                        obj.temp_Wd = rWd;
                    end
                end              
            end
            
            % Calculate numeric gradient wrt Wo
            for i = 1: size(obj.temp_Wo,1)
                for j = 1: size(obj.temp_Wo,2)
                    for k = 1: size(obj.temp_Wo,3)
                    
                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wo(i,j,k) = obj.temp_Wo(i,j,k) + eps;
                        obj.forward_backward(input_batch);
                        err1 = obj.d_hat_t_vec(:, :, k);

                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wo(i,j,k) = obj.temp_Wo(i,j,k) - 2*eps;
                        obj.forward_backward(input_batch);
                        err2 = obj.d_hat_t_vec(:, :, k);

                        dWo(i,j,k) = (err1-err2)/(2*eps);
                        
                        obj.temp_Wo = rWo;
                    end
                end          
            end
   
            % Calculate numeric gradient wrt Wi
            for i = 1: size(obj.temp_Wi,1)
                for j = 1: size(obj.temp_Wi,2)
                    for k = 1: size(obj.temp_Wi,3)
                    
                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wi(i,j,k) = obj.temp_Wi(i,j,k) + eps;
                        obj.forward_backward(input_batch);
                        err1 = obj.d_hat_t_vec(:, :, k);

                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wi(i,j,k) = obj.temp_Wi(i,j,k) - 2*eps;
                        obj.forward_backward(input_batch);
                        err2 = obj.d_hat_t_vec(:, :, k);

                        dWi(i,j,k) = (err1-err2)/(2*eps);
                    
                        obj.temp_Wi = rWi;
                    end
                end                
            end
            
            % Calculate numeric gradient wrt Wf
            for i = 1: size(obj.temp_Wf,1)
                for j = 1: size(obj.temp_Wf,2)
                    for k = 1: size(obj.temp_Wf,3)
                    
                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wf(i,j,k) = obj.temp_Wf(i,j,k) + eps;
                        obj.forward_backward(input_batch);
                        err1 = obj.d_hat_t_vec(:, :, k);

                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wf(i,j,k) = obj.temp_Wf(i,j,k) - 2*eps;
                        obj.forward_backward(input_batch);
                        err2 = obj.d_hat_t_vec(:, :, k);

                        dWf(i,j,k) = (err1-err2)/(2*eps);
                    
                        obj.temp_Wf = rWf;
                    end
                end                
            end
            
            % Calculate numeric gradient wrt Wz
            for i = 1: size(obj.temp_Wz,1)
                for j = 1: size(obj.temp_Wz,2)
                    for k = 1: size(obj.temp_Wz,3)
                    
                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wz(i,j,k) = obj.temp_Wz(i,j,k) + eps;
                        obj.forward_backward(input_batch);
                        err1 = obj.d_hat_t_vec(:, :, k);

                        obj.h_t_pre = rh_pre;
                        obj.c_t_pre = rc_pre;
                        obj.temp_Wz(i,j,k) = obj.temp_Wz(i,j,k) - 2*eps;
                        obj.forward_backward(input_batch);
                        err2 = obj.d_hat_t_vec(:, :, k);

                        dWz(i,j,k) = (err1-err2)/(2*eps);
                    
                        obj.temp_Wz = rWz;
                    end 
                end                
            end

            
            
       end
       
   end
   
end



