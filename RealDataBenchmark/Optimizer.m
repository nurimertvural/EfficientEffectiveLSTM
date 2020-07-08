classdef Optimizer < handle
    properties
        lr      % learning_rate          - Float
        alpha   % first  order momentum  - Float
        beta    % second order momentum  - Float
        beta_1  % first  order momentum  - Float (For Adam)
        beta_2  % second  order momentum - Float (For Adam)
        eps     % Epsilon
        
        Vh     % First auxiliary matrix  - Float Matrix
        Sh     % Second auxiliary matrix - Float Matrix
        
        idx     % Time step for Adam correction
        
        
        optimizer
        optimizer_name
        
        Ptx
        Tx
        
        Q_init
        Q_last
        R_init
        R_last
        
    end
    
    methods
        function obj = Optimizer(opt, varargin)       
            p = inputParser;
            
            expectedFun = {'SGD','Adam','RmsProp', 'EKF', 'DEKF'};
            
            p.addRequired('Optimizer' , @(x) any(validatestring(x,expectedFun)) );
            
            p.addOptional('LRate' , 0.005 , @(x) x>0  & 1>x);
            p.addOptional('Alpha' , 0     , @(x) x>=0 & 1>x);
            p.addOptional('Beta'  , 0.99  , @(x) x>0  & 1>x);
            p.addOptional('Beta1' , 0.9   , @(x) x>0  & 1>x);
            p.addOptional('Beta2' , 0.999 , @(x) x>0  & 1>x);
            p.addOptional('eps'   , 1e-8  , @(x) x>0  & 1>x);
            
            p.addOptional('Rinit' , 1   , @(x) x>0 );
            p.addOptional('Rlast' , 0.5    , @(x) x>0 );
            p.addOptional('Qinit' , 1e-4 , @(x) x>0 );
            p.addOptional('Qlast' , 1e-4 , @(x) x>0 );
            

            p.parse(opt,varargin{:});
            
            obj.lr      = p.Results.LRate;
            obj.alpha   = p.Results.Alpha;
            obj.beta    = p.Results.Beta;
            obj.beta_1  = p.Results.Beta1;
            obj.beta_2  = p.Results.Beta2;
            obj.eps     = p.Results.eps;
            
            obj.Q_init = p.Results.Qinit;
            obj.Q_last = p.Results.Qlast;
            obj.R_init = p.Results.Rinit;
            obj.R_last = p.Results.Rlast;

            
            obj.idx   = 0;
            
            obj.optimizer_name = p.Results.Optimizer;
            
            if(obj.optimizer_name == "SGD")
                obj.optimizer = @obj.SGD;
            elseif(obj.optimizer_name == "Adam")
                obj.optimizer = @obj.Adam;
            elseif(obj.optimizer_name == "RmsProp")
                obj.optimizer = @obj.RmsProp;
            elseif(obj.optimizer_name == "EKF")
                obj.optimizer = @obj.EKF; 
            elseif(obj.optimizer_name == "DEKF")
                obj.optimizer = @obj.DEKF; 
            end

        end
        
        function [theta, S1, V1] = update(obj, theta, dtheta)
            
            [theta, S1, V1] = obj.optimizer( theta , dtheta);
            
        end
    end
    
    methods(Access = private)
        function [theta, V1, S1 ] = SGD(obj, theta, dtheta, error_vec)
            
            % Copy object variables to the local variables and clean them.
            V1 = obj.Vh; S1 = obj.Sh;
            obj.Vh = []; obj.Sh = [];
            
            % SGD
            dtheta = - dtheta * error_vec;
            
            V1 = obj.lr*dtheta + obj.alpha * V1;
            
            theta = theta - V1;
            

        end
        
        function [theta, V1, S1] = RmsProp(obj, theta, dtheta, error_vec)
            % Kaynak: Andrew NG'nin dersi.
            
            % Copy object variables to the local variables and clean them.
            V1 = obj.Vh; S1 = obj.Sh;
            obj.Vh = []; obj.Sh = [];
            
            % RmsProp
            dtheta = - dtheta * error_vec;
            
            S1 = obj.beta*S1 + (1-obj.beta) * dtheta.^2;
            
            nWh = sqrt(S1) + obj.eps; % Normalization factors
            
            theta = theta - obj.lr * dtheta./nWh;
            
            
              
        end
        
        function [theta, V1, S1] = Adam(obj, theta, dtheta, error_vec)
            % Kaynak: Andrew NG'nin dersi.
            
            % Copy object variables to the local variables and clean them.
            V1 = obj.Vh; S1 = obj.Sh;
            obj.Vh = []; obj.Sh = [];
            
            % Adam
            dtheta = - dtheta * error_vec;
            
            obj.idx = obj.idx + 1;
            
            S1 = obj.beta_2*S1 + (1-obj.beta_2) * dtheta.^2;
            
            S_h_cor = S1/(1-obj.beta_2^obj.idx);   
            
            V1 = obj.beta_1*V1 + (1-obj.beta_1) * dtheta;
            
            V_h_cor = V1/(1-obj.beta_1^obj.idx);  
            
            nWh = sqrt(S_h_cor) + obj.eps; % Normalization factors
            
            theta = theta - obj.lr * V_h_cor./nWh;
            
            
        end
        
        function [theta, V1, S1] = EKF(obj, theta, dtheta, error_vec)
            
            % Copy object variables to the local variables and clean them.
            V1 = obj.Vh; S1 = obj.Sh;
            obj.Vh = []; obj.Sh = [];
            
            % EKF
            obj.idx = obj.idx + 1;
            Q = obj.Q_init + (obj.idx-1)/(obj.Tx-1) *  (obj.Q_last - obj.Q_init);
            R = obj.R_init + (obj.idx-1)/(obj.Tx-1) *  (obj.R_last - obj.R_init);
            
            intm  = V1 * dtheta;
            intm2 = dtheta' * intm;
            
            G_t   = intm * 1/(intm2 + R);
            
            theta = theta + G_t * error_vec; 
            
            V1 = V1 - G_t * intm' + Q * eye( size(V1,1) );
        end
        
        function [theta_hid, theta_out, V1, S1] = DEKF(obj, theta_hid, theta_out, dtheta_hid, dtheta_out, error_vec)
            
            % Copy object variables to the local variables and clean them.
            V1 = obj.Vh; S1 = obj.Sh;
            obj.Vh = []; obj.Sh = [];
            
            % EKF
            obj.idx = obj.idx + 1;
            Q = obj.Q_init + (obj.idx-1)/(obj.Tx-1) *  (obj.Q_last - obj.Q_init);
            R = obj.R_init + (obj.idx-1)/(obj.Tx-1) *  (obj.R_last - obj.R_init);
            
            intm_hid  = mtimesx( V1, dtheta_hid );
            intm2_hid = mtimesx( dtheta_hid, 'C' , intm_hid ); 
            
            intm_out  = S1 * dtheta_out;
            intm2_out = dtheta_out' * intm_out; 
            
            intm2_all = sum(intm2_hid, 3) + intm2_out;
            
            % Calculate Gain Matrices
            G_hid = intm_hid ./ (intm2_all + R);
            G_out = intm_out ./ (intm2_out + R);
            
            % Updates
            theta_hid = theta_hid + G_hid * error_vec;
            theta_out = theta_out + G_out * error_vec;
            
            G_hid( G_hid == 0) = 1e-15;
            V1 = (V1./G_hid - permute(intm_hid, [ 2 1 3])).*G_hid ...
                                    + Q * eye( size(V1, 1) ) .* ones(1, 1, size(V1, 3) );
            
            G_out( G_out == 0) = 1e-15;
            S1 = (S1./G_out - intm_out').*G_out + Q * eye( size( S1, 1) );
        end   
        
    end
    
    
end