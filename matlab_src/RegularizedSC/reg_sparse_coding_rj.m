function [B, S, stat] = reg_sparse_coding_rj(X, num_bases, Sigma, beta, gamma, num_iters, batch_size, initB, fname_save)
%
% Regularized sparse coding
%
% Inputs
%       X           -data samples, column wise
%       num_bases   -number of bases
%       Sigma       -smoothing matrix for regularization
%       beta        -smoothing regularization
%       gamma       -sparsity regularization
%       num_iters   -number of iterations 
%       batch_size  -batch size
%       initB       -initial dictionary
%       fname_save  -file name to save dictionary
%
% Outputs
%       B           -learned dictionary
%       S           -sparse codes
%       stat        -statistics about the training
%
% Written by Jianchao Yang @ IFP UIUC, Sep. 2009.
% 
% %%%%%%%  Edit log:   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Robert Jones, 03-25-2024
%       [ Edited L1QP_FeatureSign_Set_rj, getObjective_RegSc_rj,
%       l2ls_learn_basis_dual_rj; Added fields to `stat` struct ]

pars = struct;
pars.patch_size = size(X,1);
pars.num_patches = size(X,2);
pars.num_bases = num_bases;
pars.num_trials = num_iters;
pars.beta = beta;
pars.gamma = gamma;
pars.VAR_basis = 1; % maximum L2 norm of each dictionary atom

if ~isa(X, 'double'),
    X = cast(X, 'double');
end

if isempty(Sigma),
	Sigma = eye(pars.num_bases);
end

if exist('batch_size', 'var') && ~isempty(batch_size)
    pars.batch_size = batch_size; 
else
    pars.batch_size = size(X, 2);
end

if exist('fname_save', 'var') && ~isempty(fname_save)
    pars.filename = fname_save;
else
    pars.filename = sprintf('NewResults/reg_sc_b%d_%s', num_bases, datestr(now, 30));	
end

pars

% initialize basis
if ~exist('initB') || isempty(initB)
    B = rand(pars.patch_size, pars.num_bases)-0.5;
	B = B - repmat(mean(B,1), size(B,1),1);
    B = B*diag(1./sqrt(sum(B.*B)));
else
    disp('Using initial B...');
    B = initB;
end

[L M]=size(B);

t=0;
% statistics variable
stat= [];
stat.fobj_avg = [];
stat.elapsed_time=0;
stat.sparsity = [];
stat.stime=[];
stat.btime=[];
stat.fresidue = [];
stat.fsparsity = [];
stat.fregs = [];
stat.fobj = [];

% optimization loop
while t < pars.num_trials
    t=t+1;
    fprintf('  t=%d\n',t);

    start_time= cputime;
    stat.fobj_total=0;    
    % Take a random permutation of the samples
    indperm = randperm(size(X,2));
        
    for batch=1:(size(X,2)/pars.batch_size)
        % This is data to use for this step
        batch_idx = indperm((1:pars.batch_size)+pars.batch_size*(batch-1));
        Xb = X(:,batch_idx);
        
        % learn coefficients (conjugate gradient)  
        fprintf('  L1QP_FeatureSign_Set...\n');
        stic=tic;
%         S = L1QP_FeatureSign_Set(Xb, B, Sigma, pars.beta, pars.gamma);
        S = L1QP_FeatureSign_Set_rj(Xb, B, Sigma, pars.beta, pars.gamma);
        
        stoc=toc(stic);
        stat.stime(t)=stoc;
        
%         sparsity(end+1) = length(find(S(:) ~= 0))/length(S(:));
        stat.sparsity(t) = length(find(S(:) ~= 0))/length(S(:));
        
        % get objective
        fprintf('  getObjective_RegSc...\n');
%         [fobj, fresidue, fsparsity, fregs] = getObjective_RegSc(Xb, B, S, Sigma, pars.beta, pars.gamma);  
        [fobj, fresidue, fsparsity, fregs] = getObjective_RegSc_rj(Xb, B, S, Sigma, pars.beta, pars.gamma);  
        stat.fobj_total = stat.fobj_total + fobj;
        stat.fobj(t) = fobj;
        stat.fresidue(t) = fresidue;
        stat.fsparsity(t) = fsparsity;
        stat.fregs(t) = fregs;

        % update basis
        fprintf('  l2ls_learn_basis_dual...\n');
        btic=tic;
%         B = l2ls_learn_basis_dual(Xb, S, pars.VAR_basis);
%         [B,opts,res] = l2ls_learn_basis_dual_rj(Xb, S, pars.VAR_basis);
        [B,~,~] = l2ls_learn_basis_dual_rj(Xb, S, pars.VAR_basis);
%         [B,opts,res] = l2ls_learn_basis_dual_rj_intpnt(Xb, S, pars.VAR_basis);
        btoc=toc(btic);
        stat.btime(t)=btoc;

    end
    
    % get statistics
    stat.fobj_avg(t)      = stat.fobj_total / pars.num_patches;
    stat.elapsed_time(t)  = cputime - start_time;
    
    fprintf(['epoch= %d, sparsity = %f, fobj= %f, took %0.2f ' ...
             'seconds\n'], t, mean(stat.sparsity), stat.fobj_avg(t), stat.elapsed_time(t));
         
    % save results
    fprintf('saving results ...\n');
%     experiment = [];
%     experiment.matfname = sprintf('%s.mat', pars.filename);     
%     save(experiment.matfname, 't', 'pars', 'B', 'stat');
%     fprintf('saved as %s\n', experiment.matfname);

    experiment = [];
    experiment.matfname = sprintf('%s_iter%d.mat', pars.filename, t);     
    save(experiment.matfname, 't', 'pars', 'B', 'stat');
    fprintf('saved as %s\n', experiment.matfname);
end

return

function retval = assert(expr)
retval = true;
if ~expr 
    error('Assertion failed');
    retval = false;
end
return
