function [ fit, x, pcg_out, k_fm] = mt_fit_fcn_v10( dM_in, dM_in_indices, Mn, Cfull, km, ...
    tse_traj, U , tar_pxls , full_msk_pxls, xprev, exp_str, kfilter,pad)


%%                             Precomputations                           %%

%%% Currently hardcoded values
iters = 20;
lambda = 0;

[nlin, ncol, nsli, ~] = size(U);
fixed_pxls = setdiff(full_msk_pxls,tar_pxls);

% reshape motion vectors
dM_in_all = zeros(numel(Mn),1);
dM_in_all(dM_in_indices) = dM_in;
dM_in_all_mtx = reshape(dM_in_all, size(Mn));
Ms = Mn + dM_in_all_mtx;


%% call pcg

% x vector "fixed" pixels (not being updated in pcg)
x_v_f = xprev(fixed_pxls);

% find RHS (right hand side) for normal equations
Afxf = A_v10(x_v_f,U,Cfull,tse_traj,Ms,fixed_pxls,pad);
AtsAfxf = Astar_v10(Afxf,U,Cfull,tse_traj,Ms,tar_pxls,pad);
RHS = Astar_v10(km,U,Cfull,tse_traj,Ms,tar_pxls,pad) - AtsAfxf;

% find x_v_t (x vector "targetted" pixels)
if (~isempty(xprev))
    [x_v_t, f, rr, it] = pcg(@(x)...
        LHS_v10(x,U,Cfull,tse_traj,Ms,lambda,tar_pxls,pad), RHS, 1e-3, iters, [], [],...
        reshape(xprev(tar_pxls),numel(tar_pxls),1));
else
    [x_v_t, f, rr, it] = pcg(@(x)...
        LHS_v10(x,U,Cfull,tse_traj,Ms,lambda,tar_pxls,pad), RHS, 1e-3, iters);
end
pcg_out = [f, rr, it];

%% Re-evaluate Forward Model with new x

% combine x_fixed and x_targetted into one vector
x_vol = zeros(nlin,ncol,nsli);
x_vol(fixed_pxls) = x_v_f; x_vol(tar_pxls) = x_v_t;
x_v_all = x_vol(full_msk_pxls);

% shift x based on zero-padding used
pxl_per_sli = numel(full_msk_pxls)/nsli;
x_v = zeros(numel(full_msk_pxls),1);
x_v(pad*pxl_per_sli+1:end-pad*pxl_per_sli) = x_v_all(pad*pxl_per_sli+1:end-pad*pxl_per_sli);

% call forward model
k_fm = A_v10(x_v,U,Cfull,tse_traj,Ms,full_msk_pxls,pad);

% weight the kspace data
km_filt = km .* kfilter;
k_fm_filt = k_fm .* kfilter;

% calculate data consistency L2 norm
fit_filt = norm(k_fm_filt(:)-km_filt(:))/norm(km_filt(:));

fit = fit_filt;

% save intermediate steps of optimization if and experiment string is
% passed into the function
if (~isempty(exp_str))
    save(strcat(exp_str,'_tmp.mat'),'Mn','dM_in','fit')
end

%% put x back in full image matrix
x = zeros(nlin,ncol,nsli);
x(full_msk_pxls) = x_v;

end


%% "Left Hand Side" fucntion, i.e. A*A + Tik in normal equations     %%%%%%%%%%
function [output] = LHS_v10(x,U,Cfull,tse_traj,Ms,lambda,nz_pxls,pad)

AsAx = AsA_v10(x,U,Cfull,tse_traj,Ms,nz_pxls,pad);
output = AsAx + lambda*x;

end







