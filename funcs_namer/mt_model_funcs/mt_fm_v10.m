function [ fit, k_fm_filt ] = mt_fm_v10( dM_in, dM_in_indices, Mn, Cfull, km, ...
    tse_traj, U , full_msk_pxls, xprev, exp_str, kfilter,pad)
% v10 of the motion forward model

%%                             Precomputations                           %%

% reshape motion vectors
dM_in_all = zeros(numel(Mn),1);
dM_in_all(dM_in_indices) = dM_in;
dM_in_all_mtx = reshape(dM_in_all, size(Mn));
Ms = Mn + dM_in_all_mtx;

%% evaluate forward model
k_fm = A_v10(xprev(full_msk_pxls),U,Cfull,tse_traj,Ms,full_msk_pxls,pad);

% weight the kspace data
km_filt = km .* kfilter;
k_fm_filt = k_fm .* kfilter;

% apply k-space weighting filter
fit_filt = norm(k_fm_filt(:)-km_filt(:))/norm(km_filt(:));

fit = fit_filt;

if (~isempty(exp_str))
    save(strcat(exp_str,'_fixed_hr_tmp.mat'),'Mn','dM_in','fit')
end

end










