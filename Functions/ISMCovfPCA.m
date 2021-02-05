function [covfPCAOut] = ISMCovfPCA(cov_sqrt,Psi,forwardOp,laplmat,massmat,R,loglambda,niter)
% 
% ISMCovfPCA: Reconstruction model for covariance objects in an inverse problem setting 
% 
% Model: $C_i = \sum_{r=1}^R \gamma_{ir} f_r \otimes f_r$,
% with $C_i$ subject-specific latent covariance on the brain, $\gamma_{ir}$ subject-specific
% variance and $f_r$ $r$th common PC function
%
% Reference: Lila, Arridge, Aston (2020) Representation and reconstruction of covariance operators in 
% linear inverse problems. Inverse Problems. doi:10.1088/1361-6420/ab8713
%
% Input:
%    cov_sqrt   = list of square root decompositions of the covariances on the sensors space, 
%                 s.t. cov(i) = cov_sqrt(i)*cov_sqrt(i)'
%    Psi        = #dipoles x #(mesh nodes) matrix. Entry in pos (k, l) is the pointwise evaluation of the finite element 
%                 \psi_l at the location of the kth dipole p_k -- This is an identity matrix if all mesh nodes are modelled as dipoles
%    forwardOp  = #detectors x #dipoles matrix. Represents the linear forward operator relating the signal on the dipoles to the signal on the detectors
%    laplmat    = Laplacian matrix (defined as A in Lila et al (2020))
%    massmat    = L^2 Mass matrix (defined as M in Lila et al (2020))
%    R          = Number of PCs to be computed
%    loglambda  = The logarithm of the penalizing coeff \lambda
%    niter_fpca = Number of iteration for each PC
%
% Output:
%   covfPCAOut.f_pca         = #(mesh nodes) x R: FE coefficients of the R PC functions (L^2 normalized)
%   covfPCAOut.v_fpca        = #dipoles x R: Evaluations of the (L^2 norm normalized) R PC functions at the dipoles' locations 
%   covfPCAOut.var_fpca      = n x R: Subject-specific variances on the brain space
%   covfPCAOut.var_fpca_sens = n x R: Subject-specific variances on the sensors space


%disp('Compute PC functions');

p_chan = size(cov_sqrt{1}, 1); % Assume same forward operator
n = length(cov_sqrt);          %Number of subjects

YY_v_concat = zeros(p_chan*n,p_chan);
for i = 1:n
    indices_i = ((i-1)*p_chan+1):(i*p_chan);
    YY_v_concat(indices_i,:) = cov_sqrt{i}';
end


fPCAOut = ISMfPCA(YY_v_concat,Psi,forwardOp,laplmat,massmat,R,loglambda,niter);

% L2 ortogonalize PC functions
f_fpca_orth = fPCAOut.f_fpca(:,1:3)*...
    inv(chol(fPCAOut.f_fpca(:,1:3)'*massmat*fPCAOut.f_fpca(:,1:3)));
v_fpca_orth = fPCAOut.v_fpca(:,1:3)*...
    inv(chol(fPCAOut.v_fpca(:,1:3)'*massmat*fPCAOut.v_fpca(:,1:3)));

covfPCAOut.f_fpca = f_fpca_orth;
covfPCAOut.v_fpca = v_fpca_orth;

% Norm scores post-orthogonalization
norm_scores_source_orth = zeros(n,R);
norm_scores_sens_orth = zeros(n,R);
norm_scores_source = zeros(n,R);
norm_fpc_sens = sqrt(sum((forwardOp*fPCAOut.f_fpca).^2))';
for i = 1:n
    indices_i = ((i-1)*p_chan+1):(i*p_chan);
    [qq,rr] = qr(fPCAOut.u_fpca(indices_i,:),0);
    norm_scores_source_orth(i,:) = diag(rr.^2)'.*(fPCAOut.sd_fpca.^2)';
    norm_scores_sens_orth(i,:) = diag(rr.^2)'.*(norm_fpc_sens.^2)';
    norm_scores_source(i,:) = sum(fPCAOut.u_fpca(indices_i,:).^2,1)*(fPCAOut.sd_fpca.^2);
end

covfPCAOut.var_fpca = norm_scores_source_orth;
covfPCAOut.var_fpca_sens = norm_scores_sens_orth;

