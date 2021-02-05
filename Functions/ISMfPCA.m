function [fPCAOut] = ISMfPCA(Y,Psi,forwardOp,laplmat,massmat,R,loglambda,niter_fpca)
%
% ISMfPCA Reconstruct PC functions in an inverse problem setting 
% Model: $X_i = \sum_{r=1}^R \sqrt(\gamma_r) u_i f_r$,
% with $X_i$ subject-specific latent signal on the brain, $\gamma_{r} variance and $f_r$ $r$th common PC function
%
% Reference: Lila, Arridge, Aston (2020) Representation and reconstruction of covariance operators in 
% linear inverse problems. Inverse Problems. doi:10.1088/1361-6420/ab8713
%
%
% Input:
%    Y          = n x #detectors data-matrix. Entry Y(i,j) is the signal of the ith subject detected by the jth MEG detector
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
%   fPCAOut.f_pca  = #(mesh nodes) x R: FE coefficients of the R PC functions (L^2 normalized)
%   fPCAOut.v_fpca = #dipoles x R: Evaluations of the (L^2 norm normalized) R PC functions at the dipoles' locations 
%   fPCAOut.u_fpca = n x R matrix: Score vectors associated with the PC functions


%disp('Compute PC functions');

[n,~] = size(Y); % n is number of detectors
ph = size(laplmat,1); %ph is number of mesh nodes
p_ev = size(forwardOp,2);   % number of dipoles

% lsqr setting
tol  = 1e-8; %maxit = min(2*ndat,ns);
niter_lsqr = 100;
%disp(['-Linear System: maxit ' num2str(maxit)]);

lambda = 10^loglambda;
mass_lumped_inv = sparse(1:ph,1:ph,1./sum(massmat,2));
A = [forwardOp*Psi; sqrt(lambda)*sqrt(mass_lumped_inv)*laplmat];

Y = Y - repmat(mean(Y),size(Y,1),1);
YY = Y; % Remove mean from data matrix


fPCAOut.u_fpca = zeros(n,R);     % Scores vector output
fPCAOut.f_fpca = zeros(ph,R);    % PC function FE coeffs output
fPCAOut.v_fpca = zeros(p_ev,R);  % PC function ptws evaluations output
fPCAOut.sd_fpca = zeros(R,1);     % PC sd estimates

for r = 1:R
    %disp(['SM-FPCA PC #',num2str(m)]);
    [u_init,~] = svd(YY,'econ'); 
    u = u_init(:,1);               % Initialize score vector with svd solution
    %fs = zeros(p_ev,1);              % Pointwise eval PC function
    f = zeros(ph,1);               % PC function FE coeffs
    for ad_iter = 1:niter_fpca
        % update PC function estimate
        [f,~] = lsqr(A,[YY'*u; zeros(ph,1)], tol, niter_lsqr);
        %f = A\[YY'*u; zeros(ph,1)];
        fs = Psi'*f;
        % update scores estimate
        u = YY*forwardOp*fs/norm(YY*forwardOp*fs);
        %figure(5);%clf; 
        %hM.Display(fs,'lighting',0,'showcolorbar',0); 
        %title(['Inverse-Sparse fPCA #' num2str(i)]);
        %drawnow;
    end
    
    % Normalized wrt L2 functional norm
    f = f/sqrt(f'*massmat*f); fs = Psi'*f;
    fPCAOut.f_fpca(:,r)= f; fPCAOut.v_fpca(:,r)= fs;
    
    
    % Remove compoment from data
    proj_v = forwardOp*fs/norm(forwardOp*fs);
    YY = YY - YY*proj_v*(proj_v');
    
    % Compute \sigma minimizing \|Y - \sigma * u*(Kfs)'\|^2
    u_norm = Y*forwardOp*fs./sqrt(sum((Y*forwardOp*fs).^2));
    fPCAOut.sd_fpca(r) = u_norm'*Y*(forwardOp*fs)/(fs'*forwardOp'*forwardOp*fs);    
end

fPCAOut.u_fpca = Y*forwardOp*fPCAOut.v_fpca./sqrt(sum((Y*forwardOp*fPCAOut.v_fpca).^2,1)); %Norm squared!


