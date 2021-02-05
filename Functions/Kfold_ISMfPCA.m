function [fPCAOut_kfold] = Kfold_ISMfPCA(Y,Psi,forwardOp,laplmat,massmat,R,loglambdaseq,niter,K)
%
% Kfold_ISMfPCA Reconstruct PC functions in an inverse problem setting with K-fold cross-validated choice of the hyperparameter \lambda
%
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
%    loglambda  = grid (vector) of log(\lambda) to be cross-validated
%    niter_fpca = Number of iteration for each PC
%    K          = Number of folds for cross-validation
%
% Output:
%   fPCAOut.f_pca  = #(mesh nodes) x R: FE coefficients of the R PC functions at the mesh nodes (L^2 norm normalized)
%   fPCAOut.v_fpca = #dipoles x R: Evaluation of the (L^2 norm normalized) R PC functions at the dipoles' locations 
%   fPCAOut.u_fpca = n x R matrix: Scores vectors associated with the PC functions



[n,~] = size(Y); % n is number of detectors
ph = size(laplmat,1); %ph is number of mesh nodes
p_ev = size(forwardOp,2);   % number of dipoles

% Copy data matrix
Y = Y - ones(size(Y,1),1)*mean(Y);
YY = Y;

nlambda = length(loglambdaseq);
%f_hat_mat = zeros(nbasis, nlambda);

fPCAOut_kfold.v_fpca = zeros(ph,R);
fPCAOut_kfold.u_fpca = zeros(n,R);
fPCAOut_kfold.CVseq = cell(R,1);
for r = 1:R
    disp(['Computing PC function ',num2str(r)]);
    CVseq = zeros(1,nlambda);
    %parfor
    for ilambda = 1:nlambda
        disp(['   Computing CV for loglambda = ', num2str(loglambdaseq(ilambda))]);
        loglambda = loglambdaseq(ilambda);
        
        folds = cvpartition(size(YY,1),'KFold',K);

        for ifold = 1:K
            YY_train = YY(folds.training(ifold),:);
            YY_valid = YY(folds.test(ifold),:);
        
            fPCA = ISMfPCA(YY_train,Psi,forwardOp,laplmat,massmat,1,loglambda,niter);
            fs = fPCA.v_fpca(:,1);
            proj_v = forwardOp*fs/norm(forwardOp*fs);
            CVseq(ilambda) = CVseq(ilambda) + sum(sum((YY_valid - YY_valid*proj_v*(proj_v')).^2))/numel(YY_valid);
        end
    %disp(['CV index computed for log(lambda)=' num2str(loglambdaseq(ilambda))])
    end
    
    fPCAOut_kfold.CVseq{r} = CVseq;
    [~,ilambdachosen] = min(CVseq);
    loglambda = loglambdaseq(ilambdachosen);
    disp(['PC function ' num2str(r) '; Optimal log(lambda) = ' num2str(loglambda) ' index: ' num2str(ilambdachosen)])
    if (ilambdachosen == 1 || ilambdachosen == nlambda)
        disp('WARNING');
    end
    
    % Compute best solution and store the output
    fPCA = ISMfPCA(YY,Psi,forwardOp,laplmat,massmat,1,loglambda,niter);
    fPCAOut_kfold.v_fpca(:,r)= fPCA.v_fpca(:,1);
    fPCAOut_kfold.u_fpca(:,r)= fPCA.u_fpca(:,1);
    fPCAOut_kfold.f_fpca(:,r)= fPCA.f_fpca(:,1);
    fPCAOut_kfold.sd_fpca(:,r)= fPCA.sd_fpca(:,1);
    
    
    % remove compoment from data
    fs = fPCA.v_fpca(:,1);
    proj_v = forwardOp*fs/norm(forwardOp*fs);
    YY = YY - YY*proj_v*(proj_v');
end