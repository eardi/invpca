%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PC functions estimation in an inverse problem context
%
%  Requirements: 
%   -Felicity: https://github.com/walkersw/felicity-finite-element-toolbox/wiki
%   -FieldTrip: https://www.fieldtriptoolbox.org/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% environment setup
clear
run Libs/fieldtrip-20170912/ft_defaults; % Load fieldtrip(Change "Libs/fieldtrip-20170912" to path of fieldtrip library)
addpath(genpath('Libs/FELICITY'));       % Replace "Libs/FELICITY" with path to FELICITY library
addpath(genpath('Functions'))            % Path to internal functions

Init                                     % Compile FELICITY scripts for FE matrices computation

global ft_default                        % Fieldtrip param setting
ft_default.trackcallinfo = 'no';
%% Load data and plot brain/ MEG sensors

% Data in subject_data (sensors positions, cortical surface,..)
load('Data/subject_data')

% Plot head sensors, head model and cortical surface for subject 100307
h1 = figure(1); hold on; 
title('MEG Detectors and brain model');
ft_plot_sens(subject_data.grad , 'style', '*b');
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black', 'facecolor', 'white', 'vertexalpha', 0); view(0,0);

% Plot MEG sensors layout for subject 100307
cfg = []; cfg.layout = '4D248_helmet.mat';
cfg.layout = ft_prepare_layout(cfg);
figure; ft_plot_lay(cfg.layout);
  
% Load forward operator

forwardOP = subject_data.forwardOP;
[p_chan,p_dipoles] = size(forwardOP);

%% Example of a simple signal reconstruction

% Object geom defines cortical surface geometry
geom.x = subject_data.sourcemodel.pos; % mesh nodes locations
geom.G = subject_data.sourcemodel.tri; % mesh triangles
p_nodes = size(geom.x,1);

% Compute FE mass and stiffness matrices from cortical surface
[mass,stiff] = computeFEM(geom); %L2 mass matrix and laplacian
mass_lumped_inv = sparse(1:p_nodes,1:p_nodes,1./sum(mass,2)); % Lumped inverse mass matrix
stiff_lh = stiff(1:p_nodes/2,1:p_nodes/2); %mass matrix left hemisphere 

% Compute LB eigenfunctions
[V_lb,~] = eigs(stiff_lh, 20,'smallestabs','Tolerance',1e-8);

ncomp = 4; % Choose LB eigenfunctions to use
function_on_cortex = [V_lb(:,ncomp);V_lb(:,ncomp)];   %Function (vector-valued) on the brain space 
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',function_on_cortex)

% Noisy generation of signal on sensors space
sigma = 0.01;
signal_detected = forwardOP*function_on_cortex + normrnd(zeros(size(forwardOP,1),1),sigma); %Signal on the sensors space 

lambda = 1; %smoothing parameter
A = [forwardOP; sqrt(lambda)*sqrt(mass_lumped_inv)*stiff]; % solve inverse problem

%Reconstruct signal on brain space from forwardOP*signal (on sensors space)
[f_source,~] = lsqr(A,[signal_detected; zeros(p_dipoles,1)], 1e-5); 

% Plot reconstruction
h2 = figure(2);
subplot(1,3,1); title('Original signal')
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',function_on_cortex); view(-90,90);
subplot(1,3,2)
tmpclass = subject_data.comp_class;
tmpclass.topo = signal_detected;
ft_topoplotIC(subject_data.cfgtopo, tmpclass);
subplot(1,3,3); title('Reconstructed signal');
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_source); view(-90,90);

%% Define modes of variation of PCA model

% Compute LB eigenfunctions
[V_lb,~] = eigs(stiff_lh,20,'smallestabs','Tolerance',1e-8);

% Generate and reconstruct signals on sensors space 
% to define subspace of reconstructible signals on the brain space
signals_brain = [V_lb(:,2:10);V_lb(:,2:10)];
signals_sens = forwardOP*signals_brain;
A_rec = [forwardOP; 10*sqrt(mass_lumped_inv)*stiff];
signals_brain_rec = zeros(size(signals_brain));

for i=1:size(signals_brain,2)
      signals_brain_rec(:,i) = lsqr(A_rec,[signals_sens(:,i); zeros(p_dipoles,1)], 1e-8, 100); 
end

% Compute orthogonal fuction basis on brain space
[U_rec,D_rec,F_rec] = svd(signals_brain_rec');

% Define L2-orthogonal PC functions (modes of variation) to be estimates
f_mode = F_rec([2,3,5],:)'; 
SC = f_mode'*mass*f_mode;
SQ_sq_inv = inv(chol(SC));
f_mode = f_mode*SQ_sq_inv;

f_mode'*mass*f_mode % check L2-orthogonality

% Plot modes of variation
h_modes = figure(4);
subplot(1,3,1); title('PC function 1')
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_mode(:,1));
subplot(1,3,2); title('PC function 2')
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_mode(:,2));
subplot(1,3,3); title('PC function 3')
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_mode(:,3));

%% Generate data on brain space

sig_mode = [3;2;1];  % standard deviation of each mode of variation
n = 50;       % Number of observations for each dataset

% Generate n functions on the brain space
rng(0)
X_no_noise = zeros(n,p_dipoles);
uscore = zeros(n,3);
for i = 1:n
    uscore(i,:) = sig_mode.*randn(3,1);
    for r = 1:length(sig_mode)
       X_no_noise(i,:) = X_no_noise(i,:) + f_mode(:,r)'*uscore(i,r);
    end
end

% Plot the n functions on the brain space
for i = 1:n
    figure(4); clf;
    ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',X_no_noise(i,:)');
    title(['signal sample ',num2str(i)]);
    pause(0.02);
end

% Push forward data on sensors space without noise
Y = (forwardOP*X_no_noise')';

% Sd of the noise on the sensors space
sig_noise = 1;

% Add noise on the sensors space
YN = Y + sig_noise*randn(size(Y));

%% Plot brain and sensor space data

i_obs = 1; % Number observation to be plotted

figure; hold on;
subplot(1,3,1)
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',X_no_noise(1,:)');

subplot(1,3,2)
tmpclass = subject_data.comp_class; tmpclass.topo = Y(i_obs,:)';
ft_topoplotIC(subject_data.cfgtopo, tmpclass); view([90 90]);
colorbar('off') 

subplot(1,3,3)
tmpclass = subject_data.comp_class; tmpclass.topo = YN(i_obs,:)';
ft_topoplotIC(subject_data.cfgtopo, tmpclass); view([90 90]);
colorbar('off')

%% Reconstruction fixed lambda

% See ISMfPCA help for definintion. Here location dipoles and mesh nodes are the same, so Psi = Id
Psi = speye(p_dipoles); 

% Number iterations Inverse SM-FPCA algorithm
niter_fpca = 10;
% Number of PCs to be computed
R_comp = 3;

loglambda = 0;

% Apply Inverse SM-FPCA algorithm
% Data here is YN: n x #dipoles
fPCAOut = ISMfPCA(YN,Psi,forwardOP,stiff,mass,R_comp,loglambda,niter_fpca);

% L2 ortogonalize PC functions
f_fpca_orth = fPCAOut.f_fpca(:,1:3)*...
    inv(chol(fPCAOut.f_fpca(:,1:3)'*mass*fPCAOut.f_fpca(:,1:3)));

% Plot PC function reconstructions
figure;
for r = 1:R_comp
    subplot(2,3,r);ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_mode(:,r));
    title(['PC function ' ,num2str(r)])
    if(f_fpca_orth(:,r)'*f_mode(:,r) < 0) 
        f_fpca_orth(:,r) = -f_fpca_orth(:,r); 
    end
    
    subplot(2,3,r+3); ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_fpca_orth(:,r));   
    title('Estimation')
end

% Compare unnormalized estimated scores with true scores
figure;
for r = 1:R_comp
    subplot(1,3,r); scatter(fPCAOut.u_fpca(:,r)*fPCAOut.sd_fpca(r), uscore(:,r))
    title(['Estimated vs True Scores ', num2str(r)])
    daspect([1 1 1])
end

%% Reconstruction cross-validated lambda

% Number folds cross-validation
Kfolds = 5;
% Sequence of log-lambdas
loglambdaseq = 0:1:3;

fPCAOut_kfold = Kfold_ISMfPCA(YN,Psi,forwardOP,stiff,mass,R_comp,loglambdaseq,niter_fpca,Kfolds);

% L2 ortogonalize PC functions
f_fpca_orth = fPCAOut_kfold.f_fpca(:,1:3)*...
    inv(chol(fPCAOut_kfold.f_fpca(:,1:3)'*mass*fPCAOut_kfold.f_fpca(:,1:3)));

% Plot PC function reconstructions
figure;
for r = 1:R_comp
    subplot(2,3,r);ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_mode(:,r));
    title(['PC function ' ,num2str(r)])
    if(f_fpca_orth(:,r)'*f_mode(:,r) < 0)
        f_fpca_orth(:,r) = -f_fpca_orth(:,r);
    end
    subplot(2,3,r+3); ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_fpca_orth(:,r));   
    title('Estimation')
end

% Compare unnormalized estimated scores with true scores
figure;
for r = 1:R_comp
    subplot(1,3,r)
    scatter(fPCAOut_kfold.u_fpca(:,r)*fPCAOut_kfold.sd_fpca(r), uscore(:,r))
    title(['Estimated vs True Scores ', num2str(r)])
    daspect([1 1 1])
end

