%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inverse FPCA model on brain geometry of HCP subject 100307
% This script uses the functions: 
%   ISMfPCA, ISMfPCA_Kfold
% Plots and regularization FieldTrip
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Requirements: 
%   -Felicity
%   -FieldTrip (Libs/fieldtrip-20170912/ft_defaults)

% setup the execution environment
%opengl software;
clear
run Libs/fieldtrip-20170912/ft_defaults; % Change "Libs/fieldtrip-20170912"  to path of fieldtrip library
addpath(genpath('Libs/FELICITY'));       % Change "Libs/FELICITY" to path of FELICITY library
addpath(genpath('Functions'))            % Path to some functions

Init                                     % Compile FELICITY scripts for FE matrices computation


% Fieldtrip param setting
global ft_default
ft_default.trackcallinfo = 'no';
%% Load data of subject 100307

% Data in subject_data (sensors positions, headmodel, cortical surface,..)
load('Data/subject_data_100307')

%% Plot brain/ MEG sensors and forward operator conversion to matrix form

% Plot head sensors, head model and cortical surface for subject 100307
h1 = figure(1); hold on; 
subplot(1,2,1); title('Sensors and head model');
ft_plot_sens(subject_data.grad , 'style', '*b');
ft_plot_vol(subject_data.headmodel, 'facealpha', 1, 'edgealpha', 0.1); view(0,0);
subplot(1,2,2); title('Cortical surface');
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black', 'facecolor', 'white', 'vertexalpha', 0); view(0,0);

% Plot MEG sensors layout for subject 100307
cfg = []; cfg.layout = '4D248_helmet.mat';
cfg.layout = ft_prepare_layout(cfg);
figure; ft_plot_lay(cfg.layout);
  
% Count the number of channels and leadfield components
%p_chan = size(subject_data.gridLF.leadfield{1},1); %Number of MEG sensors (channels)
%Nsource = 0;                                       %Number of sources 3x8004 (mesh nodes)
%for i=1:size(subject_data.gridLF.pos,1)
%Nsource = Nsource + size(subject_data.gridLF.leadfield{i}, 2);
%end

% Pick direction to project 3D dipoles and reduce dimension problem (just
% for speeding up simulations)
% perp_dir = normals(subject_data.sourcemodel.pos,subject_data.sourcemodel.tri,'vertex');
perp_dir = subject_data.sourcemodel.pos-ones(size(subject_data.sourcemodel.pos,1),1)*mean(subject_data.sourcemodel.pos);
perp_dir = perp_dir./(sqrt(sum(perp_dir.^2,2))*ones(1,3));

p_dipoles = size(subject_data.gridLF.pos,1); % Number of dipoles locations
p_chan = length(subject_data.channels);
forwardOP = zeros(p_chan, p_dipoles);        % Forward operator
for i=1:p_dipoles
forwardOP(:,i) = subject_data.gridLF.leadfield{i}*(perp_dir(i,:)');
end

% Forward Operator matrix
size(forwardOP)

%% Example of the classical inverse problem

% Object geom defines cortical surface geometry
geom.x = subject_data.sourcemodel.pos; % mesh nodes locations
geom.G = subject_data.sourcemodel.tri; % mesh triangles
p_nodes = size(subject_data.sourcemodel.pos,1);

% Compute FE mass and stiffness matrices from cortical surface
[mass,stiff] = computeFEM(geom);
mass_lumped_inv = sparse(1:p_nodes,1:p_nodes,1./sum(mass,2));

stiff_lh = stiff(1:p_nodes/2,1:p_nodes/2); %mass matrix left hemisphere 

% Comput LB eigenfunctions
[V_lb,~] = eigs(stiff_lh, 20,'smallestabs','Tolerance',1e-8);

ncomp = 4; % Choose LB eig to use
function_on_cortex = [V_lb(:,ncomp);V_lb(:,ncomp)];   %Function (vector-valued) on the brain space 
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',function_on_cortex)

% Noisy generation of signal on sensors space
sigma = 0.01;
signal_detected = forwardOP*function_on_cortex + normrnd(zeros(size(forwardOP,1),1),sigma); %Signal on the sensors space 


lambda = 100; %smoothing parameter
A = [forwardOP; sqrt(lambda)*sqrt(mass_lumped_inv)*stiff]; % solve inverse problem

%Reconstruct signal on brain space from forwardOP*signal (on sensors space)
[f_source,~] = lsqr(A,[signal_detected; zeros(p_dipoles,1)], 1e-5); 

% Plot reconstruction
h2 = figure(2);
subplot(1,3,1); title('Original signal (Energy)')
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',function_on_cortex); view(-90,90);
subplot(1,3,2)
tmpclass = subject_data.comp_class;
tmpclass.topo = signal_detected;
ft_topoplotIC(subject_data.cfgtopo, tmpclass);
subplot(1,3,3); title('Reconstructed signal');
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_source); view(-90,90);

%% Define modes of variation of PCA model

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

f_mode'*mass*f_mode % check orthogonality

% Plot modes of variation
h_modes = figure(4);
subplot(1,3,1); title('PC function 1')
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_mode(:,1));
subplot(1,3,2); title('PC function 2')
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_mode(:,2));
subplot(1,3,3); title('PC function 3')
ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_mode(:,3));

%% Generate data from PC covariance functions

sig_mode = [5;3;2];  % standard deviation of each mode of variation
sig_noise = 0;     % variance of noise on SENSORS space
n = 50;       % Number of observations for each dataset

Cov = cell(n,1);
rng(0);

% Generate sqrt of subj-specific variances
u_covscore = (ones(n,1)*sig_mode').*randn(n,3);
%sqrt(mean(uscore.^2))

for i = 1:n
    n_detectors = size(forwardOP,1);
    Cov{i}.var = u_covscore(i,:).^2;
    Cov{i}.err = sig_noise.*randn(p_chan,p_chan);

    % Generate covariance on sensors space from PC covariance functions on
    % brain space (OBS: covariance on brain space not storable)
    Cov{i}.grad = zeros(p_chan,p_chan);
    for r = 1:3
        Cov{i}.grad = Cov{i}.grad + Cov{i}.var(r)*(forwardOP*f_mode(:,r))*(forwardOP*f_mode(:,r))';
    end
    %Covariance on sensors space
    Cov{i}.grad = Cov{i}.grad + Cov{i}.err*Cov{i}.err';

    %Eigenvalue dec of covariance on sensors space
    [Cov{i}.V, Cov{i}.D] = eig(Cov{i}.grad);
    Cov{i}.V = real(Cov{i}.V); Cov{i}.D = real(Cov{i}.D);
    Cov{i}.D = diag(Cov{i}.D);   
    Cov{i}.D(Cov{i}.D<0) = 0; %Rounding issue
    [~,permutation]=sort(Cov{i}.D, 'descend'); Cov{i}.D=Cov{i}.D(permutation);Cov{i}.V=Cov{i}.V(:,permutation);
    %Square root representation of the covariance on the sensors space
    Cov{i}.S2 = Cov{i}.V * diag(sqrt(Cov{i}.D));
    %Cov{i}.C = chol(Cov{i}.grad, 'lower');
    Cov{i} = rmfield(Cov{i},{'D','V'});
end

%% Plot covariances

% for i = 1:n
%     indices = cellfun(@(x) sscanf(x, 'A%d'),subject_data.channels);
%     figure(2); clf;
%     CovYY = NaN(248,248); 
%     CovYY(indices,indices) = Cov{i}.grad;
%     imagesc(CovYY)    
%     title(['Covariance ' num2str(i)]);
%     colorbar;pause(2);
% end

%% Fixed choice of lambda


Psi = speye(p_dipoles); % Location dipoles and mesh nodes same, so Psi = Id

% Number iterations Inverse SM-FPCA algorithm
niter_fpca = 10;
% Number of PCs to be computed
R_comp = 3;

loglambda = 1;

% Apply Inverse SM-FPCA algorithm
Cov_sqrt = cellfun(@(x) x.S2,Cov,'UniformOutput',false);
covfPCAOut = ISMCovfPCA(Cov_sqrt,Psi,forwardOP,stiff,mass,R_comp,loglambda,niter_fpca);

% Plot PC functions reconstruction
figure;
for r = 1:R_comp
    subplot(2,3,r);ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',f_mode(:,r));
    title(['PC function ' ,num2str(r)])
    if(covfPCAOut.f_fpca(:,r)'*f_mode(:,r) < 0) 
        covfPCAOut.f_fpca(:,r) = -covfPCAOut.f_fpca(:,r); 
    end
    
    subplot(2,3,r+3); ft_plot_mesh(subject_data.sourcemodel,'edgecolor', 'black','edgealpha', 0.01,'vertexcolor',covfPCAOut.f_fpca(:,r));   
    title('Estimation')
end


% Scores
figure;
for r = 1:R_comp
    subplot(1,3,r); scatter(cellfun(@(x) x.var(r),Cov), covfPCAOut.var_fpca(:,r))
    title(['Estimated vs True Scores ', num2str(r)])
    daspect([1 1 1])
end
