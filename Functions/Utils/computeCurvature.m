clear all
close all

cd '/store/DPMMS/el425/bitbucket/FunctionalShapes/'
path_data = '/store/DPMMS/el425/bitbucket/FunctionalShapes/Data/Synthetic/'

% Add fshapes toolkit location
addpath(genpath('/store/DPMMS/el425/bitbucket/fshapes/Bin/'))
addpath(genpath('/store/DPMMS/el425/bitbucket/CurvatureEstimation/'))
%-------------------------
%           Load Surfaces
%-------------------------

filenames = dir([path_data 'original*.vtk']);
surfaces = cell(length(filenames),1);
for i=1:length(surfaces)
    surfaces{i,1} = import_fshape_vtk([path_data,filenames(i).name]);
end

for i=1:length(surfaces)
    FV = struct('faces', surfaces{i}.G,'vertices', surfaces{i}.x);
    [PrincipalCurvatures,PrincipalDir1,PrincipalDir2] = GetCurvatures(FV,0);
    surfaces{i}.f = (PrincipalCurvatures(1,:).*PrincipalCurvatures(2,:))';
    export_fshape_vtk(surfaces{i}, filenames(i).name);
end

[vertex_original,face_original,signal_original] = read_vtk_el([path_data 'canonical_synth.vtk'], false);
FV = struct('faces', face_original,'vertices', vertex_original);
[PrincipalCurvatures,PrincipalDir1,PrincipalDir2] = GetCurvatures(FV,0);
signal_original = (PrincipalCurvatures(1,:).*PrincipalCurvatures(2,:))';
hypertemplate = struct('x', vertex_original, 'G', face_original, 'f', signal_original);
export_fshape_vtk(hypertemplate, 'canonical.vtk');