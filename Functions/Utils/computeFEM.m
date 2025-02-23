function [R0,R1] = computeFEM(manifold)

% Mesh in the form 
Mesh = MeshTriangle(double(manifold.G), double(manifold.x), 'Gamma');

% define function spaces (i.e. the DoFmaps)
Vh_DoFmap = uint32(Mesh.ConnectivityList);

% assemble
tic
FEM = feval(str2func(fullfile('Assem_Laplace_Penalty_Surface')),[], ...
            Mesh.Points,uint32(Mesh.ConnectivityList),[],[],Vh_DoFmap);
toc

% How matrices are stored
R0  = FEM(1).MAT;
R1 = FEM(2).MAT;
end