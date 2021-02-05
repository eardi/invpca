% % Compile MEX files FEM Manifold discretization
mex -setup
m_script = 'MatAssem_Laplace_Penalty_Surface';
MEX_File = 'Assem_Laplace_Penalty_Surface';
[status, Path_To_Mex] = Convert_Mscript_to_MEX(fullfile('Functions','Utils'),m_script,MEX_File);
if status~=0
    disp('Compile MEX_getFEM_FELICITY did not succeed.');
    return;
end