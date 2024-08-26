% Read the data from the .dat file
data = importdata('H:\phd stuff\tidy3d\generate_stl\FromEndsFiles\RCP 18.01 End Points\1_sample_L18_lines_cut.dat');

% Assuming each row represents a set of data, transpose the data matrix
r = data;

% Save the data into a MATLAB .mat file
save('1_sample_L18_lines_cut.mat', 'r');
