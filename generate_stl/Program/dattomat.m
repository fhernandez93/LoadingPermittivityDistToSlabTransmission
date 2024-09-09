% Read the data from the .dat file
data = importdata('H:\phd stuff\tidy3d\structures\End2EndFiles\Florescu LSU 14.3\ak4_1000_ends.dat');

% Assuming each row represents a set of data, transpose the data matrix
r = data;

% Save the data into a MATLAB .mat file
save('ak4_1000_ends.mat', 'r');
