clear all; close all; clc;

image_dir = 'data/rgb/4';

edge_dir = 'data/mv-rnn/4';

out_folder = 'data/mv-rnn-display/4';

num_views = 60;

for loop = 1 : num_views
    file_name = sprintf('%03d', loop - 1);
    I = single(imread([image_dir, '/RGB-', file_name, '.png']));
    [E, mymap] = imread([edge_dir, '/MV-RNN-', file_name, '.png']);
    if (~isempty(mymap))
        E = single(255.0 * (E ~= 0));
    else
        E = single(E);
    end
    J = zeros(size(I));
    for i = 1 : size(J, 1)
        for j = 1 : size(J, 2)
            if (E(i, j) < 50)
                J(i, j) = I(i, j);
            else
                J(i, j) = 255 - E(i, j);
            end
        end
    end
    J(J > 255.0) = 255.0;
    J = uint8(J);
    imwrite(J, [out_folder, '/MV-RNN-', file_name, '.png']);
end
