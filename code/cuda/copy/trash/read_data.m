
% Reads the result file from the fast t-SNE implementation
function [X, costs] = read_data(file_name, max_iter)
    h = fopen(file_name, 'rb');
	n = fread(h, 1, 'integer*4');
	d = fread(h, 1, 'integer*4');
	X = fread(h, n * d, 'double');
    max_iter = fread(h, 1, 'integer*4');
    costs = fread(h, max_iter, 'double');     
    X = reshape(X, [d n])';
	fclose(h);
end