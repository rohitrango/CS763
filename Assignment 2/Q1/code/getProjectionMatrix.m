
function [P] = getProjectionMatrix(p1, p2)
%GETPROJECTIONMATRIX
% p1 - I x 3 matrix giving world coordinates (X, Y, Z)
% p2 - I x 2 matrix giving pixel coordinates (x, y)
size_P = 12;
num_points = size(p1, 1);
M = zeros(size(p1, 1), size_P);
for i=1:num_points
    M(2 * i - 1, 1 : 3) = -p1(i, 1: 3);
    M(2 * i - 1, 4) = -1;
    M(2 * i - 1, 9 : 11) = p1(i, 1: 3) * p2(i, 1);
    M(2 * i - 1, 12) = p2(i, 1);
    M(2 * i, 5 : 7) = -p1(i, 1 : 3);
    M(2 * i, 8) = -1;
    M(2 * i, 9 : 11) = p1(i, 1: 3) * p2(i, 2);
    M(2 * i, 12) = p2(i, 2);
end

[~, ~, V] = svd(M);
P_lin = V(:, size(V, 2));
P = zeros(3, 4);
P(1, :) = P_lin(1 : 4);
P(2, :) = P_lin(5 : 8);
P(3, :) = P_lin(9 : 12);
end

