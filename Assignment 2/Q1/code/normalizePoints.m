
function [U, p_normalized] = normalizePoints(p, mean_dist)
%NORMALIZEPOINTS
    p_centroid = mean(p);
    p_mean_dist = mean(sqrt(sum((p - p_centroid).^2, 2)));
    
    size_p = size(p, 2);
    U = eye(size_p + 1);
    U = U * sqrt(mean_dist) / p_mean_dist;
    U(size_p + 1, size_p + 1) = 1;
    V = eye(size_p + 1);
    V(1:size_p,size_p + 1) = -p_centroid';
    U = U * V;
    p_normalized = ones(size(p, 1), size(p, 2) + 1);
    for i=1:size(p, 1)
    p_normalized(i, :) = U * [p(i, :)'; 1];
    end
end

