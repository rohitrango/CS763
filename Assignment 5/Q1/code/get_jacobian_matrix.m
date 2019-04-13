function jacobian = get_jacobian_matrix(patchsize)
    x = squeeze(1:patchsize)';
    [xx, yy] = meshgrid(x, x);
    jacobian = zeros(patchsize*patchsize, 2, 6);
    jacobian(:, 1, 1) = xx(:);
    jacobian(:, 1, 2) = yy(:);
    jacobian(:, 1, 3) = 1;
    
    jacobian(:, 2, 4) = xx(:);
    jacobian(:, 2, 5) = yy(:);
    jacobian(:, 2, 6) = 1;    
end