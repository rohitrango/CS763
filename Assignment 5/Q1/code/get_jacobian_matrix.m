function jacobian = get_jacobian_matrix(patchsize, xg, yg)
    x = squeeze(1:patchsize)';
    [xx, yy] = meshgrid(x, x);
    xx = xx + double(xg - patchsize/2);
    yy = yy + double(yg - patchsize/2);
    jacobian = zeros(2, 6, patchsize*patchsize);
    jacobian(1, 1, :) = xx(:);
    jacobian(1, 2, :) = yy(:);
    jacobian(1, 3, :) = 1;
    
    jacobian(2, 4, :) = xx(:);
    jacobian(2, 5, :) = yy(:);
    jacobian(2, 6, :) = 1;
end