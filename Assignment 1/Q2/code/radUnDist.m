function imOut = radUnDist(imIn, k1, k2, nSteps)
    % Your code here
    [m, n] = size(imIn);   
    [x, y] = meshgrid(1:n, 1:m);
    cx = n/2;
    cy = m/2;
    
    % ax, ay is for reference
    ax = (x - cx)/cx;
    ay = (y - cy)/cy;
    
    % x, y are my estimates
    x = ax + 0;
    y = ay + 0;
    
    for iters=1:nSteps,
       % iterations
       r2 = sqrt(x.^2 + y.^2);
       dr = -k1*r2 - k2*r2.^2;   
       x = ax + dr.*x;
       y = ay + dr.*y;
    end
    
    x = x*cx + cx;
    y = y*cy + cy;
    
    imOut = interp2(imIn, x, y, 'cubic');
end