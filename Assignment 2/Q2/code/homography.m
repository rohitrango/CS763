function [ H ] = homography( p1, p2 )


    % constructing matrix on which to do SVD
    num_points = size(p1, 1);
    H_size = 9;
    A = zeros(num_points * 2, H_size);
    for i=1:num_points
        A(2 * i - 1, 1) = p1(i, 1);
        A(2 * i - 1, 2) = p1(i, 2);
        A(2 * i - 1, 3) = 1;
        A(2 * i - 1, 7) = -p2(i, 1) * p1(i, 1);
        A(2 * i - 1, 8) = -p2(i, 1) * p1(i, 2);
        A(2 * i - 1, 9) = -p2(i, 1);

        A(2 * i, 4) = p1(i, 1);
        A(2 * i, 5) = p1(i, 2);
        A(2 * i, 6) = 1;
        A(2 * i, 7) = -p2(i, 2) * p1(i, 1);
        A(2 * i, 8) = -p2(i, 2) * p1(i, 2);
        A(2 * i, 9) = -p2(i, 2);
    end

    [~, ~, V] = svd(A);
    H_elongated = V(:, size(V, 2));
    H = zeros(sqrt(H_size), sqrt(H_size));
    H(1, :) = H_elongated(1 : 3);
    H(2, :) = H_elongated(4 : 6);
    H(3, :) = H_elongated(7 : 9);

end
