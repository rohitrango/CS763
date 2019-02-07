
function [K, R, T] = decomposeProjectionMatrix(P)
%DECOMPOSEP
    H_inf = P(1 : 3, 1 : 3);
    [R_t, K_inv] = qr(inv(H_inf));
    R_z_pi = [-1 0 0; 0 -1 0; 0 0 1];
    R = R_z_pi * transpose(R_t);
    K = inv(K_inv) * R_z_pi;
    T = -1 * inv(K * R) * P(1 : 3, 4);
    K = K / K(3, 3);
end

