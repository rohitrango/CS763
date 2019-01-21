function rot_matrix = angles2rot(angles_list)
    %% Your code here
    % angles_list: [theta1, theta2, theta3] about the x,y and z axes,
    % respectively.
    N = size(angles_list, 1);
    rot_matrix = zeros(N, 3, 3);
    for i=1:N,
        % Find angles 
        wx = angles_list(i, 1);
        wy = angles_list(i, 2);
        wz = angles_list(i, 3);
        Rx = [  1, 0, 0;
                0, cos(wx), -sin(wx); 
                0, sin(wx), cos(wx)
             ];
        Ry = [  cos(wy), 0, sin(wy);
                0, 1, 0; 
               -sin(wy), 0, cos(wy)
             ];
        Rz = [  cos(wz), -sin(wz), 0;
                sin(wz), cos(wz), 0;
                0, 0, 1;
             ];
        rot_matrix(i, :, :) = Rz*Ry*Rx;
    end
end




