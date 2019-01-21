function [result_pose, composed_rot] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_positoin: the index of the root in pose vector.
    % Your code here
    
    % Composite rotation maintained here, one for each point
    composed_rot = zeros(size(pose, 1), 3, 3);
    for i=1:size(composed_rot, 1)
        composed_rot(i, :, :) = eye(3);
    end
    
    % Bone indices to initialize bones
    boneIndices = size(kinematic_chain, 1);    
    
    % Initialize the composite roations first
    for boneIndex = 1:boneIndices,
       % Given a bine rotation, the child is rotated
       childIndex = kinematic_chain(boneIndex, 1);
       parentIndex = kinematic_chain(boneIndex, 2);
       composed_rot(childIndex, :, :) = squeeze(rotations(boneIndex, :, :))*squeeze(composed_rot(parentIndex, :, :));
       
       disp(childIndex)
       disp(squeeze(composed_rot(childIndex, :, :)))
       
    end
    
    fprintf('-----------------------------------------------------------');

    % Initialized the root joint
    result_pose = zeros(16, 3);
    result_pose(root_location, :) = pose(root_location, :);
    
    for boneIndex = 1:boneIndices,
       % For every joint, we need to find parent, child
       childIndex  = kinematic_chain(boneIndex, 1);
       childCoord  = pose(childIndex, :);
       
       parentIndex = kinematic_chain(boneIndex, 2);
       parentCoord = pose(parentIndex, :);
     
       % Now find the new child coord
       newChildCoord = squeeze(composed_rot(childIndex, :, :))*(childCoord - parentCoord)' + result_pose(parentIndex, :)';
       result_pose(childIndex, :) = newChildCoord';
    end
    
    %sum((result_pose(8, :) - result_pose(9, :)).*(result_pose(9, :) - result_pose(13, :)))
    
end