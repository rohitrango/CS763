clear;
close all;
clc;

%% Your Code Here
% number of frames
% parameters here
patchsize = 40;
topK = 5;
NITERS = 100;

% patch offset for cropping 
offset = int32(patchsize/2);

% get jacobian
% 2 * 6 * P
jacobian_matrix = get_jacobian_matrix(patchsize);

% Start
for i=1:247
    % Read starting frame
    if mod(i, 10) == 1
        % Read from frame        
        firstFrame = im2double(imread(sprintf('../input/%d.jpg', i)));
        [template_patches, x_good, y_good] = selectGoodFeatures(firstFrame, patchsize, topK);
    else
        % Subsequent frames, run your algorithm on it 
        % estimate empty affine matrix
        % initialize new patches
        nFrame = im2double(imread(sprintf('../input/%d.jpg', i)));
        p = zeros(topK, 2, 3);
        p(:, 1, 1) = 1;
        p(:, 2, 2) = 1;
        new_patches = zeros(size(template_patches));
        
        % Solve for each patch
        for patchNum = 1:topK
           % Iteratively solve the optimization
           xg = x_good(patchNum);
           yg = y_good(patchNum);              
           % For every patch, calculate the new patch from the current
           % frame
           for iters = 1:NITERS
              % Init errors to all zeros
              error = zeros(topK); 
              % Get current affine matrix, warped image
              affMat = squeeze(p(patchNum, :, :));
              tform = affine2d([affMat; 0, 0, 1]');
              warpedIm = imwarp(nFrame, tform);
              new_patches(patchNum, :, :) = warpedIm(yg - offset:yg+offset-1, xg - offset:xg + offset-1);
                
              % Calculate L2 error
              error(patchNum) = mean(mean((new_patches(patchNum, :, :) - template_patches(patchNum, :, :)).^2));
              fprintf('Error : %f\n', error(patchNum));
              
              % Image gradients
              [Ix, Iy] = getSobelGradients(warpedIm);
              IxCrop = Ix(yg - offset:yg+offset-1, xg - offset:xg + offset-1);
              IyCrop = Iy(yg - offset:yg+offset-1, xg - offset:xg + offset-1);
              
              % Find delI = 2 * 1600
              delI = zeros(2, patchsize*patchsize);
              delI(1, :) = IxCrop(:)';
              delI(2, :) = IyCrop(:)';
              
              % This will be 6 * 1600
              IdWdp = zeros(6, patchsize*patchsize);
              for pii = 1:patchsize*patchsize
                  % delI = 2 * 1 , jacobian_matrix = 2 * 6
                  IdWdp(:, pii) = delI(:, pii)'*jacobian_matrix(:, :, pii);
              end
              
              % calculate H matrix (6 * 6)
              H = inv(IdWdp*IdWdp');
              
              % TIW = (T - I)*Idwp
              TIW = - new_patches(patchNum, :, :) + template_patches(patchNum, :, :);
              TIW = TIW(:)';
              TIW = TIW.*IdWdp;
              % This is 6 * 1600
              
              % Add this error to p
              TIW = sum(TIW, 2)';
              TIW = TIW*H;
              p(patchNum, :, :) = p(patchNum, :, :) + permute(reshape(TIW, 1, 3, 2), [1, 3, 2]);
       
           end
        end
        
    end
end





%% Save all the trajectories frame by frame
% variable trackedPoints assumes that you have an array of size 
% No of frames * 2(x, y) * No Of Features
% noOfFeatures is the number of features you are tracking
% Frames is array of all the frames(assumes grayscale)
noOfPoints = 1;
for i=1:N
    NextFrame = Frames(i,:,:);
    imshow(uint8(NextFrame)); hold on;
    for nF = 1:noOfFeatures
        plot(trackedPoints(1:noOfPoints, 1, nF), trackedPoints(1:noOfPoints, 2, nF),'*')
    end
    hold off;
    saveas(gcf, strcat('output/',num2str(i),'.jpg'));
    close all;
    noOfPoints = noOfPoints + 1;
end 
   
