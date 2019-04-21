clear;
close all;
clc;

%% Your Code Here
% number of frames
% parameters here
patchsize = 40;
noOfFeatures = 2;
NITERS = 20;
N = 247;
lambda = 0.7;
sigma = 2;

% patch offset for cropping 
offset = int32(patchsize/2);

% Matrix containing the tracked points
trackedPoints = zeros([N, 2, noOfFeatures]);

firstFrame = im2double(imread(sprintf('../input/1.jpg')));
[h,w]  =  size(firstFrame);
Frames = zeros([N, h, w]);

% get jacobian
% 2 * 6 * P
jacobian_matrix = get_jacobian_matrix(patchsize, 0, 0);

p = zeros(noOfFeatures, 2, 3);
p(:, 1, 1) = 1;
p(:, 2, 2) = 1;

% Start
noOfPoints = 1;

for i=1:N	
	% Read starting frame
	if mod(i, 20) == 1
		% Read from frame        
		firstFrame = im2double(imread(sprintf('../input/%d.jpg', i)));
		Frames(i,:,:) = firstFrame;

		% Smoothing the image to remove noise
		firstFrame = imgaussfilt(firstFrame, sigma);
        [template_patches, x_good, y_good] = selectGoodFeatures(firstFrame, patchsize, noOfFeatures, 1, [140, 190], [240, 280]);
%         [template_patches, x_good, y_good] = selectGoodFeatures(firstFrame, patchsize, noOfFeatures, 1, [0, 1000], [0, 1000]);

       % Store the tracked points here
	   for patchNum = 1:noOfFeatures
		   trackedPoints(i, 1, patchNum) = x_good(patchNum);
		   trackedPoints(i, 2, patchNum) = y_good(patchNum);
	   end

	else
		% Subsequent frames, run your algorithm on it 
		% estimate empty affine matrix
		% initialize new patches
		nFrame = im2double(imread(sprintf('../input/%d.jpg', i)));
		Frames(i,:,:) = nFrame;
        % Apply smoothing
		nFrame = imgaussfilt(nFrame, sigma);
		new_patches = zeros(size(template_patches));
		
		% Solve for each patch
		for patchNum = 1:noOfFeatures
		   % Iteratively solve the optimization
		   xg = x_good(patchNum);
		   yg = y_good(patchNum);              
		   % For every patch, calculate the new patch from the current frame
		   jacobian_matrix = get_jacobian_matrix(patchsize, xg, yg);

		   for iters = 1:NITERS
			  % Init errors to all zeros
			  error = zeros(noOfFeatures); 
			  % Get current affine matrix, warped image
			  affMat = squeeze(p(patchNum, :, :));  % 2 * 3

              % Get forward operation
              [xx, yy] = meshgrid(1:patchsize);
              xx = xx + double(xg) - patchsize/2;
              yy = yy + double(yg) - patchsize/2;
              coord_old = ones(3, patchsize*patchsize);
              coord_old(1, :) = xx(:);
              coord_old(2, :) = yy(:);
              % Get new coords = 2 * 1600
              coord_new = affMat*coord_old;
              xx_new = reshape(coord_new(1, :), patchsize, patchsize);
              yy_new = reshape(coord_new(2, :), patchsize, patchsize);
              % Get image coordinates from here
              warpedIm = interp2(nFrame, xx_new, yy_new);
              warpedIm(isnan(warpedIm)) = 0;
              
              % Get new patches, also take care of gradients
              new_patches(patchNum, :, :) = warpedIm;
              [IxCrop, IyCrop] = getSobelGradients(warpedIm);

              
% 			  tform = affine2d([affMat; 0, 0, 1]');
% 			  warpedIm = imwarp(nFrame, tform);

			  % Getting the warpedIm to be the same size as template
% 			  [htemp, wtemp] = size(warpedIm);
% 			  if htemp < h
% 				warpedIm = [warpedIm; zeros([h-htemp, wtemp])];
% 			  end
% 			  [htemp, wtemp] = size(warpedIm);
% 			  if wtemp < w
% 				warpedIm = [warpedIm, zeros([htemp, w-wtemp])];
% 			  end  
% 			  new_patches(patchNum, :, :) = warpedIm(yg - offset:yg+offset-1, xg - offset:xg + offset-1);
				
			  % Calculate L2 error
			  error(patchNum) = mean(mean((new_patches(patchNum, :, :) - template_patches(patchNum, :, :)).^2));
			  fprintf('Frame : %d Patchnum: %d, Error : %f\n', i, patchNum, error(patchNum));

			  % Image gradients
			  % [Ix, Iy] = getSobelGradients(warpedIm);
			  % IxCrop = Ix(yg - offset:yg+offset-1, xg - offset:xg + offset-1);
			  % IyCrop = Iy(yg - offset:yg+offset-1, xg - offset:xg + offset-1);
			  
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
              % This is correct -> Checked with manual for loop
			  H = inv(IdWdp*IdWdp');
			  
			  % TIW = (T - I)*Idwp
			  TIW = template_patches(patchNum, :, :) - new_patches(patchNum, :, :);
			  TIW = repmat(TIW(:)', [6, 1]);
			  TIW = TIW.*IdWdp;
			  % This is 6 * 1600
			  
			  % Add this error to p
			  TIW = sum(TIW, 2)';
			  TIW = TIW*H;

%               delP = zeros(1, 2, 3);
%               delP(1, 1, 1) = TIW(1);
%               delP(1, 1, 2) = TIW(3);
%               delP(1, 1, 3) = TIW(5);
%               delP(1, 2, 1) = TIW(2);
%               delP(1, 2, 2) = TIW(4);
%               delP(1, 2, 3) = TIW(6);
              
%               p(patchNum, :, :) = p(patchNum, :, :) + 1e-2*delP;
% 			  p(patchNum, :, :) = p(patchNum, :, :) + lambda*permute(reshape(TIW, 1, 3, 2), [1, 3, 2]);
              p(patchNum, :, :) = p(patchNum, :, :) + permute(reshape(TIW, 1, 3, 2), [1, 3, 2]);

			 %  if error(patchNum) < 0.001
				% break
			 %  end
		   end

		   % Storing the coordinates of the tracked Point
		   affMat = squeeze(p(patchNum, :, :));
		   affInv = inv([affMat; 0, 0, 1]);
		   coord  = affInv*[double(xg); double(yg); 1];
		   trackedPoints(i, 1, patchNum) = coord(1);
		   trackedPoints(i, 2, patchNum) = coord(2);



		end        
		% Displaying the tracked points on the frames for debugging %
		NextFrame = Frames(i,:,:);
		ColFrame = zeros([h, w, 3]);		    
		ColFrame(:,:,1) = NextFrame;
		ColFrame(:,:,2) = NextFrame;
		ColFrame(:,:,3) = NextFrame;

		ColFrame = ColFrame * 255;
		imshow(uint8(ColFrame)); 
		hold on;
		for nF = 1:noOfFeatures
			plot(trackedPoints(1:noOfPoints, 1, nF), trackedPoints(1:noOfPoints, 2, nF),'*')
		end
		waitforbuttonpress;
		hold off;
		close all;
		noOfPoints = noOfPoints + 1;
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
	ColFrame = zeros([h,w,3]);
	
	ColFrame(:,:,1) = NextFrame;
	ColFrame(:,:,2) = NextFrame;
	ColFrame(:,:,3) = NextFrame;

	ColFrame = ColFrame * 255;
	imshow(uint8(ColFrame)); 
	hold on;
	for nF = 1:noOfFeatures
		plot(trackedPoints(1:noOfPoints, 1, nF), trackedPoints(1:noOfPoints, 2, nF),'*')
	end
	hold off;
	saveas(gcf, strcat('../output/',num2str(i),'.jpg'));
	close all;
	noOfPoints = noOfPoints + 1;
end 
