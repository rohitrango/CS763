clear;
close all;
clc;

%% Your Code Here
% number of frames
% parameters here
patchsize = 60;
noOfFeatures = 2;
NITERS = 100;
N = 247;
lambda = 1;

% patch offset for cropping 
offset = int32(patchsize/2);

% Matrix containing the tracked points
trackedPoints = zeros([N,2,noOfFeatures]);

firstFrame = im2double(imread(sprintf('../input/1.jpg')));
[h,w]  =  size(firstFrame);
Frames = zeros([N,h,w]);

% get jacobian
% 2 * 6 * P
jacobian_matrix = get_jacobian_matrix(patchsize);

% Start
noOfPoints = 1;

for i=1:N	
	% Read starting frame
	if mod(i, 100) == 1
		% Read from frame        
		firstFrame = im2double(imread(sprintf('../input/%d.jpg', i)));
		Frames(i,:,:) = firstFrame;

		% Smoothing the image to remove noise
		firstFrame = imgaussfilt(firstFrame, 3);

		[template_patches, x_good, y_good] = selectGoodFeatures(firstFrame, patchsize, noOfFeatures, 1);

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

		nFrame = imgaussfilt(nFrame, 3);


		p = zeros(noOfFeatures, 2, 3);
		p(:, 1, 1) = 1;
		p(:, 2, 2) = 1;
		new_patches = zeros(size(template_patches));
		
		% Solve for each patch
		for patchNum = 1:noOfFeatures
		   % Iteratively solve the optimization
		   xg = x_good(patchNum);
		   yg = y_good(patchNum);              
		   % For every patch, calculate the new patch from the current
		   % frame
		   for iters = 1:NITERS
			  % Init errors to all zeros
			  error = zeros(noOfFeatures); 
			  % Get current affine matrix, warped image
			  affMat = squeeze(p(patchNum, :, :));
			  tform = affine2d([affMat; 0, 0, 1]');
			  warpedIm = imwarp(nFrame, tform);

			  % Getting the warpedIm to be the same size as template
			  [htemp, wtemp] = size(warpedIm);
			  if htemp < h
				warpedIm = [warpedIm; zeros([h-htemp, wtemp])];
			  end
			  [htemp, wtemp] = size(warpedIm);
			  if wtemp < w
				warpedIm = [warpedIm, zeros([htemp, w-wtemp])];
			  end  

			  new_patches(patchNum, :, :) = warpedIm(yg - offset:yg+offset-1, xg - offset:xg + offset-1);
				
			  % Calculate L2 error
			  error(patchNum) = mean(mean((new_patches(patchNum, :, :) - template_patches(patchNum, :, :)).^2));
			  fprintf('Frame : %d Patchnum: %d, Error : %f\n', i, patchNum, error(patchNum));

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

			  p(patchNum, :, :) = p(patchNum, :, :) + lambda*permute(reshape(TIW, 1, 3, 2), [1, 3, 2]);

			  if error(patchNum) < 0.001
				break
			  end
		   end

		   % Storing the coordinates of the tracked Point
		   affMat = squeeze(p(patchNum, :, :));
		   affInv = inv([affMat; 0, 0, 1]);
		   coord  = affInv*[double(xg);double(yg);1];
		   trackedPoints(i, 1, patchNum) = coord(1);
		   trackedPoints(i, 2, patchNum) = coord(2);



		end        
		% Displaying the tracked points on the frames for debugging %
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
   
