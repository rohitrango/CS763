%% Read Video & Setup Environment
clear
clc
close all hidden
[FileName,PathName] = uigetfile({'*.avi'; '*.mp4'},'Select shaky video file');

cd mmread
vid=mmread(strcat(PathName,FileName));
cd ..
s=vid.frames;

%% Your code here

% number of frames
size_vid = size(s);
N = size_vid(2);

% Parameter Matrices for the transformation
transformations = zeros([N, 3, 3]);

% parameters here
noOfFeatures = 4;
NITERS = 100;
lambda = 1;

% Initialising the previous frame
prevFrame = s(1).cdata;

% Height and width of the frames involved
[h, w] = size(prevFrame);

% Loop over all frames to estimate the motion
for i = 2:N
	currFrame = s(i).cdata;
	
	% Calculate transformation parameters between current and previous frames
	prevFramegray = rgb2gray(prevFrame);
	currFramegray = rgb2gray(currFrame);

	% Detect SURF Features
	prevPoints = detectSURFFeatures(prevFramegray);
	currPoints = detectSURFFeatures(currFramegray);

	% Extract features
	[f1,vpts1] = extractFeatures(prevFramegray,prevPoints);
	[f2,vpts2] = extractFeatures(currFramegray,currPoints);

	% Find the matching pairs between img 1 and img 2
	indexPairs = matchFeatures(f1,f2,'Unique',true);
	matched_points1  = vpts1(indexPairs(:,1));
	matched_points2 = vpts2(indexPairs(:,2));


	% We want the projections to differ only by threshold pixels in euclidean distance
	threshold = 0.5;
	H = ransacHomography(matched_points1, matched_points2, threshold);
	transformations(i, :, :) = H;

	tform = projective2d(H');
	warpedIm = imwarp(prevFrame, tform);

	imagesc(prevFrame);
	waitforbuttonpress;
	imagesc(warpedIm);
	waitforbuttonpress;
	imagesc(currFrame);
	waitforbuttonpress;
	close all;

	prevFrame = currFrame;
end

%% Write Video
vfile=strcat(PathName,'combined_', FileName);
ff = VideoWriter(vfile);
ff.FrameRate = 30;
open(ff)

for i=1:N+1
    f1 = s(i).cdata;
    f2 = outV(i).cdata;
    vframe=cat(1,f1, f2);
    writeVideo(ff, vframe);
end
close(ff)

%% Display Video
figure
msgbox(strcat('Combined Video Written In ', vfile), 'Completed') 



displayvideo(outV, 0.01)
