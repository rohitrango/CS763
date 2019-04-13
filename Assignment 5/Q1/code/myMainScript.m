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
% parameters here
N = size(s, 2);
patchsize = 40;
topK = 5;

% Start
firstFrame = im2double(s(1).cdata);
im_patches = selectGoodFeatures(firstFrame, patchsize, topK);

% get jacobian
jacobian_matrix = get_jacobian_matrix(patchsize);


return






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
