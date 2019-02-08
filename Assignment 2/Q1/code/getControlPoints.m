close all
clear
clc

image = imread('../input/real_image.jpg');

imgray = rgb2gray(image);
imgray = 1.0 * (imgray > 120);
% imagesc(imgray);

k = 0.6;
% Get kernel for left part
% N = 20;
% Kleft = zeros(N , N) - 1;
% Kleft(1:N/2, 1:N/2) = 1;
% Kleft(N/2+1:N, N/2+1:N) = 1;
% 
% leftImage = imfilter(imgray, Kleft);
% leftPts = leftImage > 0.7*max(leftImage(:));

% Get bottom kernel
N = 20;
[xx, yy] = meshgrid(-N/2:N/2, -N/2:N/2);
Kbottom = sign((yy - k*xx).*(yy + k*xx));

bottomImage = imfilter(imgray, Kbottom);
bottomPts = bottomImage > 0.8*max(bottomImage(:));

% Get right kernel
Ktop = sign((yy - 0.2*xx).*(yy + 1/0.2*xx));
topImage = imfilter(imgray, Ktop);
topPts = topImage > 0.6*max(topImage(:));

% imagesc(topPts)
% figure
% imagesc(topImage)

imagesc(bottomImage)
figure
imagesc(bottomPts)

% Get connected components
label = bwlabel(bottomPts);
for i = 1:

