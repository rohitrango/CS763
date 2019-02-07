close all
clc
clear

% Define variables
angle = 23.5;
translate = [-3.0, 0];
kernel = 5;
sigma = 5;
MAX = 8;

% Parameters of search
tx = 12;
theta = 60;
step_tx = 1;
step_theta = 1;
bin_size = 10;

% Load all the images
barbaraOrig = imread('../input/barbara.png');
barbaraNeg  = imread('../input/negative_barbara.png');
flashImg = imread('../input/flash1.jpg');
noFlashImg = imread('../input/noflash1.jpg');

% Change to grayscale
flashImg = rgb2gray(flashImg);
noFlashImg = rgb2gray(noFlashImg);

% Perform Gaussian smoothing for the flash images
gaussFilt = fspecial('gaussian', kernel, sigma);
flashImg = imfilter(flashImg, gaussFilt);
noFlashImg = imfilter(noFlashImg, gaussFilt);

% Corrupt the negative images
barbaraNeg = imrotate(barbaraNeg, angle, 'crop');
barbaraNeg = imtranslate(barbaraNeg, translate);
barbaraNeg = barbaraNeg + uint8(randi(MAX, size(barbaraNeg)) - 1);

noFlashImg = imrotate(noFlashImg , angle, 'crop');
noFlashImg = imtranslate(noFlashImg, translate);
noFlashImg = noFlashImg + uint8(randi(MAX, size(noFlashImg)) - 1);

% See images
% figure; imagesc(barbaraOrig); colormap(gray);
% figure; imagesc(barbaraNeg); colormap(gray);
% figure; imagesc(flashImg); colormap(gray);
% figure; imagesc(noFlashImg); colormap(gray);


% Get the joint entropy distribution
% The size is of theta * tx
[distr1, optTheta1, optTx1] = getJointDistribution(barbaraOrig, barbaraNeg, -tx, tx, -theta, theta, step_tx, step_theta, bin_size);
[distr2, optTheta2, optTx2] = getJointDistribution(flashImg, noFlashImg, -tx, tx, -theta, theta, step_tx, step_theta, bin_size);
