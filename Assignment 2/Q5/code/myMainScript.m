close all
clc
clear

% Define variables
angle = 23.5;
translate = [-3.0, 0];
kernel = 5;
sigma = 5;
MAX = 9;

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
% Then downsampling to get
gaussFilt = fspecial('gaussian', kernel, sigma);
flashImg = imfilter(flashImg, gaussFilt);
flashImg = imresize(flashImg, 0.5);
noFlashImg = imfilter(noFlashImg, gaussFilt);
noFlashImg = imresize(noFlashImg, 0.5);

% Corrupt the negative images
barbaraNeg = imrotate(barbaraNeg, angle, 'crop');
barbaraNeg = imtranslate(barbaraNeg, translate);
barbaraNeg = barbaraNeg + uint8(randi(MAX, size(barbaraNeg)) - 1);

noFlashImg = imrotate(noFlashImg , angle, 'crop');
noFlashImg = imtranslate(noFlashImg, translate);
noFlashImg = noFlashImg + uint8(randi(MAX, size(noFlashImg)) - 1);

% For the last part
bNegAdv = imread('../input/negative_barbara.png');
bNegAdv = imrotate(bNegAdv, 90, 'crop');
bNegAdv = imtranslate(bNegAdv, [-5, 0]);
bNegAdv = bNegAdv + uint8(randi(MAX, size(bNegAdv)) - 1);

% See images
% figure; imagesc(barbaraOrig); colormap(gray);
% figure; imagesc(barbaraNeg); colormap(gray);
% figure; imagesc(flashImg); colormap(gray);
% figure; imagesc(noFlashImg); colormap(gray);

% Get the joint entropy distribution
% The size is of theta * tx
fprintf('Please wait, this may take about a minute...\n');
[distr1, optTheta1, optTx1] = getJointDistribution(barbaraOrig, barbaraNeg, -tx, tx, -theta, theta, step_tx, step_theta, bin_size);
[distr2, optTheta2, optTx2] = getJointDistribution(flashImg, noFlashImg, -tx, tx, -theta, theta, step_tx, step_theta, bin_size);
[distr3, optTheta3, optTx3] = getJointDistribution(barbaraOrig, bNegAdv, -tx, tx, -theta, theta, step_tx, step_theta, bin_size);

% Print optimal values
fprintf('Optimal theta for first pair = %f, optimal tx = %f\n', optTheta1, optTx1);
fprintf('Optimal theta for second pair = %f, optimal tx = %f\n', optTheta2, optTx2);
fprintf('Optimal theta for third pair = %f, optimal tx = %f\n', optTheta3, optTx3);

% Show the figures
myPlot(barbaraOrig, barbaraNeg, optTx1, optTheta1);
myPlot(flashImg, noFlashImg, optTx2, optTheta2);
myPlot(barbaraOrig, bNegAdv, optTx3, optTheta3);
