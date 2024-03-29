close all;
clc;
clear;

img = imread('Painting.jpg');
[H, W, C] = size(img);

% Extract RGB values
red = img(:, :, 1);
green = img(:, :, 2);
blue = img(:, :, 3);

greenmask = (green > (red + blue + 100));
[y, x] = find(greenmask);
horizonY = mean(y);

yellowmask = (red > 210)&(green > 210)&(blue < 50);
leftMask = yellowmask(:, 1:W/2);
rightMask = yellowmask(:, (W/2+1):W);

realHeightLeft = 180;

% Find all the points
% All coordinates are stored in (x, y) format
[ym, xm] = find(leftMask);
[yr, xr] = find(rightMask);

leftPersonHeight = max(ym) - min(ym);
leftPersonFoot = [mean(xm(ym == max(ym))), max(ym)];

rightPersonHeight = max(yr) - min(yr);
rightPersonFoot = [W/2 + 1 + mean(xr(yr == max(yr))), max(yr)]; 

% Here, we find the necessary values required.
horizonX = leftPersonFoot(1) + (horizonY - leftPersonFoot(2))*(rightPersonFoot(1) - leftPersonFoot(1))/(rightPersonFoot(2) - leftPersonFoot(2));

% Visualize the horizontal line joining their feet at the horizon
imagesc(img);
hold on;
line([horizonX, rightPersonFoot(1)], [horizonY, rightPersonFoot(2)], 'LineWidth', 3, 'Color', [0.8, 0, 0.7]);

% Find ration Z_1/Z_0, 
ratioZ = (rightPersonFoot(2) - horizonY)/(leftPersonFoot(2) - horizonY);
realHeightRight = realHeightLeft*rightPersonHeight/leftPersonHeight/ratioZ;

disp(realHeightRight);
