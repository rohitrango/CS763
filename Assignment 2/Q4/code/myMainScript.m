%% MyMainScript
clear;

%% Setting the color scale
my_num_of_colors = 256;
col_scale =  [0:1/(my_num_of_colors-1):1]';
my_color_scale = [col_scale,col_scale,col_scale];
is_color = 1;

tic;
%% Your code here

folder_path = '../input/ownpic/';
listing = dir(folder_path);
	
img1 = imread(strcat(folder_path,'1.jpg'));
img2 = imread(strcat(folder_path,'2.jpg'));

flag_3 = true;

img3 = img2;
if size(listing,1) - 2 == 3
	img3 = imread(strcat(folder_path,'3.jpg'));
else
	flag_3 = false;
end

img1gray = rgb2gray(img1);
img2gray = rgb2gray(img2);
img3gray = rgb2gray(img3);

% savefig(my_color_scale,img1,"1",is_color)

% Detect SURF Features
points1 = detectSURFFeatures(img1gray);
points2 = detectSURFFeatures(img2gray);
points3 = detectSURFFeatures(img3gray);

% Extract features
[f1,vpts1] = extractFeatures(img1gray,points1);
[f2,vpts2] = extractFeatures(img2gray,points2);
[f3,vpts3] = extractFeatures(img3gray,points3);

% Find the matching pairs between img 1 and img 2
indexPairs = matchFeatures(f1,f2,'Unique',true);
matched_points1  = vpts1(indexPairs(:,1)); %The matched_points2 is SURF points type, not only location
matched_points21 = vpts2(indexPairs(:,2));

% % Display matched points
% figure;
% showMatchedFeatures(img1gray,img2gray,matched_points1,matched_points21);
% legend('matched points 1','matched points 2');

% We want the projections to differ only by threshold pixels in euclidean distance
threshold = 0.5;
H12 = ransacHomography(matched_points1,matched_points21,threshold);
% disp(H12);

%% Stiching together the images after Homography matrices are obtained
net_size = size(img2gray)*3;
stiched_image = zeros([net_size,3]);

% Note that x is along columns and y is along rows as that is what was taken by SURF

xfin = net_size(2);
yfin = net_size(1);

o2 = size(img2gray);
x2 = o2(2);
y2 = o2(1);
ox = x2;
oy = y2;

o1 = size(img1gray);
x1 = o1(2);
y1 = o1(1);

o3 = size(img3gray);
x3 = o3(2);
y3 = o3(1);

% % Forward Projection, leaves holes
% for x=1:x1
% 	for y=1:y1
% 		proj = H12*[x;y;1];
% 		proj = proj/proj(3);
% 		proj_x = round(proj(1));
% 		proj_y = round(proj(2));

% 		stiched_image(oy+proj_y,ox+proj_x,:) = double(img1(y,x,:))/255;
% 	end
% end

% Backward Projection, for now, nearest neighbor
for x=1:xfin
	for y=1:yfin
		
		proj = inv(H12)*[x-ox;y-oy;1];
		proj = proj/proj(3);
		proj_x = round(proj(1));
		proj_y = round(proj(2));

		if (proj_x <= 0) || (proj_y <= 0) || (proj_x > x1) || (proj_y > y1)
			continue
		end
		stiched_image(y,x,:) = double(img1(proj_y,proj_x,:))/255;
	end
end

savefig(my_color_scale,stiched_image,"Warped Image 1",is_color)

% % Find the matching pairs between img 2 and img 3
indexPairs = matchFeatures(f2,f3,'Unique',true);
matched_points23 = vpts2(indexPairs(:,1));
matched_points3  = vpts3(indexPairs(:,2));

% % Display matched points
% figure;
% showMatchedFeatures(img2gray,img3gray,matched_points23,matched_points3);
% legend('matched points 2','matched points 3');

if flag_3
	threshold = 0.5;
	H32 = ransacHomography(matched_points3,matched_points23,threshold);
	% disp(H32);

	for x=1:xfin
		for y=1:yfin
			
			proj = inv(H32)*[x-ox;y-oy;1];
			proj = proj/proj(3);
			proj_x = round(proj(1));
			proj_y = round(proj(2));

			if (proj_x <= 0) || (proj_y <= 0) || (proj_x > x3) || (proj_y > y3)
				continue
			end
			stiched_image(y,x,:) = double(img3(proj_y,proj_x,:))/255;
		end
	end

	savefig(my_color_scale,stiched_image,"Image 1 and 3",is_color)

end
	stiched_image(oy+1 : oy+y2 ,ox+1: ox+x2,:) = double(img2)/255;
	savefig(my_color_scale,stiched_image,"Final stitched image",is_color)

toc;

% Helper function to display and save processed images %
function savefig(my_color_scale,modified_pic,title_name,is_color)
	fig = figure; colormap(my_color_scale);

	if is_color == 1
		colormap jet;
	else
		colormap(gray);
	end
	
	imagesc(modified_pic), title(title_name), colorbar, daspect([1 1 1]), axis tight;
	impixelinfo();
end
