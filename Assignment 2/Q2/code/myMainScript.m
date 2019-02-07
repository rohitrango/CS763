%% Rigit Transform between 2 sets of 3D Points

%% Load Data
img = imread('../input/wembley.jpeg');

%% Getting pixel coordinates of ends of Dee

num_points = 4;
% p1 is world coordinates of corners of Dee and p2, its pixel coordinates
p1 = zeros(num_points, 2);
p2 = zeros(num_points, 2);
p1(1, :) = [0, 0];
p1(2, :) = [18, 0];
p1(3, :) = [0, 44];
p1(4, :) = [18, 44];
imagesc(img); daspect([1, 1, 1]);
impixelinfo();
p2(1, :) = [845, 679];
p2(2, :) = [1058, 720];
p2(3, :) = [958, 534];
p2(4, :) = [1124, 557];


H = homography(p1, p2);

% p4(1, :) and p4(2, :) are coordinates of ends of a line along the field and coinciding with
% Dee's edges
% p4(3, :) and p4(4, :) are coordinates of ends of the mid line across
% field
p3 = zeros(num_points, 3);
p4 = zeros(num_points, 3);
p4(1, :) = [1058, 720, 1];
p4(2, :) = [64, 541, 1];
p4(3, :) = [373, 669, 1];
p4(4, :) = [702, 469, 1];
for i=1:num_points
    p3(i, :) = transpose(H \ transpose(p4(i, :)));
    p3(i, 1) = p3(i, 1) / p3(i, 3);
    p3(i, 2) = p3(i, 2) / p3(i, 3);
    p3(i, 3) = 1;
end

fprintf('length: %f\n', norm(p3(1, :) - p3(2, :)));
fprintf('breadth: %f\n', norm(p3(3, :) - p3(4, :)));

% % Helper function to display and save processed images %
% function savefig(my_color_scale,modified_pic,title_name,file_name,is_color,to_save)
% 	
% 	fig = figure; colormap(my_color_scale);
% 
% 	if is_color == 1
% 		colormap jet;
% 	else
% 		colormap(gray);
% 	end
% 	
% 	imagesc(modified_pic), title(title_name), colorbar, daspect([1 1 1]), axis tight;
% 	impixelinfo();
%     
% end