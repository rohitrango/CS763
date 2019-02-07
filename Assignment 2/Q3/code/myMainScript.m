%% Using Invariance of the Cross Ratio

%% Setting the color scale
my_num_of_colors = 256;
col_scale =  [0:1/(my_num_of_colors-1):1]';
my_color_scale = [col_scale,col_scale,col_scale];

%% Load Data
wembley = imread('../input/wembley.jpeg');

% savefig(my_color_scale,wembley,"wembley","wembley.jpg",1,0);

% waitforbuttonpress;

%% Finding the breadth

% Real World Coordinates
% AB = x
% BC = 44
% CD = x

% Cross Ratio = AC/AD : BC/BD = (x+44)*(44+x)/((44 + 2x)*44)

% Pixel Coordinates --> Found using impixelinfo();
a = [1023;808];
b = [1058;720];
c = [1124;557];
d = [1140;519];

ac = euclidean(a,c);
ad = euclidean(a,d);
bd = euclidean(b,d);
bc = euclidean(b,c);
cross_ratio = (ac * bd)/(ad * bc)

% Cross Ratio here = (ac * bd)/(ad * bc) = 1.0712
% Solving the quadratic for x gives us x = 15.28 yrd
% Thus, breadth = 44 + 15.28*2 = 74.56 yrd

%% Finding the length

% Real World Coordinates
% AB = 18
% BC = x
% CD = 18

% Cross Ratio = AC/AD : BC/BD = (x+18)*(18+x)/((36 + x)*x)

% Pixel Coordinates --> Found using impixelinfo();
a = [64;540];
b = [177;560];
c = [845;679];
d = [1058;720];

ac = euclidean(a,c);
ad = euclidean(a,d);
bd = euclidean(b,d);
bc = euclidean(b,c);
cross_ratio = (ac * bd)/(ad * bc)

% Cross Ratio here = (ac * bd)/(ad * bc) = 1.0363
% Solving the quadratic for x gives us x = 78.17 yrd
% Thus, length = 2*18 + 78.17 = 114.17 yrd

% Helper function to display and save processed images %
function savefig(my_color_scale,modified_pic,title_name,file_name,is_color,to_save)
	
	fig = figure; colormap(my_color_scale);

	if is_color == 1
		colormap jet;
	else
		colormap(gray);
	end
	
	imagesc(modified_pic), title(title_name), colorbar, daspect([1 1 1]), axis tight;
	impixelinfo();
    
end
