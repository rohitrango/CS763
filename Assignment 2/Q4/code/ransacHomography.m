function [ H ] = ransacHomography(x1,x2,thresh)
%% RANSAC HOMOGRAPHY Summary of this function goes here

	% Fixed number of iterations for now, use theoretical bounds later
	
	num_iter = 50;
	num_indices = x1.size(1);

	loc1 = x1.Location;
	loc2 = x2.Location;

	best_losses = ones(num_indices,1);
	max_inliers = 0;
	max_inlier_set = [];

	for i=1:num_iter
		losses = ones(num_indices,1);
		% Choose k random points %
		k = 4;
		indices_perm = randperm(num_indices);
		indices = indices_perm(1:k);
		p1 = loc1(indices,:);
		p2 = loc2(indices,:);

		H_curr = homography(p1,p2);
		num_inliers = 0;

		% Confirm this, we are assuming that the points used to construct are necessary inliers, which of course is not true theoretically
		inlier_set = indices;

		% Right now we are not counting the chosen four points, but we can/should include them, too?
		for index=indices_perm(k+1:num_indices)
			cord1 = ones(3,1);
			cord1(1:2) = loc1(index,:)';
			proj = H_curr*cord1;
			proj = proj/proj(3);
			loss = norm(loc2(index,:) - proj(1:2)');
			losses(index) = loss;

			if loss < thresh
				num_inliers = num_inliers + 1;
				inlier_set = [inlier_set,index];
			end
		end

		if num_inliers > max_inliers
			max_inliers = num_inliers;
			best_losses = losses;
			max_inlier_set = inlier_set;
		end
		% disp(max_inliers);
	end

	% Construct the best model from the inliers

	p1 = loc1(max_inlier_set,:);
	p2 = loc2(max_inlier_set,:);
	H = homography(p1,p2);
end