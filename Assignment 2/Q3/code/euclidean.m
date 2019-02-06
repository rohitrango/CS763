function diff_dist = euclidean(p1,p2)
	
	difference = abs(p1-p2);
	diff_sqr = difference.^2;
	diff_sum = sum(diff_sqr);
	diff_dist = sqrt(diff_sum);

end