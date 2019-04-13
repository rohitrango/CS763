function [im_patches ] = selectGoodFeatures(firstFrame, patchsize, topK)
    [H, W, ~] = size(firstFrame);
    grayFirstFrame = rgb2gray(firstFrame);
    features = detectHarrisFeatures(grayFirstFrame);
    points = features.Location;
    x = points(:, 1);
    y = points(:, 2);

    filter_edges = (x > patchsize/2)&(x <= W - patchsize/2);
    filter_edges = filter_edges&(y > patchsize/2)&(y <= H - patchsize/2);
    x = x(filter_edges);
    y = y(filter_edges);

    % Display first frame and overlay the features
    imagesc(firstFrame);
    hold on;
    scatter(x, y);
    hold off;

    % Get structure tensors
    sobel_mask = fspecial('sobel');
    Ix = imfilter(grayFirstFrame, sobel_mask');
    Iy = imfilter(grayFirstFrame, sobel_mask);

    Ix2 = Ix.*Ix;
    Iy2 = Iy.*Iy;
    Ixy = Ix.*Iy;

    % Select Ix, Iy from the features
    Ix2_features = interp2(Ix2, x, y);
    Iy2_features = interp2(Iy2, x, y);
    Ixy_features = interp2(Ixy, x, y);

    % Select good structure tensors
    trace = Ix2_features + Iy2_features;
    det = Ix2_features.*Iy2_features - Ixy_features.*Ixy_features;

    % threshold by second eigenvalue
    eigen_2 = (trace - sqrt(trace.^2 - 4*det))/2;
    [~, I] = sort(eigen_2, 1, 'descend');
    I = I(1:topK);
    x_good = int32(x(I));
    y_good = int32(y(I));

    Pby2 = int32(patchsize/2);
    im_patches = zeros(topK, patchsize, patchsize);
    for i=1:topK,
        im_patches(i, :, :) = grayFirstFrame(y_good(i) - Pby2: y_good(i) + Pby2 - 1, x_good(i) - Pby2: x_good(i) + Pby2 - 1);
    end
end