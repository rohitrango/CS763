function [im_patches, x_good, y_good ] = selectGoodFeatures(grayFrame, patchsize, topK, disp_image)
    % Get good patches, and their coordinates given the frame, patch size,
    % and top K patches to pick from
    [H, W, ~] = size(grayFrame);
    features = detectHarrisFeatures(grayFrame);
    points = features.Location;
    x = points(:, 1);
    y = points(:, 2);

    filter_edges = (x > patchsize/2)&(x <= W - patchsize/2);
    filter_edges = filter_edges&(y > patchsize/2)&(y <= H - patchsize/2);
    x = x(filter_edges);
    y = y(filter_edges);

    % Display first frame and overlay the features
    if disp_image == 1
        colormap gray;
        imagesc(grayFrame);
        hold on;
        scatter(x, y);
        hold off;
        waitforbuttonpress;
        close all;
    end

    [Ix, Iy] = getSobelGradients(grayFrame);

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
        im_patches(i, :, :) = grayFrame(y_good(i) - Pby2: y_good(i) + Pby2 - 1, x_good(i) - Pby2: x_good(i) + Pby2 - 1);
    end
    % Display first frame and overlay the features
    if disp_image == 1
        colormap gray;
        imagesc(grayFrame);
        hold on;
        scatter(x_good(1:topK), y_good(1:topK));
        hold off;
        waitforbuttonpress;
        close all;
    end
end