function [Ix, Iy] = getSobelGradients(grayFrame)
    % Get structure tensors
    sobel_mask = fspecial('sobel');
    Ix = imfilter(grayFrame, sobel_mask');
    Iy = imfilter(grayFrame, sobel_mask);

    % Smoothing the gradients as suggested in PS
    Ix = imgaussfilt(Ix, 1);
    Iy = imgaussfilt(Iy, 1);

end