function [Ix, Iy] = getSobelGradients(grayFrame)
    % Get structure tensors
    sobel_mask = fspecial('sobel');
    Ix = imfilter(grayFrame, sobel_mask');
    Iy = imfilter(grayFrame, sobel_mask);
end