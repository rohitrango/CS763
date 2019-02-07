function [V] = myPlot(image, misaligned, optTx, optTheta)
    figure;
    subplot(1, 3, 1);
    imagesc(image);
    daspect([1, 1, 1]);
    title('Original image');

    subplot(1, 3, 2);
    imagesc(imrotate(imtranslate(misaligned, [optTx, 0]), optTheta, 'crop'));
    daspect([1, 1, 1]);
    title('Re-aligned image');

    subplot(1, 3, 3);
    imagesc(misaligned);
    daspect([1, 1, 1]);
    colormap(gray);
    title('Moving misaligned image');    
end