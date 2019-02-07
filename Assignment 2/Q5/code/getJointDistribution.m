function [entropy, bestTheta, bestTx] = getJointDistribution(img, movImg, minTx, maxTx, minTheta, maxTheta, step_tx, step_theta, bin_size)
    % Set up some variables
    numTx = (maxTx - minTx + 1)/step_tx;
    numTheta = (maxTheta - minTheta + 1)/step_theta;
    entropy = zeros(numTheta, numTx);
    
    % Keep track of best values here
    bestTheta = inf;
    bestTx = inf;
    bestVal = inf;
    [M, N] = size(img);
    X = zeros(M*N, 2);
    X(:, 1) = img(:);
    
    for tx = minTx:step_tx:maxTx,
        for theta = minTheta:step_theta:maxTheta,
            % Align the moving image back
            movTmpImg = imtranslate(movImg, [tx, 0]);
            movTmpImg = imrotate(movTmpImg, theta, 'crop');
            X(:, 2) = movTmpImg(:);
            % Create 3d histogram and make it into a prob density
            histogram  = hist3(X, [bin_size, bin_size]);
            prob_density = histogram/sum(histogram(:));
            entropyVal = -prob_density.*log(prob_density + 1e-100);
            entropyVal = sum(entropyVal(:));
            % Put it in the corresponding bin
            
            coordTx = (tx - minTx)/step_tx + 1;
            coordTheta = (theta - minTheta)/step_theta + 1;
            
            fprintf('Running for theta = %f, tx = %f\n', theta, tx);
            
            entropy(coordTheta, coordTx) = entropyVal;
            if(entropyVal < bestVal)
               bestVal = entropyVal;
               bestTx = tx;
               bestTheta = theta;
            end
            
        end
    end

end