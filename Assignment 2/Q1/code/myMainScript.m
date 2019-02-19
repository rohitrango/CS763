% p1 - world (N, 3), p2 - pixel (N, 2)
% p1 = [1,2,3;2,3,4;3,4,5];

img = imread('../input/calib.jpg');
% imagesc(img); daspect([1 1 1]);
% impixelinfo();
p1 = [1 0 1;
      2 0 3;
      3 0 5;
      5 0 7;
      
      2 1 0;
      1 4 0;
      6 3 0;
      3 6 0;
      
      0 2 2;
      0 5 3;
      0 4 7;
      0 1 3
    ];
p2 = [1141 1453;
      1029 1287;
      894 1086;
      549 887;
      
      1136 1688;
      1532 1860;
      850 2177;
      1543 2215;
      
      1470 1430;
      1836 1553;
      1844 904;
      1378 1248
     ];
[U, p1_normalized] = normalizePoints(p1, sqrt(3));
[T, p2_normalized] = normalizePoints(p2, sqrt(2));
% U = eye(4);
% T = eye(3);
% p1_normalized = p1;
% p2_normalized = p2;

P_normalized = getProjectionMatrix(p1_normalized, p2_normalized);
P = inv(T) * P_normalized * U;
[K, R, T] = decomposeProjectionMatrix(P);

p1_test = [2 0 4;
           2 3 0;
           0 2 4;
           5 5 0;
           0 4 5;
           5 0 6
          ];
p2_test = [1028 1167;
           1325 1850;
           1502 1195;
           1190 2299;
           1767 1215;
           576 1057
          ];
p2_pred = P * [p1_test'; ones(1, size(p1_test, 1))];
p2_pred = p2_pred';
for i=1:size(p2_pred, 1)
    p2_pred(i, :) = p2_pred(i, :) / p2_pred(i, size(p2_pred, 2));
end

RMSE = sqrt(mean(sum((p2_pred(:, 1:2) - p2_test).^2, 2)));
fprintf("RMSE for test points: %f\n", RMSE); 
imshow(img);
hold on;
scatter(p2_pred(:, 1), p2_pred(:, 2), 5);

 