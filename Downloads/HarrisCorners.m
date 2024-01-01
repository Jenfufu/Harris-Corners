originalImage = imread('Image1.jpg');
grayImage = rgb2gray(originalImage);

% Values can be customized depending on your needs
Sigma = 0.8; 
N = 3;       
D = 10;      
M = 100;     

[corners, R] = detectHarrisCorners(grayImage, Sigma, N, D, M);

% R-score image
figure;
imshow(R, []);
title('R-score Image');

% Show corners on the image
figure; 
imshow(originalImage); 
hold on;
title('Detected Harris Corners');
for i = 1:size(corners, 1)
    rectangle('Position', [corners(i,1)-D/2, corners(i,2)-D/2, D, D], ...
              'EdgeColor', 'r', 'LineWidth', 1);
end
hold off;

patchSize = 9;
patches = extractPatches(grayImage, corners, patchSize);

% Display the patches
numCorners = size(corners, 1);
numCols = ceil(sqrt(numCorners));
numRows = ceil(numCorners / numCols);

figure;
for i = 1:numCorners
    subplot(numRows, numCols, i);
    imshow(patches(:,:,i), []);
    title(sprintf('Patch %d', i));
end

function [corners, R] = detectHarrisCorners(Image, Sigma, N, D, M)
     % Image smoothing
     kernel_size = 2 * floor(3 * Sigma) + 1; % Size of Gaussian kernel
     G = fspecial('gaussian', [kernel_size, kernel_size], Sigma);
     smoothedImage = imfilter(double(Image), G, 'same','conv');

    % Compute gradient images (gradient function should get Gx and Gy)
    [Gx, Gy] = gradient(smoothedImage);

    % Product of derivatives; no need to loop through the whole picture
    % since it will be done for each pixel through these products
    Gx2 = Gx .^ 2;
    Gy2 = Gy .^ 2;
    GxGy = Gx .* Gy;

    % Compute the sums of products within the NxN neighborhood
    boxFilter = ones(N, N);
    Sx2 = imfilter(Gx2, boxFilter, 'same', 'replicate');
    Sy2 = imfilter(Gy2, boxFilter, 'same', 'replicate');
    Sxy = imfilter(GxGy, boxFilter, 'same', 'replicate');

    % Define Harris corner value R(x,y)
    k = 0.05;
    R = (Sx2 .* Sy2 - Sxy .^ 2) - k * (Sx2 + Sy2) .^ 2;

    %Non-maxima suppression to extract the top M corners
    corners = zeros(M, 2);
    Rmax = imregionalmax(R); %regional maxima (binary img)

    %Threshold R to avoid non-significant corners
    threshold = 0.01 * max(R(:));
    R_thresholded = R;
    R_thresholded(R < threshold | ~Rmax) = 0;

    %Get the M strongest corners based on the R values
    for i = 1:M
        [~, index] = max(R_thresholded(:));
        [row, col] = ind2sub(size(R_thresholded), index);
        corners(i, :) = [col, row];  % Save the corner location

        % Suppress the neighbors of the corner
        row_min = max(1, row - D);
        row_max = min(size(Image, 1), row + D);
        col_min = max(1, col - D);
        col_max = min(size(Image, 2), col + D);
        R_thresholded(row_min:row_max, col_min:col_max) = 0; % Set the neighborhood to zero
    end
end

function patches = extractPatches(img, corners, patchSize)

    halfSize = floor(patchSize / 2);
    numCorners = size(corners, 1);
    patches = zeros(patchSize, patchSize, numCorners);

    for i = 1:numCorners
        cornerX = corners(i, 1);
        cornerY = corners(i, 2);

        % Patch boundaries
        xStart = max(cornerX - halfSize, 1);
        xEnd = min(cornerX + halfSize, size(img, 2));
        yStart = max(cornerY - halfSize, 1);
        yEnd = min(cornerY + halfSize, size(img, 1));

        patch = img(yStart:yEnd, xStart:xEnd); % Extract patches

        patches(1:size(patch, 1), 1:size(patch, 2), i) = patch; % Store patches
    end
end

