%% Section 2.1
clc
close all

% Openning image :
img1 = imread('01.jpg');

% Showing image :
figure, imshow(img1);
title('Original Image "01.jpg"','color','r');

% Showing histogram :
figure, imhist(rgb2gray(img1));
title('Histogram Of Image "01.jpg" Converted To Grayscale','color','r');

%% Section 2.2
clc
close all

% Filter designing :
img1FiltGauss = fspecial('gaussian');
img1FiltAvg = fspecial('average');

% Filtering :
img1FilteredGauss = imfilter(img1, img1FiltGauss);
img1FilteredAvg = imfilter(img1, img1FiltAvg);
img1FilteredMed = medfilt2(rgb2gray(img1));
img1FilteredAll = imfilter(img1FilteredGauss , img1FiltAvg);
img1FilteredAll = medfilt2(rgb2gray(img1FilteredAll));

figure, imshowpair(img1FilteredGauss,img1,'montage');
title('Gaussian Filtered Image "01.jpg" (Left) & The Original Image (Right)','color','r');

figure, imshowpair(img1FilteredAvg,img1,'montage');
title('Average Filtered Image "01.jpg" (Left) & The Original Image (Right)','color','r');

figure, imshowpair(img1FilteredMed,img1,'montage');
title('Median Filtered Image "01.jpg" (Left) & The Original Image (Right)','color','r');

figure, imshowpair(img1FilteredAll,img1,'montage');
title('All Filters Used For Image "01.jpg" (Left) & The Original Image (Right)','color','r');

%% Section 2.3
clc
close all

% First Method

% Fourier transform :
img1FT = FFT2(img1);

% Plotting the results :
figure, imshow(abs(img1FT), [0 200000]);
title('Method 1 : Absolute Value Of Fourier Transform Of "01.jpg" Image','color','r');
figure, imshow(angle(img1FT), [-pi pi]);
title('Method 1 : Phase Of Fourier Transform Of "01.jpg" Image','color','r');

% Second Method
img1FT2 = fft2(img1);
img1FT2shift = fftshift(img1FT2);
img1FT2mag = abs(img1FT2shift); % magnitude
img1FT2phase = angle(img1FT2shift);
img1FT2mag = log(img1FT2mag+1); % for perceptual scaling, and +1 since log(0) is undefined
img1FT2mag = mat2gray(img1FT2mag); % Using mat2gray to scale the image between 0 and 1

figure,imshow(img1FT2mag,[]); % Display the result
title('Method 2 : Absolute Value Of Fourier Transform Of "01.jpg" Image','color','r');

img1FT2phase = mat2gray(img1FT2phase);
figure,imshow(img1FT2phase,[]);
title('Method 2 : Phase Of Fourier Transform Of "01.jpg" Image','color','r');

%% Section 2.4
clc
close all

% Openning image :
img3 = imread('03.jpg');

% Showing image :
figure, imshow(img3);
title('Original Image "03.jpg"','color','r');

% Fourier transform :
img3FT = FFT2(img3);

% Plotting the results :
figure, imshow(abs(img3FT), [0 500000]);
title('Method 1 : Absolute Value Of Fourier Transform Of "03.jpg" Image','color','r');
figure, imshow(angle(img3FT), [-pi pi]);
title('Method 1 : Phase Of Fourier Transform Of "03.jpg" Image','color','r');

% Second Method
img1FT2 = fft2(img3);
img1FT2shift = fftshift(img1FT2);
img1FT2mag = abs(img1FT2shift); % magnitude
img1FT2phase = angle(img1FT2shift);
img1FT2mag = log(img1FT2mag+1); % for perceptual scaling, and +1 since log(0) is undefined
img1FT2mag = mat2gray(img1FT2mag); % Using mat2gray to scale the image between 0 and 1

figure,imshow(img1FT2mag,[]); % Display the result
title('Method 2 : Absolute Value Of Fourier Transform Of "03.jpg" Image','color','r');

img1FT2phase = mat2gray(img1FT2phase);
figure,imshow(img1FT2phase,[]);
title('Method 2 : Phase Of Fourier Transform Of "03.jpg" Image','color','r');

%% Section 3.1
clc
close all
k = [2 4 8 12 16 20 24 28];

for i = 1 : length(k)
    % Filter designing :
    MA = fspecial('average', 2*k(i)+1); 
    
    % Filtering :
    img1FilteredAvg = imfilter(img1, MA);
    
    % Showing the result :
    figure, imshow(abs(img1FilteredAvg));
    title(sprintf('Average Filtered Image "01.jpg" with k = %d', k(i)),'color','r');
    
    % Fourier transform First Method :
    img1FilteredAvgFT = FFT2(img1FilteredAvg);
    
    % Fourier transform Second Method :
    imgFT = fft2(img1FilteredAvg);
    imgFTshift = fftshift(imgFT);
    imgFTmag = abs(imgFTshift); % magnitude
    imgFTmag = log(imgFTmag+1); % for perceptual scaling, and +1 since log(0) is undefined
    imgFTmag = mat2gray(imgFTmag); % Using mat2gray to scale the image between 0 and 1
    
    % Showing absolute value of fourier transform Using Both Methods:
    
    % First Method
    figure
    subplot(1,2,1)
    imshow(abs(img1FilteredAvgFT), [0 200000]);
    title(sprintf('Method 1 : Absolute Value Of Fourier Transform Of Average Filtered Image "01.jpg" with k = %d', k(i)),'color','r');
    
    % Second Method
    subplot(1,2,2)
    imshow(imgFTmag,[]); % Display the result
    title(sprintf('Method 2 : Absolute Value Of Fourier Transform Of "01.jpg" Image with k = %d', k(i)),'color','r');
    
    % Showing histogram :
    figure, imhist(rgb2gray(img1FilteredAvg));
    title(sprintf('Histogram Of Average Filtered Image "01.jpg" with k = %d', k(i)),'color','r');
end

%% Section 3.2
clc
close all

for i = 1 : length(k) 
    % Filter designing :
    MA = fspecial('average', 2*k(i)+1); 
    % Filtering :
    img1FilteredAvg = imfilter(img1, MA);
    
    % Difference between original & blurred image :
    img1FilteredMed = medfilt2(rgb2gray(img1));
    img1FilteredAvg = img1FilteredMed - rgb2gray(img1FilteredAvg);
    
    % Showing the result :
    figure, imshow(img1FilteredAvg*3);
    title(sprintf('Difference Between Original & Blured Versions of "01.jpg" for k = %d', k(i)),'color','r');
end

%% Section 3.3
clc
close all

k = [4 10 16 22 28];

for i = 1 : length(k)
    % Filter designing :
    % Sig = 0.2 * k
    MA = fspecial('gaussian', 2*k(i)+1, 0.2*k(i)); 
    
    % Filtering :
    img1FilteredGauss = imfilter(img1, MA);
    
    % Showing the result :
    figure, imshow(abs(img1FilteredGauss));
    title(sprintf('Gaussian Filtered Image "01.jpg" with k = %d', k(i)),'color','r');
    
    % Fourier transform First Method :
    img1FilteredGaussFT = FFT2(img1FilteredGauss);
    
    % Fourier transform Second Method :
    imgFT = fft2(img1FilteredGauss);
    imgFTshift = fftshift(imgFT);
    imgFTmag = abs(imgFTshift); % magnitude
    imgFTmag = log(imgFTmag+1); % for perceptual scaling, and +1 since log(0) is undefined
    imgFTmag = mat2gray(imgFTmag); % Using mat2gray to scale the image between 0 and 1
    
    % Showing absolute value of fourier transform Using Both Methods:
    
    % First Method
    figure
    subplot(1,2,1)
    imshow(abs(img1FilteredAvgFT), [0 200000]);
    title(sprintf('Method 1 : Absolute Value Of Fourier Transform Of Gaussian Filtered Image "01.jpg" with k = %d', k(i)),'color','r','fontsize',10);
    
    % Second Method
    subplot(1,2,2)
    imshow(imgFTmag,[]); % Display the result
    title(sprintf('Method 2 : Absolute Value Of Fourier Transform Of Gaussian Filtered Image "01.jpg" with k = %d', k(i)),'color','r','fontsize',10);
    
    % Showing histogram :
    figure, imhist(rgb2gray(img1FilteredGauss));
    title(sprintf('Histogram Of Gaussian Filtered Image "01.jpg" with k = %d', k(i)),'color','r');
    
end

for i = 1 : length(k) 
    % Filter designing :
    % Sig = 0.2 * k
    MA = fspecial('gaussian', 2*k(i)+1, 0.2*k(i)); 
    
    % Filtering :
    img1FilteredGauss = imfilter(img1, MA);
    
    % Difference between original & blurred image :
    img1FilteredMed = medfilt2(rgb2gray(img1));
    img1FilteredGauss = img1FilteredMed - rgb2gray(img1FilteredGauss);
    
    % Showing the result :
    figure, imshow(img1FilteredGauss*3);
    title(sprintf('Difference Between Original & Blured Versions of "01.jpg" for k = %d', k(i)),'color','r');
end


%% Section 3.4
clc
close all

% Filter designing :
win = 7;
sigma = 0.5;
gaussKernel = fspecial('gaussian', win, sigma);
xDerivGauss = imfilter(gaussKernel, [-1, 1]);
yDerivGauss = imfilter(gaussKernel, [-1; 1]);

% Filtering :
img1xDerivGauss = imfilter((img1), xDerivGauss);
img1yDerivGauss = imfilter((img1), yDerivGauss);

% Showing the result :
figure, imshow(3*img1xDerivGauss);
title('First Derivative Of Blurred Image With Respect To X','color','r');
figure, imshow(3*img1yDerivGauss);
title('First Derivative Of Blurred Image With Respect To Y','color','r');

% Increasing contrast :
figure, imshow(imadjust(img1xDerivGauss, [0 0 0; 0.3 0.3 0.3], []));
title('First Derivative Of Blurred Image With Respect To X (Contrast Increased)','color','r');
figure, imshow(imadjust(img1yDerivGauss, [0 0 0; 0.3 0.3 0.3], []));
title('First Derivative Of Blurred Image With Respect To Y (Contrast Increased)','color','r');

%% Section 3.5
clc
close all

% Magnitude of gradient :
img1Gmag = uint8(sqrt(pow2(double(img1xDerivGauss))+pow2(double(img1yDerivGauss))));

% Showing the result :
figure, imshow(img1Gmag);
title('Magnitude Of Gradient Of image "01.jpg"','color','r');

%% Section 3.6
clc
close all

[img1EdgeSobel, thresh] = edge(rgb2gray(img1), 'sobel');                % With default threshold in both directions
[img1EdgeSobelT] = edge(rgb2gray(img1), 'sobel', 1.5*thresh);           % With threshold = 0.7*thresh in both directions
[img1EdgeSobelX] = edge(rgb2gray(img1), 'sobel', thresh, 'horizontal'); % With default threshold in x direction
[img1EdgeSobelY] = edge(rgb2gray(img1), 'sobel', thresh, 'vertical');   % With default threshold in y direction

figure, imshow(img1EdgeSobel);
title('Edges Of Image 01.jpg With Sobel Edge Detector','color','r');
figure, imshow(img1EdgeSobelT);
title(sprintf('Edges Of Image 01.jpg With Sobel Edge Detector With Threshold = %d', 1.5 * thresh),'color','r');
figure, imshow(img1EdgeSobelX);
title('Horizontal Edges Of Image 01.jpg With Sobel Edge Detector','color','r');
figure, imshow(img1EdgeSobelY);
title('Vertical Edges Of Image 01.jpg With Sobel Edge Detector','color','r');

%% Section 4.1
clc
close all

% Openning image :
img2 = imread('02.jpg');

% Fourier transform :
img1FT = FFT2(img1);
img2FT = FFT2(img2);

% Combining fourier transforms phases and magnitudes :
img_1FT = abs(img1FT).*exp(1i*angle(img2FT));
img_2FT = abs(img2FT).*exp(1i*angle(img1FT));

% Inverse fourier transform :
img_1 = ifft2(ifftshift(img_1FT));
img_2 = ifft2(ifftshift(img_2FT));

% Showing the result :
figure, imshow(uint8(img_1));
title('Image 3 : Magnitude Of Image "01.jpg" & Phase Of Image "02.jpg"','color','r');
figure, imshow(uint8(img_2));
title('Image 4 : Magnitude Of Image "02.jpg" & Phase Of Image "01.jpg"','color','r');

%% Section 4.2
clc
close all

gaussMask = fspecial('gaussian', 100, 10);

img1LP = imfilter(img1, gaussMask);
img1HP = img1 - img1LP;
img2LP = imfilter(img2, gaussMask);
img2HP = img2 - img2LP;

% Increasing contrast of high frequency parts:
img1HPHC = imadjust(img1HP, [0 0 0; 0.4 0.4 0.4], []);
img2HPHC = imadjust(img2HP, [0 0 0; 0.5 0.5 0.5], []);


figure, imshow(0.6 * (img1LP + img2HPHC));
title('Hybrid Image : "01.jpg" Lowpassed & "02.jpg" Highpassed','color','r');
figure, imshow(0.6 * (img2LP + img1HPHC));
title('Hybrid Image : "02.jpg" Lowpassed & "01.jpg" Highpassed','color','r');

%% Section 4.3 
clc
close all

% I1 = 0.2989 * R1 + 0.5870 * G1 + 0.1140 * B1 :
I1 = rgb2gray(img1);

% Spliting image2 into it's components :
R2 = img2(:, :, 1);
G2 = img2(:, :, 2);
B2 = img2(:, :, 3);

k1 = G2./R2;
k2 = B2./R2;

% New comopnents of image2 after combining with image1 :
R1New = I1./(0.2989+0.587*k1+0.114*k2);
G1New = (I1./(0.2989+0.587*k1+0.114*k2)).*k1;
B1New = (I1./(0.2989+0.587*k1+0.114*k2)).*k2;

% Reconstructing image1 from it's new components :
img1New = uint8(zeros(size(img1)));
img1New(:, :, 1) = R1New;
img1New(:, :, 2) = G1New;
img1New(:, :, 3) = B1New;

% Showing the results :
figure, imshow(img1New);
title('Image 01.jpg That Is Colored With Image 02.jpg','color','r');


%% Section 5.1
clc
close all

% Openning image :
img4 = imread('04.jpg');

% Showing image :
figure, imshow(img4);

% Fourier transform First Method :
img4FT = FFT2(img4);

% Showing the result :
figure
subplot(1,2,1)
imshow(abs(img4FT), [0 200000]);
title('Absolute Value Of Image 4 Fourier Transform Method 1','color','r');
% Fourier transform Second Method :
subplot(1,2,2)
imshow(mat2gray(log(abs(fftshift(fft2(double(rgb2gray(img4)))))+1)),[]);
title('Absolute Value Of Image 4 Fourier Transform Method 2','color','r');


% Section 5.1 (Continued) FT Of a Zeros

% Openning image :
imgZeros = imread('zeros.jpg');

% Showing image :
figure, imshow(imgZeros);

% Fourier transform :
imgZerosFT = FFT2(imgZeros);

% Showing the result :
figure
subplot(1,2,1)
imshow(abs(imgZerosFT), [0 200000]);
title('Absolute Value Of Zeros Fourier Transform Method 1','color','r');
% Fourier transform Second Method :
subplot(1,2,2)
imshow(mat2gray(log(abs(fftshift(fft2(double(rgb2gray(imgZeros)))))+1)),[]);
title('Absolute Value Of Zeros Fourier Transform Method 2','color','r');


% Section 5.1 (Continued) FT Of a Single 0

% Openning image :
imgZero = imread('Zero1.jpg');

% Showing image :
figure, imshow(imgZero);

% Fourier transform :
imgZeroFT = FFT2(imgZero);

% Showing the result :
figure
subplot(1,2,1)
imshow(abs(imgZeroFT), [0 200000]);
title('Absolute Value Of a Single Zero Fourier Transform Method 1','color','r');
% Fourier transform Second Method :
subplot(1,2,2)
imshow(mat2gray(log(abs(fftshift(fft2(double(rgb2gray(imgZero)))))+1)),[]);
title('Absolute Value Of a Single Zero Fourier Transform Method 2','color','r');


% Section 5.1 (Continued) FT Of Ones

% Openning image :
imgOnes = imread('ones.jpg');

% Showing image :
figure, imshow(imgOnes);

% Fourier transform :
imgOnesFT = FFT2(imgOnes);

% Showing the result :
figure
subplot(1,2,1)
imshow(abs(imgOnesFT), [0 200000]);
title('Absolute Value Of Ones Fourier Transform Method 1','color','r');
% Fourier transform Second Method :
subplot(1,2,2)
imshow(mat2gray(log(abs(fftshift(fft2(double(rgb2gray(imgOnes)))))+1)),[]);
title('Absolute Value Of Ones Fourier Transform Method 2','color','r');


% Section 5.1 (Continued) FT Of a Single 1

% Openning image :
imgOne = imread('One1.jpg');

% Showing image :
figure, imshow(imgOne);

% Fourier transform :
imgOneFT = FFT2(imgOne);

% Showing the result :
figure
subplot(1,2,1)
imshow(abs(imgOneFT), [0 200000]);
title('Absolute Value Of a Single 1 Fourier Transform Method 1','color','r');
% Fourier transform Second Method :
subplot(1,2,2)
imshow(mat2gray(log(abs(fftshift(fft2(double(rgb2gray(imgOne)))))+1)),[]);
title('Absolute Value Of a Single 1 Fourier Transform Method 1','color','r');




%% Counting Zeros
clc
close all

imgZero = img4(10:36,37:54,:); % Cropping A Single Zero
imgZeroComp = imcomplement(rgb2gray(imgZero)); % Complementing Images For Better Use
img4Comp = imcomplement(rgb2gray(img4)); % Complementing Images For Better Use

KernelZero = double(imgZeroComp)/(255 * 27 * 17); % Scaling

img4CorZero = xcorr2(KernelZero, double(img4Comp)); % Calculating The Result Of Cross Correlation
img4XCorZero = normxcorr2(KernelZero, double(img4Comp));

XcorResultZero = (img4XCorZero > 0.52) * 255; % Creating White Spots Which Have Passed Threshold
figure, imshow(XcorResultZero);
title('Result Of Cross Correlation With A Single Zero','color','r');
figure, imshow(img4);
title('Original Image For Comparison Of White Spots With Zeros','color','r');
[A,B] = max(XcorResultZero);
ZeroCounter = 0;

for i = 1 : length(A) 
        
        while A(i)>200
            ZeroCounter = ZeroCounter + 1;
            XcorResultZero(B(i)-10:B(i)+10,i-10:i+10) = 0 ;% Range Which is Being Cleaned From White Spots
            [A,B] = max(XcorResultZero);
        end
        
end
clc
sprintf('Number Of Zeros : %d',ZeroCounter)

%% Counting Ones

clc
close all

imgOne = img4(10:36,19:36,:); % Cropping A Single One
imgOneComp = imcomplement(rgb2gray(imgOne)); % Complementing Images For Better Use
img4Comp = imcomplement(rgb2gray(img4)); % Complementing Images For Better Use

KernelOne = double(imgOneComp)/(255 * 27 * 17); % Scaling

img4CorOne = xcorr2(KernelOne, double(img4Comp)); % Calculating The Result Of Cross Correlation
img4XCorOne = normxcorr2(KernelOne, double(img4Comp));

XcorResultOne = (img4XCorOne > 0.71) * 255; % Creating White Spots Which Have Passed Threshold
figure, imshow(XcorResultOne)
title('Result Of Cross Correlation With A Single One','color','r');
[A,B] = max(XcorResultOne);
OneCounter = 0;
for i = 1 : length(A) 
        
        while A(i)>200
            OneCounter = OneCounter + 1;
            XcorResultOne(B(i)-10:B(i)+10,i-10:i+10) = 0 ;% Range Which is Being Cleaned From White Spots
            [A,B] = max(XcorResultOne);
        end
        
end
clc
sprintf('Number Of Ones : %d',OneCounter)





