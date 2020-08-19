% In this file we will draw desirable plots. With the help of previous
% functions we create two plots to compare our approaches with the naive
% ones. 
% In the first figure, we compare time stretching approaches, and in the 
% second one, we compare time compressing approaches. Both with factor 2.
clear all

[input,Fs] = audioread('test3.wav');

% Check if the voice is mono or stereo and make it mono in case it is stereo
inputsize = size(input);
if inputsize(2) == 2
    input = (input(:,1) + input(:,2))/2;
end

%% Part one: time stretching
% In this part, we will plot 4 plots in a single figure with subplot
% command to show our methods do not change the shape of main signal in
% frequency domain too much. the stretching factor in all of them would be 
% euql to 2.

figure(1)

% Abs(FFT) of main signal
subplot(4,1,1)
plot(linspace(-pi,pi,length(input)),abs(fftshift(fft(input))))
xlim([-pi pi])
xlabel('From -\pi to \pi')
title({'Upsampling with factor 2';'FFT of input'});

% Abs(FFT) of output of "Sum of cosine" approach
subplot(4,1,2)
output = voice_timevar(2, 256, 256, 1024, input, 2, 2);
plot(linspace(-pi,pi,length(output)),abs(fftshift(fft(output))))
xlim([-pi pi])
title('FFT of output of "Sum of cosine" approach');

% Abs(FFT) of output of "Direct FFT" approach
subplot(4,1,3)
output = voice_timevar(3, 256, 256, 1024, input, 2, 2);
plot(linspace(-pi,pi,length(output)),abs(fftshift(fft(output))))
xlim([-pi pi])
title('FFT of output of "Direct FFT" approach');

% Abs(FFT) of output of simple upsampling approach
subplot(4,1,4)
up = upsample(input,2);
plot(linspace(-pi,pi,length(up)),abs(fftshift(fft(up))))
xlim([-pi pi])
title('FFT of simple upsampling of the input');

% Saving plot
saveas(1,'Results/Section2/Upsampling', 'jpg');

figure(3)

% Abs(FFT) of output of simple upsampling approach
subplot(2,1,1)
up = upsample(input,2);
plot(linspace(-pi,pi,length(up)),abs(fftshift(fft(up))))
xlim([-pi pi])
title('FFT of simple upsampling of the input');

L = length(input);
up2 = zeros(2*L,1);
up2([1:2:end]) = input([1:end]);
up2([2:2:end]) = input([1:end]);

subplot(2,1,2)
plot(linspace(-pi,pi,length(up2)),abs(fftshift(fft(up2))))
xlim([-pi pi])
title('FFT of modified upsampling (up2) of the input');

% Saving plot
saveas(3,'Results/Section2/Upsampling2', 'jpg');


%% Part two: time compressing
% In this part, we will plot 4 plots in a single figure with subplot
% command to show our methods do not change the shape of main signal in
% frequency domain too much. the compressing factor in all of them would be 
% euql to 2. One should keep this in his or her mind that compressing with 
% factor of 2 is equal to stretching with factor of 1/2. 

figure(2)

% Abs(FFT) of main signal
subplot(4,1,1)
plot(linspace(-pi,pi,length(input)),abs(fftshift(fft(input))))
xlim([-pi pi])
xlabel('From -\pi to \pi')
title({'Downsampling with factor 2';'FFT of input'});

% Abs(FFT) of output of "Sum of cosine" approach
subplot(4,1,2)
output = voice_timevar(2, 256, 256, 1024, input, 1/2, 1/2);
plot(linspace(-pi,pi,length(output)),abs(fftshift(fft(output))))
xlim([-pi pi])
title('FFT of output of "Sum of cosine" approach');

% Abs(FFT) of output of "Direct FFT" approach
subplot(4,1,3)
output = voice_timevar(3, 256, 256, 1024, input, 1/2, 1/2);
plot(linspace(-pi,pi,length(output)),abs(fftshift(fft(output))))
xlim([-pi pi])
title('FFT of output of "Direct FFT" approach');

% Abs(FFT) of output of simple downsampling approach
subplot(4,1,4)
down = downsample(input,2);
plot(linspace(-pi,pi,length(down)),abs(fftshift(fft(down))))
xlim([-pi pi])
title('FFT of simple downsampling of the input');

% Saving plot
saveas(2,'Results/Section2/Downsampling', 'jpg');

