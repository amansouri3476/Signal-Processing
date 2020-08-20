function x_new = FFT2 (x)
    
    x_new = double(rgb2gray(x));
    x_new = fftshift(fft2(x_new));

end