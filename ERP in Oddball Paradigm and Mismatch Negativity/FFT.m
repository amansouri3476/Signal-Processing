function [y0, f0] = FFT(x,fs,flag)
m = length(x);          % Window length
n = pow2(nextpow2(m));  % Transform length
y = fft(x,n);           % DFT

y0 = fftshift(y);          % Rearrange y values
f0 = (-n/2:n/2-1)*(fs/n);  % 0-centered frequency range
if flag == 1
    plot(f0,abs(y0));
    xlim([-100 100]);
end
end