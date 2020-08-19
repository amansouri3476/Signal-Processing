function y = bh92transform(x,N)
% Calculate transform of the Blackman-Harris 92dB window
% x: bin positions to compute (real values)
% y: transform values
% N: DFT size

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                 Write your code here                        %%%%%%%
%%%%%%%                         _||_                                %%%%%%%
%%%%%%%                         \  /                                %%%%%%%
%%%%%%%                          \/                                 %%%%%%%

% Your job is to calculate the Fourier transform of a zero-centered 
% Blackman-Harris 92 dB in given positions in the array x. At the end 
% normalize the transform valus by dividing them to (0.35875 * N).


% Default code:

blackmanwindow = blackmanharris(N);
modifiedwindow = blackmanwindow./0.35875/N;
% using fftshift to have a zero-centered spectrum with center at index
% N/2+1
y = fftshift(fft(modifiedwindow));
% shifting the bin to match the significant part of the spectrum
x = x + N/2 + 1;
% clearing phase effect and picking the amplitude of real-even(blackman) transform
y = abs(y(x));


%%%%%%%                         /\                                  %%%%%%%
%%%%%%%                        /  \                                 %%%%%%%
%%%%%%%                         ||                                  %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
