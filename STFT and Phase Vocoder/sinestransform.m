function Y = sinestransform(ploc, pmag, pphase, N, bn)
% Compute a spectrum from a series of sine values
% ploc: sine locations in STFT spectrum [not neccessarily an integer 
% because of the refinements in locating the peaks]
% pmag: sine magnitudes in dB
% pphase: sine phasesin radian
% N: size of complex spectrum
% bn: number of bins before and after a peak where we estimate the filtered
% sinusoidal spectrum
% Y: generated complex spectrum of sines


% Default code:
Y = zeros(N,1);
hN = N/2+1; % size of positive freq. spectrum
bhw = bh92transform((-bn:bn),N);
for i=1:length(ploc) % generate all sine spectral lobes
    loc = ploc(i); % location of peak (zero-based indexing)
    % it should be in range ]0,hN-1[
    if (loc<=1||loc>=hN-1)
        continue;
    end % avoid frequencies out of range

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%%                 Write your code here                        %%%%%%%
	%%%%%%%                         _||_                                %%%%%%%
	%%%%%%%                         \  /                                %%%%%%%
	%%%%%%%                          \/                                 %%%%%%%
	% Your job here is very simple. In each peak location (loc) and with the 
	% help of bh92transform function, calculate the magnitudes of sine spectrum
	% (lmag) related to this peak. In Blackman-Harris filter use only bn bins 
	% before and after the peak location for calculating spectrum magnitudes. 
	% ([round(loc)-bn:round(loc)+bn])
    
    % default code
    lmag = zeros((2 * bn + 1),1); % lobe magnitudes of the complex exponential
    lmag = 10^(pmag(i)/20)*bhw;

	%%%%%%%                         /\                                  %%%%%%%
	%%%%%%%                        /  \                                 %%%%%%%
	%%%%%%%                         ||                                  %%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % complex exponential
    b = 1+[round(loc)-bn:round(loc)+bn]; % spectrum bins to fill
    % (1-based indexing)
    for m=1:(2 * bn + 1)
        if (b(m)<1) % peak lobe croses DC bin
            Y(2-b(m)) = Y(2-b(m)) + lmag(m)*exp(-1i*pphase(i));
        elseif (b(m)>hN) % peak lobe croses Nyquist bin
            Y(2*hN-b(m)) = Y(2*hN-b(m)) + lmag(m)*exp(-1i*pphase(i));
        else % peak lobe in positive freq. range
            Y(b(m)) = Y(b(m)) + lmag(m)*exp(1i*pphase(i)) + lmag(m)*exp(-1i*pphase(i))*(b(m)==1||b(m)==hN);
        end
    end
    Y(hN+1:end) = conj(Y(hN-1:-1:2)); % fill the rest of the spectrum  
end   
end


