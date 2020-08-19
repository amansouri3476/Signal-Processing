function bank = myownfilt_bank(N,L,Win,is_PM,Fs,B)
%FILT_BANK	Filter bank generator template
%   BANK = FILT_BANK(N,L,winType,is_PM,Fs,B) generates a bank of non-overlapping filters where
%   N is the number of filter bands, L is the length of each FIR filter,  Fs is
%   the sampling frequency in Hz,B is the width of each band in Hz and Win is 
%
%   window type :  'kaiser' or 'rectangular'
%
%   BANK is an LxN matrix, where each of the N columns of BANK contains an L-point FIR
%
%   BANK = FILT_BANK(N,L,,winType,is_PM,Fs) automatically selects the bandwidth B so that the N
%   filters span the spectrum from 0 Hz to 3600 Hz.
%
%   BANK = FILT_BANK(N,L,winType,is_PM) sets Fs to 8000 Hz, and automatically selects the
%   bandwidth B so that the N filters span the spectrum from 0 Hz to 3600 Hz.
%

% Set Defaults
%-------------
if nargin < 6
    B       = 3600/N;  % set default width of each band in Hz
end
if nargin < 5
    Fs      = 8000;   % set default sampling frequency in Hz
    B = 3600/N/Fs;
end

start       = B/2;     % First center freq. in Hz

% preallocate output for speed
bank        = zeros(L,N);

% x-axis vector for plotting and saving filterBank figures
% note that you should change it into correct vector for plotting
% freqAxis    = 1:L;
freqAxis = Fs/2*linspace(-1,1,L);
if(~is_PM)
    % Design a prototype lowpass filter
    %----------------------------------
    lpf  = zeros(1,L); 
    switch Win
        case 'kaiser'
    %        create your low pass filter using kaiser window ...
    %        ....
        lpf = fir1(L-1,B/2,'low',kaiser((L),3));
        case 'rectangular'
    %       create your low pass filter using rectangular window ...
    %       ....
        lpf = fir1(L-1,B/2,'low',rectwin(L));
    end

    lpf = lpf(:);

    % Create bandpass filters
    %------------------------
    bandDistance = 2*pi/N;
    for i=1:N
        
        bank(:,i) = lpf.*exp(1j*[0:L-1]'*bandDistance*i);
        
    end

else
    % Design an appropriate Parks-McClellan filter 
    %----------------------------------
    pmOrder = L-1;
    bandDistance = 2*pi/N;
%     lpf = firpm(pmOrder,[0 B/2*0.6 B/2*0.7 B/2*0.8 B/2*0.9 B/2*1.1 B/2*1.25 B/2/0.7 B/2/0.6 1],[1 1 1 1 1 0 0 0 0 0]);
    lpf = firpm(pmOrder,[0 B/2 B/2+0.1 1],[1 1 0 0]);
    
    for i=1:N
        
        bank(:,i) = lpf.*exp(1j*[0:L-1]*bandDistance*i);
    
    end
    
    
end
                                
% plotting and saving frequency response of your filterBank
%---------------------------------------------------------
size(bank)
for i = 1:N
    figure(i)
    length(freqAxis)
    length(bank(:,i))
    plot(freqAxis,abs(fftshift(fft(bank(:,i)))));
    xlabel('f(Hz)');
    ylabel('|H(e^{jw})|');
    title(['magnitude of frequency response of band' ,num2str(i)]); 
    if(~is_PM)
        switch Win
            case 'kaiser'
                if exist(['./filtBank figures/Kaiser/',num2str(N),'/',num2str(L)],'dir')~= 7  % checking directory existence
                    mkdir(['./filtBank figures/Kaiser/',num2str(N),'/',num2str(L)]);            % making directory
                end
                hgsave(['./filtBank figures/Kaiser/' , num2str(N),'/',num2str(L),'/band',num2str(i),'.fig']);

            case 'rectangular'
                if exist(['./filtBank figures/Rectangular/',num2str(N),'/',num2str(L)],'dir')~= 7
                    mkdir(['./filtBank figures/Rectangular/',num2str(N),'/',num2str(L)]);
                end
                hgsave(['./filtBank figures/Rectangular/' , num2str(N),'/',num2str(L) ,'/band',num2str(i),'.fig']);
        end
    else
        % saving in different folders based on order of PM filter 
        if exist(['./filtBank figures/P_M/',num2str(pmOrder),'/',num2str(L)],'dir')~= 7  % checking directory existence
            mkdir(['./filtBank figures/P_M/',num2str(pmOrder),'/',num2str(L)]);            % making directory
        end
        hgsave(['./filtBank figures/P_M/' , num2str(pmOrder),'/',num2str(L),'/band',num2str(i),'.fig']);
    end
end

if(~is_PM)
    
    switch Win
               case 'kaiser'
                    figure
                    for i = 1:N
                        plot(freqAxis,abs(fftshift(fft(bank(:,i)))));
                        hold on
                        xlabel('f(Hz)');
                        ylabel('|H(e^{jw})|');
                        title('magnitude of frequency response of kaiser window'); 

                    end

                
               case 'rectangular'
                   figure
                    for i = 1:N
                        plot(freqAxis,abs(fftshift(fft(bank(:,i)))));
                        hold on
                        xlabel('f(Hz)');
                        ylabel('|H(e^{jw})|');
                        title('magnitude of frequency response of rectangular window'); 

                    end
                
    end
    
else
    figure
                    for i = 1:N
                        plot(freqAxis,abs(fftshift(fft(bank(:,i)))));
                        hold on
                        xlabel('f(Hz)');
                        ylabel('|H(e^{jw})|');
                        title('magnitude of frequency response of Parks Mcclellan window'); 

                    end
end
% figure
% sumbank = sum(bank,2);
% polt(abs(fftshift(fft(sumbank))));
end
