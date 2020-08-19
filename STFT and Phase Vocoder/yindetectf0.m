function f0 = yindetectf0(x,fs,ws,minf0,maxf0)
% Fundamental frequency detection function with yin algorithm
% x: input signal
% fs: sampling rate
% ws: integration window length
% minf0: minimum f0; f0 should not be below this frquency.
% maxf0: maximum f0; f0 should not be above this frquency.
% f0: fundamental frequency detected in Hz

maxlag = ws-2; % maximum lag
th = 0.1; % set threshold
d = zeros(maxlag,1); % init variable (d(tau))
d2 = zeros(maxlag,1); % init variable (d’(tau))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                 Write your code here                        %%%%%%%
%%%%%%%                         _||_                                %%%%%%%
%%%%%%%                         \  /                                %%%%%%%
%%%%%%%                          \/                                 %%%%%%%
% Your job here is to implement Yin algorithm to detect the fundamental
% frequency of a given frame. After calculating d’(tau) function, limit the
% search to the target range by minf0 and maxf0. Compute lags corresponding
% to these frequencies and set the values of d’(tau) in lags smaller than
% the lag corresponding to maxf0 or lags greater than the lag corresponding
% to minf0 to a high number like 100 to avoid detecting them as f0. Then
% find minima of this function (d’(tau)). Consider a threshold of 0.1 for
% this function. Pick the first lag from the found set of minima that has
% a smaller function value than the threshold(0.1). If none of the found
% minima has smaller function value than the threshold, pick the lag
% corresponding to the smallest function value. Now you have a candidate lag
% value. Use the d’(tau) function values before and after the candidate lag
% value to perform a parabolic interpolation in order to refine the candidate
% lag value (find where the minimum of this interpolated parabola occurs and
% set it as the candidate lag value). At last compute candidate frequency
% in Hz by dividing the sampling frequency by candidate lag value. If the
% minimum of d’(tau) function is greater than 0.2, set f0 to 0.


% Default code:
f0 = 0;
% determining minimum and maximum lags according to maximum and minimum
% frequencies
minlag = floor(fs/maxf0) ;
% for maximum lag we should consider maxlag variable which is given, too.
maxlag = min(maxlag, ceil(fs/minf0) - 1) ;

% calculating d using 3 terms one of which is constant and details can be
% found in the report.
summation_constant = sum(x(1:ws).^2);
d = summation_constant + conv(x(1:(maxlag+ws)).^2,ones(1,ws),'val')-2*conv(x(1:(maxlag+ws)),flip(x(1:ws)),'val');

% calculating d2 by the formula given in the text.
d2 = [1:maxlag+1].*d'./cumsum(d)';
d2(d(:)==0) = 1;

% applying limitations determined in comments of this function to avoid low
% and high lags from being estimated as the fundamental period.(by setting
% their values to 100 which is far higher than the 0.1 threshold
d2(1:round(minlag)) = 100;
d2((maxlag+1):end) = 100;

% calculating the minimum lag having the condition mentioned in the text.
% first we find the local minimas according to the condition of being less
% than 0.1
[~,taumins] = findpeaks(-d2,'MinPeakHeight',-0.1);

% if no such taumin is found, according to the comments at the first part
% of this function we have to set taumin to the index of d2's minimum
% value
if isempty(taumins)
    
    [~,taumin] = min(d2);

else % such taumin is found
    % picking the first element which is the first local minima less than 0.1
    taumin = taumins(1);

end

% here we fit a second degree polynomial to interpolate the minimum lag.
% But it's important to note due to large number of warnings indicating bad
% conditioned polynomial, we use 2 points before and after the local minima
% for interpolation.
interppoints  = [taumin-2,taumin,taumin+2];

% handling the case in which indeces exceed the dimensions.(It is told to
% set f0 equal to zero at such cases.)
if taumin+2>length(d2)
    f0 = 0;
    
elseif taumin-2<=0
        
    f0 = 0;
    
else
    parinterp  = polyfit(interppoints, d2(interppoints),2);
    taumin = -parinterp(2)/2/parinterp(1);
        
end
    
% calculating the fundamental frequency according to sampling freqeuncy
f0 = fs/taumin;

% handling another last case which d2 at this local minima is larger than
% 0.2. We will return fundamental frequency = 0 at such case.
if min(d2)>0.2
    f0 = 0;
end
%%%%%%%                         /\                                  %%%%%%%
%%%%%%%                        /  \                                 %%%%%%%
%%%%%%%                         ||                                  %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
