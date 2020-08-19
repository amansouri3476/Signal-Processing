function [output] = voice_timevar(selector, s1, s2i, ...
     s_win, input, rmin, rmax)
% [output,Fs] = voice_timevar(selector, s1, s2i, ...
%    s_win, input, rmin, rmax)
%
% This function is designed to implement three tasks, in which we try to
% change an input files duration (speech speed.) 
% The first one is a simple up-down sampling which is designed to show how
% bad would it be to change file durating recklessly and without paying
% attention to the DSP details.
% The third taks, which uses our second approach - direct FFT, is a way
% faster than the second task.
%
% Inputs:
% selector          1: Simple up-down sampling
%                   2: Sum of cosine method
%                   3: Direct FFT method
% s1                 : Step size that in each step the input window goes
%                      forward by this factor
% s2i                : Initial output step size which is deigned to be
%                      changed by the ratio factor 
% s_win              : Window size
% input              : The input file
% rmin               : Minimum stretching ratio
% rmax               : Maximum stretching ratio
%
% Outputs:
% output             : The output file

tic

%% Initializing
% Creating ratio vector regarding lower and upper limit of ratio 
ratio        = [rmin rmax];

l            = length(input);

win          = hanning(s_win);

% Zero padding to become sure the input signal has more than s_win elements, 
% and with this s_win and step size it will not exceed number of elements.
input        = [zeros(s_win,1);input;zeros(s_win-mod(l,s1),1)];

% Normalizing the input signal
input        = input/max(abs(input));

% Number of segments 
n_seg        = (length(input)-s_win)/s1;

% Setting ratios decreasing liniarly during procedure
Ratios       = linspace(max(ratio),min(ratio),n_seg);
s2           = round(s2i*Ratios);

% Based on the fact that we know the output step is varying during
% procedure, and we set it to decrease liniarly, we sould write:
output       = zeros(s_win + ceil(length(input)*mean(ratio)),1);


    switch selector
        case 1
            %% Simple up-down sampling
            down   = downsample(input, 2);
            up     = upsample(input, 2);
            L      = length(input);
            
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%           Write your code here              %%%%%%%
                %%%%%%%                   _||_                      %%%%%%%
                %%%%%%%                   \  /                      %%%%%%%
                %%%%%%%                    \/                       %%%%%%%

                % Your job is to create the 'up2' signal based on what has
                % been described in the readme file.

                % Available variables:
                % input:                        Is the input signal
                % L:                            Length of input signal
                %
                % Outputs of this part of code:
                % up2                       	Is the upsampled signal
                %                               of the input with the 
                %                               described technique. 

                % Default code:
                up2 = zeros(2*L,1);
                % we have to just copy input values for both odd and even
                % indeces! This is done below.
                up2([1:2:end]) = input([1:end]);
                up2([2:2:end]) = input([1:end]); 
        
                %%%%%%%                    /\                       %%%%%%%
                %%%%%%%                   /  \                      %%%%%%%
                %%%%%%%                    ||                       %%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % Creating a single output file
            output = [down;up;up2];

            % Normalizing the output
            output = output/max(output);
    

        case 2
            %% Approach no.1: Sum of cosines
            
            % Since the input signal has real values, we only need half of
            % its frequency samples!
            s_half  = s_win/2;
            
            % Building constant part of sim. frequency
            omega   = 2*pi*s1*(0:s_half-1)'/s_win;   
            
            % Phase of previous segment
            phi0    = zeros(s_half,1);
            % Amplitude of previous segment
            a0      = zeros(s_half,1);
            % Phase of previous output 
            Phi     = zeros(s_half,1);    
            
            pointer = 0;                        

            for i   = 1:n_seg
                
                % Initializing output of each part
                result        = zeros(s2(i),1); 
                
                % Getting the input segment multiplied by window
                segment       = input((i-1)*s1+1:(i-1)*s1+s_win).*win;
                
                % Performing FFT of the segment and choosing half of it
                f_seg         = fft(fftshift(segment));
                f_half        = f_seg(1:s_half);     
                a             = abs(f_half);
                phi           = angle(f_half);
      
                % Computing total changes in phase vector
                deltaPhi      = omega + phasewrapper(phi-phi0-omega);
                
                % Computing total changes in amplitude vector
                dA            = (a-a0)/s2(i);
                dPhi          = deltaPhi/s1;
                
                
                % Producing output signal incrementaly bt sum of cosine
                for k         = 1:s2(i)
                    a0        = a0 + dA;
                    Phi       = Phi + dPhi;
                    result(k) = a0'*cos(Phi); 
                end
                
                % Initializing the next iteration
                phi0          = phi;
                a0            = a;
                Phi           = phasewrapper(Phi);
                pointer       = pointer + s2(i);
   
                output(pointer+1:pointer+s2(i)) = ...
                                result;
   
            end

            % Normalizing the output
            output = output/max(output);
        

        case 3
            %% Approach no.2: Direct FFT

            % Building constant part of sim. frequency
            omega   = 2*pi*s1*(0:s_win-1)'/s_win;
            
            % Phase of previous segment
            phi0    = zeros(s_win,1);
            % Phase of previous output
            Phi     = zeros(s_win,1);
            
            pointer = 0;

            for j   = 1:n_seg
                
                % Getting the input segment multiplied by window
                segment  = input((j-1)*s1+1:(j-1)*s1+s_win).*win;
                
                % Performing FFT of the segment
                f_seg    = fft(fftshift(segment));
                a        = abs(f_seg);
                phi      = angle(f_seg);
   
                % Computing total changes in sim. freq
                deltaPhi = omega + phasewrapper(phi-phi0-omega);
                
                % Computing delta phi per sample
                Phi      = phasewrapper(Phi+deltaPhi*(s2(j)/s1));
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%           Write your code here              %%%%%%%
                %%%%%%%                   _||_                      %%%%%%%
                %%%%%%%                   \  /                      %%%%%%%
                %%%%%%%                    \/                       %%%%%%%

                % Your job is to create the 'ft' vector. A complex vector
                % which took its amplitude from vector 'a' and its phase
                % from vector 'Phi'.

                % Available variable:
                % a:                            Amplitude vector. 
                % Phi                           Redefined phase vector
                %
                % Outputs of this part of code:
                % ft                            A vector with a's amplitude
                %                               and Phi's phase
    
                % Default code:
                % a is amplitude and phi is the desired phases
                ft = a.*exp(1i*Phi);
            

                %%%%%%%                    /\                       %%%%%%%
                %%%%%%%                   /  \                      %%%%%%%
                %%%%%%%                    ||                       %%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                % Performing IFFT
                result   = ifftshift(real(ifft(ft))).*win;
   
                output(pointer+1:pointer+s_win) = ...
                           output(pointer+1:pointer+s_win) + result;
                
                % Initializing for the next step
                pointer  = pointer + s2(j);
                phi0     = phi;
   
            end

            % Normalizing the output
            output = output/max(output);
        
    end

toc
    
end

