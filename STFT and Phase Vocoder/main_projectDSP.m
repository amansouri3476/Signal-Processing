% In the name of god
% This is our main file
%
% This is the main file which should be run to produce the outputs. If this
% is your first time with MATLAB, either press F5 or click on the green
% triangular button (RUN).
%
% In this file you may change values of the following variables
%   "sample_number"
%   "section_enable"
%   "sub_section"
%   "hoarsening_coeff"
%   "win_multiplier"
%   "drawplot"
%   "rmin"
%   "rmax"
%   "bn"
%   "input_bins"
%   "effect_number"
%	"yin_window_length"
%
% Your main task in this project is to edit the following m files
%   'phasewrapper.m'
%   'voice_effects.m'
%   'voice_timevar.m'
%   'bh92transform.m'
%   'sinestransform.m'
%   'yindetectf0.m'
% located in "DSP Final Project/Functions/". Please avoid modifying
% other parts of this code and other files.

%% Initialization 

% This part is to clear the command window, remove all the pre-defined
% parameters and close all open figures
clear all;
close all;
clc;


% Adding functions and input-voices path to the matlab directory

addpath(genpath('Input_voices'));
addpath(genpath('Functions'));
addpath(genpath('Results'));

% Choose which sample file you want to apply effects to:
% if you want to test the effects with your own voice choose 5 as sample_number.
% Remember your recorded voice shoud be converted to wav format and it sampling frequency must
% be 44100 Hz.


% Choose the sample file to apply effects to
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                          You can change sample file                            %%%%%%%
%%%%%%%                                      _||_                                      %%%%%%%
%%%%%%%                                      \  /                                      %%%%%%%
%%%%%%%                                       \/                                       %%%%%%%

sample_number = 3;    
						% 1 = 'basket.wav'
						% 2 = 'meeting.wav'
						% 3 = 'ACS.wav'
						% 4 = 'Metamorphosis.wav'
						% 5 = 'test3.wav'
                        % 6 = 'Your_Voice.wav'
if sample_number == 1
	[x,fs] = audioread('Input_voices/basket.wav');
elseif sample_number == 2
	[x,fs] = audioread('Input_voices/meeting.wav');
elseif sample_number == 3
	[x,fs] = audioread('Input_voices/ACS.wav');
elseif sample_number == 4
	[x,fs] = audioread('Input_voices/Metamorphosis.wav');
elseif sample_number == 5
    [x,fs] = audioread('Input_voices/test3.wav');
else
    [x,fs] = audioread('Input_voices/Your_voice.wav');
end

%%%%%%%                                       /\                                       %%%%%%%
%%%%%%%                                      /  \                                      %%%%%%%
%%%%%%%                                       ||                                       %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Check if the voice is mono or stereo and make it mono in case it is stereo
xsize = size(x);
if xsize(2) == 2
    x = (x(:,1) + x(:,2))/2;
end



%% Section Enable

% Choose the section to run due to Readme file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%                          You can change section enable                         %%%%%%%
%%%%%%%                                      _||_                                      %%%%%%%
%%%%%%%                                      \  /                                      %%%%%%%
%%%%%%%                                       \/                                       %%%%%%%

section_enable        = 3;
                            % 1: Adding simple effects to the input voice
                            % 2: Stretching/compressing in the time domain    
                            % without destroying voice's characteristics
                            % 3: Changing gender/age of the voice

%%%%%%%                                       /\                                       %%%%%%%
%%%%%%%                                      /  \                                      %%%%%%%
%%%%%%%                                       ||                                       %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




switch section_enable
    case 1
        %% Adding simple effects to the input signal
        
        % Here you should design the 'phasewrapper' function. 
        phasewrapper(0);
        % Plot the output of the 'phasewrapper' function for phi = [-3pi 3pi]
        phi = -3*pi: 0.01*pi: 3*pi;
        plot(phi, phasewrapper(phi));
        axis equal;
        title('output of phasewrapper function');
        
        % Save the plot above
        saveas(1,'Results/Section1/output_of_phasewrapper', 'jpg');
        
        
        % Herein we will examine various levels of hoarsening for the first
        % part the various window sizes for the second part.
         
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%        You can change sub_section, hoarsening_coeff, and win_multiplier        %%%%%%%
        %%%%%%%                                      _||_                                      %%%%%%%
        %%%%%%%                                      \  /                                      %%%%%%%
        %%%%%%%                                       \/                                       %%%%%%%

        sub_section             = 2;
                                    % 1: hoarsening effect
                                    % 2: robotizing effect
        hoarsening_coeff        = 0.8;
                                    % Choose a number between 0 (no  
                                    % hoarsening) to 1 (complete hoarsening)
        win_multiplier          = 32;
                                    % Choose a discrete number >= 1;

        %%%%%%%                                       /\                                       %%%%%%%
        %%%%%%%                                      /  \                                      %%%%%%%
        %%%%%%%                                       ||                                       %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Creating the output signal
        output     = voice_effects(sub_section, 256, ...
                    1024* win_multiplier, x, hoarsening_coeff);
                
        % Playing the output signal
        soundsc(output,fs);
        
        % Creating the name of output signal
        if (sub_section == 1)
           sub_name     = 'hoarsening';
           number       = num2str(hoarsening_coeff);
        else
           sub_name     = 'robotization';
           number       = num2str(win_multiplier);
        end
        %Saving the output signal
        filename        = ['Section1_' num2str(sample_number)...
                          sub_name number  '.wav'];
        audiowrite(strcat('Results/Section1/', filename), output, fs);
        
        
    case 2
        %% Stretching/compressing in the time domain

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%          You can change sub_section, rmin, rmax, and drowplot here             %%%%%%%
        %%%%%%%                                     _||_                                       %%%%%%%
        %%%%%%%                                     \  /                                       %%%%%%%
        %%%%%%%                                      \/                                        %%%%%%%

        sub_section                 = 3;
                                        % 1: Simple Up/Down sampling
                                        % 2: Sum of cosine method
                                        % 3: Direct FFT method
        drawplot                    = 1;
                                        % 0: Do not draw plot
                                        % 1: Draw plot
        rmin                        = 2;
                                        % Minimum stretching ratio
        rmax                        = 2;
                                        % Maximum stretching 	

        %%%%%%%                                      /\                                        %%%%%%%
        %%%%%%%                                     /  \                                       %%%%%%%
        %%%%%%%                                      ||                                        %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        % Fristly, you should complete the 'voice_timevar' function.
        % After that, you should go to the next task:
        
        if (drawplot == 1)
            % Here we will draw two plots to see what happens with our code 
            % and compare it with rough approaches.
            drawplots;
        else
            
            % Now, based on the instructions in the readme file, complete
            % assigned tasks.
            output = voice_timevar(sub_section, 128, 128, 1024, ...
                x, rmin, rmax);
            
            % Play the output result
            soundsc(output,fs);
        
            % Creating the output name
            rationame    = [num2str(min(rmin)) 'to' num2str(max(rmax))];
            filename     = ['Section2-subsec' num2str(sample_number)...
                           num2str(sub_section) rationame '.wav'];
            % Saving the output signal
            audiowrite(strcat('Results/Section2/', filename), output, fs);
        end
        
    

    case 3
    	fs = 44100;
        %% Changing gender/age of the voice
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%                        You can change sub_section, bn, input_bins,             %%%%%%%
        %%%%%%%						   effect_number here                                	   %%%%%%%
        %%%%%%%                                      _||_                                      %%%%%%%
        %%%%%%%                                      \  /                                      %%%%%%%
        %%%%%%%                                       \/                                       %%%%%%%
        
        
        sub_section                 = 4; 
                                                % 1: Zero-centered Blackman-Hariis 92 dB transform
                                                % 2: Generate sine spectrum
                                                % 3: Detect fundamental frequency of a frame
                                                % 4: Changing gender/age of the voice


        input_bins                  = [-20:20]; % frequency bins in which Fourier transform of 
                                                % Blackman-Hariis window is calculated

       
        bn                          = 4;        % bn: number of bins before and after a peak where we 
                                                % estimate the filtered sinusoidal spectrum 


        yin_window_length           = 0.0125;   % Integeration window length in time domain (second)



        effect_number               = 3;
                                                % 1: Male to female
                                                % 2: Female to male
                                                % 3: Male to child


        fscale 						= 2;        % frequency scale factor


        timbremapping   =  [0 3600 fs/2;   		% input frequency for mapping 
                            0 5000 fs/2]; 		% output frequency for mapping
        

                                                

        %%%%%%%                                       /\                                       %%%%%%%
        %%%%%%%                                      /  \                                      %%%%%%%
        %%%%%%%                                       ||                                       %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        if (sub_section == 1)
            ybh92 = bh92transform(input_bins,1024);
            plot(input_bins, abs(ybh92));
            title('Zero-centered Blackman-Harris Window Transform');
            % Save the plot above
            saveas(1,strcat('Results/Section3/Blackman-Hariis_transform_', int2str(input_bins(end))) , 'jpg');
        end
        if (sub_section == 2)
            sinespecparams = load('Signals/sinespecparams.mat');
            hloc = sinespecparams.hloc;
            hmag = sinespecparams.hmag;
            hphase = sinespecparams.hphase;
            
            sinespec = sinestransform(hloc(1:200),hmag,hphase,1024, bn);
            plot(1:1024, abs(sinespec));
            title('Sinusoidal Spectrum');
            % Save the plot above
            saveas(1, strcat('Results/Section3/sines_spectrum_bn', int2str(bn)), 'jpg');
        end
        if (sub_section == 3)
            test_signal = load('Signals/test_signal.mat');
            test_signal = test_signal.test_signal;
            test_yinws = round(44100 * yin_window_length);
            test_yinws = test_yinws+mod(test_yinws,2); % make window length even
            test_f0 = yindetectf0(test_signal,44100,test_yinws,100,400);
            fprintf('Detected Fundamental Frequency: %f Hz\n', test_f0);
        end
         if (sub_section == 4) 
         	w = [blackmanharris(1024);0];
            if (effect_number == 1)
                %-----male to female-----%
                [y,yh,yr] = apply_effects(x,fs,w,2048,-150,200,100,400,1,.2,10,[],fscale,timbremapping,bn);
                audiowrite(strcat('Results/Section3/MaleToFemale_sample', int2str(sample_number), '.wav'),y,fs);
            end
            if (effect_number == 2)
                %-----female to male-----%
                [y,yh,yr] = apply_effects(x,fs,w,2048,-150,200,100,400,1,.2,10,[],fscale,timbremapping,bn);
                audiowrite(strcat('Results/Section3/FemaleToMale_sample', int2str(sample_number), '.wav'),y,fs);
            end
            if (effect_number == 3)
                %-----male to child-----%
                [y,yh,yr] = apply_effects(x,fs,w,2048,-150,200,100,400,1,.2,10,[],fscale,timbremapping,bn);
                audiowrite(strcat('Results/Section3/MaleToChild_sample', int2str(sample_number), '.wav'),y,fs);
            end
        end  
end
