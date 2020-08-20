%% Part 3
%  Section 1,2
clc
close all

S1 = load('S01.mat'); % First person data
S2 = load('S02.mat'); % Second person data

% First person :
numberOfProbes = 10;
FS = 512;
probe_passive = zeros(numberOfProbes, length(S1.data{1,1}.X(:,1)))'; % Preallocation for passive probes
probe_active = zeros(numberOfProbes, length(S1.data{1,2}.X(:,1)))'; % Preallocation for active probes

t_passive = ((1:length(probe_passive(: , 1)))')/FS;
t_active = ((1:length(probe_active(: , 1)))')/FS;

%Initializing : 
%Passive
for  i = 1:numberOfProbes
    probe_passive(:, i) = S1.data{1,1}.X(:, i);
end
%Active
for  i = 1:numberOfProbes
    probe_active(:, i) = S1.data{1,2}.X(:, i);
end

%Plotting the results :

nPassive = pow2(nextpow2(length(t_passive))); % Fourier Transform length for passive
nActive = pow2(nextpow2(length(t_active))); % Fourier Transform length for active
probe_passive_ft = zeros(nPassive,numberOfProbes); % Preallocating
probe_active_ft = zeros(nActive, numberOfProbes); % Preallocating
probe_passive_Freq = ((-nPassive/2:nPassive/2-1)*(FS/nPassive))'; % Frequency range for passive
probe_active_Freq = ((-nActive/2:nActive/2-1)*(FS/nActive))'; % Frequency range for active

%Passive_FT
for i = 1:numberOfProbes
    figure
    probe_passive_ft(:, i) = FFT(probe_passive(:, i), FS, 1);
    title(strcat('Channel', sprintf(' : %d', i)));
end
suptitle('Passive mode fourier transform');

%Active_FT
for i = 1:numberOfProbes
    figure
    probe_active_ft(:, i) = FFT(probe_active(:,i), FS, 1);
    title(strcat('Channel', sprintf(' : %d', i)));
end
suptitle('Active mode fourier transform');

%% Section 3
clc
close all

% Frequency domain signals :
frequencyBandsTransformPassive = zeros(nPassive, 4*numberOfProbes); % Preallocating
frequencyBandsTransformActive = zeros(nActive, 4*numberOfProbes); % Preallocating

% Time domain signals :
frequencyBandsSignalPassive = zeros(nPassive, 4*numberOfProbes); % Preallocating
frequencyBandsSignalActive = zeros(nActive, 4*numberOfProbes); % Preallocating

% Time axis :
t = (0:nPassive-1)/FS;

% Delta band :
for i = 1:numberOfProbes
    % Plotting :
    figure
    
    subplot(2,2,1);
    hold on
    frequencyBandsTransformPassive(:, 4*i-3) = FTP(probe_passive_ft(:, i), probe_passive_Freq, 0, 4, 1);
    FFT(probe_passive(:, i), FS, 1); % Plotting whole signal fourier transform in passive mode
    title('Passive FT');
    hold off
    
    subplot(2,2,2);
    hold on
    frequencyBandsTransformActive(:, 4*i-3) = FTP(probe_active_ft(:, i), probe_active_Freq, 0, 4, 1);
    FFT(probe_active(:,i), FS, 1);  % Plotting whole signal fourier transform in active mode
    title('Active FT');
    hold off
    
    subplot(2,2,3);
    hold on
    frequencyBandsSignalPassive(:, 4*i-3) = ifft(ifftshift(frequencyBandsTransformPassive(:, 4*i-3)), 'symmetric');
    plot(t, frequencyBandsSignalPassive(:, 4*i-3));
    plot(t_passive, probe_passive(:, i));
    xlim([0 t_passive(end)]);
    title('Passive IFT');
    hold off
    
    subplot(2,2,4);
    hold on
    frequencyBandsSignalActive(:, 4*i-3) = ifft(ifftshift(frequencyBandsTransformActive(:, 4*i-3)), 'symmetric');
    plot(t, frequencyBandsSignalActive(:, 4*i-3));
    plot(t_active, probe_active(:, i));
    xlim([0 t_active(end)]);
    title('Active IFT');
    hold off
    
    suptitle(sprintf('Channel : %d Delta', i));
    
end

%%
clc
close all

% Alpha band :
for i = 1:numberOfProbes
    % Plotting :
    figure
    
    subplot(2,2,1);
    hold on
    frequencyBandsTransformPassive(:, 4*i-2) = FTP(probe_passive_ft(:, i), probe_passive_Freq, 8, 13, 1);
    FFT(probe_passive(:, i), FS, 1); % Plotting whole signal fourier transform in passive mode
    title('Passive FT');
    hold off
    
    subplot(2,2,2);
    hold on
    frequencyBandsTransformActive(:, 4*i-2) = FTP(probe_active_ft(:, i), probe_active_Freq, 8, 13, 1);
    FFT(probe_active(:,i), FS, 1);  % Plotting whole signal fourier transform in active mode
    title('Active FT');
    hold off
    
    subplot(2,2,3);
    hold on
    frequencyBandsSignalPassive(:, 4*i-2) = ifft(ifftshift(frequencyBandsTransformPassive(:, 4*i-2)), 'symmetric');
    plot(t, frequencyBandsSignalPassive(:, 4*i-2), 'r');
    plot(t_passive, probe_passive(:, i), 'b');
    xlim([0 t_passive(end)]);
    title('Passive IFT');
    hold off
    
    subplot(2,2,4);
    hold on
    frequencyBandsSignalActive(:, 4*i-2) = ifft(ifftshift(frequencyBandsTransformActive(:, 4*i-2)), 'symmetric');
    plot(t, frequencyBandsSignalActive(:, 4*i-2), 'r');
    plot(t_active, probe_active(:, i), 'b');
    xlim([0 t_active(end)]);
    title('Active IFT');
    hold off
    
    suptitle(sprintf('Channel : %d Alpha', i));
    
end

%%
clc
close all

% Beta band :
for i = 1:numberOfProbes
    % Plotting :
    figure
    
    subplot(2,2,1);
    hold on
    frequencyBandsTransformPassive(:, 4*i-1) = FTP(probe_passive_ft(:, i), probe_passive_Freq, 13, 30, 1);
    FFT(probe_passive(:, i), FS, 1); % Plotting whole signal fourier transform in passive mode
    title('Passive FT');
    hold off
    
    subplot(2,2,2);
    hold on
    frequencyBandsTransformActive(:, 4*i-1) = FTP(probe_active_ft(:, i), probe_active_Freq, 13, 30, 1);
    FFT(probe_active(:,i), FS, 1);  % Plotting whole signal fourier transform in active mode
    title('Active FT');
    hold off
    
    subplot(2,2,3);
    hold on
    frequencyBandsSignalPassive(:, 4*i-1) = ifft(ifftshift(frequencyBandsTransformPassive(:, 4*i-1)), 'symmetric');
    plot(t, frequencyBandsSignalPassive(:, 4*i-1), 'r');
    plot(t_passive, probe_passive(:, i), 'b');
    xlim([0 t_passive(end)]);
    title('Passive IFT');
    hold off
    
    subplot(2,2,4);
    hold on
    frequencyBandsSignalActive(:, 4*i-1) = ifft(ifftshift(frequencyBandsTransformActive(:, 4*i-1)), 'symmetric');
    plot(t, frequencyBandsSignalActive(:, 4*i-1), 'r');
    plot(t_active, probe_active(:, i), 'b');
    xlim([0 t_active(end)]);
    title('Active IFT');
    hold off
    
    suptitle(sprintf('Channel : %d Beta', i));
    
end

%%
clc
close all

% Gamma band :
for i = 1:numberOfProbes
    % Plotting :
    figure
    
    subplot(2,2,1);
    hold on
    frequencyBandsTransformPassive(:, 4*i) = FTP(probe_passive_ft(:, i), probe_passive_Freq, 30, 50, 1);
    FFT(probe_passive(:, i), FS, 1); % Plotting whole signal fourier transform in passive mode
    title('Passive FT');
    hold off
    
    subplot(2,2,2);
    hold on
    frequencyBandsTransformActive(:, 4*i) = FTP(probe_active_ft(:, i), probe_active_Freq, 30, 50, 1);
    FFT(probe_active(:,i), FS, 1);  % Plotting whole signal fourier transform in active mode
    title('Active FT');
    hold off
    
    subplot(2,2,3);
    hold on
    frequencyBandsSignalPassive(:, 4*i) = ifft(ifftshift(frequencyBandsTransformPassive(:, 4*i)), 'symmetric');
    plot(t, frequencyBandsSignalPassive(:, 4*i), 'r');
    plot(t_passive, probe_passive(:, i), 'b');
    xlim([0 t_passive(end)]);
    title('Passive IFT');
    hold off
    
    subplot(2,2,4);
    hold on
    frequencyBandsSignalActive(:, 4*i) = ifft(ifftshift(frequencyBandsTransformActive(:, 4*i)), 'symmetric');
    plot(t, frequencyBandsSignalActive(:, 4*i), 'r');
    plot(t_active, probe_active(:, i), 'b');
    xlim([0 t_active(end)]);
    title('Active IFT');
    hold off
    
    suptitle(sprintf('Channel : %d Gamma', i));
    
end

%% Part 4
%  Section 1
clc
close all

% Calculating statistical information :

signalMean = zeros(62, 1); % 1-31 are passive & 32-62 are active
signalStd = zeros(62, 1);
signalVar = zeros(62, 1);
signalMax = zeros(62, 1);
signalMin = zeros(62, 1);
signalMedian = zeros(62, 1);
signalMode = zeros(62, 1);

for i = 1:31
    % Mean :
    signalMean(i) = mean(S1.data{1,1}.X(:, i));
    signalMean(i+31) = mean(S1.data{1,2}.X(:, i));
    
    % Std :
    signalStd(i) = std(S1.data{1,1}.X(:, i));
    signalStd(i+31) = std(S1.data{1,2}.X(:, i));
    
    % Var :
    signalVar(i) = var(S1.data{1,1}.X(:, i));
    signalVar(i+31) = var(S1.data{1,2}.X(:, i));
    
    % Max :
    signalMax(i) = max(S1.data{1,1}.X(:, i));
    signalMax(i+31) = max(S1.data{1,2}.X(:, i));
    
    % Min :
    signalMin(i) = min(S1.data{1,1}.X(:, i));
    signalMin(i+31) = min(S1.data{1,2}.X(:, i));
    
    % Median :
    signalMedian(i) = median(S1.data{1,1}.X(:, i));
    signalMedian(i+31) = median(S1.data{1,2}.X(:, i));
    
    % Mode :
    signalMode(i) = mode(S1.data{1,1}.X(:, i));
    signalMode(i+31) = mode(S1.data{1,2}.X(:, i));
end

% Histogram :
for i = 1:31
    figure
    subplot(1,2,1);
    histogram(S1.data{1,1}.X(:, i), 100); % Passive
    title(sprintf('Channel : %d Passive', i));
    subplot(1,2,2);
    histogram(S1.data{1,2}.X(:, i), 100); % Active
    title(sprintf('Channel : %d Active', i));
end




% Plotting the signals :

%Passive
figure
title('Passive mode');
for i = 1:numberOfProbes
    subplot(2, 5, i);
    plot(t_passive, probe_passive(:, i));
    xlim([0 t_passive(end)]);
    title(strcat('Channel', sprintf(' : %d', i)));
end
suptitle('Passive mode');

%Active
figure
for i = 1:numberOfProbes
    subplot(2, 5, i);
    plot(t_active, probe_active(:, i));
    xlim([0 t_active(end)]);
    title(strcat('Channel', sprintf(' : %d', i)));
end
suptitle('Active mode');

%% Section 2
% Cross Correlation :
clc
close all

passiveTruncateStart = 18000;
passiveTruncateEnd = 242000;
passiveTruncatedLength = passiveTruncateEnd-passiveTruncateStart+1;
activeTruncateStart = 12000;
activeTruncateEnd = 235000;
activeTruncatedLength = activeTruncateEnd-activeTruncateStart+1;
probe_passive_modified = zeros(passiveTruncatedLength, 31);
probe_active_modified = zeros(activeTruncatedLength, 31);

% Modifying signals :
for i = 1:31
    probe_passive_modified(:,i) = modifySignal(S1.data{1,1}.X(:,i),passiveTruncateStart,passiveTruncateEnd);
    probe_active_modified(:,i) = modifySignal(S1.data{1,2}.X(:,i),activeTruncateStart,activeTruncateEnd);
end

% Cross correlation :
crossCorrelationPassivePearson = corr(probe_passive_modified);
crossCorrelationActivePearson = corr(probe_active_modified);
% Cross correlation :
crossCorrelationPassiveSpearman = corr(probe_passive_modified,'type','spearman');
crossCorrelationActiveSpearman = corr(probe_active_modified,'type','spearman');


%% Section 3 
clc
close all

k = 4; % Number of clusters
numrep = 20;

%  Clustering pearson
idxPassivePearson = kmeans(crossCorrelationPassivePearson, k,'Distance','sqeuclidean','Replicates',numrep);
idxActivePearson = kmeans(crossCorrelationActivePearson, k,'Distance','sqeuclidean','Replicates',numrep);

%  Clustering Spearman
idxPassiveSpearman = kmeans(crossCorrelationPassiveSpearman, k,'Distance','sqeuclidean','Replicates',numrep);
idxActiveSpearman = kmeans(crossCorrelationActiveSpearman, k,'Distance','sqeuclidean','Replicates',numrep);

% Silhouette plot for Pearson :
figure
[silPassivePearson, ~] = silhouette(crossCorrelationPassivePearson,idxPassivePearson);
silPassivePearsonMean = mean(silPassivePearson);

figure;
[silActivePearson, ~] = silhouette(crossCorrelationActivePearson,idxActivePearson);
silActivePearsonMean = mean(silActivePearson);

% Silhouette plot for Spearman :
figure;
[silPassiveSpearman, ~] = silhouette(crossCorrelationPassiveSpearman,idxPassiveSpearman);
silPassiveSpearmanMean = mean(silPassiveSpearman);

figure;
[silActiveSpearman, ~] = silhouette(crossCorrelationActiveSpearman,idxActiveSpearman);
silActiveSpearmanMean = mean(silActiveSpearman);

%% Part 5
%  Section 1
clc
close all

lim1 = 230;
lim2 = 307;

% Each stimulus is separated and each of following contains 480(num of
% stimuli)*31(num of probes)*737(num of data in each separateed signal)
separatedResponsePassive = responseSeparator(probe_passive_modified, S1.data{1,1}.trial - passiveTruncateStart, lim1, lim2);
separatedResponseActive = responseSeparator(probe_active_modified, S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);


%%
%  Section 2
clc
close all

% Finding the most informative probe of each cluster (using spearman correlation) using variance :

type1StimulusPassive = find(S1.data{1,1}.y == 1);
type2StimulusPassive = find(S1.data{1,1}.y == 2);
type1StimulusActive = find(S1.data{1,2}.y == 1);
type2StimulusActive = find(S1.data{1,2}.y == 2);

interval = lim1 + lim2; % Time interval of response

meanResponseType1Passive = zeros(interval, k); % 'k' is the number of clusters
meanResponseType1Active = zeros(interval, k);
meanResponseType2Passive = zeros(interval, k);
meanResponseType2Active = zeros(interval, k);

indexInformativePassive = zeros(k, 1);
indexInformativeActive = zeros(k, 1);

probe_passive_modified_32 = [zeros(length(probe_passive_modified),1) probe_passive_modified];
probe_active_modified_32 = [zeros(length(probe_active_modified),1) probe_active_modified];


for i = 1:k % 'k' is the number of clusters
    indexPassiveCluster = (1:31)' .* (idxPassiveSpearman == i); % Passive
    indexActiveCluster = (1:31)' .* (idxActiveSpearman == i); % Active
    
    [~,indexInformativePassive(i, 1)] = max(var(probe_passive_modified_32(:, indexPassiveCluster+1))); % Efficient probe
    [~,indexInformativeActive(i, 1)] = max(var(probe_active_modified_32(:, indexActiveCluster+1))); % Efficient probe
    
    meanResponseType1Passive(:, i) = mean(separatedResponsePassive(type1StimulusPassive, indexInformativePassive(i, 1), :));
    meanResponseType2Passive(:, i) = mean(separatedResponsePassive(type2StimulusPassive, indexInformativePassive(i, 1), :));
    meanResponseType1Active(:, i) = mean(separatedResponseActive(type1StimulusActive, indexInformativeActive(i, 1), :));
    meanResponseType2Active(:, i) = mean(separatedResponseActive(type2StimulusActive, indexInformativeActive(i, 1), :));
end

% Horizontal vectors
confidenceIntervalResponseType1Passive = mean(std(separatedResponsePassive(type1StimulusPassive, indexInformativePassive(:,1), :), 0, 3))/sqrt(60);
confidenceIntervalResponseType2Passive = mean(std(separatedResponsePassive(type2StimulusPassive, indexInformativePassive(:,1), :), 0, 3))/sqrt(420);
confidenceIntervalResponseType1Active = mean(std(separatedResponseActive(type1StimulusActive, indexInformativeActive(:,1), :), 0, 3))/sqrt(60);
confidenceIntervalResponseType2Active = mean(std(separatedResponseActive(type2StimulusActive, indexInformativeActive(:,1), :), 0, 3))/sqrt(420);

% Time Vector :
t = 0:(1000/FS):1000*((lim1+lim2-1)/FS) ;

% Plotting Responses of Each Informated Probe :
for i = 1:k
    % Type 1 Response Passive
    figure;
    plot(t,meanResponseType1Passive(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive, Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive
    figure;
    plot(t,meanResponseType2Passive(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive, Cluster No. %d',i),'color','r');
    
    
    
    
    % Type 1 Response Active
    figure;
    plot(t,meanResponseType1Active(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active, Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active
    figure;
    plot(t,meanResponseType2Active(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active, Cluster No. %d',i),'color','r');
end

%% Section 3

clc
close all

nPassiveModified = pow2(nextpow2(length(probe_passive_modified(: , 1)))); % Fourier Transform length for passive modified signal
nActiveModified = pow2(nextpow2(length(probe_active_modified(: , 1)))); % Fourier Transform length for active modified signal

probe_passive_modified_ft = zeros(nPassiveModified, k); % Preallocating
probe_active_modified_ft = zeros(nActiveModified, k); % Preallocating

probe_passive_modified_Freq = ((-nPassiveModified/2:nPassiveModified/2-1)*(FS/nPassiveModified))'; % Frequency range for passive
probe_active_modified_Freq = ((-nActiveModified/2:nActiveModified/2-1)*(FS/nActiveModified))'; % Frequency range for active

% Preallocation of Delta
probe_passive_modified_delta_ft = zeros(length(probe_passive_modified_Freq),k);
probe_active_modified_delta_ft = zeros(length(probe_active_modified_Freq),k);

% Preallocation of Theta
probe_passive_modified_theta_ft = zeros(length(probe_passive_modified_Freq),k);
probe_active_modified_theta_ft = zeros(length(probe_active_modified_Freq),k);

% Preallocation of Alpha
probe_passive_modified_alpha_ft = zeros(length(probe_passive_modified_Freq),k);
probe_active_modified_alpha_ft = zeros(length(probe_active_modified_Freq),k);

% Preallocation of Beta
probe_passive_modified_beta_ft = zeros(length(probe_passive_modified_Freq),k);
probe_active_modified_beta_ft = zeros(length(probe_active_modified_Freq),k);

% Preallocation of Delta IFT :
delta_passive_ift = zeros(length(probe_passive_modified_ft),k);
delta_active_ift = zeros(length(probe_active_modified_ft),k);

% Preallocation of Theta IFT :
theta_passive_ift = zeros(length(probe_passive_modified_ft),k);
theta_active_ift = zeros(length(probe_active_modified_ft),k);

% Preallocation of Alpha IFT :
alpha_passive_ift = zeros(length(probe_passive_modified_ft),k);
alpha_active_ift = zeros(length(probe_active_modified_ft),k);

% Preallocation of Beta IFT :
beta_passive_ift = zeros(length(probe_passive_modified_ft),k);
beta_active_ift = zeros(length(probe_active_modified_ft),k);

% Preallocation of Delta Channel Separated Signals :
separatedResponsePassiveDelta = zeros(480, k, lim1+lim2); % 480 : num of stimuli, k : num of clusters, lim1+lim2 : length of each separated signal 
separatedResponseActiveDelta = zeros(480, k, lim1+lim2);

% Preallocation of Theta Channel Separated Signals :
separatedResponsePassiveTheta = zeros(480, k, lim1+lim2);
separatedResponseActiveTheta = zeros(480, k, lim1+lim2);

% Preallocation of Alpha Channel Separated Signals :
separatedResponsePassiveAlpha = zeros(480, k, lim1+lim2);
separatedResponseActiveAlpha = zeros(480, k, lim1+lim2);

% Preallocation of Beta Channel Separated Signals :
separatedResponsePassiveBeta = zeros(480, k, lim1+lim2);
separatedResponseActiveBeta = zeros(480, k, lim1+lim2);

% Fourier transform :
for i = indexInformativePassive(:, 1)'
    [probe_passive_modified_ft(:, find(indexInformativePassive(:, 1) == i)), ~] = FFT(probe_passive_modified(:, find(indexInformativePassive(:, 1) == i)), FS, 0);
    
    probe_passive_modified_delta_ft(:,find(indexInformativePassive(:, 1) == i)) = FTP(probe_passive_modified_ft(:,find(indexInformativePassive(:, 1) == i)), probe_passive_modified_Freq, 0, 4, 0);
    delta_passive_ift(:,find(indexInformativePassive(:, 1) == i)) = ifft(ifftshift(probe_passive_modified_delta_ft(:,find(indexInformativePassive(:, 1) == i))), 'symmetric');
    separatedResponsePassiveDelta(:,find(indexInformativePassive(:, 1) == i),:) = responseSeparator(delta_passive_ift(:,find(indexInformativePassive(:, 1) == i)), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_theta_ft(:,find(indexInformativePassive(:, 1) == i)) = FTP(probe_passive_modified_ft(:,find(indexInformativePassive(:, 1) == i)), probe_passive_modified_Freq, 4, 8, 0);
    theta_passive_ift(:,find(indexInformativePassive(:, 1) == i)) = ifft(ifftshift(probe_passive_modified_theta_ft(:,find(indexInformativePassive(:, 1) == i))), 'symmetric');
    separatedResponsePassiveTheta(:,find(indexInformativePassive(:, 1) == i),:) = responseSeparator(theta_passive_ift(:,find(indexInformativePassive(:, 1) == i)), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_alpha_ft(:,find(indexInformativePassive(:, 1) == i)) = FTP(probe_passive_modified_ft(:,find(indexInformativePassive(:, 1) == i)), probe_passive_modified_Freq, 8, 13, 0);
    alpha_passive_ift(:,find(indexInformativePassive(:, 1) == i)) = ifft(ifftshift(probe_passive_modified_alpha_ft(:,find(indexInformativePassive(:, 1) == i))), 'symmetric');
    separatedResponsePassiveAlpha(:,find(indexInformativePassive(:, 1) == i),:) = responseSeparator(alpha_passive_ift(:,find(indexInformativePassive(:, 1) == i)), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_beta_ft(:,find(indexInformativePassive(:, 1) == i)) = FTP(probe_passive_modified_ft(:,find(indexInformativePassive(:, 1) == i)), probe_passive_modified_Freq, 13, 30, 0);
    beta_passive_ift(:,find(indexInformativePassive(:, 1) == i)) = ifft(ifftshift(probe_passive_modified_beta_ft(:,find(indexInformativePassive(:, 1) == i))), 'symmetric');
    separatedResponsePassiveBeta(:,find(indexInformativePassive(:, 1) == i),:) = responseSeparator(beta_passive_ift(:,find(indexInformativePassive(:, 1) == i)), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
end


for i = indexInformativeActive(:, 1)'
    [probe_active_modified_ft(:, find(indexInformativeActive(:, 1) == i)), ~] = FFT(probe_active_modified(:, find(indexInformativeActive(:, 1) == i)), FS, 0);
    
    probe_active_modified_delta_ft(:,find(indexInformativeActive(:, 1) == i)) = FTP(probe_active_modified_ft(:,find(indexInformativeActive(:, 1) == i)), probe_active_modified_Freq, 0, 4, 0);
    delta_active_ift(:,find(indexInformativeActive(:, 1) == i)) = ifft(ifftshift(probe_active_modified_delta_ft(:,find(indexInformativeActive(:, 1) == i))), 'symmetric');
    separatedResponseActiveDelta(:,find(indexInformativeActive(:, 1) == i),:) = responseSeparator(delta_active_ift(:,find(indexInformativeActive(:, 1) == i)), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_theta_ft(:,find(indexInformativeActive(:, 1) == i)) = FTP(probe_active_modified_ft(:,find(indexInformativeActive(:, 1) == i)), probe_active_modified_Freq, 4, 8, 0);
    theta_active_ift(:,find(indexInformativeActive(:, 1) == i)) = ifft(ifftshift(probe_active_modified_theta_ft(:,find(indexInformativeActive(:, 1) == i))), 'symmetric');
    separatedResponseActiveTheta(:,find(indexInformativeActive(:, 1) == i),:) = responseSeparator(theta_active_ift(:,find(indexInformativeActive(:, 1) == i)), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_alpha_ft(:,find(indexInformativeActive(:, 1) == i)) = FTP(probe_active_modified_ft(:,find(indexInformativeActive(:, 1) == i)), probe_active_modified_Freq, 8, 13, 0);
    alpha_active_ift(:,find(indexInformativeActive(:, 1) == i)) = ifft(ifftshift(probe_active_modified_alpha_ft(:,find(indexInformativeActive(:, 1) == i))), 'symmetric');
    separatedResponseActiveAlpha(:,find(indexInformativeActive(:, 1) == i),:) = responseSeparator(alpha_active_ift(:,find(indexInformativeActive(:, 1) == i)), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_beta_ft(:,find(indexInformativeActive(:, 1) == i)) = FTP(probe_active_modified_ft(:,find(indexInformativeActive(:, 1) == i)), probe_active_modified_Freq, 13, 30, 0);
    beta_active_ift(:,find(indexInformativeActive(:, 1) == i)) = ifft(ifftshift(probe_active_modified_beta_ft(:,find(indexInformativeActive(:, 1) == i))), 'symmetric');
    separatedResponseActiveBeta(:,find(indexInformativeActive(:, 1) == i),:) = responseSeparator(beta_active_ift(:,find(indexInformativeActive(:, 1) == i)), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
end



% Preallocation of Each Channel, type1 & type2, passive :
meanResponseType1PassiveDelta = zeros(lim1+lim2,k);
meanResponseType1PassiveTheta = zeros(lim1+lim2,k);
meanResponseType1PassiveAlpha = zeros(lim1+lim2,k);
meanResponseType1PassiveBeta = zeros(lim1+lim2,k);
meanResponseType2PassiveDelta = zeros(lim1+lim2,k);
meanResponseType2PassiveTheta = zeros(lim1+lim2,k);
meanResponseType2PassiveAlpha = zeros(lim1+lim2,k);
meanResponseType2PassiveBeta = zeros(lim1+lim2,k);

% Preallocation of Each Channel, type1 & type2, active :
meanResponseType1ActiveDelta = zeros(lim1+lim2,k);
meanResponseType1ActiveTheta = zeros(lim1+lim2,k);
meanResponseType1ActiveAlpha = zeros(lim1+lim2,k);
meanResponseType1ActiveBeta = zeros(lim1+lim2,k);
meanResponseType2ActiveDelta = zeros(lim1+lim2,k);
meanResponseType2ActiveTheta = zeros(lim1+lim2,k);
meanResponseType2ActiveAlpha = zeros(lim1+lim2,k);
meanResponseType2ActiveBeta = zeros(lim1+lim2,k);
    
% Evaluating Response of Each Channel
for i = 1:k
    % Passive Mode :
    meanResponseType1PassiveDelta(:, i) = mean(separatedResponsePassiveDelta(type1StimulusPassive, i, :));
    meanResponseType1PassiveTheta(:, i) = mean(separatedResponsePassiveTheta(type1StimulusPassive, i, :));
    meanResponseType1PassiveAlpha(:, i) = mean(separatedResponsePassiveAlpha(type1StimulusPassive, i, :));
    meanResponseType1PassiveBeta(:, i) = mean(separatedResponsePassiveBeta(type1StimulusPassive, i, :));
    
    meanResponseType2PassiveDelta(:, i) = mean(separatedResponsePassiveDelta(type2StimulusPassive, i, :));
    meanResponseType2PassiveTheta(:, i) = mean(separatedResponsePassiveTheta(type2StimulusPassive, i, :));
    meanResponseType2PassiveAlpha(:, i) = mean(separatedResponsePassiveAlpha(type2StimulusPassive, i, :));
    meanResponseType2PassiveBeta(:, i) = mean(separatedResponsePassiveBeta(type2StimulusPassive, i, :));
    
    % Active Mode :
    meanResponseType1ActiveDelta(:, i) = mean(separatedResponseActiveDelta(type1StimulusActive, i, :));
    meanResponseType1ActiveTheta(:, i) = mean(separatedResponseActiveTheta(type1StimulusActive, i, :));
    meanResponseType1ActiveAlpha(:, i) = mean(separatedResponseActiveAlpha(type1StimulusActive, i, :));
    meanResponseType1ActiveBeta(:, i) = mean(separatedResponseActiveBeta(type1StimulusActive, i, :));
    
    meanResponseType2ActiveDelta(:, i) = mean(separatedResponseActiveDelta(type2StimulusActive, i, :));
    meanResponseType2ActiveTheta(:, i) = mean(separatedResponseActiveTheta(type2StimulusActive, i, :));
    meanResponseType2ActiveAlpha(:, i) = mean(separatedResponseActiveAlpha(type2StimulusActive, i, :));
    meanResponseType2ActiveBeta(:, i) = mean(separatedResponseActiveBeta(type2StimulusActive, i, :));
end 

% Preallocation of Each Channel Confidence Interval Passive & Active Mode :
confidenceIntervalResponseType1PassiveDelta = zeros(k,1);
confidenceIntervalResponseType1PassiveTheta = zeros(k,1);
confidenceIntervalResponseType1PassiveAlpha = zeros(k,1);
confidenceIntervalResponseType1PassiveBeta = zeros(k,1);

confidenceIntervalResponseType2PassiveDelta = zeros(k,1);
confidenceIntervalResponseType2PassiveTheta = zeros(k,1);
confidenceIntervalResponseType2PassiveAlpha = zeros(k,1);
confidenceIntervalResponseType2PassiveBeta = zeros(k,1);

confidenceIntervalResponseType1ActiveDelta = zeros(k,1);
confidenceIntervalResponseType1ActiveTheta = zeros(k,1);
confidenceIntervalResponseType1ActiveAlpha = zeros(k,1);
confidenceIntervalResponseType1ActiveBeta = zeros(k,1);

confidenceIntervalResponseType2ActiveDelta = zeros(k,1);
confidenceIntervalResponseType2ActiveTheta = zeros(k,1);
confidenceIntervalResponseType2ActiveAlpha = zeros(k,1);
confidenceIntervalResponseType2ActiveBeta = zeros(k,1);

for j = 1:k
    
    % Type 1 Confidence Intervals (Passive Mode)
    confidenceIntervalResponseType1PassiveDelta(j,1) = mean(std(separatedResponsePassiveDelta(type1StimulusPassive, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType1PassiveTheta(j,1) = mean(std(separatedResponsePassiveTheta(type1StimulusPassive, j, :), 0, 3))/sqrt(420);
    confidenceIntervalResponseType1PassiveAlpha(j,1) = mean(std(separatedResponsePassiveAlpha(type1StimulusPassive, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType1PassiveBeta(j,1) = mean(std(separatedResponsePassiveBeta(type1StimulusPassive, j, :), 0, 3))/sqrt(420);
    
    % Type 2 Confidence Intervals (Passive Mode)
    confidenceIntervalResponseType2PassiveDelta(j,1) = mean(std(separatedResponsePassiveDelta(type1StimulusPassive, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType2PassiveTheta(j,1) = mean(std(separatedResponsePassiveTheta(type1StimulusPassive, j, :), 0, 3))/sqrt(420);
    confidenceIntervalResponseType2PassiveAlpha(j,1) = mean(std(separatedResponsePassiveAlpha(type1StimulusPassive, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType2PassiveBeta(j,1) = mean(std(separatedResponsePassiveBeta(type1StimulusPassive, j, :), 0, 3))/sqrt(420);
    
    % Type 1 Confidence Intervals (Active Mode)
    confidenceIntervalResponseType1ActiveDelta(j,1) = mean(std(separatedResponseActiveDelta(type1StimulusActive, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType1ActiveTheta(j,1) = mean(std(separatedResponseActiveTheta(type1StimulusActive, j, :), 0, 3))/sqrt(420);
    confidenceIntervalResponseType1ActiveAlpha(j,1) = mean(std(separatedResponseActiveAlpha(type1StimulusActive, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType1ActiveBeta(j,1) = mean(std(separatedResponseActiveBeta(type1StimulusActive, j, :), 0, 3))/sqrt(420);
    
    % Type 2 Confidence Intervals (Active Mode)
    confidenceIntervalResponseType2ActiveDelta(j,1) = mean(std(separatedResponseActiveDelta(type2StimulusActive, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType2ActiveTheta(j,1) = mean(std(separatedResponseActiveTheta(type2StimulusActive, j, :), 0, 3))/sqrt(420);
    confidenceIntervalResponseType2ActiveAlpha(j,1) = mean(std(separatedResponseActiveAlpha(type2StimulusActive, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType2ActiveBeta(j,1) = mean(std(separatedResponseActiveBeta(type2StimulusActive, j, :), 0, 3))/sqrt(420);
end


% Plotting Responses of Each Channel
for i = 1:k 
    % Type 1 Response Passive Delta
    figure;
    plot(t,meanResponseType1PassiveDelta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive Delta,Cluster No. %d',i),'color','r');
    
    % Type 1 Response Active Delta
    figure;
    plot(t,meanResponseType1ActiveDelta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active Delta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive Delta
    figure;
    plot(t,meanResponseType2PassiveDelta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive Delta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active Delta
    figure;
    plot(t,meanResponseType2ActiveDelta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active Delta,Cluster No. %d',i),'color','r');
    
    
    
    
    % Type 1 Response Passive Theta
    figure;
    plot(t,meanResponseType1PassiveTheta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive Theta,Cluster No. %d',i),'color','r');
    
    % Type 1 Response Active Theta
    figure;
    plot(t,meanResponseType1ActiveTheta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active Theta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive Theta
    figure;
    plot(t,meanResponseType2PassiveTheta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive Theta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active Theta
    figure;
    plot(t,meanResponseType2ActiveTheta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active Theta,Cluster No. %d',i),'color','r');
    
    
    
    
    % Type 1 Response Passive Alpha
    figure;
    plot(t,meanResponseType1PassiveAlpha(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive Alpha,Cluster No. %d',i),'color','r');
    
    % Type 1 Response Active Alpha
    figure;
    plot(t,meanResponseType1ActiveAlpha(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active Alpha,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive Alpha
    figure;
    plot(t,meanResponseType2PassiveAlpha(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive Alpha,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active Alpha
    figure;
    plot(t,meanResponseType2ActiveAlpha(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active Alpha,Cluster No. %d',i),'color','r');
    
    
    
    
    % Type 1 Response Passive Beta
    figure;
    plot(t,meanResponseType1PassiveBeta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive Beta,Cluster No. %d',i),'color','r');
    
    % Type 1 Response Active Beta
    figure;
    plot(t,meanResponseType1ActiveBeta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active Beta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive Beta
    figure;
    plot(t,meanResponseType2PassiveBeta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive Beta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active Beta
    figure;
    plot(t,meanResponseType2ActiveBeta(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active Beta,Cluster No. %d',i),'color','r');
end

%% Section 4
% included in report
%% Section 5
clc
close all

% Preallocation for mean of responses to type1 & type2, active and passive
% experiments :
% Each of the following is a 537(lenght of interval around stimuli) *
% 31(num of probes) matrix
meanResponseType1PassiveAllProbes = zeros(lim1+lim2, 31);
meanResponseType2PassiveAllProbes = zeros(lim1+lim2, 31);
meanResponseType1ActiveAllProbes = zeros(lim1+lim2, 31);
meanResponseType2ActiveAllProbes = zeros(lim1+lim2, 31);

for i = 1:31
    meanResponseType1PassiveAllProbes(:, i) = mean(separatedResponsePassive(type1StimulusPassive, i, :));
    meanResponseType2PassiveAllProbes(:, i) = mean(separatedResponsePassive(type2StimulusPassive, i, :));
    meanResponseType1ActiveAllProbes(:, i) = mean(separatedResponseActive(type1StimulusActive, i, :));
    meanResponseType2ActiveAllProbes(:, i) = mean(separatedResponseActive(type2StimulusActive, i, :));
end

% Calculating the Pearson correlation between type1 & type2 stimuli for each probe,
% passive & active :
corrCoeffType1Type2PassivePearson = diag(corr(meanResponseType1PassiveAllProbes, meanResponseType2PassiveAllProbes));
corrCoeffType1Type2ActivePearson = diag(corr(meanResponseType1ActiveAllProbes, meanResponseType2ActiveAllProbes));

% Calculating the Spearman correlation between type1 & type2 stimuli for each probe,
% passive & active :
corrCoeffType1Type2PassiveSpearman = diag(corr(meanResponseType1PassiveAllProbes, meanResponseType2PassiveAllProbes, 'type', 'spearman'));
corrCoeffType1Type2ActiveSpearman = diag(corr(meanResponseType1ActiveAllProbes, meanResponseType2ActiveAllProbes, 'type', 'spearman'));

% Plotting response to type1 & type2 stimuli for abs(corrCoeff)>0.4  and Pearson correlation :
indexOfPlotPassive = (abs(corrCoeffType1Type2PassivePearson) > 0.4) .* (1:31)';
indexOfPlotActive = (abs(corrCoeffType1Type2ActivePearson) > 0.4) .* (1:31)';

for i = indexOfPlotPassive(indexOfPlotPassive ~= 0)'
    figure
    plot(meanResponseType2PassiveAllProbes(:, i), meanResponseType1PassiveAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode', i),'color','r');
end

for i = indexOfPlotActive(indexOfPlotActive ~= 0)'
    figure
    plot(meanResponseType2ActiveAllProbes(:, i), meanResponseType1ActiveAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode', i),'color','r');
end

%% Section 6
clc
close all

% Preallocation of fourier transform of all probes :
probe_passive_modified_ft_all_probes = zeros(nPassiveModified, 31);
probe_active_modified_ft_all_probes = zeros(nActiveModified, 31);

% Preallocation of FT of channels (delta, ...) for all probes, passive mode :
probe_passive_modified_delta_ft_all_probes = zeros(length(probe_passive_modified_Freq), 31);
probe_passive_modified_theta_ft_all_probes = zeros(length(probe_passive_modified_Freq), 31);
probe_passive_modified_alpha_ft_all_probes = zeros(length(probe_passive_modified_Freq), 31);
probe_passive_modified_beta_ft_all_probes = zeros(length(probe_passive_modified_Freq), 31);

% Preallocation of IFT of channels for all probes, passive mode :
delta_passive_ift_all_probes = zeros(length(probe_passive_modified_Freq), 31);
theta_passive_ift_all_probes = zeros(length(probe_passive_modified_Freq), 31);
alpha_passive_ift_all_probes = zeros(length(probe_passive_modified_Freq), 31);
beta_passive_ift_all_probes = zeros(length(probe_passive_modified_Freq), 31);

% Preallocation of mean of responses to type1 & type2 stimuli for all
% probes, passive mode :
separatedResponsePassiveDeltaAllProbes = zeros(480, 31, lim1+lim2); % 480 * 31 * 537
separatedResponsePassiveThetaAllProbes = zeros(480, 31, lim1+lim2);
separatedResponsePassiveAlphaAllProbes = zeros(480, 31, lim1+lim2);
separatedResponsePassiveBetaAllProbes = zeros(480, 31, lim1+lim2);

% Fourier transform of all probes in passive mode + 
% Separating channels(delta, ...) from transforms + 
% IFT from each channel +
% Separating type1 & type2 stimuli for each channel 
% for passive mode :
for i = 1:31
    [probe_passive_modified_ft_all_probes(:, i), ~] = FFT(probe_passive_modified(:, i), FS, 0);
    
    probe_passive_modified_delta_ft_all_probes(:,i) = FTP(probe_passive_modified_ft_all_probes(:, i), probe_passive_modified_Freq, 0, 4, 0);
    delta_passive_ift_all_probes(:,i) = ifft(ifftshift(probe_passive_modified_delta_ft_all_probes(:,i)), 'symmetric');
    separatedResponsePassiveDeltaAllProbes(:,i,:) = responseSeparator(delta_passive_ift_all_probes(:,i), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_theta_ft_all_probes(:,i) = FTP(probe_passive_modified_ft_all_probes(:, i), probe_passive_modified_Freq, 4, 8, 0);
    theta_passive_ift_all_probes(:,i) = ifft(ifftshift(probe_passive_modified_theta_ft_all_probes(:,i)), 'symmetric');
    separatedResponsePassiveThetaAllProbes(:,i,:) = responseSeparator(theta_passive_ift_all_probes(:,i), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_alpha_ft_all_probes(:,i) = FTP(probe_passive_modified_ft_all_probes(:, i), probe_passive_modified_Freq, 8, 13, 0);
    alpha_passive_ift_all_probes(:,i) = ifft(ifftshift(probe_passive_modified_alpha_ft_all_probes(:,i)), 'symmetric');
    separatedResponsePassiveAlphaAllProbes(:,i,:) = responseSeparator(alpha_passive_ift_all_probes(:,i), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_beta_ft_all_probes(:,i) = FTP(probe_passive_modified_ft_all_probes(:, i), probe_passive_modified_Freq, 13, 30, 0);
    beta_passive_ift_all_probes(:,i) = ifft(ifftshift(probe_passive_modified_beta_ft_all_probes(:,i)), 'symmetric');
    separatedResponsePassiveBetaAllProbes(:,i,:) = responseSeparator(beta_passive_ift_all_probes(:,i), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
end

% Preallocation of FT of channels (delta, ...) for all probes, active mode :
probe_active_modified_delta_ft_all_probes = zeros(length(probe_active_modified_Freq), 31);
probe_active_modified_theta_ft_all_probes = zeros(length(probe_active_modified_Freq), 31);
probe_active_modified_alpha_ft_all_probes = zeros(length(probe_active_modified_Freq), 31);
probe_active_modified_beta_ft_all_probes = zeros(length(probe_active_modified_Freq), 31);

% Preallocation of IFT of channels for all probes, active mode :
delta_active_ift_all_probes = zeros(length(probe_active_modified_Freq), 31);
theta_active_ift_all_probes = zeros(length(probe_active_modified_Freq), 31);
alpha_active_ift_all_probes = zeros(length(probe_active_modified_Freq), 31);
beta_active_ift_all_probes = zeros(length(probe_active_modified_Freq), 31);

% Preallocation of mean of responses to type1 & type2 stimuli for all
% probes, active mode :
separatedResponseActiveDeltaAllProbes = zeros(480, 31, lim1+lim2); % 480 * 31 * 537
separatedResponseActiveThetaAllProbes = zeros(480, 31, lim1+lim2);
separatedResponseActiveAlphaAllProbes = zeros(480, 31, lim1+lim2);
separatedResponseActiveBetaAllProbes = zeros(480, 31, lim1+lim2);

% Fourier transform of all probes in passive mode + 
% Separating channels(delta, ...) from transforms + 
% IFT from each channel +
% Separating type1 & type2 stimuli for each channel 
% for active mode :
for i = 1:31
    [probe_active_modified_ft_all_probes(:, i), ~] = FFT(probe_active_modified(:, i), FS, 0);
    
    probe_active_modified_delta_ft_all_probes(:,i) = FTP(probe_active_modified_ft_all_probes(:, i), probe_active_modified_Freq, 0, 4, 0);
    delta_active_ift_all_probes(:,i) = ifft(ifftshift(probe_active_modified_delta_ft_all_probes(:,i)), 'symmetric');
    separatedResponseActiveDeltaAllProbes(:,i,:) = responseSeparator(delta_active_ift_all_probes(:,i), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_theta_ft_all_probes(:,i) = FTP(probe_active_modified_ft_all_probes(:, i), probe_active_modified_Freq, 4, 8, 0);
    theta_active_ift_all_probes(:,i) = ifft(ifftshift(probe_active_modified_theta_ft_all_probes(:,i)), 'symmetric');
    separatedResponseActiveThetaAllProbes(:,i,:) = responseSeparator(theta_active_ift_all_probes(:,i), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_alpha_ft_all_probes(:,i) = FTP(probe_active_modified_ft_all_probes(:, i), probe_active_modified_Freq, 8, 13, 0);
    alpha_active_ift_all_probes(:,i) = ifft(ifftshift(probe_active_modified_alpha_ft_all_probes(:,i)), 'symmetric');
    separatedResponseActiveAlphaAllProbes(:,i,:) = responseSeparator(alpha_active_ift_all_probes(:,i), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_beta_ft_all_probes(:,i) = FTP(probe_active_modified_ft_all_probes(:, i), probe_active_modified_Freq, 13, 30, 0);
    beta_active_ift_all_probes(:,i) = ifft(ifftshift(probe_active_modified_beta_ft_all_probes(:,i)), 'symmetric');
    separatedResponseActiveBetaAllProbes(:,i,:) = responseSeparator(beta_active_ift_all_probes(:,i), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
end

% Calculating the mean of separated signals for each channel of each probe
% over 420 stimuli (type2) and also 60 stimuli (type1) :

% Preallocation for the mean of separated signals for each channel of each
% probe :
% Delta :
meanResponseType1PassiveDeltaAllProbes = zeros(lim1+lim2, 31);
meanResponseType2PassiveDeltaAllProbes = zeros(lim1+lim2, 31);
meanResponseType1ActiveDeltaAllProbes = zeros(lim1+lim2, 31);
meanResponseType2ActiveDeltaAllProbes = zeros(lim1+lim2, 31);
% Theta :
meanResponseType1PassiveThetaAllProbes = zeros(lim1+lim2, 31);
meanResponseType2PassiveThetaAllProbes = zeros(lim1+lim2, 31);
meanResponseType1ActiveThetaAllProbes = zeros(lim1+lim2, 31);
meanResponseType2ActiveThetaAllProbes = zeros(lim1+lim2, 31);
% Alpha :
meanResponseType1PassiveAlphaAllProbes = zeros(lim1+lim2, 31);
meanResponseType2PassiveAlphaAllProbes = zeros(lim1+lim2, 31);
meanResponseType1ActiveAlphaAllProbes = zeros(lim1+lim2, 31);
meanResponseType2ActiveAlphaAllProbes = zeros(lim1+lim2, 31);
% Beta :
meanResponseType1PassiveBetaAllProbes = zeros(lim1+lim2, 31);
meanResponseType2PassiveBetaAllProbes = zeros(lim1+lim2, 31);
meanResponseType1ActiveBetaAllProbes = zeros(lim1+lim2, 31);
meanResponseType2ActiveBetaAllProbes = zeros(lim1+lim2, 31);

for i = 1:31
    % Delta :
    meanResponseType1PassiveDeltaAllProbes(:, i) = mean(separatedResponsePassiveDeltaAllProbes(type1StimulusPassive,i, :)); % 31 passive, delta, type1
    meanResponseType2PassiveDeltaAllProbes(:, i) = mean(separatedResponsePassiveDeltaAllProbes(type2StimulusPassive,i, :)); % 31 passive, delta, type2
    meanResponseType1ActiveDeltaAllProbes(:, i) = mean(separatedResponseActiveDeltaAllProbes(type1StimulusActive,i, :)); % 31 active, delta, type1
    meanResponseType2ActiveDeltaAllProbes(:, i) = mean(separatedResponseActiveDeltaAllProbes(type2StimulusActive,i, :)); % 31 active, delta, type2
    
    % Theta :
    meanResponseType1PassiveThetaAllProbes(:, i) = mean(separatedResponsePassiveThetaAllProbes(type1StimulusPassive,i, :)); % 31 passive, theta, type1
    meanResponseType2PassiveThetaAllProbes(:, i) = mean(separatedResponsePassiveThetaAllProbes(type2StimulusPassive,i, :)); % 31 passive, theta, type2
    meanResponseType1ActiveThetaAllProbes(:, i) = mean(separatedResponseActiveThetaAllProbes(type1StimulusActive,i, :)); % 31 active, theta, type1
    meanResponseType2ActiveThetaAllProbes(:, i) = mean(separatedResponseActiveThetaAllProbes(type2StimulusActive,i, :)); % 31 active, theta, type2
    
    % Alpha :
    meanResponseType1PassiveAlphaAllProbes(:, i) = mean(separatedResponsePassiveAlphaAllProbes(type1StimulusPassive,i, :)); % 31 passive, alpha, type1
    meanResponseType2PassiveAlphaAllProbes(:, i) = mean(separatedResponsePassiveAlphaAllProbes(type2StimulusPassive,i, :)); % 31 passive, alpha, type2
    meanResponseType1ActiveAlphaAllProbes(:, i) = mean(separatedResponseActiveAlphaAllProbes(type1StimulusActive,i, :)); % 31 active, alpha, type1
    meanResponseType2ActiveAlphaAllProbes(:, i) = mean(separatedResponseActiveAlphaAllProbes(type2StimulusActive,i, :)); % 31 active, alpha, type2
    
    % Beta :
    meanResponseType1PassiveBetaAllProbes(:, i) = mean(separatedResponsePassiveBetaAllProbes(type1StimulusPassive,i, :)); % 31 passive, beta, type1
    meanResponseType2PassiveBetaAllProbes(:, i) = mean(separatedResponsePassiveBetaAllProbes(type2StimulusPassive,i, :)); % 31 passive, beta, type2
    meanResponseType1ActiveBetaAllProbes(:, i) = mean(separatedResponseActiveBetaAllProbes(type1StimulusActive,i, :)); % 31 active, beta, type1
    meanResponseType2ActiveBetaAllProbes(:, i) = mean(separatedResponseActiveBetaAllProbes(type2StimulusActive,i, :)); % 31 active, beta, type2
end

% Calculating the Pearson correlation between type1 & type2 stimuli for
% each probe, each channel, passive & active :
% Passive :
corrCoeffType1Type2PassiveDeltaPearson = diag(corr(meanResponseType1PassiveDeltaAllProbes, meanResponseType2PassiveDeltaAllProbes));
corrCoeffType1Type2PassiveThetaPearson = diag(corr(meanResponseType1PassiveThetaAllProbes, meanResponseType2PassiveThetaAllProbes));
corrCoeffType1Type2PassiveAlphaPearson = diag(corr(meanResponseType1PassiveAlphaAllProbes, meanResponseType2PassiveAlphaAllProbes));
corrCoeffType1Type2PassiveBetaPearson = diag(corr(meanResponseType1PassiveBetaAllProbes, meanResponseType2PassiveBetaAllProbes));
% Active :
corrCoeffType1Type2ActiveDeltaPearson = diag(corr(meanResponseType1ActiveDeltaAllProbes, meanResponseType2ActiveDeltaAllProbes));
corrCoeffType1Type2ActiveThetaPearson = diag(corr(meanResponseType1ActiveThetaAllProbes, meanResponseType2ActiveThetaAllProbes));
corrCoeffType1Type2ActiveAlphaPearson = diag(corr(meanResponseType1ActiveAlphaAllProbes, meanResponseType2ActiveAlphaAllProbes));
corrCoeffType1Type2ActiveBetaPearson = diag(corr(meanResponseType1ActiveBetaAllProbes, meanResponseType2ActiveBetaAllProbes));

% Plotting response to type1 & type2 stimuli for each channel for abs(corrCoeff)>0.4  and Pearson correlation :
% Passive :
indexOfPlotPassiveDelta = (abs(corrCoeffType1Type2PassiveDeltaPearson) > 0.4) .* (1:31)';
indexOfPlotPassiveTheta = (abs(corrCoeffType1Type2PassiveThetaPearson) > 0.4) .* (1:31)';
indexOfPlotPassiveAlpha = (abs(corrCoeffType1Type2PassiveAlphaPearson) > 0.4) .* (1:31)';
indexOfPlotPassiveBeta = (abs(corrCoeffType1Type2PassiveBetaPearson) > 0.4) .* (1:31)';
% Active :
indexOfPlotActiveDelta = (abs(corrCoeffType1Type2ActiveDeltaPearson) > 0.4) .* (1:31)';
indexOfPlotActiveTheta = (abs(corrCoeffType1Type2ActiveThetaPearson) > 0.4) .* (1:31)';
indexOfPlotActiveAlpha = (abs(corrCoeffType1Type2ActiveAlphaPearson) > 0.4) .* (1:31)';
indexOfPlotActiveBeta = (abs(corrCoeffType1Type2ActiveBetaPearson) > 0.4) .* (1:31)';

% Passive :
% Delta :
for i = indexOfPlotPassiveDelta(indexOfPlotPassiveDelta ~= 0)'
    figure
    plot(meanResponseType2PassiveDeltaAllProbes(:, i), meanResponseType1PassiveDeltaAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode, Delta Channel', i),'color','r');
end

% Theta :
for i = indexOfPlotPassiveTheta(indexOfPlotPassiveTheta ~= 0)'
    figure
    plot(meanResponseType2PassiveThetaAllProbes(:, i), meanResponseType1PassiveThetaAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode, Theta Channel', i),'color','r');
end

% Alpha :
for i = indexOfPlotPassiveAlpha(indexOfPlotPassiveAlpha ~= 0)'
    figure
    plot(meanResponseType2PassiveAlphaAllProbes(:, i), meanResponseType1PassiveAlphaAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode, Alpha Channel', i),'color','r');
end

% Beta :
for i = indexOfPlotPassiveBeta(indexOfPlotPassiveBeta ~= 0)'
    figure
    plot(meanResponseType2PassiveBetaAllProbes(:, i), meanResponseType1PassiveBetaAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode, Beta Channel', i),'color','r');
end

% Active :
% Delta :
for i = indexOfPlotActiveDelta(indexOfPlotActiveDelta ~= 0)'
    figure
    plot(meanResponseType2ActiveDeltaAllProbes(:, i), meanResponseType1ActiveDeltaAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode, Delta Channel', i),'color','r');
end

% Tetha :
for i = indexOfPlotActiveTheta(indexOfPlotActiveTheta ~= 0)'
    figure
    plot(meanResponseType2ActiveThetaAllProbes(:, i), meanResponseType1ActiveThetaAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode, Theta Channel', i),'color','r');
end

% Alpha :
for i = indexOfPlotActiveAlpha(indexOfPlotActiveAlpha ~= 0)'
    figure
    plot(meanResponseType2ActiveAlphaAllProbes(:, i), meanResponseType1ActiveAlphaAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode, Alpha Channel', i),'color','r');
end

% Beta :
for i = indexOfPlotActiveBeta(indexOfPlotActiveBeta ~= 0)'
    figure
    plot(meanResponseType2ActiveBetaAllProbes(:, i), meanResponseType1ActiveBetaAllProbes(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode, Beta Channel', i),'color','r');
end

%% Section 7
clc
close all

% Finding the most informative cluster by meaning on the correlation
% coefficients :
meanOfCorrCoeffsOfClusters = zeros(k, 1);
for i = 1:k
   meanOfCorrCoeffsOfClusters(i, 1) = mean(abs(corrCoeffType1Type2PassivePearson(find(idxPassivePearson == i))));
end
[~, indexOfInformativeClusterPassive] = max (meanOfCorrCoeffsOfClusters);

for i = 1:k
   meanOfCorrCoeffsOfClusters(i, 1) = mean(abs(corrCoeffType1Type2ActivePearson(find(idxActivePearson == i))));
end
[~, indexOfInformativeClusterActive] = max (meanOfCorrCoeffsOfClusters);



% Calculation of maximum of cross-correlations in each cluster, passive mode :
indexOfInformativeProbesPassive = zeros(k, 1);
for i = 1:k
   [~, temp1] = max(abs(corrCoeffType1Type2PassivePearson(find(idxPassivePearson == i))));
   temp2 = find(idxPassivePearson == i);
   indexOfInformativeProbesPassive(i, 1) = temp2(temp1);
end

% Calculation of maximum of cross-correlations in each cluster, active mode :
indexOfInformativeProbesActive = zeros(k, 1);
for i = 1:k
   [~, temp1] = max(abs(corrCoeffType1Type2ActivePearson(find(idxActivePearson == i))));
   temp2 = find(idxActivePearson == i);
   indexOfInformativeProbesActive(i, 1) = temp2(temp1);
end


%% Section 8
clc
close all

% Averaging over type1 & type2 stimuli and calculating their difference:
meanResponseType1PassiveInformative = zeros(lim1+lim2, k);
meanResponseType2PassiveInformative = zeros(lim1+lim2, k);
meanResponseType1ActiveInformative = zeros(lim1+lim2, k);
meanResponseType2ActiveInformative = zeros(lim1+lim2, k);
meanResponseType1Type2PassiveInformativeDiff = zeros(lim1+lim2, k);
meanResponseType1Type2ActiveInformativeDiff = zeros(lim1+lim2, k);

for i = 1:k
    meanResponseType1PassiveInformative(:, i) = mean(separatedResponsePassive(type1StimulusPassive, indexOfInformativeProbesPassive(i, 1), :));
    meanResponseType2PassiveInformative(:, i) = mean(separatedResponsePassive(type2StimulusPassive, indexOfInformativeProbesPassive(i, 1), :));
    meanResponseType1ActiveInformative(:, i) = mean(separatedResponseActive(type1StimulusActive, indexOfInformativeProbesActive(i, 1), :));
    meanResponseType2ActiveInformative(:, i) = mean(separatedResponseActive(type2StimulusActive, indexOfInformativeProbesActive(i, 1), :));
    meanResponseType1Type2PassiveInformativeDiff(:, i) = meanResponseType1PassiveInformative(:, i) - meanResponseType2PassiveInformative(:, i);
    meanResponseType1Type2ActiveInformativeDiff(:, i) = meanResponseType1ActiveInformative(:, i) - meanResponseType2ActiveInformative(:, i);
end


% Plotting :
for i = 1:k
   figure
   plot(t, meanResponseType1PassiveInformative(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Passive Mode, Type1', indexOfInformativeProbesPassive(i)),'color','r');
   
   figure
   plot(t, meanResponseType2PassiveInformative(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Passive Mode, Type2', indexOfInformativeProbesPassive(i)),'color','r');
   
   figure
   plot(t, meanResponseType1ActiveInformative(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Active Mode, Type1', indexOfInformativeProbesActive(i)),'color','r');
    
   figure
   plot(t, meanResponseType2ActiveInformative(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Active Mode, Type2', indexOfInformativeProbesActive(i)),'color','r');
   
   figure
   plot(t, meanResponseType1Type2PassiveInformativeDiff(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Passive Mode, Type1-Type2', indexOfInformativeProbesPassive(i)),'color','r');
   
   figure
   plot(t, meanResponseType1Type2ActiveInformativeDiff(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Active Mode, Type1-Type2', indexOfInformativeProbesActive(i)),'color','r');
   
end
%% section 9
% included in report
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 6
%  Section 1

clc
close all

probe_passive_modified_woman = zeros(length(S2.data{1,1}.X(:, 1)) - passiveTruncateStart + 1, 31);
probe_active_modified_woman = zeros(length(S2.data{1,2}.X(:, 1)) - activeTruncateStart + 1, 31);

% Modifying signals :
for i = 1:31
    probe_passive_modified_woman(:,i) = modifySignal(S2.data{1,1}.X(:,i),passiveTruncateStart,length(S2.data{1,1}.X(:,1)));
    probe_active_modified_woman(:,i) = modifySignal(S2.data{1,2}.X(:,i),activeTruncateStart,length(S2.data{1,2}.X(:,1)));
end

clc
close all

% Each stimulus is separated and each of the following contains 480(num of
% stimuli)*31(num of probes)*537(num of data in each separateed signal)
separatedResponsePassiveWoman = responseSeparator(probe_passive_modified_woman, S2.data{1,1}.trial - passiveTruncateStart, lim1, lim2);
separatedResponseActiveWoman = responseSeparator(probe_active_modified_woman, S2.data{1,2}.trial - activeTruncateStart, lim1, lim2);

%%
clc
close all

% Cross Correlation  Pearson :
crossCorrelationPassivePearsonWoman = corr(probe_passive_modified_woman);
crossCorrelationActivePearsonWoman = corr(probe_active_modified_woman);

%% 
clc
close all
k = 3;
% Clustering pearson
idxPassivePearsonWoman = kmeans(crossCorrelationPassivePearsonWoman, k,'Distance','sqeuclidean','Replicates',numrep);
idxActivePearsonWoman = kmeans(crossCorrelationActivePearsonWoman, k,'Distance','sqeuclidean','Replicates',numrep);

% Silhouette plot for pearson :
figure
[silPassivePearsonWoman, ~] = silhouette(crossCorrelationPassivePearsonWoman,idxPassivePearsonWoman);
silPassivePearsonMeanWoman = mean(silPassivePearsonWoman);

figure;
[silActivePearsonWoman, ~] = silhouette(crossCorrelationActivePearsonWoman,idxActivePearsonWoman);
silActivePearsonMeanWoman = mean(silActivePearsonWoman);

%% section 2
clc
close all

% Finding the most informative probe of each cluster (using pearson correlation) using variance :

type1StimulusPassiveWoman = find(S2.data{1,1}.y == 1);
type2StimulusPassiveWoman = find(S2.data{1,1}.y == 2);
type1StimulusActiveWoman = find(S2.data{1,2}.y == 1);
type2StimulusActiveWoman = find(S2.data{1,2}.y == 2);

intervalWoman = lim1 + lim2; % Time interval of response

meanResponseType1PassiveWoman = zeros(intervalWoman, k); % 'k' is the number of clusters
meanResponseType1ActiveWoman = zeros(intervalWoman, k);
meanResponseType2PassiveWoman = zeros(intervalWoman, k);
meanResponseType2ActiveWoman = zeros(intervalWoman, k);

indexInformativePassiveWoman = zeros(k, 1);
indexInformativeActiveWoman = zeros(k, 1);

probe_passive_modified_32_woman = [zeros(length(probe_passive_modified_woman),1) probe_passive_modified_woman];
probe_active_modified_32_woman = [zeros(length(probe_active_modified_woman),1) probe_active_modified_woman];


for i = 1:k % 'k' is the number of clusters
    indexPassiveClusterWoman = (1:31)' .* (idxPassivePearsonWoman == i); % Passive
    indexActiveClusterWoman = (1:31)' .* (idxActivePearsonWoman == i); % Active
    
    [~,indexInformativePassiveWoman(i, 1)] = max(var(probe_passive_modified_32_woman(:, indexPassiveClusterWoman+1))); % Efficient probe
    [~,indexInformativeActiveWoman(i, 1)] = max(var(probe_active_modified_32_woman(:, indexActiveClusterWoman+1))); % Efficient probe
    
    meanResponseType1PassiveWoman(:, i) = mean(separatedResponsePassiveWoman(type1StimulusPassiveWoman, indexInformativePassiveWoman(i, 1), :));
    meanResponseType2PassiveWoman(:, i) = mean(separatedResponsePassiveWoman(type2StimulusPassiveWoman, indexInformativePassiveWoman(i, 1), :));
    meanResponseType1ActiveWoman(:, i) = mean(separatedResponseActiveWoman(type1StimulusActiveWoman, indexInformativeActiveWoman(i, 1), :));
    meanResponseType2ActiveWoman(:, i) = mean(separatedResponseActiveWoman(type2StimulusActiveWoman, indexInformativeActiveWoman(i, 1), :));
end

% Horizontal vectors
confidenceIntervalResponseType1PassiveWoman = mean(std(separatedResponsePassiveWoman(type1StimulusPassiveWoman, indexInformativePassiveWoman(:,1), :), 0, 3))/sqrt(60);
confidenceIntervalResponseType2PassiveWoman = mean(std(separatedResponsePassiveWoman(type2StimulusPassiveWoman, indexInformativePassiveWoman(:,1), :), 0, 3))/sqrt(420);
confidenceIntervalResponseType1ActiveWoman = mean(std(separatedResponseActiveWoman(type1StimulusActiveWoman, indexInformativeActiveWoman(:,1), :), 0, 3))/sqrt(60);
confidenceIntervalResponseType2ActiveWoman = mean(std(separatedResponseActiveWoman(type2StimulusActiveWoman, indexInformativeActiveWoman(:,1), :), 0, 3))/sqrt(420);

% Plotting Responses of Each Informated Probe :
close all
for i = 1:k
    % Type 1 Response Passive
    figure;
    plot(t,meanResponseType1PassiveWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive, Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive
    figure;
    plot(t,meanResponseType2PassiveWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive, Cluster No. %d',i),'color','r');
    
    
    
    
    % Type 1 Response Active
    figure;
    plot(t,meanResponseType1ActiveWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active, Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active
    figure;
    plot(t,meanResponseType2ActiveWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active, Cluster No. %d',i),'color','r');
end

%% section 3
clc
close all

nPassiveModifiedWoman = pow2(nextpow2(length(probe_passive_modified_woman(: , 1)))); % Fourier Transform length for passive modified signal
nActiveModifiedWoman = pow2(nextpow2(length(probe_active_modified_woman(: , 1)))); % Fourier Transform length for active modified signal

probe_passive_modified_ft_woman = zeros(nPassiveModifiedWoman, k); % Preallocating
probe_active_modified_ft_woman = zeros(nActiveModifiedWoman, k); % Preallocating

probe_passive_modified_Freq_woman = ((-nPassiveModifiedWoman/2:nPassiveModifiedWoman/2-1)*(FS/nPassiveModifiedWoman))'; % Frequency range for passive
probe_active_modified_Freq_woman = ((-nActiveModifiedWoman/2:nActiveModifiedWoman/2-1)*(FS/nActiveModifiedWoman))'; % Frequency range for active

% Preallocation of Delta
probe_passive_modified_delta_ft_woman = zeros(length(probe_passive_modified_Freq_woman),k);
probe_active_modified_delta_ft_woman = zeros(length(probe_active_modified_Freq_woman),k);

% Preallocation of Theta
probe_passive_modified_theta_ft_woman = zeros(length(probe_passive_modified_Freq_woman),k);
probe_active_modified_theta_ft_woman = zeros(length(probe_active_modified_Freq_woman),k);

% Preallocation of Alpha
probe_passive_modified_alpha_ft_woman = zeros(length(probe_passive_modified_Freq_woman),k);
probe_active_modified_alpha_ft_woman = zeros(length(probe_active_modified_Freq_woman),k);

% Preallocation of Beta
probe_passive_modified_beta_ft_woman = zeros(length(probe_passive_modified_Freq_woman),k);
probe_active_modified_beta_ft_woman = zeros(length(probe_active_modified_Freq_woman),k);

% Preallocation of Delta IFT :
delta_passive_ift_woman = zeros(length(probe_passive_modified_ft_woman),k);
delta_active_ift_woman = zeros(length(probe_active_modified_ft_woman),k);

% Preallocation of Theta IFT :
theta_passive_ift_woman = zeros(length(probe_passive_modified_ft_woman),k);
theta_active_ift_woman = zeros(length(probe_active_modified_ft_woman),k);

% Preallocation of Alpha IFT :
alpha_passive_ift_woman = zeros(length(probe_passive_modified_ft_woman),k);
alpha_active_ift_woman = zeros(length(probe_active_modified_ft_woman),k);

% Preallocation of Beta IFT :
beta_passive_ift_woman = zeros(length(probe_passive_modified_ft_woman),k);
beta_active_ift_woman = zeros(length(probe_active_modified_ft_woman),k);

% Preallocation of Delta Channel Separated Signals :
separatedResponsePassiveDeltaWoman = zeros(480, k, lim1+lim2); % 480 : num of stimuli, k : num of clusters, lim1+lim2 : length of each separated signal 
separatedResponseActiveDeltaWoman = zeros(480, k, lim1+lim2);

% Preallocation of Theta Channel Separated Signals :
separatedResponsePassiveThetaWoman = zeros(480, k, lim1+lim2);
separatedResponseActiveThetaWoman = zeros(480, k, lim1+lim2);

% Preallocation of Alpha Channel Separated Signals :
separatedResponsePassiveAlphaWoman = zeros(480, k, lim1+lim2);
separatedResponseActiveAlphaWoman = zeros(480, k, lim1+lim2);

% Preallocation of Beta Channel Separated Signals :
separatedResponsePassiveBetaWoman = zeros(480, k, lim1+lim2);
separatedResponseActiveBetaWoman = zeros(480, k, lim1+lim2);

% Fourier transform :

% Passive Mode

for i = indexInformativePassiveWoman(:, 1)'
    [probe_passive_modified_ft_woman(:, find(indexInformativePassiveWoman(:, 1) == i)), ~] = FFT(probe_passive_modified_woman(:, find(indexInformativePassiveWoman(:, 1) == i)), FS, 0);
    
    probe_passive_modified_delta_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i)) = FTP(probe_passive_modified_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i)), probe_passive_modified_Freq_woman, 0, 4, 0);
    delta_passive_ift_woman(:,find(indexInformativePassiveWoman(:, 1) == i)) = ifft(ifftshift(probe_passive_modified_delta_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i))), 'symmetric');
    separatedResponsePassiveDeltaWoman(:,find(indexInformativePassiveWoman(:, 1) == i),:) = responseSeparator(delta_passive_ift_woman(:,find(indexInformativePassiveWoman(:, 1) == i)), S2.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_theta_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i)) = FTP(probe_passive_modified_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i)), probe_passive_modified_Freq_woman, 4, 8, 0);
    theta_passive_ift_woman(:,find(indexInformativePassiveWoman(:, 1) == i)) = ifft(ifftshift(probe_passive_modified_theta_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i))), 'symmetric');
    separatedResponsePassiveThetaWoman(:,find(indexInformativePassiveWoman(:, 1) == i),:) = responseSeparator(theta_passive_ift_woman(:,find(indexInformativePassiveWoman(:, 1) == i)), S2.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_alpha_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i)) = FTP(probe_passive_modified_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i)), probe_passive_modified_Freq_woman, 8, 13, 0);
    alpha_passive_ift_woman(:,find(indexInformativePassiveWoman(:, 1) == i)) = ifft(ifftshift(probe_passive_modified_alpha_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i))), 'symmetric');
    separatedResponsePassiveAlphaWoman(:,find(indexInformativePassiveWoman(:, 1) == i),:) = responseSeparator(alpha_passive_ift_woman(:,find(indexInformativePassiveWoman(:, 1) == i)), S2.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_beta_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i)) = FTP(probe_passive_modified_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i)), probe_passive_modified_Freq_woman, 13, 30, 0);
    beta_passive_ift_woman(:,find(indexInformativePassiveWoman(:, 1) == i)) = ifft(ifftshift(probe_passive_modified_beta_ft_woman(:,find(indexInformativePassiveWoman(:, 1) == i))), 'symmetric');
    separatedResponsePassiveBetaWoman(:,find(indexInformativePassiveWoman(:, 1) == i),:) = responseSeparator(beta_passive_ift_woman(:,find(indexInformativePassiveWoman(:, 1) == i)), S2.data{1,1}.trial - activeTruncateStart, lim1, lim2);
end

% Active Mode

for i = indexInformativeActiveWoman(:, 1)'
    [probe_active_modified_ft_woman(:, find(indexInformativeActiveWoman(:, 1) == i)), ~] = FFT(probe_active_modified_woman(:, find(indexInformativeActiveWoman(:, 1) == i)), FS, 0);
    
    probe_active_modified_delta_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i)) = FTP(probe_active_modified_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i)), probe_active_modified_Freq_woman, 0, 4, 0);
    delta_active_ift_woman(:,find(indexInformativeActiveWoman(:, 1) == i)) = ifft(ifftshift(probe_active_modified_delta_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i))), 'symmetric');
    separatedResponseActiveDeltaWoman(:,find(indexInformativeActiveWoman(:, 1) == i),:) = responseSeparator(delta_active_ift_woman(:,find(indexInformativeActiveWoman(:, 1) == i)), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_theta_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i)) = FTP(probe_active_modified_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i)), probe_active_modified_Freq_woman, 4, 8, 0);
    theta_active_ift_woman(:,find(indexInformativeActiveWoman(:, 1) == i)) = ifft(ifftshift(probe_active_modified_theta_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i))), 'symmetric');
    separatedResponseActiveThetaWoman(:,find(indexInformativeActiveWoman(:, 1) == i),:) = responseSeparator(theta_active_ift_woman(:,find(indexInformativeActiveWoman(:, 1) == i)), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_alpha_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i)) = FTP(probe_active_modified_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i)), probe_active_modified_Freq_woman, 8, 13, 0);
    alpha_active_ift_woman(:,find(indexInformativeActiveWoman(:, 1) == i)) = ifft(ifftshift(probe_active_modified_alpha_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i))), 'symmetric');
    separatedResponseActiveAlphaWoman(:,find(indexInformativeActiveWoman(:, 1) == i),:) = responseSeparator(alpha_active_ift_woman(:,find(indexInformativeActiveWoman(:, 1) == i)), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_beta_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i)) = FTP(probe_active_modified_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i)), probe_active_modified_Freq_woman, 13, 30, 0);
    beta_active_ift_woman(:,find(indexInformativeActiveWoman(:, 1) == i)) = ifft(ifftshift(probe_active_modified_beta_ft_woman(:,find(indexInformativeActiveWoman(:, 1) == i))), 'symmetric');
    separatedResponseActiveBetaWoman(:,find(indexInformativeActiveWoman(:, 1) == i),:) = responseSeparator(beta_active_ift_woman(:,find(indexInformativeActiveWoman(:, 1) == i)), S1.data{1,2}.trial - activeTruncateStart, lim1, lim2);
end

% Preallocation of Each Channel, type1 & type2, passive :
meanResponseType1PassiveDeltaWoman = zeros(lim1+lim2,k);
meanResponseType1PassiveThetaWoman = zeros(lim1+lim2,k);
meanResponseType1PassiveAlphaWoman = zeros(lim1+lim2,k);
meanResponseType1PassiveBetaWoman = zeros(lim1+lim2,k);
meanResponseType2PassiveDeltaWoman = zeros(lim1+lim2,k);
meanResponseType2PassiveThetaWoman = zeros(lim1+lim2,k);
meanResponseType2PassiveAlphaWoman = zeros(lim1+lim2,k);
meanResponseType2PassiveBetaWoman = zeros(lim1+lim2,k);

% Preallocation of Each Channel, type1 & type2, active :
meanResponseType1ActiveDeltaWoman = zeros(lim1+lim2,k);
meanResponseType1ActiveThetaWoman = zeros(lim1+lim2,k);
meanResponseType1ActiveAlphaWoman = zeros(lim1+lim2,k);
meanResponseType1ActiveBetaWoman = zeros(lim1+lim2,k);
meanResponseType2ActiveDeltaWoman = zeros(lim1+lim2,k);
meanResponseType2ActiveThetaWoman = zeros(lim1+lim2,k);
meanResponseType2ActiveAlphaWoman = zeros(lim1+lim2,k);
meanResponseType2ActiveBetaWoman = zeros(lim1+lim2,k);
    
% Evaluating Response of Each Channel
for i = 1:k
    % Passive Mode :
    meanResponseType1PassiveDeltaWoman(:, i) = mean(separatedResponsePassiveDeltaWoman(type1StimulusPassiveWoman, i, :));
    meanResponseType1PassiveThetaWoman(:, i) = mean(separatedResponsePassiveThetaWoman(type1StimulusPassiveWoman, i, :));
    meanResponseType1PassiveAlphaWoman(:, i) = mean(separatedResponsePassiveAlphaWoman(type1StimulusPassiveWoman, i, :));
    meanResponseType1PassiveBetaWoman(:, i) = mean(separatedResponsePassiveBetaWoman(type1StimulusPassiveWoman, i, :));
    
    meanResponseType2PassiveDeltaWoman(:, i) = mean(separatedResponsePassiveDeltaWoman(type2StimulusPassiveWoman, i, :));
    meanResponseType2PassiveThetaWoman(:, i) = mean(separatedResponsePassiveThetaWoman(type2StimulusPassiveWoman, i, :));
    meanResponseType2PassiveAlphaWoman(:, i) = mean(separatedResponsePassiveAlphaWoman(type2StimulusPassiveWoman, i, :));
    meanResponseType2PassiveBetaWoman(:, i) = mean(separatedResponsePassiveBetaWoman(type2StimulusPassiveWoman, i, :));
    
    % Active Mode :
    meanResponseType1ActiveDeltaWoman(:, i) = mean(separatedResponseActiveDeltaWoman(type1StimulusActiveWoman, i, :));
    meanResponseType1ActiveThetaWoman(:, i) = mean(separatedResponseActiveThetaWoman(type1StimulusActiveWoman, i, :));
    meanResponseType1ActiveAlphaWoman(:, i) = mean(separatedResponseActiveAlphaWoman(type1StimulusActiveWoman, i, :));
    meanResponseType1ActiveBetaWoman(:, i) = mean(separatedResponseActiveBetaWoman(type1StimulusActiveWoman, i, :));
    
    meanResponseType2ActiveDeltaWoman(:, i) = mean(separatedResponseActiveDeltaWoman(type2StimulusActiveWoman, i, :));
    meanResponseType2ActiveThetaWoman(:, i) = mean(separatedResponseActiveThetaWoman(type2StimulusActiveWoman, i, :));
    meanResponseType2ActiveAlphaWoman(:, i) = mean(separatedResponseActiveAlphaWoman(type2StimulusActiveWoman, i, :));
    meanResponseType2ActiveBetaWoman(:, i) = mean(separatedResponseActiveBetaWoman(type2StimulusActiveWoman, i, :));
end 

% Preallocation of Each Channel Confidence Interval Passive & Active Mode :
confidenceIntervalResponseType1PassiveDeltaWoman = zeros(k,1);
confidenceIntervalResponseType1PassiveThetaWoman = zeros(k,1);
confidenceIntervalResponseType1PassiveAlphaWoman = zeros(k,1);
confidenceIntervalResponseType1PassiveBetaWoman = zeros(k,1);

confidenceIntervalResponseType2PassiveDeltaWoman = zeros(k,1);
confidenceIntervalResponseType2PassiveThetaWoman = zeros(k,1);
confidenceIntervalResponseType2PassiveAlphaWoman = zeros(k,1);
confidenceIntervalResponseType2PassiveBetaWoman = zeros(k,1);

confidenceIntervalResponseType1ActiveDeltaWoman = zeros(k,1);
confidenceIntervalResponseType1ActiveThetaWoman = zeros(k,1);
confidenceIntervalResponseType1ActiveAlphaWoman = zeros(k,1);
confidenceIntervalResponseType1ActiveBetaWoman = zeros(k,1);

confidenceIntervalResponseType2ActiveDeltaWoman = zeros(k,1);
confidenceIntervalResponseType2ActiveThetaWoman = zeros(k,1);
confidenceIntervalResponseType2ActiveAlphaWoman = zeros(k,1);
confidenceIntervalResponseType2ActiveBetaWoman = zeros(k,1);

for j = 1:k
    
    % Type 1 Confidence Intervals (Passive Mode)
    confidenceIntervalResponseType1PassiveDeltaWoman(j,1) = mean(std(separatedResponsePassiveDeltaWoman(type1StimulusPassiveWoman, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType1PassiveThetaWoman(j,1) = mean(std(separatedResponsePassiveThetaWoman(type1StimulusPassiveWoman, j, :), 0, 3))/sqrt(420);
    confidenceIntervalResponseType1PassiveAlphaWoman(j,1) = mean(std(separatedResponsePassiveAlphaWoman(type1StimulusPassiveWoman, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType1PassiveBetaWoman(j,1) = mean(std(separatedResponsePassiveBetaWoman(type1StimulusPassiveWoman, j, :), 0, 3))/sqrt(420);
    
    % Type 2 Confidence Intervals (Passive Mode)
    confidenceIntervalResponseType2PassiveDeltaWoman(j,1) = mean(std(separatedResponsePassiveDeltaWoman(type2StimulusPassiveWoman, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType2PassiveThetaWoman(j,1) = mean(std(separatedResponsePassiveThetaWoman(type2StimulusPassiveWoman, j, :), 0, 3))/sqrt(420);
    confidenceIntervalResponseType2PassiveAlphaWoman(j,1) = mean(std(separatedResponsePassiveAlphaWoman(type2StimulusPassiveWoman, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType2PassiveBetaWoman(j,1) = mean(std(separatedResponsePassiveBetaWoman(type2StimulusPassiveWoman, j, :), 0, 3))/sqrt(420);
    
    % Type 1 Confidence Intervals (Active Mode)
    confidenceIntervalResponseType1ActiveDeltaWoman(j,1) = mean(std(separatedResponseActiveDeltaWoman(type1StimulusActiveWoman, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType1ActiveThetaWoman(j,1) = mean(std(separatedResponseActiveThetaWoman(type1StimulusActiveWoman, j, :), 0, 3))/sqrt(420);
    confidenceIntervalResponseType1ActiveAlphaWoman(j,1) = mean(std(separatedResponseActiveAlphaWoman(type1StimulusActiveWoman, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType1ActiveBetaWoman(j,1) = mean(std(separatedResponseActiveBetaWoman(type1StimulusActiveWoman, j, :), 0, 3))/sqrt(420);
    
    % Type 2 Confidence Intervals (Active Mode)
    confidenceIntervalResponseType2ActiveDeltaWoman(j,1) = mean(std(separatedResponseActiveDeltaWoman(type2StimulusActiveWoman, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType2ActiveThetaWoman(j,1) = mean(std(separatedResponseActiveThetaWoman(type2StimulusActiveWoman, j, :), 0, 3))/sqrt(420);
    confidenceIntervalResponseType2ActiveAlphaWoman(j,1) = mean(std(separatedResponseActiveAlphaWoman(type2StimulusActiveWoman, j, :), 0, 3))/sqrt(60);
    confidenceIntervalResponseType2ActiveBetaWoman(j,1) = mean(std(separatedResponseActiveBetaWoman(type2StimulusActiveWoman, j, :), 0, 3))/sqrt(420);
end
%%

% Plotting Responses of Each Channel
for i = 1:k 
    % Type 1 Response Passive Delta
    figure;
    plot(t,meanResponseType1PassiveDeltaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive Delta,Cluster No. %d',i),'color','r');
    
    % Type 1 Response Active Delta
    figure;
    plot(t,meanResponseType1ActiveDeltaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active Delta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive Delta
    figure;
    plot(t,meanResponseType2PassiveDeltaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive Delta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active Delta
    figure;
    plot(t,meanResponseType2ActiveDeltaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active Delta,Cluster No. %d',i),'color','r');
    
    
    
    
    % Type 1 Response Passive Theta
    figure;
    plot(t,meanResponseType1PassiveThetaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive Theta,Cluster No. %d',i),'color','r');
    
    % Type 1 Response Active Theta
    figure;
    plot(t,meanResponseType1ActiveThetaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active Theta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive Theta
    figure;
    plot(t,meanResponseType2PassiveThetaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive Theta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active Theta
    figure;
    plot(t,meanResponseType2ActiveThetaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active Theta,Cluster No. %d',i),'color','r');
    
    
    
    
    % Type 1 Response Passive Alpha
    figure;
    plot(t,meanResponseType1PassiveAlphaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive Alpha,Cluster No. %d',i),'color','r');
    
    % Type 1 Response Active Alpha
    figure;
    plot(t,meanResponseType1ActiveAlphaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active Alpha,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive Alpha
    figure;
    plot(t,meanResponseType2PassiveAlphaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive Alpha,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active Alpha
    figure;
    plot(t,meanResponseType2ActiveAlphaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active Alpha,Cluster No. %d',i),'color','r');
    
    
    
    
    % Type 1 Response Passive Beta
    figure;
    plot(t,meanResponseType1PassiveBetaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Passive Beta,Cluster No. %d',i),'color','r');
    
    % Type 1 Response Active Beta
    figure;
    plot(t,meanResponseType1ActiveBetaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 1 Response Active Beta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Passive Beta
    figure;
    plot(t,meanResponseType2PassiveBetaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Passive Beta,Cluster No. %d',i),'color','r');
    
    % Type 2 Response Active Beta
    figure;
    plot(t,meanResponseType2ActiveBetaWoman(:, i));
    xlabel('Time(milliseconds)','color','b');
    ylabel('Amplitude','color','b');
    title(sprintf('Type 2 Response Active Beta,Cluster No. %d',i),'color','r');
end



%% Section 4
% included in report
%% Section 5
clc
close all

% Preallocation for mean of responses to type1 & type2, active and passive
% experiments :
% Each of the following is a 537(lenght of interval around stimuli) *
% 31(num of probes) matrix
meanResponseType1PassiveAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2PassiveAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType1ActiveAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2ActiveAllProbesWoman = zeros(lim1+lim2, 31);

for i = 1:31
    meanResponseType1PassiveAllProbesWoman(:, i) = mean(separatedResponsePassiveWoman(type1StimulusPassiveWoman, i, :));
    meanResponseType2PassiveAllProbesWoman(:, i) = mean(separatedResponsePassiveWoman(type2StimulusPassiveWoman, i, :));
    meanResponseType1ActiveAllProbesWoman(:, i) = mean(separatedResponseActiveWoman(type1StimulusActiveWoman, i, :));
    meanResponseType2ActiveAllProbesWoman(:, i) = mean(separatedResponseActiveWoman(type2StimulusActiveWoman, i, :));
end

% Calculating the Pearson correlation between type1 & type2 stimuli for each probe,
% passive & active :
corrCoeffType1Type2PassivePearsonWoman = diag(corr(meanResponseType1PassiveAllProbesWoman, meanResponseType2PassiveAllProbesWoman));
corrCoeffType1Type2ActivePearsonWoman = diag(corr(meanResponseType1ActiveAllProbesWoman, meanResponseType2ActiveAllProbesWoman));

% Plotting response to type1 & type2 stimuli for abs(corrCoeff)>0.4  and Pearson correlation :
indexOfPlotPassiveWoman = (abs(corrCoeffType1Type2PassivePearsonWoman) > 0.4) .* (1:31)';
indexOfPlotActiveWoman = (abs(corrCoeffType1Type2ActivePearsonWoman) > 0.4) .* (1:31)';

for i = indexOfPlotPassiveWoman(indexOfPlotPassiveWoman ~= 0)'
    figure
    plot(meanResponseType2PassiveAllProbesWoman(:, i), meanResponseType1PassiveAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode', i),'color','r');
end

for i = indexOfPlotActiveWoman(indexOfPlotActiveWoman ~= 0)'
    figure
    plot(meanResponseType2ActiveAllProbesWoman(:, i), meanResponseType1ActiveAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode', i),'color','r');
end


%% Section 6
clc
close all

% Preallocation of fourier transform of all probes :
probe_passive_modified_ft_all_probes_woman = zeros(nPassiveModifiedWoman, 31);
probe_active_modified_ft_all_probes_woman = zeros(nActiveModifiedWoman, 31);

% Preallocation of FT of channels (delta, ...) for all probes, passive mode :
probe_passive_modified_delta_ft_all_probes_woman = zeros(length(probe_passive_modified_Freq_woman), 31);
probe_passive_modified_theta_ft_all_probes_woman = zeros(length(probe_passive_modified_Freq_woman), 31);
probe_passive_modified_alpha_ft_all_probes_woman = zeros(length(probe_passive_modified_Freq_woman), 31);
probe_passive_modified_beta_ft_all_probes_woman = zeros(length(probe_passive_modified_Freq_woman), 31);

% Preallocation of IFT of channels for all probes, passive mode :
delta_passive_ift_all_probes_woman = zeros(length(probe_passive_modified_Freq_woman), 31);
theta_passive_ift_all_probes_woman = zeros(length(probe_passive_modified_Freq_woman), 31);
alpha_passive_ift_all_probes_woman = zeros(length(probe_passive_modified_Freq_woman), 31);
beta_passive_ift_all_probes_woman = zeros(length(probe_passive_modified_Freq_woman), 31);

% Preallocation of mean of responses to type1 & type2 stimuli for all
% probes, passive mode :
separatedResponsePassiveDeltaAllProbesWoman = zeros(480, 31, lim1+lim2); % 480 * 31 * 537
separatedResponsePassiveThetaAllProbesWoman = zeros(480, 31, lim1+lim2);
separatedResponsePassiveAlphaAllProbesWoman = zeros(480, 31, lim1+lim2);
separatedResponsePassiveBetaAllProbesWoman = zeros(480, 31, lim1+lim2);

% Fourier transform of all probes in passive mode + 
% Separating channels(delta, ...) from transforms + 
% IFT from each channel +
% Separating type1 & type2 stimuli for each channel 
% for passive mode :
for i = 1:31
    [probe_passive_modified_ft_all_probes_woman(:, i), ~] = FFT(probe_passive_modified_woman(:, i), FS, 0);
    
    probe_passive_modified_delta_ft_all_probes_woman(:,i) = FTP(probe_passive_modified_ft_all_probes_woman(:, i), probe_passive_modified_Freq_woman, 0, 4, 0);
    delta_passive_ift_all_probes_woman(:,i) = ifft(ifftshift(probe_passive_modified_delta_ft_all_probes_woman(:,i)), 'symmetric');
    separatedResponsePassiveDeltaAllProbesWoman(:,i,:) = responseSeparator(delta_passive_ift_all_probes_woman(:,i), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_theta_ft_all_probes_woman(:,i) = FTP(probe_passive_modified_ft_all_probes_woman(:, i), probe_passive_modified_Freq_woman, 4, 8, 0);
    theta_passive_ift_all_probes_woman(:,i) = ifft(ifftshift(probe_passive_modified_theta_ft_all_probes_woman(:,i)), 'symmetric');
    separatedResponsePassiveThetaAllProbesWoman(:,i,:) = responseSeparator(theta_passive_ift_all_probes_woman(:,i), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_alpha_ft_all_probes_woman(:,i) = FTP(probe_passive_modified_ft_all_probes_woman(:, i), probe_passive_modified_Freq_woman, 8, 13, 0);
    alpha_passive_ift_all_probes_woman(:,i) = ifft(ifftshift(probe_passive_modified_alpha_ft_all_probes_woman(:,i)), 'symmetric');
    separatedResponsePassiveAlphaAllProbesWoman(:,i,:) = responseSeparator(alpha_passive_ift_all_probes_woman(:,i), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
    
    probe_passive_modified_beta_ft_all_probes_woman(:,i) = FTP(probe_passive_modified_ft_all_probes_woman(:, i), probe_passive_modified_Freq_woman, 13, 30, 0);
    beta_passive_ift_all_probes_woman(:,i) = ifft(ifftshift(probe_passive_modified_beta_ft_all_probes_woman(:,i)), 'symmetric');
    separatedResponsePassiveBetaAllProbesWoman(:,i,:) = responseSeparator(beta_passive_ift_all_probes_woman(:,i), S1.data{1,1}.trial - activeTruncateStart, lim1, lim2);
end

% Preallocation of FT of channels (delta, ...) for all probes, active mode :
probe_active_modified_delta_ft_all_probes_woman = zeros(length(probe_active_modified_Freq_woman), 31);
probe_active_modified_theta_ft_all_probes_woman = zeros(length(probe_active_modified_Freq_woman), 31);
probe_active_modified_alpha_ft_all_probes_woman = zeros(length(probe_active_modified_Freq_woman), 31);
probe_active_modified_beta_ft_all_probes_woman = zeros(length(probe_active_modified_Freq_woman), 31);

% Preallocation of IFT of channels for all probes, active mode :
delta_active_ift_all_probes_woman = zeros(length(probe_active_modified_Freq_woman), 31);
theta_active_ift_all_probes_woman = zeros(length(probe_active_modified_Freq_woman), 31);
alpha_active_ift_all_probes_woman = zeros(length(probe_active_modified_Freq_woman), 31);
beta_active_ift_all_probes_woman = zeros(length(probe_active_modified_Freq_woman), 31);

% Preallocation of mean of responses to type1 & type2 stimuli for all
% probes, active mode :
separatedResponseActiveDeltaAllProbesWoman = zeros(480, 31, lim1+lim2); % 480 * 31 * 537
separatedResponseActiveThetaAllProbesWoman = zeros(480, 31, lim1+lim2);
separatedResponseActiveAlphaAllProbesWoman = zeros(480, 31, lim1+lim2);
separatedResponseActiveBetaAllProbesWoman = zeros(480, 31, lim1+lim2);

% Fourier transform of all probes in passive mode + 
% Separating channels(delta, ...) from transforms + 
% IFT from each channel +
% Separating type1 & type2 stimuli for each channel 
% for active mode :
for i = 1:31
    [probe_active_modified_ft_all_probes_woman(:, i), ~] = FFT(probe_active_modified_woman(:, i), FS, 0);
    
    probe_active_modified_delta_ft_all_probes_woman(:,i) = FTP(probe_active_modified_ft_all_probes_woman(:, i), probe_active_modified_Freq_woman, 0, 4, 0);
    delta_active_ift_all_probes_woman(:,i) = ifft(ifftshift(probe_active_modified_delta_ft_all_probes_woman(:,i)), 'symmetric');
    separatedResponseActiveDeltaAllProbesWoman(:,i,:) = responseSeparator(delta_active_ift_all_probes_woman(:,i), S2.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_theta_ft_all_probes_woman(:,i) = FTP(probe_active_modified_ft_all_probes_woman(:, i), probe_active_modified_Freq_woman, 4, 8, 0);
    theta_active_ift_all_probes_woman(:,i) = ifft(ifftshift(probe_active_modified_theta_ft_all_probes_woman(:,i)), 'symmetric');
    separatedResponseActiveThetaAllProbesWoman(:,i,:) = responseSeparator(theta_active_ift_all_probes_woman(:,i), S2.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_alpha_ft_all_probes_woman(:,i) = FTP(probe_active_modified_ft_all_probes_woman(:, i), probe_active_modified_Freq_woman, 8, 13, 0);
    alpha_active_ift_all_probes_woman(:,i) = ifft(ifftshift(probe_active_modified_alpha_ft_all_probes_woman(:,i)), 'symmetric');
    separatedResponseActiveAlphaAllProbesWoman(:,i,:) = responseSeparator(alpha_active_ift_all_probes_woman(:,i), S2.data{1,2}.trial - activeTruncateStart, lim1, lim2);
    
    probe_active_modified_beta_ft_all_probes_woman(:,i) = FTP(probe_active_modified_ft_all_probes_woman(:, i), probe_active_modified_Freq_woman, 13, 30, 0);
    beta_active_ift_all_probes_woman(:,i) = ifft(ifftshift(probe_active_modified_beta_ft_all_probes_woman(:,i)), 'symmetric');
    separatedResponseActiveBetaAllProbesWoman(:,i,:) = responseSeparator(beta_active_ift_all_probes_woman(:,i), S2.data{1,2}.trial - activeTruncateStart, lim1, lim2);
end

% Calculating the mean of separated signals for each channel of each probe
% over 420 stimuli (type2) and also 60 stimuli (type1) :

% Preallocation for the mean of separated signals for each channel of each
% probe :
% Delta :
meanResponseType1PassiveDeltaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2PassiveDeltaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType1ActiveDeltaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2ActiveDeltaAllProbesWoman = zeros(lim1+lim2, 31);
% Theta :
meanResponseType1PassiveThetaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2PassiveThetaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType1ActiveThetaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2ActiveThetaAllProbesWoman = zeros(lim1+lim2, 31);
% Alpha :
meanResponseType1PassiveAlphaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2PassiveAlphaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType1ActiveAlphaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2ActiveAlphaAllProbesWoman = zeros(lim1+lim2, 31);
% Beta :
meanResponseType1PassiveBetaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2PassiveBetaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType1ActiveBetaAllProbesWoman = zeros(lim1+lim2, 31);
meanResponseType2ActiveBetaAllProbesWoman = zeros(lim1+lim2, 31);

for i = 1:31
    % Delta :
    meanResponseType1PassiveDeltaAllProbesWoman(:, i) = mean(separatedResponsePassiveDeltaAllProbesWoman(type1StimulusPassiveWoman,i, :)); % 31 passive, delta, type1
    meanResponseType2PassiveDeltaAllProbesWoman(:, i) = mean(separatedResponsePassiveDeltaAllProbesWoman(type2StimulusPassiveWoman,i, :)); % 31 passive, delta, type2
    meanResponseType1ActiveDeltaAllProbesWoman(:, i) = mean(separatedResponseActiveDeltaAllProbesWoman(type1StimulusActiveWoman,i, :)); % 31 active, delta, type1
    meanResponseType2ActiveDeltaAllProbesWoman(:, i) = mean(separatedResponseActiveDeltaAllProbesWoman(type2StimulusActiveWoman,i, :)); % 31 active, delta, type2
    
    % Theta :
    meanResponseType1PassiveThetaAllProbesWoman(:, i) = mean(separatedResponsePassiveThetaAllProbesWoman(type1StimulusPassiveWoman,i, :)); % 31 passive, theta, type1
    meanResponseType2PassiveThetaAllProbesWoman(:, i) = mean(separatedResponsePassiveThetaAllProbesWoman(type2StimulusPassiveWoman,i, :)); % 31 passive, theta, type2
    meanResponseType1ActiveThetaAllProbesWoman(:, i) = mean(separatedResponseActiveThetaAllProbesWoman(type1StimulusActiveWoman,i, :)); % 31 active, theta, type1
    meanResponseType2ActiveThetaAllProbesWoman(:, i) = mean(separatedResponseActiveThetaAllProbesWoman(type2StimulusActiveWoman,i, :)); % 31 active, theta, type2
    
    % Alpha :
    meanResponseType1PassiveAlphaAllProbesWoman(:, i) = mean(separatedResponsePassiveAlphaAllProbesWoman(type1StimulusPassiveWoman,i, :)); % 31 passive, alpha, type1
    meanResponseType2PassiveAlphaAllProbesWoman(:, i) = mean(separatedResponsePassiveAlphaAllProbesWoman(type2StimulusPassiveWoman,i, :)); % 31 passive, alpha, type2
    meanResponseType1ActiveAlphaAllProbesWoman(:, i) = mean(separatedResponseActiveAlphaAllProbesWoman(type1StimulusActiveWoman,i, :)); % 31 active, alpha, type1
    meanResponseType2ActiveAlphaAllProbesWoman(:, i) = mean(separatedResponseActiveAlphaAllProbesWoman(type2StimulusActiveWoman,i, :)); % 31 active, alpha, type2
    
    % Beta :
    meanResponseType1PassiveBetaAllProbesWoman(:, i) = mean(separatedResponsePassiveBetaAllProbesWoman(type1StimulusPassiveWoman,i, :)); % 31 passive, beta, type1
    meanResponseType2PassiveBetaAllProbesWoman(:, i) = mean(separatedResponsePassiveBetaAllProbesWoman(type2StimulusPassiveWoman,i, :)); % 31 passive, beta, type2
    meanResponseType1ActiveBetaAllProbesWoman(:, i) = mean(separatedResponseActiveBetaAllProbesWoman(type1StimulusActiveWoman,i, :)); % 31 active, beta, type1
    meanResponseType2ActiveBetaAllProbesWoman(:, i) = mean(separatedResponseActiveBetaAllProbesWoman(type2StimulusActiveWoman,i, :)); % 31 active, beta, type2
end

% Calculating the Pearson correlation between type1 & type2 stimuli for
% each probe, each channel, passive & active :
% Passive :
corrCoeffType1Type2PassiveDeltaPearsonWoman = diag(corr(meanResponseType1PassiveDeltaAllProbesWoman, meanResponseType2PassiveDeltaAllProbesWoman));
corrCoeffType1Type2PassiveThetaPearsonWoman = diag(corr(meanResponseType1PassiveThetaAllProbesWoman, meanResponseType2PassiveThetaAllProbesWoman));
corrCoeffType1Type2PassiveAlphaPearsonWoman = diag(corr(meanResponseType1PassiveAlphaAllProbesWoman, meanResponseType2PassiveAlphaAllProbesWoman));
corrCoeffType1Type2PassiveBetaPearsonWoman = diag(corr(meanResponseType1PassiveBetaAllProbesWoman, meanResponseType2PassiveBetaAllProbesWoman));
% Active :
corrCoeffType1Type2ActiveDeltaPearsonWoman = diag(corr(meanResponseType1ActiveDeltaAllProbesWoman, meanResponseType2ActiveDeltaAllProbesWoman));
corrCoeffType1Type2ActiveThetaPearsonWoman = diag(corr(meanResponseType1ActiveThetaAllProbesWoman, meanResponseType2ActiveThetaAllProbesWoman));
corrCoeffType1Type2ActiveAlphaPearsonWoman = diag(corr(meanResponseType1ActiveAlphaAllProbesWoman, meanResponseType2ActiveAlphaAllProbesWoman));
corrCoeffType1Type2ActiveBetaPearsonWoman = diag(corr(meanResponseType1ActiveBetaAllProbesWoman, meanResponseType2ActiveBetaAllProbesWoman));

% Plotting response to type1 & type2 stimuli for each channel for abs(corrCoeff)>0.4  and Pearson correlation :
% Passive :
indexOfPlotPassiveDeltaWoman = (abs(corrCoeffType1Type2PassiveDeltaPearsonWoman) > 0.4) .* (1:31)';
indexOfPlotPassiveThetaWoman = (abs(corrCoeffType1Type2PassiveThetaPearsonWoman) > 0.4) .* (1:31)';
indexOfPlotPassiveAlphaWoman = (abs(corrCoeffType1Type2PassiveAlphaPearsonWoman) > 0.4) .* (1:31)';
indexOfPlotPassiveBetaWoman = (abs(corrCoeffType1Type2PassiveBetaPearsonWoman) > 0.4) .* (1:31)';
% Active :
indexOfPlotActiveDeltaWoman = (abs(corrCoeffType1Type2ActiveDeltaPearsonWoman) > 0.4) .* (1:31)';
indexOfPlotActiveThetaWoman = (abs(corrCoeffType1Type2ActiveThetaPearsonWoman) > 0.4) .* (1:31)';
indexOfPlotActiveAlphaWoman = (abs(corrCoeffType1Type2ActiveAlphaPearsonWoman) > 0.4) .* (1:31)';
indexOfPlotActiveBetaWoman = (abs(corrCoeffType1Type2ActiveBetaPearsonWoman) > 0.4) .* (1:31)';
%% Plotting
% Passive :
% Delta :
for i = indexOfPlotPassiveDeltaWoman(indexOfPlotPassiveDeltaWoman ~= 0)'
    figure
    plot(meanResponseType2PassiveDeltaAllProbesWoman(:, i), meanResponseType1PassiveDeltaAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode, Delta Channel', i),'color','r');
end

% Theta :
for i = indexOfPlotPassiveThetaWoman(indexOfPlotPassiveThetaWoman ~= 0)'
    figure
    plot(meanResponseType2PassiveThetaAllProbesWoman(:, i), meanResponseType1PassiveThetaAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode, Theta Channel', i),'color','r');
end

% Alpha :
for i = indexOfPlotPassiveAlphaWoman(indexOfPlotPassiveAlphaWoman ~= 0)'
    figure
    plot(meanResponseType2PassiveAlphaAllProbesWoman(:, i), meanResponseType1PassiveAlphaAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode, Alpha Channel', i),'color','r');
end

% Beta :
for i = indexOfPlotPassiveBetaWoman(indexOfPlotPassiveBetaWoman ~= 0)'
    figure
    plot(meanResponseType2PassiveBetaAllProbesWoman(:, i), meanResponseType1PassiveBetaAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Passive Mode, Beta Channel', i),'color','r');
end

% Active :
% Delta :
for i = indexOfPlotActiveDeltaWoman(indexOfPlotActiveDeltaWoman ~= 0)'
    figure
    plot(meanResponseType2ActiveDeltaAllProbesWoman(:, i), meanResponseType1ActiveDeltaAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode, Delta Channel', i),'color','r');
end

% Tetha :
for i = indexOfPlotActiveThetaWoman(indexOfPlotActiveThetaWoman ~= 0)'
    figure
    plot(meanResponseType2ActiveThetaAllProbesWoman(:, i), meanResponseType1ActiveThetaAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode, Theta Channel', i),'color','r');
end

% Alpha :
for i = indexOfPlotActiveAlphaWoman(indexOfPlotActiveAlphaWoman ~= 0)'
    figure
    plot(meanResponseType2ActiveAlphaAllProbesWoman(:, i), meanResponseType1ActiveAlphaAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode, Alpha Channel', i),'color','r');
end

% Beta :
for i = indexOfPlotActiveBetaWoman(indexOfPlotActiveBetaWoman ~= 0)'
    figure
    plot(meanResponseType2ActiveBetaAllProbesWoman(:, i), meanResponseType1ActiveBetaAllProbesWoman(:, i), '.');
    xlabel('Type2','color','b');
    ylabel('Type1','color','b');
    title(sprintf('Probe No. %d, Active Mode, Beta Channel', i),'color','r');
end
%% Section 7

clc
close all

% Finding the most informative cluster by meaning on the correlation
% coefficients :
meanOfCorrCoeffsOfClustersWoman = zeros(k, 1);
for i = 1:k
   meanOfCorrCoeffsOfClustersWoman(i, 1) = mean(corrCoeffType1Type2PassivePearsonWoman(find(idxPassivePearsonWoman == i)));
end
[~, indexOfInformativeClusterPassiveWoman] = max (meanOfCorrCoeffsOfClustersWoman);

for i = 1:k
   meanOfCorrCoeffsOfClustersWoman(i, 1) = mean(corrCoeffType1Type2ActivePearsonWoman(find(idxActivePearsonWoman == i)));
end
[~, indexOfInformativeClusterActiveWoman] = max (meanOfCorrCoeffsOfClustersWoman);



% Calculation of maximum of cross-correlations in each cluster, passive mode :
indexOfInformativeProbesPassiveWoman = zeros(k, 1);
for i = 1:k
   [~, temp1] = max(abs(corrCoeffType1Type2PassivePearsonWoman(find(idxPassivePearsonWoman == i))));
   temp2 = find(idxPassivePearsonWoman == i);
   indexOfInformativeProbesPassiveWoman(i, 1) = temp2(temp1);
end

% Calculation of maximum of cross-correlations in each cluster, active mode :
indexOfInformativeProbesActiveWoman = zeros(k, 1);
for i = 1:k
   [~, temp1] = max(abs(corrCoeffType1Type2ActivePearsonWoman(find(idxActivePearsonWoman == i))));
   temp2 = find(idxActivePearsonWoman == i);
   indexOfInformativeProbesActiveWoman(i, 1) = temp2(temp1);
end
%% Section 8

clc
close all

% Averaging over type1 & type2 stimuli and calculating their difference:
meanResponseType1PassiveInformativeWoman = zeros(lim1+lim2, k);
meanResponseType2PassiveInformativeWoman = zeros(lim1+lim2, k);
meanResponseType1ActiveInformativeWoman = zeros(lim1+lim2, k);
meanResponseType2ActiveInformativeWoman = zeros(lim1+lim2, k);
meanResponseType1Type2PassiveInformativeDiffWoman = zeros(lim1+lim2, k);
meanResponseType1Type2ActiveInformativeDiffWoman = zeros(lim1+lim2, k);

for i = 1:k
    meanResponseType1PassiveInformativeWoman(:, i) = mean(separatedResponsePassiveWoman(type1StimulusPassiveWoman, indexOfInformativeProbesPassiveWoman(i, 1), :));
    meanResponseType2PassiveInformativeWoman(:, i) = mean(separatedResponsePassiveWoman(type2StimulusPassiveWoman, indexOfInformativeProbesPassiveWoman(i, 1), :));
    meanResponseType1ActiveInformativeWoman(:, i) = mean(separatedResponseActiveWoman(type1StimulusActiveWoman, indexOfInformativeProbesActiveWoman(i, 1), :));
    meanResponseType2ActiveInformativeWoman(:, i) = mean(separatedResponseActiveWoman(type2StimulusActiveWoman, indexOfInformativeProbesActiveWoman(i, 1), :));
    meanResponseType1Type2PassiveInformativeDiffWoman(:, i) = meanResponseType1PassiveInformativeWoman(:, i) - meanResponseType2PassiveInformativeWoman(:, i);
    meanResponseType1Type2ActiveInformativeDiffWoman(:, i) = meanResponseType1ActiveInformativeWoman(:, i) - meanResponseType2ActiveInformativeWoman(:, i);
end


% Plotting :
for i = 1:k
   figure
   plot(t, meanResponseType1PassiveInformativeWoman(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Passive Mode, Type1', indexOfInformativeProbesPassiveWoman(i)),'color','r');
   
   figure
   plot(t, meanResponseType2PassiveInformativeWoman(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Passive Mode, Type2', indexOfInformativeProbesPassiveWoman(i)),'color','r');
   
   figure
   plot(t, meanResponseType1ActiveInformativeWoman(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Active Mode, Type1', indexOfInformativeProbesActiveWoman(i)),'color','r');
   
   figure
   plot(t, meanResponseType2ActiveInformativeWoman(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Active Mode, Type2', indexOfInformativeProbesActiveWoman(i)),'color','r');
   
   figure
   plot(t, meanResponseType1Type2PassiveInformativeDiffWoman(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Passive Mode, Type1-Type2', indexOfInformativeProbesPassiveWoman(i)),'color','b');
   
   figure
   plot(t, meanResponseType1Type2ActiveInformativeDiffWoman(:, i));
   xlabel('Time','color','b');
   ylabel('Amplitude','color','b');
   title(sprintf('Probe %d, Active Mode, Type1-Type2', indexOfInformativeProbesActiveWoman(i)),'color','r');
end

