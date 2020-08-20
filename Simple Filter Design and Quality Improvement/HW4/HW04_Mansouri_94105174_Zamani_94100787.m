%% Part 1 : Designing Discrete-Time Filter
%% Section 1.3
clc
close all
% vector of radius values
R0 = [0.25 0.5 0.9 0.95 1.05 1.1]'; 
% FIR filter coefficients :
% coefficients of lowpass numerator
LP_FIR_Num = [ones(6, 1) R0];
% coefficients of lowpass denominator
LP_FIR_Denom = 1;

% Preallocation :
% number of points which filter response id calculated at
n = 2001; 
% complex values of filter response
h = zeros(n, 6); 
% frequencies at which ft response is calculated
w = zeros(n, 6); 

for i = 1:6
    % evaluating ft response values
    [h(:, i), w(:, i)] = freqz(LP_FIR_Num(i, :), LP_FIR_Denom, 'whole', n);
end

% plotting ft responses
for i = 1:6
    figure
    subplot(2,1,1);
    plot(w(:, i)/pi, 20*log10(abs(h(:, i))));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Absolute Value Of Filter TF Response','color','b','fontsize',8);
    title(sprintf('Filter Transfer Function Absolute Value (r0=%d)',R0(i)),'color','r');
    
    subplot(2,1,2);
    plot(w(:, i)/pi, angle(h(:, i)));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Phase Of Filter TF Response','color','b','fontsize',8);
    title(sprintf('Filter Transfer Function Phase (r0=%d)',R0(i)),'color','r');
end


%% Section 1.4
% included in report
%% Section 1.5
clc
close all

LP_IIR_Num = 1;
LP_IIR_Denom = [ones(6, 1) -R0];

for i = 1:6
    [h(:, i), w(:, i)] = freqz(LP_IIR_Num, LP_IIR_Denom(i, :), 'whole', n);
end

for i = 1:6
    figure
    subplot(2,1,1);
    plot(w(:, i)/pi, 20*log10(abs(h(:, i))));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Absolute Value','color','b');
    title(sprintf('Absolute Value Of IIR Filter Frequency Response (r0=%d)',R0(i)),'color','r');
    
    subplot(2,1,2);
    plot(w(:, i)/pi, angle(h(:, i)));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Phase','color','b');
    title(sprintf('Phase of IIR Filter Frequency Response (r0=%d)',R0(i)),'color','r');
end
%% Section 1.6
clc
close all

% FIR Filter Coefficients :
% Coefficients Of FIR High Pass Numerator Preallocation
HP_FIR_Num = [ones(6, 1) -R0];
% Coefficients Of FIR High Pass Denumerator
HP_FIR_Denom = 1;
% Coefficients Of IIR High Pass Numerator
HP_IIR_Num = 1;
% Coefficients Of IIR High Pass Denumerator Preallocation
HP_IIR_Denom = [ones(6, 1) R0];

% FIR :
% Evaluating frequency responses
for i = 1:6
    [h(:, i), w(:, i)] = freqz(HP_FIR_Num(i, :), HP_FIR_Denom, 'whole', n);
end

% Plotting frequency responses
for i = 1:6
    figure
    subplot(2,1,1);
    plot(w(:, i)/pi, 20*log10(abs(h(:, i))));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Absolute Value','color','b');
    title(sprintf('Absolute Value Of FIR HP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
    
    subplot(2,1,2);
    plot(w(:, i)/pi, angle(h(:, i)));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Phase','color','b');
    title(sprintf('Phase Of FIR HP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
end

% IIR :
% Evaluating frequency responses
for i = 1:6
    [h(:, i), w(:, i)] = freqz(HP_IIR_Num, HP_IIR_Denom(i, :), 'whole', n);
end

% Plotting frequency responses
for i = 1:6
    figure
    subplot(2,1,1);
    plot(w(:, i)/pi, 20*log10(abs(h(:, i))));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Absolute Value','color','b');
    title(sprintf('Absolute Value Of IIR HP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
    
    subplot(2,1,2);
    plot(w(:, i)/pi, angle(h(:, i)));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Phase','color','b');
    title(sprintf('Phase Of IIR HP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
end

%% Section 1.7
clc
close all

theta1 = pi/18; % 10 degrees
theta2 = pi/9; % 20 degrees

% LP FIR :
LP_FIR_Num1 = [ones(1, 6); R0'];
LP_FIR_Num2 = [ones(1, 6); R0' * (cos(theta1) - 1j*sin(theta1))];
LP_FIR_Num3 = [ones(1, 6); R0' * (cos(theta1) + 1j*sin(theta1))];
LP_FIR_Num4 = [ones(1, 6); R0' * (cos(theta2) - 1j*sin(theta2))];
LP_FIR_Num5 = [ones(1, 6); R0' * (cos(theta2) + 1j*sin(theta2))];

% Preallocation :
LP_FIR_Num_Final = zeros(6, 6); % 5 zeros ==> 6 numerator coeffs & 6*R0 ==> 6 rows
LP_FIR_Denom_Final = 1;

% Calculating the numerator coefficients :
for i = 1:6
    LP_FIR_Num_Final(i, :) = conv(LP_FIR_Num1(:, i),...
                                  conv(LP_FIR_Num2(:, i),...
                                       conv(LP_FIR_Num3(:, i),...
                                            conv(LP_FIR_Num4(:, i),...
                                                 LP_FIR_Num5(:, i)))));
end

% Frequency response :
for i = 1:6
   [h(:, i), w(:, i)] = freqz(LP_FIR_Num_Final(i, :), LP_FIR_Denom_Final, 'whole', n);
end

% Plotting the results :
for i = 1:6
    figure
    subplot(2,1,1);
    plot(w(:, i)/pi, 20*log10(abs(h(:, i))));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Absolute Value','color','b');
    title(sprintf('Absolute Value Of FIR LP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
    
    subplot(2,1,2);
    plot(w(:, i)/pi, angle(h(:, i)));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Phase','color','b');
    title(sprintf('Phase Of FIR LP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
end

% LP IIR :
LP_IIR_Denom1 = [ones(1, 6); -R0'];
LP_IIR_Denom2 = [ones(1, 6); -R0' * (cos(theta1) + 1j*sin(theta1))];
LP_IIR_Denom3 = [ones(1, 6); -R0' * (cos(theta1) - 1j*sin(theta1))];
LP_IIR_Denom4 = [ones(1, 6); -R0' * (cos(theta2) + 1j*sin(theta2))];
LP_IIR_Denom5 = [ones(1, 6); -R0' * (cos(theta2) - 1j*sin(theta2))];

% Preallocation :
LP_IIR_Denom_Final = zeros(6, 6); % 5 poles ==> 6 numerator coeffs & 6*R0 ==> 6 rows
LP_IIR_Num_Final = 1;

% Calculating the numerator coefficients :
for i = 1:6
    LP_IIR_Denom_Final(i, :) = conv(LP_IIR_Denom1(:, i),...
                                  conv(LP_IIR_Denom2(:, i),...
                                       conv(LP_IIR_Denom3(:, i),...
                                            conv(LP_IIR_Denom4(:, i),...
                                                 LP_IIR_Denom5(:, i)))));
end

% Frequency response :
for i = 1:6
   [h(:, i), w(:, i)] = freqz(LP_IIR_Num_Final, LP_IIR_Denom_Final(i, :), 'whole', n);
end

% Plotting the results :
for i = 1:6
    figure
    subplot(2,1,1);
    plot(w(:, i)/pi, 20*log10(abs(h(:, i))));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Absolute Value','color','b');
    title(sprintf('Absolute Value Of IIR LP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
    
    subplot(2,1,2);
    plot(w(:, i)/pi, angle(h(:, i)));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Phase','color','b');
    title(sprintf('Phase Of IIR LP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
end


% HP FIR :
HP_FIR_Num1 = [ones(1, 6); -R0'];
HP_FIR_Num2 = [ones(1, 6); -R0' * (cos(theta1) + 1j*sin(theta1))];
HP_FIR_Num3 = [ones(1, 6); -R0' * (cos(theta1) - 1j*sin(theta1))];
HP_FIR_Num4 = [ones(1, 6); -R0' * (cos(theta2) + 1j*sin(theta2))];
HP_FIR_Num5 = [ones(1, 6); -R0' * (cos(theta2) - 1j*sin(theta2))];

% Preallocation :
HP_FIR_Num_Final = zeros(6, 6); % 5 zeros ==> 6 numerator coeffs & 6*R0 ==> 6 rows
HP_FIR_Denom_Final = 1;

% Calculating the numerator coefficients :
for i = 1:6
    HP_FIR_Num_Final(i, :) = conv(HP_FIR_Num1(:, i),...
                                  conv(HP_FIR_Num2(:, i),...
                                       conv(HP_FIR_Num3(:, i),...
                                            conv(HP_FIR_Num4(:, i),...
                                                 HP_FIR_Num5(:, i)))));
end

% Frequency response :
for i = 1:6
   [h(:, i), w(:, i)] = freqz(HP_FIR_Num_Final(i, :), HP_FIR_Denom_Final, 'whole', n);
end

% Plotting the results :
for i = 1:6
    figure
    subplot(2,1,1);
    plot(w(:, i)/pi, 20*log10(abs(h(:, i))));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Absolute Value','color','b');
    title(sprintf('Absolute Value Of FIR HP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
    
    subplot(2,1,2);
    plot(w(:, i)/pi, angle(h(:, i)));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Phase','color','b');
    title(sprintf('Phase Of FIR HP Filter Frequency Response (r0=%d)',R0(i)),'color','r');

end

% HP IIR :
HP_IIR_Denom1 = [ones(1, 6); R0'];
HP_IIR_Denom2 = [ones(1, 6); R0' * (cos(theta1) - 1j*sin(theta1))];
HP_IIR_Denom3 = [ones(1, 6); R0' * (cos(theta1) + 1j*sin(theta1))];
HP_IIR_Denom4 = [ones(1, 6); R0' * (cos(theta2) - 1j*sin(theta2))];
HP_IIR_Denom5 = [ones(1, 6); R0' * (cos(theta2) + 1j*sin(theta2))];

% Preallocation :
HP_IIR_Denom_Final = zeros(6, 6); % 5 poles ==> 6 numerator coeffs & 6*R0 ==> 6 rows
HP_IIR_Num_Final = 1;

% Calculating the numerator coefficients :
for i = 1:6
    HP_IIR_Denom_Final(i, :) = conv(HP_IIR_Denom1(:, i),...
                                  conv(HP_IIR_Denom2(:, i),...
                                       conv(HP_IIR_Denom3(:, i),...
                                            conv(HP_IIR_Denom4(:, i),...
                                                 HP_IIR_Denom5(:, i)))));
end
% Frequency response :
for i = 1:6
   [h(:, i), w(:, i)] = freqz(HP_IIR_Num_Final, HP_IIR_Denom_Final(i, :), 'whole', n);
end

% Plotting the results :
for i = 1:6
    figure
    subplot(2,1,1);
    plot(w(:, i)/pi, 20*log10(abs(h(:, i))));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Absolute Value','color','b');
    title(sprintf('Absolute Value Of IIR HP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
    
    subplot(2,1,2);
    plot(w(:, i)/pi, angle(h(:, i)));
    xlabel('Normalized by pi Angular frequency','color','b');
    ylabel('Phase','color','b');
    title(sprintf('Phase Of IIR HP Filter Frequency Response (r0=%d)',R0(i)),'color','r');
    
end

%% Section 1.8
clc
close all

N = [1 2 4];
Wn = [0.03398 0.9665]; % LP, HP Fisrt Order respectively

% Best filter : R0 = 0.9 :
LP_IIR_Num_Best = 1;
HP_IIR_Num_Best = 1;

% N = 1 :
LP_IIR_Denom_Best_1 = LP_IIR_Denom(3, :); % Coefficient convolution, LP
[LP_IIR_Num_Best_1_Butter, LP_IIR_Denom_Best_1_Butter] = butter(N(1), Wn(1)); % ButterWorth, LP
HP_IIR_Denom_Best_1 = HP_IIR_Denom(3, :); % Coefficient convolution, HP
[HP_IIR_Num_Best_1_Butter, HP_IIR_Denom_Best_1_Butter] = butter(N(1), Wn(2), 'high'); % ButterWorth, HP

% Creating frequency response :
[h_LP_IIR_Best, w_LP_IIR_Best] = freqz(LP_IIR_Num_Best, LP_IIR_Denom_Best_1, 'whole', n);
[h_LP_IIR_Best_Butter, w_LP_IIR_Best_Butter] = freqz(LP_IIR_Num_Best_1_Butter, LP_IIR_Denom_Best_1_Butter, 'whole', n);
[h_HP_IIR_Best, w_HP_IIR_Best] = freqz(HP_IIR_Num_Best, HP_IIR_Denom_Best_1, 'whole', n);
[h_HP_IIR_Best_Butter, w_HP_IIR_Best_Butter] = freqz(HP_IIR_Num_Best_1_Butter, HP_IIR_Denom_Best_1_Butter, 'whole', n);

% Plotting :
% Absolute Value LP IIR First Order vs. Butter
figure
subplot(2,1,1)
hold on
plot(w_LP_IIR_Best/pi, 20*log10(abs(h_LP_IIR_Best/(max(h_LP_IIR_Best)))),'color','r');
plot(w_LP_IIR_Best_Butter/pi, 20*log10(abs(h_LP_IIR_Best_Butter/(max(h_LP_IIR_Best_Butter)))),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Absolute Value','color','b');
title(sprintf('Absolute Value Of IIR LP First Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');
% Phase LP IIR First Order vs. Butter
subplot(2,1,2)
hold on
plot(w_LP_IIR_Best/pi, angle(h_LP_IIR_Best./(max(h_LP_IIR_Best))),'color','r');
plot(w_LP_IIR_Best_Butter/pi, angle(h_LP_IIR_Best_Butter),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Phase','color','b');
title(sprintf('Phase Of IIR LP First Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');
% Absolute HP IIR First Order vs. Butter
figure
subplot(2,1,1)
hold on
plot(w_HP_IIR_Best/pi, 20*log10(abs(h_HP_IIR_Best/(max(h_HP_IIR_Best)))),'color','r');
plot(w_HP_IIR_Best_Butter/pi, 20*log10(abs(h_HP_IIR_Best_Butter/(max(h_HP_IIR_Best_Butter)))),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Absolute Value(dB)','color','b');
title(sprintf('Absolute Value Of IIR HP First Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');
% Phase HP IIR First Order vs. Butter
subplot(2,1,2)
hold on
plot(w_HP_IIR_Best/pi,angle(h_HP_IIR_Best),'color','r');
plot(w_HP_IIR_Best_Butter/pi,angle(h_HP_IIR_Best_Butter),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Phase','color','b');
title(sprintf('Phase Of IIR HP First Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');


% N = 2 :
Wn = [0.02199 0.9785]; % LP, HP Second Order respectively

LP_IIR_Denom_Best_2 = conv(LP_IIR_Denom(3, :), LP_IIR_Denom(3, :)); % Coefficient convolution, LP
[LP_IIR_Num_Best_2_Butter, LP_IIR_Denom_Best_2_Butter] = butter(N(2), Wn(1)); % ButterWorth, LP
HP_IIR_Denom_Best_2 = conv(HP_IIR_Denom(3, :), HP_IIR_Denom(3, :)); % Coefficient convolution, HP
[HP_IIR_Num_Best_2_Butter, HP_IIR_Denom_Best_2_Butter] = butter(N(2), Wn(2), 'high'); % ButterWorth, HP

% Creating frequency response :
[h_LP_IIR_Best, w_LP_IIR_Best] = freqz(LP_IIR_Num_Best, LP_IIR_Denom_Best_2, 'whole', n);
[h_LP_IIR_Best_Butter, w_LP_IIR_Best_Butter] = freqz(LP_IIR_Num_Best_2_Butter, LP_IIR_Denom_Best_2_Butter, 'whole', n);
[h_HP_IIR_Best, w_HP_IIR_Best] = freqz(HP_IIR_Num_Best, HP_IIR_Denom_Best_2, 'whole', n);
[h_HP_IIR_Best_Butter, w_HP_IIR_Best_Butter] = freqz(HP_IIR_Num_Best_2_Butter, HP_IIR_Denom_Best_2_Butter, 'whole', n);

% Plotting :
% Absolute Value LP IIR Second Order vs. Butter
figure
subplot(2,1,1)
hold on
plot(w_LP_IIR_Best/pi, 20*log10(abs(h_LP_IIR_Best/max(h_LP_IIR_Best))),'color','r');
plot(w_LP_IIR_Best_Butter/pi, 20*log10(abs(h_LP_IIR_Best_Butter/max(h_LP_IIR_Best_Butter))),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Absolute Value(dB)','color','b');
title(sprintf('Absolute Value Of IIR LP Second Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');
% Phase LP IIR First Order vs. Butter
subplot(2,1,2)
hold on
plot(w_LP_IIR_Best/pi,angle(h_LP_IIR_Best),'color','r');
plot(w_LP_IIR_Best_Butter/pi,angle(h_LP_IIR_Best_Butter),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Phase','color','b');
title(sprintf('Phase Of IIR LP Second Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');
% Absolute Value HP IIR Second Order vs. Butter
figure
subplot(2,1,1)
hold on
plot(w_HP_IIR_Best/pi, 20*log10(abs(h_HP_IIR_Best/max(h_HP_IIR_Best))),'color','r');
plot(w_HP_IIR_Best_Butter/pi, 20*log10(abs(h_HP_IIR_Best_Butter/max(h_HP_IIR_Best_Butter))),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Absolute Value(dB)','color','b');
title(sprintf('Absolute Value Of IIR HP Second Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');
% Phase HP IIR Second Order vs. Butter
subplot(2,1,2)
hold on
plot(w_HP_IIR_Best/pi,angle(h_HP_IIR_Best),'color','r');
plot(w_HP_IIR_Best_Butter/pi,angle(h_HP_IIR_Best_Butter),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Phase','color','b');
title(sprintf('Phsae Of IIR HP Second Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');


% N = 4 :
Wn = [0.01499 0.9855]; % LP, HP 4th Order respectively

LP_IIR_Denom_Best_4 = conv(LP_IIR_Denom_Best_2, LP_IIR_Denom_Best_2); % Coefficient convolution, LP
[LP_IIR_Num_Best_4_Butter, LP_IIR_Denom_Best_4_Butter] = butter(N(3), Wn(1)); % ButterWorth, LP
HP_IIR_Denom_Best_4 = conv(HP_IIR_Denom_Best_2, HP_IIR_Denom_Best_2); % Coefficient convolution, HP
[HP_IIR_Num_Best_4_Butter, HP_IIR_Denom_Best_4_Butter] = butter(N(3), Wn(2), 'high'); % ButterWorth, HP

% Creating frequency response :
[h_LP_IIR_Best, w_LP_IIR_Best] = freqz(LP_IIR_Num_Best, LP_IIR_Denom_Best_4, 'whole', n);
[h_LP_IIR_Best_Butter, w_LP_IIR_Best_Butter] = freqz(LP_IIR_Num_Best_4_Butter, LP_IIR_Denom_Best_4_Butter, 'whole', n);
[h_HP_IIR_Best, w_HP_IIR_Best] = freqz(HP_IIR_Num_Best, HP_IIR_Denom_Best_4, 'whole', n);
[h_HP_IIR_Best_Butter, w_HP_IIR_Best_Butter] = freqz(HP_IIR_Num_Best_4_Butter, HP_IIR_Denom_Best_4_Butter, 'whole', n);

% Plotting :
% Absolute Value LP IIR 4th Order vs. Butter
figure
subplot(2,1,1)
hold on
plot(w_LP_IIR_Best/pi, 20*log10(abs(h_LP_IIR_Best/max(h_LP_IIR_Best))),'color','r');
plot(w_LP_IIR_Best_Butter/pi, 20*log10(abs(h_LP_IIR_Best_Butter)),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Absolute Value(dB)','color','b');
title(sprintf('Absolute Value Of IIR LP 4th Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');
% Phase LP IIR 4th Order vs. Butter
subplot(2,1,2)
hold on
plot(w_LP_IIR_Best/pi,angle(h_LP_IIR_Best),'color','r');
plot(w_LP_IIR_Best_Butter/pi,angle(h_LP_IIR_Best_Butter),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Phase','color','b');
title(sprintf('Phase Of IIR LP 4th Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');

% Absolute Value HP IIR 4th Order vs. Butter
figure
subplot(2,1,1)
hold on
plot(w_HP_IIR_Best/pi, 20*log10(abs(h_HP_IIR_Best/max(h_HP_IIR_Best))),'color','r');
plot(w_HP_IIR_Best_Butter/pi, 20*log10(abs(h_HP_IIR_Best_Butter/max(h_HP_IIR_Best_Butter))),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Absolute Value(dB)','color','b');
title(sprintf('Absolute Value Of IIR HP 4th Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');
% Phase HP IIR 4th Order vs. Butter
subplot(2,1,2)
hold on
plot(w_HP_IIR_Best/pi,angle(h_HP_IIR_Best),'color','r');
plot(w_HP_IIR_Best_Butter/pi,angle(h_HP_IIR_Best_Butter),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Phase','color','b');
title(sprintf('Phase Of IIR HP 4th Order VS Butter Filter Frequency Response (r0=%d)',0.9),'color','r','fontsize',8);
hold off
legend('IIR','Butterworth','location','best');
%% Section 1.9
% included in report
%% Section 1.10
clc
close all

% Z = (0.7071678 + j0.7071678) & P = (0.7 + j0.7) and their complex
% conjugates :

Notch_Num = [1 -1.4142 1];
Notch_Denom = [1 -1.4 0.98];

[hNotch, wNotch] = freqz(Notch_Num, Notch_Denom, 'whole', n);

figure
subplot(2,1,1)
plot(wNotch/pi, 20*log10(abs(hNotch)),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Absolute Value(dB)','color','b');
title('Absolute Value Of Notch Filter Frequency Response','color','r');

subplot(2,1,2)
plot(wNotch/pi, angle(hNotch),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Phase','color','b');
title('Phase Of Notch Filter Frequency Response','color','r');

% Effect Of Changing Position Of Poles Or Zeros

% Getting Further
% Z = (0.7071678 + j0.7071678) & P = (0.65 + j0.65) and their complex
% conjugates :

Notch_Num = [1 -1.4142 1];
Notch_Denom = [1 -1.3 0.845];

[hNotch, wNotch] = freqz(Notch_Num, Notch_Denom, 'whole', n);

figure
subplot(2,1,1)
plot(wNotch/pi, 20*log10(abs(hNotch)),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Absolute Value(dB)','color','b');
title('Absolute Value Of Notch Filter Frequency Response(Zeros Are Put Further From Poles)','color','r','fontsize',9);

subplot(2,1,2)
plot(wNotch/pi, angle(hNotch),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Phase','color','b');
title('Phase Of Notch Filter Frequency Response(Zeros Are Put Further From Poles)','color','r','fontsize',9);

% Getting Closer
% Z = (0.7071678 + j0.7071678) & P = (0.707106 + j0.707106) and their complex
% conjugates :

Notch_Num = [1 -1.414213562 1];
Notch_Denom = [1 -1.414212 0.9999977905];

[hNotch, wNotch] = freqz(Notch_Num, Notch_Denom, 'whole', n);

figure
subplot(2,1,1)
plot(wNotch/pi,20*log10(abs(hNotch)),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Absolute Value(dB)','color','b');
title('Absolute Value Of Notch Filter Frequency Response(Zeros Are Put Further From Poles)','color','r','fontsize',9);

subplot(2,1,2)
plot(wNotch/pi, angle(hNotch),'color','b');
xlabel('Normalized by pi Angular frequency','color','b');
ylabel('Phase','color','b');
title('Phase Of Notch Filter Frequency Response(Zeros Are Put Further From Poles)','color','r','fontsize',9);

%% Part 2 : Modifying Audio Signal By Desired Filters
%% Section 2.1
clc
close all

[Sound_Edited, fs] = audioread('sound_edited.wav');

% Fourier Transform :
[Sound_Edited_FFT, f] = FFT(Sound_Edited, fs,1);

%% Section 2.2
clc
% close all

% Notch filter parameters :
w0_Notch_5k = 5e3/(fs/2);
bw_Notch_5k = w0_Notch_5k/500;
w0_Notch_10k = 1e4/(fs/2);
bw_Notch_10k = w0_Notch_10k/1000;

% Designing notch filter :
[Num_Notch_5k, Denom_Notch_5k] = iirnotch(w0_Notch_5k, bw_Notch_5k);
[Num_Notch_10k, Denom_Notch_10k] = iirnotch(w0_Notch_10k, bw_Notch_10k);

% Filtering :
Sound_Edited_Notched = filter(conv(Num_Notch_5k, Num_Notch_10k), conv(Denom_Notch_5k, Denom_Notch_10k), Sound_Edited);

% Fourier transform plotting :
[Sound_Edited_Notched_FFT, f_Notched] = FFT(Sound_Edited_Notched, fs,1);
title('Edited sound signal after notches on 5kHz & 10kHz');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

% Playing the sound without continuous beeeeeeep :
player = audioplayer(Sound_Edited_Notched, fs);

%% Section 2.3
clc
% close all

fmax = 4e3; % Hz
wn_max = fmax/(fs/2);

% Designing lowpass filter :
[Num_Noise_Removed, Denom_Noise_Removed] = butter(8, wn_max);

% Filtering :
Sound_Edited_Noise_Removed = filter(Num_Noise_Removed, Denom_Noise_Removed, Sound_Edited_Notched);

% Fourier transform plotting :
Sound_Edited_Noise_Removed_FFT = FFT(Sound_Edited_Noise_Removed, fs,1);
title('Edited Sound Signal After Lowpass Filter');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

% Playing the sound with reduced noise :
player = audioplayer(Sound_Edited_Noise_Removed, fs);

%% Section 2.4
clc
close all

% Filter coefficients :
Num_Inverse_Delay = 1;
Denom_Inverse_Delay = [1 zeros(1, 3*fs-1) 0.3864485981];

% Filtering :
Sound_Edited_Echo = filter(Num_Inverse_Delay, Denom_Inverse_Delay, Sound_Edited_Noise_Removed);

% Fourier transform plotting :
Sound_Edited_Echo_FFT = FFT(Sound_Edited_Echo, fs,1);
title('Edited Sound Signal After Removing Echo');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

% Playing the sound without echo :
player = audioplayer(Sound_Edited_Echo, fs);

%% Part 3 : Changing Sampling Rate
%% Section 3.1 , 3.2
% included in report
%% Section 3.3
clc
close all

[Sound, fs] = audioread('sound.wav');

L = [2;4;8;16;32];
Fs = fs * (L.^(-1));

% Down sampling :
Sound_DS_2 = downsample(Sound, L(1));
Sound_DS_4 = downsample(Sound, L(2));
Sound_DS_8 = downsample(Sound, L(3));
Sound_DS_16 = downsample(Sound, L(4));
Sound_DS_32 = downsample(Sound, L(5));

% Fourier transform plotting :
FFT(Sound, fs, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Original Signal','color','r');
Sound_DS_2_FFT = FFT(Sound_DS_2, 2*fs, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled Signal By Scale 2','color','r');
Sound_DS_4_FFT = FFT(Sound_DS_4, 4*fs, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled Signal By Scale 4','color','r');
Sound_DS_8_FFT = FFT(Sound_DS_8, 8*fs, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled Signal By Scale 8','color','r');
Sound_DS_16_FFT = FFT(Sound_DS_16, 16*fs, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled Signal By Scale 16','color','r');
Sound_DS_32_FFT = FFT(Sound_DS_32, 32*fs, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled Signal By Scale 32','color','r');

% Playing the sound with different samplings :
p2ds = audioplayer(Sound_DS_2, Fs(1));
p4ds = audioplayer(Sound_DS_4, Fs(2));
p8ds = audioplayer(Sound_DS_8, Fs(3));
p16ds = audioplayer(Sound_DS_16, Fs(4));
p32ds = audioplayer(Sound_DS_32, Fs(5));


%% Section 3.4
clc
close all
[~,fs]=audioread('Sound.wav');
% Up sampling :
Sound_US_2 = upsample(Sound_DS_2, L(1));
Sound_US_4 = upsample(Sound_DS_4, L(2));
Sound_US_8 = upsample(Sound_DS_8, L(3));
Sound_US_16 = upsample(Sound_DS_16, L(4));
Sound_US_32 = upsample(Sound_DS_32, L(5));

% Fourier transform plotting :
Sound_US_2_FFT = FFT(Sound_US_2, fs/2, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled & Upsampled Signal By Scale 2','color','r');
Sound_US_4_FFT = FFT(Sound_US_4, fs/4, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled & Upsampled Signal By Scale 4','color','r');
Sound_US_8_FFT = FFT(Sound_US_8, fs/8, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled & Upsampled Signal By Scale 8','color','r');
Sound_US_16_FFT = FFT(Sound_US_16, fs/16, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled & Upsampled Signal By Scale 16','color','r');
Sound_US_32_FFT = FFT(Sound_US_32, fs/32, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Downsampled & Upsampled Signal By Scale 32','color','r');


% Playing the sound with different samplings :
p = audioplayer(Sound,fs);
p2us = audioplayer(Sound_US_2, fs);
p4us = audioplayer(Sound_US_4, fs);
p8us = audioplayer(Sound_US_8, fs);
p16us = audioplayer(Sound_US_16, fs);
p32us = audioplayer(32*Sound_US_32, fs);


%% Section 3.5
clc
close all

fmax = 4e3; % Hz
wn_max = fmax/(fs/2);

% Designing lowpass filter :
[Num_Filtered, Denom_Filtered] = butter(6, wn_max);

% Filtering :
Sound_US_2_Filtered = filter(2*Num_Filtered, Denom_Filtered, Sound_US_2);
Sound_US_4_Filtered = filter(4*Num_Filtered, Denom_Filtered, Sound_US_4);
Sound_US_8_Filtered = filter(8*Num_Filtered, Denom_Filtered, Sound_US_8);
Sound_US_16_Filtered = filter(16*Num_Filtered, Denom_Filtered, Sound_US_16);
Sound_US_32_Filtered = filter(32*Num_Filtered, Denom_Filtered, Sound_US_32);

% Fourier transform plotting :
FFT(Sound, fs,1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Original Signal','color','r');
Sound_US_2_Filtered_FFT = FFT(Sound_US_2_Filtered, fs,1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Reconstructed Signal(DS Factor=2)','color','r');
Sound_US_4_Filtered_FFT = FFT(Sound_US_4_Filtered, fs,1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Reconstructed Signal(DS Factor=4)','color','r');
Sound_US_8_Filtered_FFT = FFT(Sound_US_8_Filtered, fs,1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Reconstructed Signal(DS Factor=8)','color','r');
Sound_US_16_Filtered_FFT = FFT(Sound_US_16_Filtered, fs,1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Reconstructed Signal(DS Factor=16)','color','r');
Sound_US_32_Filtered_FFT = FFT(Sound_US_32_Filtered, fs,1);
xlabel('Frequency','color','b');
ylabel('Absolute Value Of Fourier Transform','color','b');
title('Fourier Transform Of Reconstructed Signal(DS Factor=32)','color','r');

% Playing the sound with reduced noise :
pfinal2 = audioplayer(Sound_US_2_Filtered, fs);
pfinal4 = audioplayer(Sound_US_4_Filtered, fs);
pfinal8 = audioplayer(Sound_US_8_Filtered, fs);
pfinal16 = audioplayer(Sound_US_16_Filtered, fs);
pfinal32 = audioplayer(Sound_US_32_Filtered, fs);

%% Section 3.6
clc
close all

% ZeroHold up sampling :
Sound_ZeroHold_2 = zeroHold(Sound_DS_2, L(1));
Sound_ZeroHold_4 = zeroHold(Sound_DS_4, L(2));
Sound_ZeroHold_8 = zeroHold(Sound_DS_8, L(3));
Sound_ZeroHold_16 = zeroHold(Sound_DS_16, L(4));
Sound_ZeroHold_32 = zeroHold(Sound_DS_32, L(5));

% OneHold up sampling :
Sound_OneHold_2 = oneHold(Sound_DS_2, L(1));
Sound_OneHold_4 = oneHold(Sound_DS_4, L(2));
Sound_OneHold_8 = oneHold(Sound_DS_8, L(3));
Sound_OneHold_16 = oneHold(Sound_DS_16, L(4));
Sound_OneHold_32 = oneHold(Sound_DS_32, L(5));

% Fourier transform plotting :
FFT(Sound, fs,1);
title('Original Sound Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

% ZeroHold :
FFT(Sound_ZeroHold_2, fs,1);
title('Zero Hold Reconstructed Signal(L = 2) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_ZeroHold_4, fs,1);
title('Zero Hold Reconstructed Signal(L = 4) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_ZeroHold_8, fs,1);
title('Zero Hold Reconstructed Signal(L = 8) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_ZeroHold_16, fs,1);
title('Zero Hold Reconstructed Signal(L = 16) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_ZeroHold_32, fs,1);
title('Zero Hold Reconstructed Signal(L = 32) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

% OneHold :
FFT(Sound_OneHold_2, fs,1);
title('One Hold Reconstructed Signal(L = 2) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_OneHold_4, fs,1);
title('One Hold Reconstructed Signal(L = 4) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_OneHold_8, fs,1);
title('One Hold Reconstructed Signal(L = 8) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_OneHold_16, fs,1);
title('One Hold Reconstructed Signal(L = 16) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_OneHold_32, fs,1);
title('One Hold Reconstructed Signal(L = 32) Fourier Transform');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');


% Playing the sound with zerohold and onehold reconstruction :
player2Zero = audioplayer(Sound_ZeroHold_2, fs);
player4Zero = audioplayer(Sound_ZeroHold_4, fs);
player8Zero = audioplayer(Sound_ZeroHold_8, fs);
player16Zero = audioplayer(Sound_ZeroHold_16, fs);
player32Zero = audioplayer(Sound_ZeroHold_32, fs);

player2One = audioplayer(Sound_OneHold_2, fs);
player4One = audioplayer(Sound_OneHold_4, fs);
player8One = audioplayer(Sound_OneHold_8, fs);
player16One = audioplayer(Sound_OneHold_16, fs);
player32One = audioplayer(Sound_OneHold_32, fs);
%% Section 3.7
clc
close all

fmax = 4e3; % Hz
wn_max = fmax/(fs/2);

% Designing lowpass filter :
[Num_Filtered, Denom_Filtered] = butter(4, wn_max);

% Filtering :
Sound_ZeroHold_2_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_ZeroHold_2);
Sound_ZeroHold_4_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_ZeroHold_4);
Sound_ZeroHold_8_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_ZeroHold_8);
Sound_ZeroHold_16_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_ZeroHold_16);
Sound_ZeroHold_32_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_ZeroHold_32);

Sound_OneHold_2_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_OneHold_2);
Sound_OneHold_4_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_OneHold_4);
Sound_OneHold_8_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_OneHold_8);
Sound_OneHold_16_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_OneHold_16);
Sound_OneHold_32_Filtered = filter(Num_Filtered, Denom_Filtered, Sound_OneHold_32);

% Fourier transform plotting :
% ZeroHold :
FFT(Sound_ZeroHold_2_Filtered, fs,1);
title('Zero Hold Reconstructed Signal(L = 2) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');
FFT(Sound_ZeroHold_4_Filtered, fs,1);
title('Zero Hold Reconstructed Signal(L = 4) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_ZeroHold_8_Filtered, fs,1);
title('Zero Hold Reconstructed Signal(L = 8) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_ZeroHold_16_Filtered, fs,1);
title('Zero Hold Reconstructed Signal(L = 16) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_ZeroHold_32_Filtered, fs,1);
title('Zero Hold Reconstructed Signal(L = 32) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

% OneHold :
FFT(Sound_OneHold_2_Filtered, fs,1);
title('One Hold Reconstructed Signal(L = 2) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_OneHold_4_Filtered, fs,1);
title('One Hold Reconstructed Signal(L = 4) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_OneHold_8_Filtered, fs,1);
title('One Hold Reconstructed Signal(L = 8) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_OneHold_16_Filtered, fs,1);
title('One Hold Reconstructed Signal(L = 16) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

FFT(Sound_OneHold_32_Filtered, fs,1);
title('One Hold Reconstructed Signal(L = 32) Fourier Transform After Lowpass Filter', 'color', 'r');
xlabel('Frequency', 'color', 'b');
ylabel('Absolute Value Of Fourier Transform', 'color', 'b');

% Playing the sound with zerohold and onehold reconstruction & filtered :
player2Zero = audioplayer(Sound_ZeroHold_2_Filtered, fs);
player4Zero = audioplayer(Sound_ZeroHold_4_Filtered, fs);
player8Zero = audioplayer(Sound_ZeroHold_8_Filtered, fs);
player16Zero = audioplayer(Sound_ZeroHold_16_Filtered, fs);
player32Zero = audioplayer(Sound_ZeroHold_32_Filtered, fs);

player2One = audioplayer(Sound_OneHold_2_Filtered, fs);
player4One = audioplayer(Sound_OneHold_4_Filtered, fs);
player8One = audioplayer(Sound_OneHold_8_Filtered, fs);
player16One = audioplayer(Sound_OneHold_16_Filtered, fs);
player32One = audioplayer(Sound_OneHold_32_Filtered, fs);
%% Part 4 : Interpolating
%% Section 4.1
clc
close all

f0 = 1e3; % 1kHz
fs_high = 5e4; % 50kHz

t = (0:1/fs_high:0.1)'; % [0, 100ms] interval
y = sin(2*pi*f0*t); % Sinusoid with frequency f0

% Plotting sinusoid with frequency f0 on [10ms, 13ms] :
figure
plot(t,y);
xlim([0.010,0.013]);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Created By Sampling Rate 50KHz','color','r');


%% Section 4.2
clc
close all

t0 = (0.01:1/fs_high:0.013)'; % [10ms, 13ms] interval for plotting
y0 = sin(2*pi*f0*t0);

fs = 1e4; % 10kHz, Sampling frequency
N_DS = fs_high/fs; % = 5
N_US = 5;

y_DS = downsample(y, N_DS);
y_US = upsample(y_DS, N_US);

figure
hold on
plot(t0, y0);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Downsampled By Factor 5 & Then Upsampled By Factor 5','color','r');
stem(t0, y_US(find(t == 0.01) : find(t == 0.01)+length(t0)-1));
hold off

%% Section 4.3
clc
close all

% Fourier transform plotting :
y_FFT = FFT(y, fs_high, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value','color','b');
title('Fourier Transform Of The Sinusoid','color','r');
y0_FFT = FFT(y0, fs_high, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value','color','b');
title('Fourier Transform Of The Sinusoid On The Specified Interval','color','r');
y_DS_FFT = FFT(y_DS, fs_high, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value','color','b');
title('Fourier Transform Of The Downsampled Sinusoid','color','r');
y_US_FFT = FFT(y_US, fs_high, 1);
xlabel('Frequency','color','b');
ylabel('Absolute Value','color','b');
title('Fourier Transform Of The Upsampled Sinusoid','color','r');

%% Section 4.4
clc
close all


% Designing filter coefficients :
Wn = (fs/2)/(fs_high/2);
[Num, Denom] = butter(6, Wn);

% Reconstruction :
y_US_Reconstruct = filter(5 * Num, Denom, y_US);

% Plotting :
figure
hold on
plot(t0, y0,'color','b');
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Reconstructed','color','r');
stem(t0, y_US_Reconstruct(find(t == 0.01) : find(t == 0.01)+length(t0)-1),'color','r');
hold off

%% Section 4.5
clc
close all

fs = [0.5; 1.2; 2; 5] * 1e3; % Sampling frequency
N_DS = ceil(fs_high * (fs.^(-1)));
N_US = N_DS;

% Down sampling & Up sampling :
y_DS_0_5 = downsample(y, N_DS(1));
y_US_0_5 = upsample(y_DS_0_5, N_US(1));

y_DS_1_2 = downsample(y, N_DS(2));
y_US_1_2 = upsample(y_DS_1_2, N_US(2));

y_DS_2 = downsample(y, N_DS(3));
y_US_2 = upsample(y_DS_2, N_US(3));

y_DS_5 = downsample(y, N_DS(4));
y_US_5 = upsample(y_DS_5, N_US(4));

% Plotting :
% 0.5kHz sampling :
figure
hold on
plot(t0, y0);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Downsampled & Then Upsampled(fs=0.5KHz)','color','r');
stem(t0, y_US_0_5(find(t == 0.01) : find(t == 0.01)+length(t0)-1));
hold off

% 1.2kHz sampling :
figure
hold on
plot(t0, y0);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Downsampled & Then Upsampled(fs=1.2KHz)','color','r');
stem(t0, y_US_1_2(find(t == 0.01) : find(t == 0.01)+length(t0)-1));
hold off

% 2kHz sampling :
figure
hold on
plot(t0, y0);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Downsampled & Then Upsampled(fs=2KHz)','color','r');
stem(t0, y_US_2(find(t == 0.01) : find(t == 0.01)+length(t0)-1));
hold off

% 5kHz sampling :
figure
hold on
plot(t0, y0);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Downsampled & Then Upsampled(fs=5KHz)','color','r');
stem(t0, y_US_5(find(t == 0.01) : find(t == 0.01)+length(t0)-1));
hold off

% Designing filter coefficients :
Wn = (fs/2)/(fs_high/2);
filter_Order = 6;
[Num_0_5, Denom_0_5] = butter(filter_Order, Wn(1));
[Num_1_2, Denom_1_2] = butter(filter_Order, Wn(2));
[Num_2, Denom_2] = butter(filter_Order, Wn(3));
[Num_5, Denom_5] = butter(filter_Order, Wn(4));

% Reconstruction :
y_US_Reconstruct_0_5 = filter(N_US(1) * Num_0_5, Denom_0_5, y_US_0_5);
y_US_Reconstruct_1_2 = filter(N_US(2) * Num_1_2, Denom_1_2, y_US_1_2);
y_US_Reconstruct_2 = filter(N_US(3) * Num_2, Denom_2, y_US_2);
y_US_Reconstruct_5 = filter(N_US(4) * Num_5, Denom_5, y_US_5);

% Plotting :
% 0.5kHz sampling :
figure
hold on
plot(t0, y0);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Reconstructed(fs=0.5KHz)','color','r');
stem(t0, y_US_Reconstruct_0_5(find(t == 0.01) : find(t == 0.01)+length(t0)-1));
hold off

% 1.2kHz sampling :
figure
hold on
plot(t0, y0);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Reconstructed(fs=1.2KHz)','color','r');
stem(t0, y_US_Reconstruct_1_2(find(t == 0.01) : find(t == 0.01)+length(t0)-1));
hold off

% 2kHz sampling :
figure
hold on
plot(t0, y0);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Reconstructed(fs=2KHz)','color','r');
stem(t0, y_US_Reconstruct_2(find(t == 0.01) : find(t == 0.01)+length(t0)-1));
hold off

% 5kHz sampling :
figure
hold on
plot(t0, y0);
xlabel('Time','color','b');
ylabel('Sin(2*pi*1000*t)','color','b');
title('Sinusoid Reconstructed(fs=5KHz)','color','r');
stem(t0, y_US_Reconstruct_5(find(t == 0.01) : find(t == 0.01)+length(t0)-1));
hold off