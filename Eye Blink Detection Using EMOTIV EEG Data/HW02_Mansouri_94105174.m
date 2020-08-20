%% Part 1 - General Study of The Data
%% section 1 - Places of probes
%included in report
%% section 2 - Mean,standard deviation,min & max of channels
% Defining the object to access the .mat file
m = matfile('Train') ;
% Getting the variables and properties of the .mat file
whos('-file','Train.mat') ;
who('-file','Train.mat') ;
data = m.EEG(:,1:14);% defining a new matrix consisting of the first 14 columns of Train.mat which are cahnnels data
mean_val = mean(data);%calculating mean value of each channel
std_dev = std(data);%calculating standard deviation value of each channel
min_val = min(data);%calculating minimum value of each channel
max_val = max(data);%calculating maximum value of each channel
% Using this command we would have more precision(15 digits)
format LONG;
% Displaying the variable and its value on each channel
disp('mean_val');   disp(mean_val);
disp('std_dev');  disp(std_dev)
disp('min_val');  disp(min_val);
disp('max_val');  disp(max_val);
%% section 3 - Plotting the graph of signals versus time
for i = 1 : 14
    figure();
    plot(m.EEG(:,16),m.EEG(:,i));
    grid on;% making plots easier to read
    title(strcat('Channel',sprintf(':  %d' , i)) , 'color' , 'r');%string concatenation used for having variable i in title
    xlabel('Time','color','b');
    ylabel('Signal(Micro Vlots)','color','b');
    xlim([0,117]);
end
%% section 4 - EEG normal range of variation
%included in report
%% section 5 - Replacing outlier data
save data1.mat;% New .mat file created. It is used so we don't need to overwrite the original data of Train.mat file
load('data1');%loading data into memory
load('Train');%loading Train.mat data into memory
data1 = EEG(:,:);% The values of the EEG data are assigned to new file's elements.(reason explained above and in report.)
% Here the first stage of data correction is done(details in report)
for j = 1 : 4
    [min_val , min_index] = min(data1(:,1:14));
    [max_val , max_index] = max(data1(:,1:14));
    for i = 1 : 14
        %Replacing minimums
        data1(min_index(i),i) = data1((min_index(i))-1,i) + ((data1((min_index(i))+1,i))-(data1((min_index(i))-1,i))).*(data1((min_index(i)),16) - data1((min_index(i))-1,16))/(data1((min_index(i))+1,16) - data1((min_index(i))-1,16));
        %Replacing maximums
        data1(max_index(i),i) = data1((max_index(i))-1,i) + ((data1((max_index(i))+1,i))-(data1((max_index(i))-1,i))).*(data1((max_index(i)),16) - data1((max_index(i))-1,16))/(data1((max_index(i))+1,16) - data1((max_index(i))-1,16));
    end
    save('data1');%saving the changes each loop
end
% Continuing Correction of Channel 13(as discussed in report, this channel
% contains more outlier data which makes us to put its data in more loops
% for correction
for a = 1 : 35
    [min_val , min_index] = min(data1(:,1:14));%minimum values and thier indexes are again found
    [max_val , max_index] = max(data1(:,1:14));%maximum values and thier indexes are again found
    %Replacing minimums
    data1(min_index(13),13) = data1((min_index(13))-1,13) + ((data1((min_index(13))+1,13))-(data1((min_index(13))-1,13))).*(data1((min_index(13)),16) - data1((min_index(13))-1,16))/(data1((min_index(13))+1,16) - data1((min_index(13))-1,16));
    %Replacing maximums
    data1(max_index(13),13) = data1((max_index(13))-1,13) + ((data1((max_index(13))+1,13))-(data1((max_index(13))-1,13))).*(data1((max_index(13)),16) - data1((max_index(13))-1,16))/(data1((max_index(13))+1,16) - data1((max_index(13))-1,16));
    save('data1');%saving the changes each loop
end
% Here the second stage of correction is done
% Channel 1 Replacing by adjacent correct values
data1(11508:11850,1) = data1(11165:11507,1) ;
data1(10386,1) = data1(10385,1) ;
data1(898,1) = data1(897,1) ;
data1(13179,1) = data1(13180,1) ;
save('data1');
% Channel 4 Replacing by adjacent correct values
data1(10386:10586,4) = data1(10185:10385,4) ;
data1(10587:10788,4) = data1(10789:10990,4) ;
data1(13179,4) = data1(13178,4) ;
data1(898,4) = data1(897,4) ;
data1(11509,4) = data1(11508,4) ;
save('data1');
% Channel 6 Replacing by adjacent correct values
data1(898:904,6) = data1(891:897,6) ;
data1(10386,6) = data1(10385,6) ;
data1(905,6) = data1(904,6) ;
data1(906:1078,6) = data1(733:905,6) ;
data1(1079:1252,6) = data1(1253:1426,6) ;
data1(13179,6) = data1(13178,6);
data1(11509,6) = data1(11510,6) ;
save('data1');
% Channel 7 Replacing by adjacent correct values
data1(10386:10606,7) = data1(10165:10385,7) ;
data1(10607:10826,7) = data1(10827:11046,7) ;
data1(11509,7) = data1(11510,7) ;
data1(13179,7) = data1(13178,7);
data1(898,7) = data1(899,7) ;
save('data1');
% Channel 9 Replacing by adjacent correct values
data1(11509:11689,9) = data1(11328:11508,9) ;
data1(11690:11779,9) = data1(11780:11869,9) ;
% Channel 13 Replacing by adjacent correct values
data1(11509:11690,13) = data1(11691:11872,13) ;
% Channel 14 Replacing by adjacent correct values
data1(898:1098,14) = data1(697:897,14) ;
data1(1099:1298,14) = data1(1299:1498,14) ;
data1(11509,14) = data1(11510,14) ;
data1(13179,14) = data1(13178,14);
data1(10386:10516,14) = data1(10255:10385,14) ;
data1(10517:10571,14) = data1(10572:10626,14) ;

save('data1');% saving all changes
%% section 6 - Plotting edited signals versus time
for i = 1 : 14
    figure();
    plot(data1(:,16),data1(:,i),data1(:,16),50*data1(:,15));
    grid on;
    title(strcat('Edited Channel',sprintf(':  %d' , i)) , 'color' , 'r');
    xlabel('Time','color','b');
    ylabel('Signal(Micro Vlots)','color','b');
    xlim([0,117]);
end
%% section 7 - Plotting Right-side signals versus their analog of the left-side
for i = 1 : 7
    figure();
    plot(data1(:,i),data1(:,15-i),'.');
    grid on;
    title('Two Side Of Brain Signal Graph' , 'color' , 'r');
    xlabel(strcat('Channel',sprintf(':  %d' , i)),'color','b');
    ylabel(strcat('Channel',sprintf(':  %d' , 15-i)),'color','b');
    xlim([0,117]);
end
%% section 8 - Finding the pairwise correlation coefficient of the channels
[rho , pval] = corr(data1(:,1:14));% column 15 & 16 are excluded since they're not related to channels
%% section 9 - boxplot
figure();
boxplot(data1(:,1:14));
title('Box Plot Of Each Channel''s Data' , 'color' , 'r');
xlabel('Channel','color','b');
ylabel('Value','color','b');
%% Part 2 - Property Of Blinks
%% section 1 - Finding The Times When Eye Has Blinked(Either Opened Or Closed)
blink_time = zeros(1,23);% The array of times which eye has blinked is created
time_index = zeros(1,23);% The array of indexes of times which eye has blinked is created
time_index_close = zeros(1,12);% The array of indexes of times which eye has closed is created
time_index_open = zeros(1,11);% The array of indexes of times which eye has opened is created
j=1;
m=1;
n=1;
for i = 1 : 14978
   if data1( i , 15 ) ~= data1( i+1 , 15 )%indicating change in the closed/open situation of the eye
       blink_time(j) = data1( i , 16 );
       time_index(j) = i ;
       if data1( i , 15 ) - data1( i+1 , 15 ) == -1%at first, data eye is open and at the next it is closed
           time_index_close(m) = i ; 
           m = m + 1 ;
       end
       if data1( i , 15 ) - data1( i+1 , 15 ) == 1%at first, data eye is closed and at the next it is open
           time_index_open(n) = i ;
           n = n + 1 ;
       end
       j = j + 1;
   end 
end
%% section 2 - Analysing Range Of Time When Changes Of Signals Happen
% included in report
    %% section 3.a - Creating New Signals From A Range About "Closing" Eye Instances
a = 126 ;%number of data included in the region(detail of choosing included in report)
New_Signal_Close = zeros(a,1,12,14);%four dimensional array created
for i = 1 : 14
   for j = 1 : 11
       New_Signal_Close(1:a,1,j,i) = data1(time_index_close(j) - 66 : time_index_close(j) + 59 , i ) ;%signal of each closing in each channel is being created
   end       
end
for i = 1 : 14
    New_Signal_Close(1:a,1,12,i) = data1(time_index_close(12) - 66 , end ) ;%last closing didn't have enough data at its end so we derive its signal individually
end

y_Close = zeros(a,1,14);%signal of closing is averaged over each channel
for m = 1 : 14
    y_Close(1:a,1,m) = (New_Signal_Close(1:a,1,1,m)+New_Signal_Close(1:a,1,2,m)+New_Signal_Close(1:a,1,3,m)+New_Signal_Close(1:a,1,4,m)+New_Signal_Close(1:a,1,5,m)+New_Signal_Close(1:a,1,6,m)+New_Signal_Close(1:a,1,7,m)+New_Signal_Close(1:a,1,8,m)+New_Signal_Close(1:a,1,9,m)+New_Signal_Close(1:a,1,10,m)+New_Signal_Close(1:a,1,11,m)+New_Signal_Close(1:a,1,12,m))/12 ;
end
%% section 3.b - Plotting y_ms With The Times Of Closing
figure();
delta_t = (128)^-1;%sampling frequency used for time
time =  delta_t : delta_t : a*delta_t ;
%plotting all channel's averaged signal of closing in the same figure
for m = 1 : 14 
    hold on;
    plot(time,y_Close(1:a,1,m));
    title('Y_m For Closing Signals' , 'color' , 'r');
    xlabel('Relative Time From Beginning Of Signal Radiation','color','b');
    ylabel('Amplitude','color','b');
end
%scaling time of closing according to range chosen before and indicating it
%on the figure
hold on
x = [66*delta_t,66*delta_t] ;
plot(x,[-40,120]);
%% section 4 - Doing The Same Opertaions For Eye's "Opening"
no = 102 ;%number of data included in the region(detail of choosing included in report)
New_Signal_Open = zeros(no,1,12,14);
for i = 1 : 14
   for j = 1 : 11
       New_Signal_Open(1:no,1,j,i) = data1(time_index_open(j) - 69 : time_index_open(j) + 32 , i ) ;%signal of each opening in each channel is being created
   end       
end
%here we don't have the same problem for the last opening as it was for
%closing
y_Open = zeros(no,1,14);
%signal of opening is averaged over each channel
for m = 1 : 14
    y_Open(1:no,1,m) = (New_Signal_Open(1:no,1,1,m)+New_Signal_Open(1:no,1,2,m)+New_Signal_Open(1:no,1,3,m)+New_Signal_Open(1:no,1,4,m)+New_Signal_Open(1:no,1,5,m)+New_Signal_Open(1:no,1,6,m)+New_Signal_Open(1:no,1,7,m)+New_Signal_Open(1:no,1,8,m)+New_Signal_Open(1:no,1,9,m)+New_Signal_Open(1:no,1,10,m)+New_Signal_Open(1:no,1,11,m))/11 ;
end
figure();
delta_t = (128)^-1;
time =  delta_t : delta_t : no*delta_t ;
for m = 1 : 14 
    hold on;
    plot(time,y_Open(1:no,1,m));
    title('Y_m For Opening Signals' , 'color' , 'r');
    xlabel('Relative Time From Beginning Of Signal Radiation','color','b');
    ylabel('Amplitude','color','b');
end
hold on
%scaling time of closing according to range chosen before and indicating it
%on the figure
x = [69*delta_t,69*delta_t] ;
plot(x,[-60,60]);
%% section 5 - Difference Between The Signal Of Blinking And The Blinking Itself
%included in report
%% section 6.a - Calculating Variance Of Previous Signals
var_close = zeros(14,1);
var_open = zeros(14,1);
for m = 1 : 14
    var_close(m) = (std(y_Close(:,1,m))).^2 ;
end
for n = 1 : 14
    var_open(n) = (std(y_Open(:,1,n))).^2 ;
end

%% section 6.b - Finding Probes Which Have The Highest Variances And Their Position On Head
%ncluded in report
%% section 6.c - Plotting The Final Signals & Their Mean Values
signal = [ 1 2 13 14 ];
mean_val_close = zeros(4,1);
mean_val_open  = zeros(4,1);
% mean values of signals are calculated below
for m = 1 : 4
    mean_val_close(m) = mean(y_Close(:,1,signal(m))) ;
end
for n = 1 : 4
    mean_val_open(n)  = mean(y_Open(:,1,signal(n))) ;
end
time =  delta_t : delta_t : a*delta_t ;
for i = 1 : 4
   figure();
   plot(time,y_Close(1:a,1,signal(i)));
   hold on;
   plot([delta_t , a*delta_t],[mean_val_close(i),mean_val_close(i)]);
   title(strcat('Channel(Closing Eye)',sprintf(':  %d' , signal(i))) , 'color' , 'r');
   xlabel('Relative Time From Beginning Of Signal Radiation','color','b');
   ylabel('Amplitude','color','b');
end
time =  delta_t : delta_t : no*delta_t ;
for i = 1 : 4 
    figure();
    plot(time,y_Open(1:no,1,signal(i)));
    hold on;
    plot([delta_t , no*delta_t],[mean_val_open(i),mean_val_open(i)]);
    title(strcat('Channel(Opening Eye)',sprintf(':  %d' , signal(i))) , 'color' , 'r');
    xlabel('Relative Time From Beginning Of Signal Radiation','color','b');
    ylabel('Amplitude','color','b');
end
%mean of closing and opening signals
y_Close_mean = (y_Close(1:a,1,signal(1))+y_Close(1:a,1,signal(2))+y_Close(1:a,1,signal(3))+y_Close(1:a,1,signal(4)))/4;
figure();
time =  delta_t : delta_t : a*delta_t ;
plot(time,y_Close_mean(1:a,1,:));
title('Closing Signal Average' , 'color' , 'r');
xlabel('Relative Time From Beginning Of Signal Radiation','color','b');
ylabel('Amplitude','color','b');
y_Open_mean = (y_Open(1:no,1,signal(1))+y_Open(1:no,1,signal(2))+y_Open(1:no,1,signal(3))+y_Open(1:no,1,signal(4)))/4;
figure();
time =  delta_t : delta_t : no*delta_t ;
plot(time,y_Open_mean(1:no,1,:));
title('Opening Signal Average' , 'color' , 'r');
xlabel('Relative Time From Beginning Of Signal Radiation','color','b');
ylabel('Amplitude','color','b');
%% Part 3 - Creating Estimation Model
%% section 1 - Cross Correlation With The Whole signal
no_cls = a ;%number of data included in the region(detail of choosing included in report)
length = 14979 ;
signal = [ 1 2 13 14 ];%only the more variant signals will be used

%for an specified method of correlation, we have to make the signals of
%closing and the whole signal the same size,here this extended matrixes are
%created
y_Close_extend1 = zeros(14979,1) ;
y_Close_extend2 = zeros(14979,1) ;
y_Close_extend13= zeros(14979,1) ;
y_Close_extend14= zeros(14979,1) ;
%here extension completes
y_Close_extend1(1:a,1) = y_Close(1:no_cls , 1 , 1) ;
y_Close_extend2(1:a,1) = y_Close(1:no_cls , 1 , 2) ;
y_Close_extend13(1:a,1) = y_Close(1:no_cls , 1 , 13) ;
y_Close_extend14(1:a,1) = y_Close(1:no_cls , 1 , 14) ;

%correlation values are caculated, this values are normalized according to
%attribute "coeff"
[r1] = xcorr(y_Close_extend1,data1(:,1),'coeff');
[r2] = xcorr(y_Close_extend2,data1(:,2),'coeff');
[r13] = xcorr(y_Close_extend13,data1(:,13),'coeff');
[r14,t] = xcorr(y_Close_extend14,data1(:,14),'coeff');% t for all above is the same
% Closing Cross Correlation

% because of an unknown problem my plot was mirrored with respect to the
% origin and i mirrored it again ---> (-x,-y)
figure();
plot(-t/128,-(r1).^3);% the 3rd power of regression is plotted over time to make meaningful peaks more visible
xlim([0,117]);
hold on
plot(data1(:,16),0.001.*data1(:,15));%closing and opening time data is scaled in order to understand the relation of
%closing/opening signal with the whole signal
xlim([0,117]);
title(strcat('Channel 1 (Closing Eye) Cross Correlation') , 'color' , 'r');

figure();
plot(-t/128,-(r2).^3);
xlim([0,117]);
hold on
plot(data1(:,16),(0.001)*data1(:,15));
xlim([0,117]);
title(strcat('Channel 2 (Closing Eye) Cross Correlation') , 'color' , 'r');

figure();
plot(-t/128,-(r13).^3);
xlim([0,117]);
hold on
plot(data1(:,16),(0.001)*data1(:,15));
xlim([0,117]);
title(strcat('Channel 13 (Closing Eye) Cross Correlation') , 'color' , 'r');

figure();
plot(-t/128,-(r14).^3);
xlim([0,117]);
hold on
plot(data1(:,16),(0.001)*data1(:,15));
xlim([0,117]);
title(strcat('Channel 14 (Closing Eye) Cross Correlation') , 'color' , 'r');

% Opening Cross Correlation
no_opn = no ;%number of data included in the region(detail of choosing included in report)

%for an specified method of correlation, we have to make the signals of
%closing and the whole signal the same size,here this extended matrixes are
%created
y_Open_extend1 = zeros(14979,1) ;
y_Open_extend2 = zeros(14979,1) ;
y_Open_extend13 = zeros(14979,1) ;
y_Open_extend14 = zeros(14979,1) ;
%here extension completes
y_Open_extend1(1:no,1) = y_Open(1:no_opn , 1 , 1) ;
y_Open_extend2(1:no,1) = y_Open(1:no_opn , 1 , 2) ;
y_Open_extend13(1:no,1) = y_Open(1:no_opn , 1 , 13) ;
y_Open_extend14(1:no,1) = y_Open(1:no_opn , 1 , 14) ;
%correlation values are caculated, this values are normalized according to
%attribute "coeff"
[r1] = xcorr(y_Open_extend1,data1(:,1),'coeff');
[r2] = xcorr(y_Open_extend2,data1(:,2),'coeff');
[r13] = xcorr(y_Open_extend13,data1(:,13),'coeff');
[r14,t] = xcorr(y_Open_extend14,data1(:,14),'coeff');% t for all above is the same

% because of an unknown problem my plot was mirrored with respect to the
% origin and i mirrored it again ---> (-x,-y)
figure();
plot(-t/128,-(r1).^3);% the 3rd power of regression is plotted over time to make meaningful peaks more visible
xlim([0,117]);
hold on
plot(data1(:,16),(0.001)*data1(:,15));%closing and opening time data is scaled in order to understand the relation of
%closing/opening signal with the whole signal
xlim([0,117]);
title(strcat('Channel 1 (Opening Eye) Cross Correlation') , 'color' , 'r');

figure();
plot(-t/128,-(r2).^3);
xlim([0,117]);
hold on
plot(data1(:,16),(0.001)*data1(:,15));
xlim([0,117]);
title(strcat('Channel 2 (Opening Eye) Cross Correlation') , 'color' , 'r');

figure();
plot(-t/128,-(r13).^3);
xlim([0,117]);
hold on
plot(data1(:,16),(0.001)*data1(:,15));
xlim([0,117]);
title(strcat('Channel 13 (Opening Eye) Cross Correlation') , 'color' , 'r');

figure();
plot(-t/128,-(r14).^3);
xlim([0,117]);
hold on
plot(data1(:,16),(0.001)*data1(:,15));
xlim([0,117]);
title(strcat('Channel 14 (Opening Eye) Cross Correlation') , 'color' , 'r');
%% section 2 - How to find blink times if we didn't have them
% included in report
%% Part 4 - Finding The Time Of Blinks
% Everything will be done with channel one
%% Checking Wether Signals Of Probes Contain Any Outliers Or Not
% Here we plot the signals and visually check the existence of outliers
load('Test');%loading Test.mat data into memory
save('data2');%creating new .mat file to store new values according to Test.mat and not to overwrite it
load('data2');%loading new file into memory
% Getting the variables and properties of the .mat file
whos('-file','Test');
data2 = EEG_Test ; %new file's data are now the same with Test.mat file
% creating time vector
fs = 128 ; % sampling rate
time = 0 : 1/fs : 12999/fs ; % according to the size of signal which has 13000 data
% plotting all of the signals
for i = 1 : 14
   figure();
   plot(time , data2(:,i));
   title(strcat('Channel',sprintf(':  %d' , i)),'color','r');
   xlabel('Time','color','b');
   ylabel('Signal(Micro Vlots)','color','b');
   xlim([0,12999/fs]);
end
% signals were clear!
% from now on, according to previous sections we'll only use probe 1 and
% only openning signal(detail included in report)
% plotting cross correlation of signals with derived signal of opening in the previous sections

%for an specified method of correlation, we have to make the signals of
%closing and the whole signal the same size,here this extended matrixes are
%created
y_Open_extend1 = zeros(13000,1) ;
y_Open_extend1(1:no,1) = y_Open(1:no_opn , 1 , 1) ;
[r1,t] = xcorr(y_Open_extend1,data2(:,1),'coeff');
figure();
% cross correlation is plotted(r^3 instead of r)
plot(-t/128,-(r1).^3);
xlabel('Time','color','b');
ylabel('r^3','color','b');
xlim([0,12999/128]);
title(strcat('Channel 1 Test (Opening Eye) Cross Correlation') , 'color' , 'r');
%% maximums
% finding peaks above determined threshold
peak_threshold_high = 35e-5 ;
r = zeros(25999,1);
[r1,t] = xcorr(y_Open_extend1,data2(:,1),'coeff');
%here because of need with calculating with data not only observing it,the problem is solved by mirroring with respect to the origin
for i = 1 : 25999
    r(i) = -r1(26000-i,1) ;
end
r = r.^3 ;
[maximum , max_index] = max(r);%maximum value and its index are found
m = 1;
%finding other maximums(detail explanation included in report)
while(maximum > peak_threshold_high)
    r(max_index - 100 : max_index + 100) = 0;
    maxi(m) = max_index ;
    [maximum , max_index] = max(r);
    m = m + 1 ;
end
% this figure shows maximums have been completely detected and removed
figure();
plot(t/128,r);
xlabel('Time','color','b');
ylabel('r^3','color','b');
title('Cross Correlation After Removing Maximums' , 'color' , 'r');
xlim([0,12999/128]);
maxi = (sort(maxi)-13000)/128 ;
%% minimums
% finding peaks below determined threshold
peak_threshold_low = -30e-5 ;
%here we retrieve r1 which was changed during previous operations
for i = 1 : 25999
    r(i) = -r1(26000-i) ;
end
r = r.^3 ;
[minimum , min_index] = min(r);%maximum value and its index are found
m = 1;
%finding other minimums(detail explanation included in report)
while(minimum < peak_threshold_low)
    r(min_index - 100 : min_index + 100) = 0;
    mini(m) = min_index ;
    [minimum , min_index] = min(r);
    m = m + 1 ;
end
figure();
plot(t/128,r);
xlabel('Time','color','b');
ylabel('r^3','color','b');
title('Cross Correlation After Removing Minimums' , 'color' , 'r');
xlim([0,12999/128]);
mini = (sort(mini)-13000)/128 ;

%% finding times of closing
m = 1;
%detail included in report
for i = 1 : max(size(maxi))
   for j = 1 : max(size(mini))
       if (mini(1,j) - maxi(1,i)< 1.6)&&(mini(1,j) - maxi(1,i)>0) %ensuring the minima is after the maxima
           final_cls(1,m) = maxi(1,i) + 0.665 ;%closing signal detected and closing time found by adding the delay
           m = m + 1;
       end
   end
end
%% finding times of opening
%detail included in report
m = 1;
j=1;
% for i = 1 situation
       while (mini(1,1) - maxi(1,j) > 0)&&(j<max(size(final_cls))) % finding the nearest maxima before the minima
               j = j + 1 ;
       end
       if mini(1,1) - maxi(1,j-1)> 1.6 %ensuring the minima is after the maxima and just one maxima between
           final_opn(1,m) = mini(1,1) + 0.524 ;%first opening signal detected and opening time found by adding the delay
           m = m + 1;
       end % if not,nothing happens and not relevant to opening
% for others
for i = 2 : max(size(mini))
    j=1;
       while (mini(1,i) - maxi(1,j) > 0)&&(j<=max(size(mini)))% finding the nearest maxima
               j = j + 1 ;
       end
       if (mini(1,i) - maxi(1,j-1)> 1.6) %ensuring the minima is after the maxima and just one maxima between
              if maxi(1,j) - mini(1,i) > 0.8 % no close maxima must be after it
                   final_opn(1,m) = mini(1,i) + 0.524 ;
                   m = m + 1;
                   continue;
              end
              if mini(1,i) - maxi(1,j-1) < mini(1,i) - mini(1,i-1) % preventing situation of two successive minima
                  final_opn(1,m) = mini(1,i) + 0.524 ;
                  m = m + 1;
                  continue;
              end
       end
end
%fast blinks
j=1;
for i = 1 : max(size(mini)) - 1
    j=1;
       while (mini(1,i) - maxi(1,j) > 0)&&(j<=max(size(mini)))% finding the nearest maxima
               j = j + 1 ;
       end
       if (mini(1,i+1) - maxi(1,j)>0 )
           if(mini(1,i) - maxi(1,j-1)< 1.6)&&(mini(1,i+1) - maxi(1,j)< 1.6 )
                   final_opn(1,m) = mini(1,i) + 0.460 ;
                   m = m + 1;
                   continue;
           end
       end
end
final_opn = sort(final_opn);

%% creating the final array
save Result.mat; 
load('Result');
who('Result');
m = 1;
n = 1;
final = zeros(13000,1);
flag_close = 0 ;
for i = 1 : 13000
   if flag_close == 0
       if (i/128 < final_cls(1,m))&&(m<=max(size(final_cls)))
           final(i,1) = 0;
           continue;
       else
           final(i,1) = 1;
           m = m + 1 ;
           flag_close = 1;
           continue;
       end
   end
   if n <= max(size(final_opn))
       if (i/128 < final_opn(1,n))&&(n<=max(size(final_opn)))
           final(i,1) = 1;
       else
           final(i,1) = 0;
           n = n + 1 ;
           flag_close = 0;
           continue;
       end
   else
       final(i,1) = 1;
       continue;
   end
end

final = final' ;

%%plotting the final correlation versus times of blinking

[r1,t] = xcorr(y_Open_extend1,data2(:,1),'coeff');
for i = 1 : 25999
    r(i) = -r1(26000-i) ;
end
r = r.^3 ;
figure();
plot(t/128,r);
hold on
time = 0 : 1/128 : 12999/128 ;
plot(time,1e-3.*final);
xlabel('Time','color','b');
ylabel('r^3','color','b');
title('Cross Correlation And Blinks' , 'color' , 'r');
xlim([0,12999/128]);
Result = final';