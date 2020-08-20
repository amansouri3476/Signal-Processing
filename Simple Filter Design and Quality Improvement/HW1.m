clear all
clc
%% part 2-noise display and analysis

%% 1 sine genereation
delta_x = 1e-6;
x = 0 : delta_x : 10 ;
y = sin(2*pi*1000.*x);
figure();
plot(x,y);
grid on;
title('sine (frequency = 1KHz)','color','r');
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]); %sets the region of x which is asked to be displayed
%% 2 integration & differentiation of signal
%differentiation
diff_y = (sin(2*pi*1000.*(x+delta_x))- sin(2*pi*1000.*x))/delta_x ;
figure();
plot(x,diff_y);
grid on;
title('sine differentiaton','color','r');
xlabel('x','color','b');
ylabel('diff_y','color','b');
xlim([0,0.010]);%sets the region of x which i want to be displayed

%%second method using matlab built in functions
fs = 1e6 ; 
t = linspace(0,0.01,0.01*fs);
x = sin(2*pi*1000.*t);
dx(1) = x(1);
dx(2:length(x)) = diff(x)/(t(2)-t(1));
figure();
plot(t,dx);
grid on;
title('Sine Differentiaton second method','color','r');
xlabel('x','color','b');
ylabel('diff_y','color','b');
xlim([0,0.010]);

%integration
% integration using Simpson approximation rule
x = 0 : delta_x : 10 ;
integral_y_matrix = delta_x*(sin(2*pi*1000.*x) + sin(2*pi*1000.*(x-delta_x)) + 4*sin(2*pi*1000.*(x-(delta_x/2))) )/6;
integral_y = cumsum(integral_y_matrix);
figure();
plot(x,integral_y);
grid on;
title('sine integration','color','r');
xlabel('x','color','b');
ylabel('integral_y','color','b');
xlim([0,0.010]);%sets the region of x which i want to be displayed
%% 3 histgram of the signal
figure();
% here subplot is used to compare the derived form of histogram(included in
% report) to the histogram given by matlab
subplot(2,1,1);
histogram(y);
grid on;
title('histogram','color','r');
xlabel('output values','color','b');
ylabel('number of data','color','b');
% now i try to plot the the general shape of histogram using the equation derived in detail in
% report. This plot 'only' gives the general shape of the histogram.
m = -1 : 0.0001 : 1;
subplot(2,1,2);
plot( m , 1./(( 1 - (m.^2)).^0.5 ));
title('distribution of y general form','color','r');
xlabel('output values','color','b');
ylabel('distribution of y','color','b');
%% 4 noisy signal display
delta_x = 1e-7; % changing of regions' lengths for more accuracy
x = 0 : delta_x : 10 ;
y = sin(2*pi*1000.*x);
xn = (rand( 1 , 1e8 + 1 ))*0.2 - 0.1; 
x_n = y + xn ;
figure();
plot( x , x_n );
title('sine signal added with noise','color','r');
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]);%sets the region of x which i want to be displayed
%% 5 integration & differentiation of 'noisy' signal
delta_x1 = 1e-5;
%noisy signal differentiation
data = 0 : delta_x1 : 10; %indexing of the independent variable for assigning correct values to si                                           gnal
noise = xn;%indexing of the random numbers for assigning correct values to signal
n = 1 : 1 : 1e5 ;
delta_x1 = 1e-5 ;
n1 =  delta_x1 : delta_x1 : 1 ;
diff_xn = ((sin(2*pi*1000.*(data(n)+delta_x1)) + noise(1+(n)*1000)) - (sin(2*pi*1000.*data(n)) + noise(1+(n-1)*1000)))/delta_x1 ;
figure();
subplot( 2 , 2 , 1);
plot( n1 , diff_xn );
title('noisy differentiaton','color','r','FontSize',8);
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]);%sets the region of x which i want to be displayed
%integration
% noisy signal integration using Simpson approximation rule
integral_yn_matrix = delta_x1*(sin(2*pi*1000.*data(n))+ noise(1+(n)*1000) + sin(2*pi*1000.*(data(n)-delta_x1)) + noise(1+(n-1)*1000) + 4*sin(2*pi*1000.*(data(n)-(delta_x1/2))) )/6;
integral_yn = cumsum(integral_yn_matrix);
subplot( 2 , 2 , 2 );
plot( n1 , integral_yn );
title('noisy integration','color','r','FontSize',8);
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]);%sets the region of x which i want to be displayed
ylim([0,4e-4]);
delta_x = 1e-6;
x = 0 : delta_x : 10 ;
diff_y = (sin(2*pi*1000.*(x+delta_x))- sin(2*pi*1000.*x))/delta_x ;
subplot( 2 , 2 , 3 );
plot(x,diff_y);
xlim([0,0.010]);
title('noiseless differentiation','color','r','FontSize',8);
xlabel('x','color','b');
ylabel('y','color','b');
integral_y_matrix = delta_x*(sin(2*pi*1000.*x) + sin(2*pi*1000.*(x-delta_x)) + 4*sin(2*pi*1000.*(x-(delta_x/2))) )/6;
integral_y = cumsum(integral_y_matrix);
subplot( 2 , 2 , 4 );
plot(x,integral_y);
title('noiseless integration','color','r','FontSize',8);
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]);

%% noisy signal histogram
figure();
histogram(x_n,80);
title('noisy signal histogram','color','r','FontSize',12);
xlabel('output values','color','b');
ylabel('number of data','color','b');
%% part 3-noise elimination
% here we set k according to the desired window length
delta_x = 1e-6; % according to part 4, that delta_x must be used
x = 0 : delta_x : 1 ;
y = sin(2*pi*1000.*x);
xn = (rand( 1 , 1e6 + 1 ))*0.2 - 0.1;
x_n = y + xn ;

figure();
subplot(4,2,1);
plot( x , x_n );
title('sine signal added with noise','color','r');
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]);
ylim([-1.15,1.15]);

k=1;
y1u = MA( x_n , k );
subplot(4,2,2);
plot(x,y1u);
title('corrected signal t=0.1u ,1u,2u (k assumed 1)','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]);
ylim([-1.15,1.15]);

k=2;
y2u = MA( x_n , k );
subplot(4,2,3);
plot(x,y2u);
title('corrected signal t=5u','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]);
ylim([-1.15,1.15]);

k=5;
y5u = MA( x_n , k );
subplot(4,2,4);
plot(x,y5u);
title('corrected signal t=10u','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]);
ylim([-1.15,1.15]);

k=50;
y10u = MA( x_n , k );
subplot(4,2,5);
plot(x,y10u);
title('corrected signal t=100u','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
xlim([0,0.010]);
ylim([-1.15,1.15]);

k=250;
y100u = MA( x_n , k );
subplot(4,2,6);
plot(x,y100u);
title('corrected signal t=500u','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
xlim([k*delta_x,0.010]);
ylim([-1.15,1.15]);

k=500;
y500u = MA( x_n , k );
subplot(4,2,7);
plot(x,y500u);
title('corrected signal t=1m','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
xlim([k*delta_x,0.010]);
ylim([-1.15,1.15]);


%% part 4-convolution & system
x = (rand( 1 , 21 ))*2;
n = -10 : 1 : 10;
% 4 section 3
figure();
stem( n , x );
title('Random Discrete Time Signal','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
% 4 section 4

%% a
h = [ 0.5 , 0.5 ];
y = conv( h , x );
n = -9 : 1 : 12;
figure();
stem( n , y );
xlim([-10,13]);
title('Impulse Response To System a','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
%% b
h = [ 1 , -1 ];
y = conv( h , x );
n = -9 : 1 : 12;
figure();
stem( n , y );
xlim([-10,13]);
title('Impulse Response To System b','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
%% c
h = [ 1 , -2 , 1 ];
y = conv( h , x );
n = -9 : 1 : 13;
figure();
stem( n , y );
xlim([-10,14]);
title('Impulse Response To System c','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
%% d
h = [ 1 , 0 , 1 ];
y = conv( h , x );
n = -9 : 1 : 13;
figure();
stem( n , y );
xlim([-10,14]);
title('Impulse Response To System d','color','r','FontSize',10);
xlabel('x','color','b');
ylabel('y','color','b');
%% part 5-audio file
% 5 section 1
[y , fs ] = audioread('sound.wav');
p = audioplayer ( y , fs );
%displaying sampling rate in command window
disp(fs);
%% 5 section 2
n = length(y);
t = fs^-1 : fs^-1 : n*(fs^-1) ;
figure();
plot(t,y);
xlim([0,n*(fs^-1)]);
title('Sound Wave','color','r','FontSize',10);
xlabel('Time','color','b');
ylabel('Amplitude','color','b');

%% 5 section 3

play(p);


%% 5 section 4
[y , fs ] = audioread('sound.wav');
n = length(y);
x2 = ((std(y)/2))*randn(n,1);
y1 = y + x2 ;
p1 = audioplayer ( y1 , fs );
play(p1);
%% 5 section 5
y2 = MA(y1 , 30);
p2 = audioplayer ( 4*y2 , fs );
play(p2);

%% 5 section 6
delta_x = fs^-1;
x3 = fs^-1 : fs^-1 : (fs^-1)*length(y) ;
x4 = 0.2*sin(2*pi*1000.*x3);
y3 = y + x4';
p4 = audioplayer ( y3 , fs );
play(p4) 
%% 5 section 7
T = 0 : fs^-1 : (fs^-1)*length(y);
y4 = chirp(T,1000,(fs^-1)*length(y),5000);
c = audioplayer( y4 , fs);
play(c);
%% part-6 Sound Rebuilding
%edited sound ffts
[x , fs] = audioread('sound_edited.wav');
t = 0:1/fs:10-1/fs;
m = length(x);          % Window length
n = pow2(nextpow2(m));  % Transform length
y = fft(x,n);           % DFT
f = (0:n-1)*(fs/n);     % Frequency range
power = y.*conj(y)/n;   % Power of the DFT
figure();
plot(f,power)
xlabel('Frequency (Hz)')
ylabel('Power')
title('{\bf Periodogram}')
y0 = fftshift(y);          % Rearrange y values
f0 = (-n/2:n/2-1)*(fs/n);  % 0-centered frequency range
power0 = y0.*conj(y0)/n;   % 0-centered power
figure();
plot(f0,power0)
xlabel('Frequency (Hz)')
ylabel('Power')
title('{\bf 0-Centered Periodogram}')
phase = unwrap(angle(y0));

        
figure();
plot(f0,phase*180/pi)
xlabel('Frequency (Hz)')
ylabel('Phase (Degrees)')
grid on
%% beep elimination
BW = 200000;
power0(1:length(f0)/2-BW/2)=0;
power0(length(f0)/2+BW/2:end)=0;
figure();
plot(f0,power0);
BW = 200000;
y0(1:length(f0)/2-BW/2)=0;
y0(length(f0)/2+BW/2:end)=0;
y = ifft(ifftshift(y0) , 'symmetric');
p = audioplayer ( y , fs );
play(p);
%% Echo removing
[x , fs] = audioread('sound_edited.wav');
p1 = audioplayer ( x , fs );
[r,p] = xcorr(x,x);
t = -length(x)*fs^-1 + fs^-1: fs^-1 : length(x)*fs^-1 -fs^-1;
m = 3*fs ;
figure();
plot(t,r);
% y is the beep eliminated signal
y1(1:m) = y(1:m);

for i = m+1 : length(x) ;
    y1(i) = y(i) - 0.3864485981*y1(i-m) ;
end
y1(10*fs:end) = 0 ;
p2 = audioplayer ( y1 , fs );
[r,p] = xcorr(y1,y1);
t = -length(y1)*fs^-1 + fs^-1: fs^-1 : length(y1)*fs^-1 -fs^-1;
figure();
plot(t,r);
y_final = MA( y1 , 20);
p_final = audioplayer(y_final*2 , fs);
audiowrite('sound_final.wav',y_final,fs);