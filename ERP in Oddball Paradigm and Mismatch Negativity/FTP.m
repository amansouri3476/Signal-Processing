function x_new = FTP(x, f, f_start, f_end, flag) % Fourier transform partitioning
                                                 % Zero padding Outside of frequency interval
                                                 % (f_start , f_end)

x_new = x.*((f >= f_start).*(f <= f_end) + ((f<=-f_start).*(f>=-f_end)));

if flag == 1
    plot(f, abs(x_new), 'r');
end
end                                  
