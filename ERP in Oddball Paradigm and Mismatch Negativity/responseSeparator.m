function y = responseSeparator(x, stimulus, lim1, lim2)

% lim1:lim2 = 450 ms before stimulus and 600 ms(accrording to P300
% variations) after stimulus = 230 points before + 507 points after
% stimulus

% x = S1.data{1,1}.X
% stimulus = S1.data{1,1}.trial - a1
% y = 480 * 31 * 737 matrix

y = zeros(length(stimulus), length(x(1,:)), lim1+lim2); % 480*31*737

for i = 1:length(x(1,:)) % 31
    for j = 1:length(stimulus) % 480
        y(j, i, :) = x(stimulus(j)-lim1+1:stimulus(j)+lim2, i);
    end
end
end