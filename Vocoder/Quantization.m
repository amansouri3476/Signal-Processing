function [output]=Quantization(input,B)
%Quantization perform quantization
%   output = Quantization(input,B) quantize input signal to B bits;

%   preallocate output for speed
% -----------------------------------------
output = zeros(1,length(input));

% Enter your code here
%----------------------------------
%...
qstep = 1/(2^B);
output = qstep*floor((input(:) + 0.5)/qstep);

end