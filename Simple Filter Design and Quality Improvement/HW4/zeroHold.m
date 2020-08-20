function x_new = zeroHold(x, n)

x_new = zeros(n * length(x), 1);

for i = 1:length(x)
   x_new(n*(i-1)+1:n*i) = x(i);
end

end