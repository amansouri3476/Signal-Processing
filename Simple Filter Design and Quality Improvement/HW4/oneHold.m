function x_new = oneHold(x, n)

x0 = [x; x(end)];
x_new = zeros(n * length(x), 1);
t = 0:(n-1);

for i = 1:length(x)
   x_new(n*(i-1)+1:n*i) = ((x0(i+1)-x0(i))/(n)) .* t + x0(i);
end

end