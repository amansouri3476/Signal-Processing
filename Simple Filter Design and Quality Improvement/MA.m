function y1 = MA( x , k )
    y1 = 1:length(x);
for i = 1 : length(x)
    if(i-k <1 || i+k > length(x))
        y1(i)= x(i);
    else
      y1(i) = mean(x(i-k:i+k));

    end
end