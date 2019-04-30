function sigma = h_PSFFIT(data)
data = data-19;
data = data./max(max(data));
[a,b]=size(data);%a is rows (y'), b is cols(z')
m = max(data(floor(a/2),:)-data(floor(a/2),1));
f=fit((1:b).',((data(floor(a/2),:)-data(floor(a/2),1))/m).','gauss1');
c = coeffvalues(f);
sigma=c(3);
end
