function sigma = v_PSFFIT(data)
data = data-39;
data = data./max(max(data));
[a,b]=size(data);%a is rows (y'), b is cols(z')
m = max(data(:,floor(b/2))-data(1,floor(b/2)));
f=fit((1:a).',(data(:,floor(b/2))-data(1,floor(b/2)))/m,'gauss1');
c = coeffvalues(f);
sigma=c(3);
end
