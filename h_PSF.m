function h = h_PSF(data)
data = data-39;
data = data./max(max(data));
[a,b]=size(data);%a is rows (y'), b is cols(z')
m = max(data(floor(a/2),:)-data(floor(a/2),1));
h=plot(1:b,(data(floor(a/2),:)-data(floor(a/2),1))/m,'-');
xlabel('z prime');ylabel('intensity');
title('horizontal PSF');axis on;

hold on;
f=fit((1:b).',((data(floor(a/2),:)-data(floor(a/2),1)).')/m,'gauss1');
plot(f);legend('hide');hold off;
xlabel('z prime');ylabel('intensity');
end

