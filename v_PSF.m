function h = v_PSF(data)
data = data-39;
data = data./max(max(data));
[a,b]=size(data);%a is rows (y'), b is cols(z')
m = max(data(:,floor(b/2))-data(1,floor(b/2)));
h=plot(1:a,(data(:,floor(b/2))-data(1,floor(b/2)))/m,'-');
xlabel('y prime');ylabel('intensity');
title('vertical PSF');axis on;

hold on;
f=fit((1:a).',(data(:,floor(b/2))-data(1,floor(b/2)))/m,'gauss1');
plot(f);legend('hide');hold off;
xlabel('y prime');ylabel('intensity');

end
