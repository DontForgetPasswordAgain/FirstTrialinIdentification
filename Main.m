%% Step 1
% video=VideoReader('2p.mp4');%data_ = read(data,1)
% data = read(video,[25,30],'native');
% redChannel = data(:, :, 1);
% greenChannel = data(:, :, 2);
% blueChannel = data(:, :, 3);
% % Recombine separate color channels into an RGB image.
% RGBdata = cat(3, redChannel, blueChannel,greenChannel);
% imshow(RGBdata,[])
%% Step 2 output: data
video=VideoReader('2p.mp4');
data = read(video,[1,Inf],'native');
[x,y,rgb,f]=size(data);
RGBdata = zeros(x,y,3);
data_  = zeros(x,y,f);

for i=1:f
redChannel = data(:, :, 1,i);
greenChannel = data(:, :, 2,i);
blueChannel = data(:, :, 3,i);

RGBdata = cat(3, redChannel, greenChannel, blueChannel);
data_(:,:,i)=rgb2gray(RGBdata);
end



for i=1:f
imshow(data_(:,:,i),[]);colormap(gray);
end
%% Step 2.5 Add poisson to every frame and take temporal average
Poi_video = zeros(x,y,f);
Poi_average_video = zeros(x,y,10);
for i=1:f
Poi_video(:,:,i) = imnoise(data_(:,:,i)/10^12,'poisson');
imshow(Poi_video(:,:,i),[]);colormap(gray);
end
for i=1:10
Poi_average_video(:,:,i) = mean(Poi_video(:,:,10*i-9:10*i));
end
% but as the video is already preprocessing..
%% Step 3 MIP
mip = zeros(x,y);
for i=1:x
    for j=1:y
        mip(i,j) = max(data(i,j,:));       
    end
end
imshow(mip,[]);
%% Step 3.5 imregionalmax
test = imreginalmax(mip);
[loc1, loc2] = find(test==1);
imshow(mip,[]);
hold on;
scatter(loc1,loc2);
%% Step 3.8 section (to be deleted)
test=imcrop(mip,[]);
[center,radii]=imfindcircles(test,[4,15],'Sensitivity',0.95);
imshow(test,[]);
hold on;
scatter(center(:,1),center(:,2));hold on;
for k=1:size(radii)
    text(center(k,1),center(k,2),num2str(k),'Color','Red');
end

list = [12,18,19];%ghost spots
radii(list)=[];center(list,:)=[];

for k=1:size(radii)
    x=center(k,1);
    y=center(k,2);
    a = radii(k)+2;
    dat_ = test(floor(y-a):floor(y+a),floor(x-a):floor(x+a));
    subplot(4,5,k);
    h_PSF(dat_);
end

% gaussian1 fit
list=zeros(17,1);
for k=1:size(radii)
    x=center(k,1);
    y=center(k,2);
    a = radii(k)+2;
    dat_ = test(floor(y-a):floor(y+a),floor(x-a):floor(x+a));
    list(k)=h_PSFFIT(dat_);   
end
fwhm = mean(list)*2*sqrt(log(2))
% vertical: FWHM = 6.5345
% horizontal: FWHM = 7.9621
sigma_ = mean(list)/sqrt(2);
f_v = fspecial('gaussian',[10,1],2.7750);
f_h = fspecial('gaussian',[1,16],3.1812);
f_panel = f_v * f_h;
imshow(f_panel,[]);

surf(f_panel);colormap(pink);
%% Step 4 find circles
[center,radii]=imfindcircles(mip,[4,15],'Sensitivity',0.95);
cx = center(:,1);cy=center(:,2);
for k=1:size(radii)
    if k>2
        if sqrt((center(k,1)-center(k-1,1))^2+(center(k,2)-center(k-1,2))^2)<6
            center(k,:)=[];radii(k)=[];
            continue;
        end
    end
end
imshow(mip,[]);hold on;c = linspace(1,10,length(cx));
scatter(center(:,1),center(:,2),[],c);
%scatter(cx,cy,[],c);

%% Step 5 trace
cellnum = size(radii);

for i=1:20
   
    plotdata=zeros(100,1);
    for j=1:100
        %[cx cy]=deal(center(i,:)); no
        cx = floor(center(i,1));cy = floor(center(i,2));
        plotdata(j)=data_(cx,cy,j);
    end
    subplot(20,1,i);
    plot(1:100,plotdata,'-');textstr=num2str(i);legend(textstr,'Location','westoutside')
    set(gca,'YTickLabel',[],'XTickLabel',[]);
    
end

%% Step 6 Use gaussian to model 
% main goal is to get sigma_x and sigma_y
imshow(mip,[]);
list_1=[85,86,92];
radii(list_1)=[];center(list_1,:)=[];
for k=1:size(radii)
    x=center(k,1);
    y=center(k,2);
    a = radii(k)+2;
    dat_ = mip(floor(y-a):floor(y+a),floor(x-a):floor(x+a));
    subplot(5,5,k-75);
    v_PSF(dat_);
end

% gaussian1 fit
list_v = [22,77,100];
ga = [76,98];
radii(97)=[];center(97,:)=[];
list=zeros(104,1);
for k=1:size(radii)
    x=center(k,1);
    y=center(k,2);
    a = 5;
    dat_ = mip(floor(y-a):floor(y+a),floor(x-a):floor(x+a));
    list(k)=h_PSFFIT(dat_);   
end
fwhm = mean(list)*2*sqrt(log(2))
% vertical: FWHM = 5.4276
% horizontal: FWHM = 9.8105
sigma_ = mean(list);
%sigma_h = 5.8918
%sigma_v = 3.2596
%% Step 7 model the gaussian panel
f_v = fspecial('gaussian',[10,1],2.3049);
f_h = fspecial('gaussian',[1,16],4.1661);
f_panel = f_v * f_h;
imshow(f_panel,[]);

surf(f_panel);colormap(pink);
%% Step 8 cross correlation (method 1: traditional xcorr)
data_1  = xcorr2(mip,f_panel);
imshow(data_1,[])



[center_test,radii_test]=imfindcircles(data_1,[2,15],'Sensitivity',0.95);
imshow(mip,[]);hold on;scatter(center_test(:,1)-7,center_test(:,2)-5);
imshow(data_1,[]);hold on;scatter(center_test(:,1),center_test(:,2));title('cross correlation');
for k=1:size(radii_test)
    text(center_test(k,1),center_test(k,2),num2str(k),'Color','Red');
end

data_1=imcrop(data_1 ,[]);%crop away the zero padding

regmax = imregionalmax(data_1);
nnz(regmax)
[cx1,cy1]=find(regmax==1);
imshow(data,[]);title('csv centers label');hold on;
scatter(cx1,cy1);
%% Cross Correlation supplemental(method 2: template matching,not work well)
f_panel_=padarray(f_panel,[(size(mip,1)-size(f_panel,1))/2,(size(mip,2)-size(f_panel,2))/2]);
%imshow(f_panel_,[])
%gamma = F^-1(F(mip)F^*(f_panel))
gamma = ifft2(fftshift(fft2(mip))*fftshift(fft2(f_panel_)));%work not well in real world
gamma = imfilter(mip,f_panel,'conv','same');
gamma = ifft2(fftshift(fft2(mip)));
imshow(gamma,[])
for u=size(gamma,1)
    for v = size(gamma,2)
        k1 = sqrt(sum(sum((f_panel_-mean(mean(f_panel_))).^2,1),2));
        k2 = sqrt(sum(sum((mip-mean(mean(mip))).^2,1),2));
        %denominator also actually a constant as well
        gamma(u,v) = gamma(u,v)/k;
    end  
end
%% Step 9 add text to video

% for i=1:f
%   imshow(data_(:,:,i),[]);colormap(gray);
%   for j=1:size(radii)
%      textstr=num2str(j);
%      Data_(:,:,:,i) = insertText(Data_(:,:,:,i),position,textstr,'FontSize',10);
%      text(center(j,1),center(j,2),textstr,'Color','yellow','FontSize',6);
%   end
% end

RGBdata1 = zeros(x,y,3,100);

for i=1:f
redChannel = data(:, :, 1,i);
greenChannel = data(:, :, 2,i);
blueChannel = data(:, :, 3,i);

RGBdata1(:,:,:,i) = cat(3, redChannel, greenChannel, blueChannel);
end



d_rep=RGBdata1;%data_replicate
writeAVI = VideoWriter('video1.avi');
writeAVI.FrameRate = 100;
open(writeAVI);
colormap(gray);

for i = 1:f
    test = d_rep(:,:,:,i);
 imshow(test,[]); 
 for j = 1: size(radii)
%         textstr=num2str(j);
%         
%         test = insertText(test,[center(j,1),center(j,2)],textstr,...
%             'BoxOpacity',0,'TextColor','red','FontSize',6);

      
        
 end
  
 frame = getframe(gcf);
 writeVideo(writeAVI,frame);

end
close(writeAVI);% fuck, doesn't work
%% Step 10 left hand trace, right hand labelled neurons
test = RGBdata;
for j=1:94
    textstr=num2str(j);
    test = insertText(test,[center(j,1),center(j,2)],textstr,...
            'TextColor','yellow','BoxOpacity',0,'FontSize',10);
    
end
imshow(test,[]);colormap(gray)
test2 = imread('delete.tif');
montage({test2,test})
%% additional steps 
imshow(data_(:,:,36),[]);
name = data_(:,:,36);
[center_test,radii_test]=imfindcircles(name,[4,15],'Sensitivity',0.9);
imshow(name,[]);hold on;scatter(center_test(:,1),center_test(:,2));
%% overlay mask on video (works, but don't adjust the size of window when playing)
%imshow(mip, 'InitialMag', 'fit');
% black = cat(3,zeros(size(mip)), zeros(size(mip)), zeros(size(mip)));
% h=imshow(black,[]);hold on;
% 
% I=scatter(center_test(:,1)-7,center_test(:,2)-5,[],'b');
% I= double(I)
% imwrite(I,'mask.tif')
% 
% set(h, 'AlphaData', I);
% 
% mask = imread('mask.tif');
% imshow(mask,[])

%M = max(max(data_(:,:,36)));m=min(min(data_(:,:,36)));
writeAVI = VideoWriter('video1.avi');
writeAVI.FrameRate = 5;
open(writeAVI);
colormap(gray);

for i=1:80
    imshow(data_(:,:,i),[]);
%    ax=plt.gca;
%     set(gca,'clim',[m,M]);
     hold on;
     scatter(center_test(:,1)-7,center_test(:,2)-5,[],'p');
     hold off;pause(0.02);
      frame = getframe(gcf);
 writeVideo(writeAVI,frame);

end
close(writeAVI);

%% add shot noise to video
poi = imnoise(mip/10^12,'poisson');
imshow(poi,[])
imshowpair(poi,mip,'Montage');title('With poisson and Without poisson');
%% implement scale-space