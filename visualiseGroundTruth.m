%% label data
run beeWaggleLabeller.m
%% DATA POINT INTERPOLATION
videoName = "20210803t1727d200m";
load("./data/" + videoName + "_labels.mat")
TD          = [];
timeNow     = 0;
x_coor      = real(userInputArray(:,1));x_coor(x_coor==0) = NaN;
y_coor      = -real(1i*userInputArray(:,1));y_coor(y_coor==0) = NaN;
timeStamp   = 1:numel(userInputArray);
nEvents     = numel(x_coor);
timeEnd     = nEvents;

for idx = 1:nEvents
    timeNow     = timeNow + round(rand*10);
    TD.x(idx)   = ceil(rand*max(x_coor));
    TD.y(idx)   = ceil(rand*max(y_coor));
    TD.p(idx)   = 1;
    TD.ts(idx)  = timeNow;
end

TD.ts = TD.ts/TD.ts(end)*timeEnd;
% figure(453453)
% subplot(1,2,1)
% scatter3(TD.x,TD.y,TD.ts,'.r')

% interpolate radius
iFrame = 1;
objectEventIndex = 0;
nFrame = numel(userInputArray);
nObj = size(userInputArray,2);

x_coor = real(userInputArray(:,1));x_coor(x_coor==0) = NaN;
y_coor = -real(1i*userInputArray(:,1));y_coor(y_coor==0) = NaN;

objx = x_coor(iFrame);
objy = y_coor(iFrame);
objRadSquared = userInputRadiusArray(iFrame)^2;

for idx = 1:nEvents
    x = TD.x(idx);
    y = TD.y(idx);
    t =  TD.ts(idx);
    while t>timeStamp(min(iFrame+1,nFrame)) && iFrame < nFrame
        iFrame = iFrame + 1;
        objx = x_coor(iFrame);
        objy = y_coor(iFrame);
        objRadSquared = userInputRadiusArray(iFrame)^2;
        [objx; objy; objRadSquared;]
    end
    
    for iObj = 1
        if ((x - objx(iObj))^2 + (y - objy(iObj))^2)<objRadSquared(iObj)
            objectEventIndex = objectEventIndex +1;
            objectTd.x(objectEventIndex) = x;
            objectTd.y(objectEventIndex) = y;
            objectTd.ts(objectEventIndex) = t;
        end
    end
end
time = 1:numel(userInputArray);

figure(456);
subplot(1,2,1);
scatter3(x_coor,y_coor,timeStamp,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
xlim([0 1000]);
ylim([0 1000]);
subplot(1,2,2);
scatter3(objectTd.x,objectTd.y,objectTd.ts,'.');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
xlim([0 1000]);
ylim([0 1000]);

%% overlay labels on video
downsampling = 1;
waggleLabellingROI = [1200 800 300 300]; % [0 0 1920 1080];
v0 = VideoReader("20210803t1727d200m.MP4");
nFrameTotal = round(v0.FrameRate *v0.Duration);
x_coor(isnan(x_coor))=0;y_coor(isnan(y_coor))=0;

for iFrame = 1:nFrameTotal
    figure(1)
    frameIntFull = read(v0,iFrame);
    imh1 = imagesc(imresize(imcrop(frameIntFull,waggleLabellingROI),1/downsampling));
    set(gca,'visible','off')
%     plot(td.x(iFrame), td.y(iFrame),'or');
%     pos = [td.x(iFrame) td.y(iFrame) userInputRadiusArray(iFrame) userInputRadiusArray(iFrame)];
%     rectangle('Position',pos,'Curvature',[1 1])
%     tt1 = title(['iFrame = ' num2str(iFrame)]);
%     hold on
%     plot(X,Y,'o')
%     plot(X0,Y0,'o')
%     plot(X1,Y1,'o')
%     plot(X2,Y2,'o')
%     plot(X3,Y3,'o')
%     hold off
%     F(i) = getframe(gcf) ;
    imwrite(imresize(imcrop(frameIntFull,waggleLabellingROI),1/downsampling),"./fig/frames/" + num2str(iFrame) + ".png")
%     drawnow
end
