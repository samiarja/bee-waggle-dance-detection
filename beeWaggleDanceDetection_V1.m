clear
addpath("../DeepGreen/greenhouseCode")

%%%%%%%%%%%%%%%% LOAD VIDEO
videoName = "20210803t1727d200m";
videoFileName = "./data/" + videoName + ".MP4";
limits.rowStart    = 401;    limits.rowEnd      = 1000;    limits.colStart    = 201;    limits.colEnd      = 1700;
v0 = VideoReader(videoName + ".MP4");
disp("Loading data...")

%%%%%%%%%%%%%%%% LOAD TEMPLATE
load('waggle16Templates_v1.mat')
disp("Template loaded...")

%if ~exist('FULL_IMAGE')
% clear

td                          = [];
tdWithInh                   = [];
jTd                         = [];
FULL_IMAGE                  = 0;
iWaggleInh                  = 0;
iWaggleEvent                = 0;
SHOW                        = 0; %rand>5;
convMapThreshold            = 20;
nEventsForWaggleThreshold   = 10;
nDel                        = 18;
framesPerSegment            = 500;
RECORD_VIDEO                = SHOW && 1;

nTemplate = size(waggleTemplate25,3);

nFrameTotal = round(v0.FrameRate *v0.Duration);
nSegment = ceil(nFrameTotal/framesPerSegment);


if SHOW
    figNum = 536451;
    figure(figNum);clf;
end


if RECORD_VIDEO
    set(gcf,'color','k')
    globalFrameRate                 = 60;
    vid1           = VideoWriter("videos/" + videoName + ".avi");
    vid1.FrameRate = globalFrameRate; %// Change the desired frame rate here
    vid1.Quality   = 95;
    open(vid1);
end

for iSegment = 1:5
    iSegment
    %startFrame = 1;%
    startFrame = max((iSegment-1)*framesPerSegment+1-nDel,1);
    
    %endFrame = nFrameTotal;%
    endFrame = min(iSegment*framesPerSegment,nFrameTotal);%500;
    
    nFrame =  endFrame - startFrame;
    
    avgFrameDepth = 6;
    iFrameWithWaggle = 0;waggleStats = {};
    if FULL_IMAGE
        imageWidth  = v0.Width;
        imageHeight = v0.Height;
    else
        imageWidth  = limits.colEnd - limits.colStart +1;
        imageHeight = limits.rowEnd - limits.rowStart +1;
    end
    
    frameArray     =  zeros(imageHeight/2,imageWidth/2,3,nFrame,'uint8');
    dRgbFrameArray =  zeros(imageHeight/2,imageWidth/2,3,nFrame,'single');
    dGreyScaleArray    = zeros(imageHeight/4,imageWidth/4,nFrame,'single');
    
    iFrame = 0;
    for iFrameOriginal = startFrame:  endFrame
        iFrame = iFrame + 1;
        frameIntFull = read(v0,iFrameOriginal);
        if FULL_IMAGE
            frameInt = frameIntFull;
        else
            frameInt = frameIntFull(limits.rowStart:limits.rowEnd,limits.colStart:limits.colEnd,:);
        end
        
        % downsample the data by two
        frameArray(:,:,:,iFrame) =  imresize(frameInt,0.5);
        if iFrame>1
            dRgbFrameArray(:,:,:,iFrame) = single(frameArray(:,:,:,iFrame)) - single(frameArray(:,:,:,iFrame-1));
            dGreyScaleArray(:,:,iFrame) =     imresize(vecnorm(single(frameArray(:,:,:,iFrame)),2,3) -   vecnorm(single(frameArray(:,:,:,iFrame-1)),2,3),0.5);
            
        end
    end
    
    
    disp("Frame differencing finished...")
    
    disp("Start 3D convolution...")
    %%%%  3D convolution
    sigma        = 10;
    Tau          = 36;
    delArray     = 1:nDel;
    %waggleFilt   = exp(-delArray/Tau).*sin(4/(2*pi)*(delArray+6.5));  % slowestOne
    %waggleFilt   = exp(-delArray/Tau).*sin(6.5/(2*pi)*(delArray+4));  % too fast
    waggleFilt1   = exp(-delArray/Tau).*sin(5.75/(2*pi)*(delArray+5.2));
    waggleFilt4d1 = single(reshape(waggleFilt1,[1,1,1,nDel]));
    
    waggleFilt2   = exp(-delArray/Tau).*sin(5/(2*pi)*(delArray+5.2));
    waggleFilt4d2 = single(reshape(waggleFilt2,[1,1,1,nDel]));
    
    waggleFilt3   =exp(-delArray/Tau).*sin(4/(2*pi)*(delArray+6.5));
    waggleFilt4d3 = single(reshape(waggleFilt3,[1,1,1,nDel]));
    
    waggleMap1 = convn(dRgbFrameArray,waggleFilt4d1,'full');
    waggleMap1 = waggleMap1(:,:,:,1:nFrame);
    
    waggleMap2 = convn(dRgbFrameArray,waggleFilt4d2,'full');
    waggleMap2 = waggleMap2(:,:,:,1:nFrame);
    
    
    waggleMap3 = convn(dRgbFrameArray,waggleFilt4d3,'full');
    waggleMap3 = waggleMap3(:,:,:,1:nFrame);
    
    disp("Finish 3D convolution...")
    
    waggleMapMaxed =  zeros(imageHeight/2,imageWidth/2,nFrame,'single');
    maxMat = waggleMap1(:,:,:,1) +nan;
    
    disp("Start creating waggle map for all conv layers...")
    for iFrame = 1:nFrame
        maxMat(:,:,1) = vecnorm(waggleMap1(:,:,:,iFrame),2,3);
        maxMat(:,:,2) = vecnorm(waggleMap2(:,:,:,iFrame),2,3);
        maxMat(:,:,3) = vecnorm(waggleMap3(:,:,:,iFrame),2,3);
        waggleMapMaxed(:,:,iFrame) = max( maxMat,[], 3);
    end
    disp("Finish creating waggle map from conv layer 1...")
    
    waggleMap1 = [];
    waggleMap2 = [];
    waggleMap3 = [];
    
    waggleMapMaxedHalf = waggleMapMaxed;
    waggleConvResult = zeros(imageHeight/2,imageWidth/2,nTemplate,'single');
    waggleConvInh=zeros(imageHeight/4,imageWidth/4,'single');
    waggleDetectionMap = waggleConvInh;
    
    inhDisc=fspecial('disk',9)>0;
    
    avgFrameDepth = 6;
    
    angleArray = 1:180;
    angleArrayActivation = angleArray + nan;
    rad25 = 7;
    rad50 = rad25*2;
    jThresh = 20;
    
    disp("Start 2D convolution...")
    for iFrame = 1:nFrame
        meanWaggleMapFrame = mean(waggleMapMaxedHalf(:,:,max(iFrame-avgFrameDepth,1):iFrame),3);
        
        %%% 2D convolution between every meanWaggleMapFrame and each
        %%% template
        for iTemplate = 1:nTemplate
            waggleConvResult(:,:,iTemplate) = conv2(meanWaggleMapFrame,waggleTemplate25(:,:,iTemplate),'same');
        end
        
        
        [waggleConvResultMaxedVal, waggleTemplateIdx ]= max(waggleConvResult,[],3);
        
        resizedWaggleMatch          = imresize(waggleConvResultMaxedVal,.5);
        waggleConvThreshed          = resizedWaggleMatch>convMapThreshold;
        waggleConvThreshedMaxed     = waggleConvThreshed & resizedWaggleMatch==max(resizedWaggleMatch(:));% only one waggle detection per frame
        waggleDetectionMap          = (waggleConvThreshedMaxed & ~waggleConvInh);
        
        % find waggle events, find their orientation
        [iRow, iCol, ~] = find(waggleConvThreshedMaxed>0);
        [r, c, ~] = find(waggleConvThreshedMaxed>0);
        
        if ~isempty(r) && (c-rad25>0) && (c+rad25<size(dGreyScaleArray,2)) && (r-rad25>0) && (r+rad25<size(dGreyScaleArray,1))
            iWaggleEvent = iWaggleEvent + 1;
            waggleRegion  = frameArray(r-rad25:r+rad25,c-rad25:c+rad25,:,iFrame);
            dwaggleRegion = dGreyScaleArray(r-rad25:r+rad25,c-rad25:c+rad25,iFrame);
            
            for iAngle = angleArray
                waggleRegionRotated = imrotate(dwaggleRegion,iAngle);
                angleArrayActivation(iAngle) = std(sum(waggleRegionRotated,2));
            end
            
            movmedianWindow = 10;
            angleArrayActivation= movmedian(angleArrayActivation,movmedianWindow);
            [~, bestAngle] = max(angleArrayActivation);
            td.x(iWaggleEvent)= c;
            td.y(iWaggleEvent)= r;
            td.ts(iWaggleEvent)  = iFrame+startFrame-1;
            td.angle(iWaggleEvent) = bestAngle;
            td.template{iWaggleEvent} = waggleTemplateIdx(r-rad25:r+rad25,c-rad25:c+rad25);
            td.context{iWaggleEvent} =  waggleRegion;
            td.dContext{iWaggleEvent} =  dwaggleRegion;
        end
        
        [iRowInh, iColInh, ~] = find(waggleDetectionMap>0);
        if ~isempty(iRowInh) && (c-rad25>0) && (c+rad25<size(dGreyScaleArray,2)) && (r-rad25>0) && (r+rad25<size(dGreyScaleArray,1))
            for idxInh = 1:numel(iRowInh)
                iWaggleInh = iWaggleInh + 1;
                r = iRowInh(idxInh);
                c = iColInh(idxInh);
                
                waggleRegion = dGreyScaleArray(r-rad25:r+rad25,c-rad25:c+rad25,iFrame);
                
                for iAngle = angleArray
                    waggleRegionRotated = imrotate(waggleRegion,iAngle);
                    angleArrayActivation(iAngle) = std(sum(waggleRegionRotated,2));
                    
                end
                movmedianWindow = 10;
                angleArrayActivation= movmedian(angleArrayActivation,movmedianWindow);
                [~, bestAngle] = max(angleArrayActivation);
                tdWithInh.x(iWaggleInh)= c;
                tdWithInh.y(iWaggleInh)= r;
                tdWithInh.ts(iWaggleInh)  = iFrame+startFrame-1;
                tdWithInh.angle(iWaggleInh) = bestAngle;
                tdWithInh.vid{iWaggleInh} =  dGreyScaleArray((r-rad25):(r+rad25),(c-rad25):(c+rad25),max(iFrame-40,1):min(iFrame+40,nFrame));
                tdWithInh.vidFrame{iWaggleInh} = (max(iFrame-40,1):min(iFrame+40,nFrame))+startFrame-1;
                %
            end
        end
        waggleConvInh = min(max(waggleConvInh + single(conv2(waggleConvThreshed,inhDisc,'same')) - .1,0),1);
        
        if SHOW
            if iFrame ==1
                subtightplot(5,2,1); % original video
                imh1 = imagesc(uint8(frameArray(:,:,:,iFrame)));axis image;
                titleHandle = title(num2str(iFrame),'color','r','fontSize',14);
                
                subtightplot(5,2,2); % frame differencing
                imh2 =  imagesc(   uint8((dRgbFrameArray(:,:,:,iFrame))));axis image;
                
                subtightplot(5,2,3); %conv with the kernel
                imh3 =  imagesc(waggleMapMaxed(:,:,iFrame));axis image;
                caxis([0 127])
                colorbar;
                
                subtightplot(5,2,4); % avg conv with the kernel
                imh4 =  imagesc( meanWaggleMapFrame  );axis image;  colorbar;
                caxis([0 512] )
                
                subtightplot(5,2,5); % conv of the previous
                imh5 =  imagesc(waggleConvResultMaxedVal );axis image;  colorbar;
                caxis([0 512] )
                
                subtightplot(5,2,6);
                imh6 =  imagesc(waggleConvThreshed );axis image;  colorbar;
                caxis([0 1] )
                
                subtightplot(5,2,7);
                imh7 =  imagesc(waggleConvInh);axis image;  colorbar;
                caxis([0 1] )
                
                subtightplot(5,2,8);
                imh8 =  imagesc(waggleDetectionMap);axis image;  colorbar;
                
                subtightplot(5,2,9);
                imh9 =  imagesc(waggleTemplateIdx);axis image;  colorbar;
                
                subtightplot(5,2,10);
                imh10 =  imagesc(dGreyScaleArray(:,:,iFrame));axis image;  colorbar; %colormap grey;
                caxis([-100 100] )
                set(gcf, 'Name', 'All processing pipeline');
                %         subtightplot(4,2,6);
                %         imh6 =  imagesc(waggleConvThreshed );axis image;  colorbar;
                %         caxis([0 1] )
            else
                set(titleHandle,'String',num2str(iFrame))
                set(imh1,'CDATA',   uint8(frameArray(:,:,:,iFrame)));
                set(imh2,'CDATA',   uint8((dRgbFrameArray(:,:,:,iFrame))))
                set(imh3,'CDATA',  waggleMapMaxed(:,:,iFrame) )
                set(imh4,'CDATA',  meanWaggleMapFrame)
                set(imh5,'CDATA',  waggleConvResultMaxedVal)
                set(imh6,'CDATA',  waggleConvThreshed)
                set(imh7,'CDATA',  waggleConvInh)
                set(imh8,'CDATA',  waggleDetectionMap)
                set(imh9,'CDATA', waggleTemplateIdx)
                set(imh10,'CDATA', dGreyScaleArray(:,:,iFrame))
            end
            drawnow
            if RECORD_VIDEO
                frame = getframe(figNum);
                writeVideo(vid1,frame);
                %                     if iFrame ~= nFrameInWaggleConvMap
                %                         frame = getframe(figNum);
                %                         writeVideo(vid1,frame);
                %                     end
            end
        end
    end
end
if RECORD_VIDEO
    close(vid1);
end
%% post processing: visualisation and stats
%%%%%%%%%----------------------- Post Process
%load('20210803t1727d200m_td.mat')

nWaggleEvent = numel(td.x);
templateArray1 = [];templateArray2 = [];
rr = 1;
for idx = 1:nWaggleEvent
    templateArray1(idx) =  td.template{idx}(8,8);
    templateArray2(idx) = mean(mat2vec(td.template{idx}(8-rr:8+rr,8-rr:8+rr)));
    aa = (mat2vec(td.template{idx}(8-rr:8+rr,8-rr:8+rr))-1)/15*pi;
    bb = mean(exp(1i*aa*2));
    templateArray3(idx) =  angle( bb);
    SArray3(idx) =  abs( bb);
end

figure(54534); clf;
subplot(211)
plot(td.ts,'.');  grid on;
subplot(212); hold on;
plot((templateArray1-1)/15*180,'.');  grid on;
plot((templateArray2-1)/15*180,'+');  grid on;
for idx = 1:nWaggleEvent
    plot(idx,mod(templateArray3(idx)*360/2/pi/2,180),'or','markersize', 25*SArray3(idx)^3);  grid on;
end

for idx = 1:nWaggleEvent
    if idx ==1
        iWaggle = 1;
        jdx = 1;
        waggleNumberArray(idx) = iWaggle;
        waggleEventNumberArray(idx) = jdx;
        waggleCellArray{iWaggle}(jdx) = idx;
        waggleAngleCellArray{iWaggle}(jdx) = templateArray3(idx);
    else
        if td.ts(idx)-td.ts(idx-1)<=1   % add event to this waggle
            jdx = jdx + 1;
            waggleNumberArray(idx) = iWaggle;
            waggleEventNumberArray(idx) = jdx;
            waggleCellArray{iWaggle}(jdx) = idx;
            waggleAngleCellArray{iWaggle}(jdx) = templateArray3(idx);
        else
            jdx = 1;
            iWaggle = iWaggle + 1;
            waggleNumberArray(idx) = iWaggle;
            waggleEventNumberArray(idx) = jdx;
            waggleCellArray{iWaggle}(jdx) = idx;
            waggleAngleCellArray{iWaggle}(jdx) = templateArray3(idx);
        end
    end
end
nWaggle = numel(waggleCellArray);

for iWaggle = 1:nWaggle
    meanWaggleAngle(iWaggle) = mode(waggleAngleCellArray{iWaggle});
    %     meanWaggleAngle(iWaggle) = angle(mean(exp(1i*waggleAngleCellArray{iWaggle}*2)));
    meanWaggleMag(iWaggle) = abs(mean(exp(1i*waggleAngleCellArray{iWaggle}*2)));
end
waggleCellArray = {};waggleAngleCellArray={};meanWaggleAngleArray = [];meanWaggleMagArray = [];
for idx = 1:nWaggleEvent
    if idx==49
        disp('');
    end
    if idx ==1
        iWaggle = 1;
        jdx = 1;
        waggleCellArray{iWaggle}(jdx) = idx;
        waggleAngleCellArray{iWaggle}(jdx) = templateArray3(idx);
        meanWaggleAngleArray(idx) = meanWaggleAngle(iWaggle);
        meanWaggleMagArray(idx) =  meanWaggleMag(iWaggle);
    else
        if td.ts(idx)-td.ts(idx-1)<=1   % same waggle add event to the current waggle
            jdx = jdx + 1;
            waggleCellArray{iWaggle}(jdx) = idx;
            waggleAngleCellArray{iWaggle}(jdx) = templateArray3(idx);
            meanWaggleAngleArray(idx) = meanWaggleAngle(iWaggle);
            meanWaggleMagArray(idx) =  meanWaggleMag(iWaggle);
        else
            jdx = 1;
            iWaggle = iWaggle + 1;
            waggleCellArray{iWaggle}(jdx) = idx;
            waggleAngleCellArray{iWaggle}(jdx) = templateArray3(idx);
            meanWaggleAngleArray(idx) = meanWaggleAngle(iWaggle);
            meanWaggleMagArray(idx) =  meanWaggleMag(iWaggle);
        end
    end
end



figure(54531); clf;
subplot(211)
plot(td.ts,'.');  grid on;
subplot(212); hold on;
plot((templateArray1-1)/15*180,'.');  grid on;
plot((templateArray2-1)/15*180,'+');  grid on;
for idx = 1:nWaggleEvent
    plot(idx,mod(meanWaggleAngleArray(idx)*360/2/pi/2,180),'or','markersize', 25*meanWaggleMagArray(idx)^3);  grid on;
end

tdAugmented = td;
for idx = 1:nWaggleEvent
    tdAugmented.iWaggle(idx)           = waggleNumberArray(idx);
    tdAugmented.nEventsThisWaggle(idx) = numel(waggleCellArray{waggleNumberArray(idx)});
    tdAugmented.iEventThisWaggle(idx)  = waggleEventNumberArray(idx);
    tdAugmented.meanAngle(idx)         = meanWaggleAngleArray(idx);
    tdAugmented.meanMag(idx)           = meanWaggleMagArray(idx);
    tdAugmented.confidence(idx)        = (tdAugmented.nEventsThisWaggle(idx)>nEventsForWaggleThreshold)*(tdAugmented.meanMag(idx) >0.5)*tdAugmented.meanMag(idx) ;
end


tdCleaned  = [];
idxCleaned = 0;
for idx = 1:nWaggleEvent
    if tdAugmented.confidence(idx)>0
        idxCleaned = idxCleaned +1;
        tdCleaned.originalIdx(idxCleaned) = idx;
        tdCleaned.x(idxCleaned)          =  td.x(idx);
        tdCleaned.y(idxCleaned)          = td.y(idx);
        tdCleaned.ts(idxCleaned)         =  td.ts(idx);
        tdCleaned.nEventsThisWaggle(idxCleaned) =  tdAugmented.nEventsThisWaggle(idx);
        tdCleaned.iWaggle(idxCleaned)     = tdAugmented.iWaggle(idx);
        tdCleaned.meanAngle(idxCleaned)  =  meanWaggleAngleArray(idx);
        tdCleaned.confidence(idxCleaned) = tdAugmented.confidence(idx);
        tdCleaned.context{idxCleaned}    =   td.context{idx};
        tdCleaned.dcontext{idxCleaned}   =   td.dContext{idx};
    end
end



figure(42421); clf; hold on;
plot(tdCleaned.originalIdx, tdCleaned.nEventsThisWaggle,'o');
plot(1:numel(tdAugmented.x),tdAugmented.nEventsThisWaggle,'x');
grid on; box on;

figure(42422); clf; hold on;
plot(tdCleaned.originalIdx, tdCleaned.iWaggle,'o');
plot(1:numel(tdAugmented.x),tdAugmented.iWaggle,'x');
grid on; box on;

figure(42423); clf; hold on;
plot(tdCleaned.originalIdx, tdCleaned.confidence,'o');
plot(1:numel(tdAugmented.x),tdAugmented.confidence,'x');
grid on; box on;

figure(42424); clf; hold on;
plot(tdCleaned.nEventsThisWaggle,tdCleaned.meanAngle*360/2/pi,'o');
grid on; box on;

figure(42425); clf; hold on;
plot3(tdCleaned.x, tdCleaned.y,tdCleaned.ts,'.')
grid on; box on;


imageWidth  = limits.colEnd - limits.colStart +1;
imageHeight = limits.rowEnd - limits.rowStart +1;

frame0     =  zeros(imageHeight/2,imageWidth/2,3,'uint8');
ry0 = [nan nan];
cx0 = [nan nan];
nFrameTotal = round(v0.FrameRate *v0.Duration);
idx = 1;jdx=1;
rad = 30;

if SHOW
    for iFrame = 1:min(nFrameTotal,td.ts(end))
        frameIntFull = read(v0,iFrame);
        frameInt = imresize(frameIntFull(limits.rowStart:limits.rowEnd,limits.colStart:limits.colEnd,:),.5);
        frame0 = frameInt;
        frame0(:,:,1) = frame0(:,:,1)* 0;
        ry = ry0;
        cx = cx0;
        if iFrame == tdAugmented.ts(idx)
            r = td.y(idx)*2;
            c = td.x(idx)*2;
            frame0((r-rad):(r+rad),(c-rad):(c+rad),:) = frameInt((r-rad):(r+rad),(c-rad):(c+rad),:);
            idx = idx+1;
        end
        if iFrame == tdCleaned.ts(jdx)
            r = td.y(jdx)*2;
            c = td.x(jdx)*2;
            frame0((r-rad):(r+rad),(c-rad):(c+rad),:) = frameInt((r-rad):(r+rad),(c-rad):(c+rad),:)*2;
            jdx = jdx+1;
        end
        if iFrame ==1
            
            figure(5423541);  clf;
            subplot(2,1,1);
            imh1 = imagesc(frameInt); axis image;
            tt1 = title(['idx =' num2str(idx)  '   iFrame = ' num2str(iFrame)]);
            
            subplot(2,1,2);hold on;
            imh2 = imagesc(frame0); axis image;
            tt2 = title(['idx =' num2str(idx)  '   iFrame = ' num2str(iFrame)]);
            plotHandle1 = plot(cx,ry,'o:w','linewidth',1) ;
            set(gca, 'YDir','reverse')
        else
            %set(plotHandle1,'XDATA',cx,'YDATA',ry)
            set(tt1,'String',['idx =' num2str(idx)  '   iFrame = ' num2str(iFrame)])
            set(imh1,'CDATA',   frameInt);
            set(tt2,'String',['idx =' num2str(idx)  '   iFrame = ' num2str(iFrame)])
            set(imh2,'CDATA',  frame0);
        end
        drawnow
    end
end

save([videoName + '.MP4_tds.mat'],'td','tdWithInh','tdCleaned','tdAugmented','limits','videoFileName')

%% DATA POINT INTERPOLATION
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
% time = 1:numel(userInputArray);

figure(456);
subplot(1,2,1);
scatter3(x_coor,y_coor,timeStamp,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
title("Waggle Labels per frame");
xlim([0 1920]); % 1920 1080
ylim([0 1080]);
subplot(1,2,2);
scatter3(objectTd.x,objectTd.y,objectTd.ts,'.');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
title("Waggle Labels with radius");
xlim([0 1920]);
ylim([0 1080]);

%% evaluation
load("./data/" + videoName + "_labels.mat") % load ground truth
load("./data/" + videoName + ".MP4_tds.mat") % load predicted labels

objectEventIndex = 0;objectTd = [];valid = 0;
x_coor = real(userInputArray(:,1));x_coor(x_coor==0) = NaN;
y_coor = -real(1i*userInputArray(:,1));y_coor(y_coor==0) = NaN;
timeStamp = 1:numel(userInputArray);

figure(56757);
subplot(2,2,1)
scatter3(td.x,td.y,td.ts,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
xlim([0 1920]); % 1920 1080
ylim([0 1080]);
title("Detected Waggle Dance");

subplot(2,2,2)
scatter3(x_coor,y_coor,timeStamp,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
xlim([0 1920]); % 1920 1080
ylim([0 1080]);
title("Waggle Dance Ground Truth");

v0 = VideoReader(videoName + ".MP4");
nFrameTotal = round(v0.FrameRate *v0.Duration);
outputLabel = nan(numel(y_coor),1);

for iFrame = 1:nFrameTotal
    valid = valid + 1;
    
    if iFrame < numel(td.y) && iFrame < numel(y_coor)
        if ((td.x(iFrame) - x_coor(iFrame))^2 + (td.y(iFrame) - y_coor(iFrame))^2)<userInputRadiusArray(iFrame)^2
            objectEventIndex = objectEventIndex +1;
            objectTd.x(objectEventIndex)     = td.x(iFrame);
            objectTd.y(objectEventIndex)     = td.y(iFrame);
            objectTd.ts(objectEventIndex)    = td.ts(iFrame);
            outputLabel(valid) = 1;
        else
            outputLabel(valid) = 0;
        end
    end
end

Accuracy = 1 - numel(find(outputLabel==0))/numel(td.x);
% Accuracy = numel(find(outputLabel==1))/numel(objectTd.x);

subplot(2,2,[3 4])
scatter3(objectTd.x,objectTd.y,objectTd.ts,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
xlim([0 1920]); % 1920 1080
ylim([0 1080]);
title("Correctly detected waggle dance \color{magenta}[Accuracy: " + num2str(Accuracy) + "%]");
