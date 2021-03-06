clear
tic;
addpath("../DeepGreen/greenhouseCode")

%%%%%%%%%%%%%%%% LOAD VIDEO

% videoName = "20210803t1727d200m";
% dataPATH = "input_videos/20210803t1259d050m_cropped";

dataPATH = "./final_labels/20210803t1301d050m_cropped/";
load(dataPATH + '20210803t1301d050m_ground_truth.mat');
videoFileName = dataPATH + "20210803t1301d050m_cropped.MP4";

% limits.rowStart    = 401;    limits.rowEnd      = 1000;    limits.colStart    = 201;    limits.colEnd      = 1700;
% limits.rowStart    = 1;    limits.rowEnd      = 800;    limits.colStart    = 1;    limits.colEnd      = 800;
v0 = VideoReader(videoFileName);
disp("Loading data...")

%%%%%%%%%%%%%%%% LOAD TEMPLATE
load('waggle36Templates_25x25_HQ.mat')
% load('waggle16Templates_v1.mat')
disp("Template loaded...")

SHOW                        = 0;
RECORD_VIDEO                = SHOW && 1;
td                          = [];
tdWithInh                   = [];
jTd                         = [];
iWaggleInh                  = 0;
iWaggleEvent                = 0;
avgFrameDepth               = 6;
convMapThreshold            = 15;
nEventsForWaggleThreshold   = 6;
nDel                        = 18;
nTemplate                   = size(waggleTemplate,3);
nFrameTotal                 = round(v0.FrameRate *v0.Duration)+5;
numberofSegment             = 40;
framesPerSegment            = nFrameTotal/numberofSegment;
nSegment                    = ceil(nFrameTotal/framesPerSegment);
AllFrames = 0;

% writerObj = VideoWriter('./output_videos/simulation_20210803t1259d050m_bee_waggles_10fps_cropped_new.avi');
% writerObj.FrameRate = 30;
% open(writerObj);


segFrame = 0;
for iSegment = 1:numberofSegment
    if iSegment < numberofSegment
        iSegment
        
        iFrame = 0;
        %startFrame = 1;%
        startFrame = max((iSegment-1)*framesPerSegment+1-nDel,1);
        
        %endFrame = nFrameTotal;
        endFrame = min(iSegment*framesPerSegment,nFrameTotal);
        
        nFrame =  endFrame - startFrame;
        
        iFrameWithWaggle = 0;waggleStats = {};
        
        %%%%%%%% for 20210803t1727d200m
        %     imageWidth  = 286;
        %     imageHeight = 192;
        %%%%%%%%
        
        %%%%%%%% for 20210803t1259d050m_cropped
        imageWidth  = v0.Width;
        imageHeight = v0.Height;
        
        
        %     imageWidth                              = limits.colEnd - limits.colStart +1;
        %     imageHeight                             = limits.rowEnd - limits.rowStart +1;
        downsamplingFactorframeArray            = 2; %2;
        frameArray                              = zeros(round(imageHeight/downsamplingFactorframeArray),round(imageWidth/downsamplingFactorframeArray),3,nFrame,'uint8');
        downsamplingFactorRGB                   = 2; %2;
        dRgbFrameArray                          = zeros(round(imageHeight/downsamplingFactorRGB),round(imageWidth/downsamplingFactorRGB),3,nFrame,'single');
        downsamplingFactorGreyScale             = 4; %4;
        dGreyScaleArray                         = zeros(round(imageHeight/downsamplingFactorRGB),round(imageWidth/downsamplingFactorRGB),nFrame,'single');
        
        for iFrameOriginal = startFrame:endFrame
            iFrame = iFrame + 1;
            
            
            frameIntFull = read(v0,iFrameOriginal);
            
            %             frameInt = frameIntFull(limits.rowStart:limits.rowEnd,limits.colStart:limits.colEnd,:);
            frameInt = frameIntFull;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            downsamplingFactor = 0.5;
            %             frameInt = imcrop(frameInt,waggleLabellingROI);
            frameArray(:,:,:,iFrame) =  imresize(frameInt,downsamplingFactor);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if iFrame>1
                dRgbFrameArray(:,:,:,iFrame) = single(frameArray(:,:,:,iFrame)) - single(frameArray(:,:,:,iFrame-1));
                dGreyScaleArray(:,:,iFrame) =  vecnorm(single(frameArray(:,:,:,iFrame)),2,3) -   vecnorm(single(frameArray(:,:,:,iFrame-1)),2,3);
            end
        end
        
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
        
        waggleMap1 = convn(dGreyScaleArray,waggleFilt4d1,'full');
        waggleMap1 = waggleMap1(:,:,1:nFrame);
        
        waggleMap2 = convn(dGreyScaleArray,waggleFilt4d2,'full');
        waggleMap2 = waggleMap2(:,:,1:nFrame);
        
        
        waggleMap3 = convn(dGreyScaleArray,waggleFilt4d3,'full');
        waggleMap3 = waggleMap3(:,:,1:nFrame);
        
        disp("Finish 3D convolution...")
        
        waggleMapMaxed =  zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        maxMat = waggleMap1(:,:,:,1) +nan;
        
        for iFrame = 1:nFrame
            maxMat(:,:,1) = vecnorm(waggleMap1(:,:,iFrame),2,3);
            maxMat(:,:,2) = vecnorm(waggleMap2(:,:,iFrame),2,3);
            maxMat(:,:,3) = vecnorm(waggleMap3(:,:,iFrame),2,3);
            waggleMapMaxed(:,:,iFrame) = max( maxMat,[], 3);
        end
        
        waggleMap1 = [];
        waggleMap2 = [];
        waggleMap3 = [];
        
        waggleMapMaxedHalf = waggleMapMaxed;
        
        waggleConvResult = zeros(round(imageHeight/2),round(imageWidth/2),nTemplate,'single');
        waggleConvInh=zeros(round(imageHeight/4),round(imageWidth/4),'single');
        waggleDetectionMap = waggleConvInh;
        
        inhDisc=fspecial('disk',9)>0;
        
        angleArray = 1:180;
        angleArrayActivation = angleArray + nan;
        rad25 = 7;
        rad50 = rad25*2;
        jThresh = 20;
        
        waggleConvFinalMaxed =  zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        
        for iFrame = 1:nFrame
            meanWaggleMapFrame = mean(waggleMapMaxedHalf(:,:,max(iFrame-avgFrameDepth,1):iFrame),3);
            segFrame = segFrame + 1;
            AllFrames = AllFrames + 1;
            
            %%% 2D convolution between every meanWaggleMapFrame and each
            %%% template
            for iTemplate = 1:nTemplate
                waggleConvResult(:,:,iTemplate) = conv2(meanWaggleMapFrame,waggleTemplate(:,:,iTemplate),'same');
            end
            
            [waggleConvResultMaxedVal, waggleTemplateIdx ]= max(waggleConvResult,[],3);
            
            waggleConvFinalMaxed(:,:,iFrame) = max(waggleConvResult,[],3);
            
            resizedWaggleMatch          = imresize(waggleConvResultMaxedVal,.5);
            waggleConvThreshed          = resizedWaggleMatch>convMapThreshold;
            waggleConvThreshedMaxed     = waggleConvThreshed & resizedWaggleMatch==max(resizedWaggleMatch(:));% only one waggle detection per frame
            waggleDetectionMap          = (waggleConvThreshedMaxed & ~waggleConvInh);
            
            % find waggle events, find their orientation
            [iRow, iCol, ~] = find(waggleConvThreshedMaxed>0);
            [r, c, ~] = find(waggleConvThreshedMaxed>0);
            
%             figure(567);
%             subtightplot(3,3,1);
%             imh1 =  imagesc(uint8(frameArray(:,:,:,iFrame)));axis image;colorbar; title("Input frames");colormap('gray');
%             title(num2str(iFrame),'color','r','fontSize',14);
%             subtightplot(3,3,2);
%             imhx =  imagesc(dGreyScaleArray(:,:,iFrame));axis image;  colorbar; title("Normalised framediff");colormap('gray');
%             caxis([-100 100] )
%             subtightplot(3,3,3);
%             imh3 =  imagesc(waggleMapMaxed(:,:,iFrame));axis image;colorbar;title("1D convolution");colormap('gray');
%             caxis([0 150] )
%             subtightplot(3,3,4);
%             imh4 =  imagesc( meanWaggleMapFrame  );axis image;  colorbar;title("Moving average");colormap('gray');
%             caxis([0 150] )
%             subtightplot(3,3,5);
%             imh5 =  imagesc(waggleConvResultMaxedVal );axis image;  colorbar;title("2D convolution waggle map");colormap('gray');
%             caxis([0 150] )
%             subtightplot(3,3,6);
%             imh6 =  imagesc(waggleConvThreshed );axis image;  colorbar;title("Threshold waggle map");colormap('gray');
%             caxis([0 1] )
%             subtightplot(3,3,8);
%             imh7 =  imagesc(waggleDetectionMap);axis image;  colorbar;title("Detected waggle map");colormap('gray');
%             %             caxis([0 1] )
%             set(gcf,'Position',[100 100 1000 1000]);
%             
%             F = getframe(gcf) ;
%             writeVideo(writerObj, F);
            
            
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
                td.frameID(iWaggleEvent)= AllFrames;
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
                    tdWithInh.frameID(iWaggleInh) = iFrame;
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
                    caxis([0 100])
                    colorbar;
                    
                    subtightplot(5,2,4); % avg conv with the kernel
                    imh4 =  imagesc( meanWaggleMapFrame  );axis image;  colorbar;
                    caxis([0 250] )
                    
                    subtightplot(5,2,5); % conv of the previous
                    imh5 =  imagesc(waggleConvResultMaxedVal );axis image;  colorbar;
                    caxis([0 250] )
                    
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
end

% close(writerObj);
% fprintf('Sucessfully generated the video\n')


figure(6565);
scatter3(td.x*4,td.y*4,td.ts,'.r');hold on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");

% figure(67);
% subplot(2,1,1)
% plot(td_gt.x,'b');hold on
% plot(td.x*4,'r');
% subplot(2,1,2)
% plot(td_gt.y,'b');hold on
% plot(td.y*4,'r');

toc;
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

save([dataPATH + '.MP4_tds.mat'],'td','tdWithInh','tdCleaned','tdAugmented','limits','videoFileName')

%% Ground truth and predicted waggle post-processing
load('./final_labels/20210803t1727d200m_cropped/20210803t1727d200m_cropped_TD_25x25Template.mat')
load('./final_labels/20210803t1727d200m_cropped/20210803t1727d200m_ground_truth.mat')

% ground truth setup
groundtruth = [];
counter = 0;
frame_index_Gt = 0;
frameID = td_gt.frameID(1);

for idx = 1:td_gt.frameID(end)
    frame_index_Gt = frame_index_Gt + 1;
    if frameID > idx
        groundtruth.x(idx)      = 0;
        groundtruth.y(idx)      = 0;
        groundtruth.angle(idx)  = 0;
        groundtruth.c(idx)      = 0;
        groundtruth.frameID(idx)      = frame_index_Gt;
    else
        counter = counter + 1;
        groundtruth.x(idx)      = td_gt.x(counter);
        groundtruth.y(idx)      = td_gt.y(counter);
        groundtruth.angle(idx)  = td_gt.angle(counter);
        groundtruth.c(idx)      = 1;
        groundtruth.frameID(idx)      = frame_index_Gt;
    end
end
td_gt = groundtruth;

% detected waggle setup
d = diff(td.frameID);
d(d==0) = 1;
ivec = cumsum([1 d]);
y = nan(1,ivec(end));
y(ivec) = td.x;td.x = y;
td.x(isnan(td.x))=0;
y(ivec) = td.y;td.y = y;
td.y(isnan(td.y))=0;
y(ivec) = td.ts;td.ts = y;
td.ts(isnan(td.ts))=0;
y(ivec) = td.angle;td.angle = y;
td.angle(isnan(td.angle))=0;
y(ivec) = td.frameID;td.frameID = y;
td.frameID(isnan(td.frameID))=0;

counter = 0;
frame_index_pred = 0;
pred_waggle = [];
pred_waggleLabel = [];
frameID = td.frameID(1);
pred_waggleLabel(find(td.frameID==0))=0;
pred_waggleLabel(find(td.frameID~=0))=1;
for idx = 1:td.frameID(end)
    frame_index_pred = frame_index_pred + 1;
    if frameID > idx
        pred_waggle.x                   = 0;
        pred_waggle.y(idx)              = 0;
        pred_waggle.angle(idx)          = 0;
        pred_waggle.c(idx)              = 0;
        pred_waggle.frameID(idx)        = frame_index_pred;
    else
        counter = counter + 1;
        pred_waggle.x(idx)      = td.x(counter);
        pred_waggle.y(idx)      = td.y(counter);
        pred_waggle.angle(idx)  = td.angle(counter);
        pred_waggle.c(idx)      = pred_waggleLabel(counter);
        pred_waggle.frameID(idx)      = frame_index_pred;
    end
end

pred_waggle.x(td.frameID(end):td_gt.frameID(end)) = 0;
pred_waggle.y(td.frameID(end):td_gt.frameID(end)) = 0;
pred_waggle.angle(td.frameID(end):td_gt.frameID(end)) = 0;
pred_waggle.frameID(td.frameID(end):td_gt.frameID(end)) = td.frameID(end):td_gt.frameID(end);
pred_waggle.c(td.frameID(end):td_gt.frameID(end)) = 0;
td = pred_waggle;

% figure(67);
% scatter3(td.x*4,td.y*4,td.frameID,'.r');hold on
% scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');

%% Final evaluation
videoFileName = "final_labels/20210803t1727d200m_cropped/20210803t1727d200m_croppedv2.MP4";
load('final_labels/20210803t1727d200m_cropped/20210803t1727d200m_ground_truth.mat')
load('final_labels/20210803t1727d200m_cropped/20210803t1727d200m_cropped_TD_25x25Template.mat')
v0 = VideoReader(videoFileName);
nFrameTotal = round(v0.FrameRate *v0.Duration);

% nFrameTotal = 2400;
beePixelSize = 70;
missclassified = 0;
correctlyclassified = 0;
td_missclassified = [];
td_correctlyclassified = [];

for idx = 1:numel(td.x)
    if ((td.x(idx)*4 - td_gt.x(idx))^2 + (td.y(idx)*4 - td_gt.y(idx))^2)<beePixelSize^2
        correctlyclassified = correctlyclassified +1;
        td_correctlyclassified.x(correctlyclassified) = td.x(idx)*4;
        td_correctlyclassified.y(correctlyclassified) = td.y(idx)*4;
        td_correctlyclassified.ts(correctlyclassified) = td.ts(idx);
        td_correctlyclassified.frameID(correctlyclassified) = td.frameID(idx);
    else
        missclassified = missclassified +1;
        td_missclassified.x(missclassified) = td.x(idx)*4;
        td_missclassified.y(missclassified) = td.y(idx)*4;
        td_missclassified.ts(missclassified) = td.ts(idx);
    end
end

oneHotEncodedLabels_td    = nan(nFrameTotal,1); %sequence
oneHotEncodedLabels_td_gt = nan(nFrameTotal,1); %sequence

for idx = 1:nFrameTotal
    if find(td.ts==idx) > 0
        oneHotEncodedLabels_td(idx) = 1;
    else
        oneHotEncodedLabels_td(idx) = 0;
    end
end

for idx = 1:nFrameTotal
    if find(td_gt.frameID==idx) > 0
        oneHotEncodedLabels_td_gt(idx) = 1;
    else
        oneHotEncodedLabels_td_gt(idx) = 0;
    end
end

correctlyLabeledPositives_TP = sum(and(oneHotEncodedLabels_td_gt,oneHotEncodedLabels_td),1);   % TP
correctlyLabeledNegatives_TN = sum(and(~oneHotEncodedLabels_td_gt,~oneHotEncodedLabels_td),1); % TN

wronglyLabeledPositives_FP = sum((oneHotEncodedLabels_td_gt == 1) & (oneHotEncodedLabels_td == 0)); % FP
wronglyLabeledNegatives_FP = sum((oneHotEncodedLabels_td_gt == 0) & (oneHotEncodedLabels_td == 1)); % FN

actualpositives = numel(oneHotEncodedLabels_td_gt(oneHotEncodedLabels_td_gt(:,1) > 0,1)); % TP+FN
actualnegatives = numel(oneHotEncodedLabels_td_gt(oneHotEncodedLabels_td_gt(:,1) <  1,1)); % TN+FP

confusion_matrix = [correctlyLabeledPositives_TP wronglyLabeledPositives_FP ; wronglyLabeledNegatives_FP correctlyLabeledNegatives_TN]

sensitivity = correctlyLabeledPositives_TP ./ actualpositives % TP/TP+FN
specificity = correctlyLabeledNegatives_TN ./ actualnegatives % TN/TN+FP
precision = correctlyLabeledPositives_TP / (correctlyLabeledPositives_TP + wronglyLabeledPositives_FP)
recall = correctlyLabeledPositives_TP / (correctlyLabeledPositives_TP + wronglyLabeledNegatives_FP)
F1 = (2 * precision * recall) / (precision + recall)
informedness = sensitivity + specificity - 1
accuracy = (1 - missclassified/(numel(td.x)))*100
% accuracy = (1 - missclassified/(missclassified + correctlyclassified))*100


figure(6565);
subplot(1,2,1)
scatter3(td.x*4,td.y*4,td.ts,'.r');hold on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
plot3(td_gt.x,td_gt.y,td_gt.frameID, 'bo', 'MarkerSize', 5);
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
title("Detected waggles information");
legend([{'Detected Waggles'},{'Labelled Waggles'}]);
set(gca,'fontsize', 14)
subplot(1,2,2)
scatter3(td_correctlyclassified.x,td_correctlyclassified.y,td_correctlyclassified.ts,'.r');hold on
scatter3(td_missclassified.x,td_missclassified.y,td_missclassified.ts,'.k');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
title("Waggles evaluation, Accuracy: " + accuracy + "%");
legend([{'Correctly Detected Waggles'},{'Wrongly Detected Waggles'}]);
set(gca,'fontsize', 14);

figure(67);
scatter3(td_correctlyclassified.x,td_correctlyclassified.y,td_correctlyclassified.ts,'.r');hold on;grid on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
xlim([100 800]);
ylim([200 900]);
zlim([0 3000]);
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
set(gca,'fontsize', 14);

figure(70);
subplot(2,1,1)
plot(td_correctlyclassified.frameID,td_correctlyclassified.x,'-r');hold on;grid on
plot(td_gt.x,'-b');
xlabel("#Frames");
ylabel("X [px]");
title("Across X");
legend("Correcly Detected Waggle","Ground truth");
set(gca,'fontsize', 18);
subplot(2,1,2)
plot(td_correctlyclassified.frameID,td_correctlyclassified.y/1.05,'-r');hold on;grid on
plot(td_gt.y,'-b');
xlabel("#Frames");
ylabel("Y [px]");
title("Across Y");
legend("Correcly Detected Waggle","Ground truth");
set(gca,'fontsize', 18);

% %% Evaluation for single file over multiple iterationstd_gt (through threshold range)
% PATH = "final_labels/20210803t1259d050m_cropped/";
% load(PATH + '/20210803t1259d050m_ground_truth.mat');
% TD = "td_with_threshold_";
% 
% convMapThresholdRange   = 24;
% nFrameTotal             = 2400;
% beePixelSize            = 150;
% sensitivity             = nan(convMapThresholdRange,1);
% specificity             = nan(convMapThresholdRange,1);
% precision               = nan(convMapThresholdRange,1);
% recall                  = nan(convMapThresholdRange,1);
% F1                      = nan(convMapThresholdRange,1);
% informedness            = nan(convMapThresholdRange,1);
% accuracy                = nan(convMapThresholdRange,1);
% 
% for file_index = 1:convMapThresholdRange
%     missclassified          = 0;
%     correctlyclassified     = 0;
%     td_missclassified       = [];
%     td_correctlyclassified  = [];
%     load(PATH + TD + num2str(file_index) + ".mat");
%     
%     if isempty(td)
%         sensitivity(file_index)     = 0; % TP/TP+FN
%         specificity(file_index)     = 0; % TN/TN+FP
%         precision(file_index)       = 0;
%         recall(file_index)          = 0;
%         F1(file_index)              = 0;
%         informedness(file_index)    = 0;
%         accuracy(file_index)        = 0;
%     else
%         %     if ~isempty(td)
%         if numel(td.x) > numel(td_gt.x)
%             sequence = numel(td_gt.x);
%         else
%             sequence = numel(td.x);
%         end
%         %     end
%         for idx = 1:sequence
%             if ((td.x(idx)*4 - td_gt.x(idx))^2 + (td.y(idx)*4 - td_gt.y(idx))^2)<beePixelSize^2
%                 correctlyclassified = correctlyclassified +1;
%                 td_correctlyclassified.x(correctlyclassified) = td.x(idx)*4;
%                 td_correctlyclassified.y(correctlyclassified) = td.y(idx)*4;
%                 td_correctlyclassified.ts(correctlyclassified) = td.ts(idx);
%             else
%                 missclassified = missclassified +1;
%                 td_missclassified.x(missclassified) = td.x(idx)*4;
%                 td_missclassified.y(missclassified) = td.y(idx)*4;
%                 td_missclassified.ts(missclassified) = td.ts(idx);
%             end
%         end
%         oneHotEncodedLabels_td    = nan(nFrameTotal,1); %sequence
%         oneHotEncodedLabels_td_gt = nan(nFrameTotal,1); %sequence
%         for idx = 1:nFrameTotal
%             if find(td.ts==idx) > 0
%                 oneHotEncodedLabels_td(idx) = 1;
%             else
%                 oneHotEncodedLabels_td(idx) = 0;
%             end
%         end
%         for idx = 1:nFrameTotal
%             if find(td_gt.frameID==idx) > 0
%                 oneHotEncodedLabels_td_gt(idx) = 1;
%             else
%                 oneHotEncodedLabels_td_gt(idx) = 0;
%             end
%         end
%         
%         correctlyLabeledPositives_TP = sum(and(oneHotEncodedLabels_td_gt,oneHotEncodedLabels_td),1);   % TP
%         correctlyLabeledNegatives_TN = sum(and(~oneHotEncodedLabels_td_gt,~oneHotEncodedLabels_td),1); % TN
%         wronglyLabeledPositives_FP = sum((oneHotEncodedLabels_td_gt == 1) & (oneHotEncodedLabels_td == 0)); % FP
%         wronglyLabeledNegatives_FP = sum((oneHotEncodedLabels_td_gt == 0) & (oneHotEncodedLabels_td == 1)); % FN
%         actualpositives = numel(oneHotEncodedLabels_td_gt(oneHotEncodedLabels_td_gt(:,1) > 0,1)); % TP+FN
%         actualnegatives = numel(oneHotEncodedLabels_td_gt(oneHotEncodedLabels_td_gt(:,1) <  1,1)); % TN+FP
%         
%         %%%%%%%%%% FINAL EVALUATION %%%%%%%%%%
%         sensitivity(file_index) = correctlyLabeledPositives_TP ./ actualpositives; % TP/TP+FN
%         specificity(file_index) = correctlyLabeledNegatives_TN ./ actualnegatives; % TN/TN+FP
%         precision(file_index) = correctlyLabeledPositives_TP / (correctlyLabeledPositives_TP + wronglyLabeledPositives_FP);
%         recall(file_index) = correctlyLabeledPositives_TP / (correctlyLabeledPositives_TP + wronglyLabeledNegatives_FP);
%         F1(file_index) = (2 * precision(file_index) * recall(file_index)) / (precision(file_index) + recall(file_index));
%         informedness(file_index) = sensitivity(file_index) + specificity(file_index) - 1;
%         accuracy(file_index) = (1 - missclassified/(missclassified + correctlyclassified))*100;
%     end
% end
% 
% figure(66);
% subplot(2,1,1);
% plot(F1,"LineWidth",2);grid on;
% x_points = [7, 7, 11, 11];
% y_points = [0, 1, 1, 0];
% color = [0, 0, 1];
% title("F1 -Score");
% xlabel("Threshold");
% hold on;
% a = fill(x_points, y_points, color);
% a.FaceAlpha = 0.1;
% set(gca,'fontsize', 14);
% subplot(2,1,2);
% plot(informedness,"LineWidth",2);grid on;ylim([0 0.43]);
% x_points = [7, 7, 11, 11];
% y_points = [0, 0.43, 0.43, 0];
% color = [0, 0, 1];
% title("Informedness");
% xlabel("Threshold");
% hold on;
% a = fill(x_points, y_points, color);
% a.FaceAlpha = 0.1;
% set(gca,'fontsize', 14);

