clear
addpath("/media/sami/Samsung_T5/MPhil/Code/DeepGreen/greenhouseCode")

%%%%%%%%%%%%%%%% LOAD VIDEO
videoName = "20210803t1727d200m.MP4";
videoFileName = "./data/" + videoName;
limits.rowStart    = 401;    limits.rowEnd      = 1000;    limits.colStart    = 201;    limits.colEnd      = 1700;
v0 = VideoReader(videoName);
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
SHOW                        = 1; %rand>5;
convMapThreshold            = 10;
nEventsForWaggleThreshold   = 10;
nDel                        = 18;
framesPerSegment            = 500;
RECORD_VIDEO                = SHOW && 0;

nTemplate = size(waggleTemplate25,3);
if SHOW
    [nSubplotRows,nSubplotCols] = goodSubPlotRowCols(nTemplate);
    % % % %% Show waggleTemplates
    for iTemplate = 1:nTemplate
        
        figure(1);
        subplot(nSubplotRows,nSubplotCols,iTemplate);imagesc(waggleTemplate25(:,:,iTemplate)); axis image; colorbar;
        set(gcf, 'Name', 'Waggle Dance Spatial Template')
    end
end
%v0 = VideoReader('E:\googleDrive\MATLAB\ML\data\bees\waggleDance\darwinExperiments\sixHiveExp\Honeybees01\20210803t1243d050m.MP4');
%rowStart    = 1;    rowEnd      = 100;    colStart    = 1;    colEnd      = 100;

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

nFrameTotal = round(v0.FrameRate *v0.Duration);
nSegment = ceil(nFrameTotal/framesPerSegment);
%nDel         = 36;

for iSegment = 1:5
    iSegment
    %startFrame = 1;%
    startFrame = max((iSegment-1)*framesPerSegment+1-nDel,1);
    
    %endFrame = nFrameTotal;%
    endFrame = min(iSegment*framesPerSegment,nFrameTotal);%500;
    
    nFrame =  endFrame -   startFrame ;
    
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
        % %                 figure(53451);
        % %                 imagesc(frameIntFull);axis image;
        % %                 figure(53452);
        % %                 imagesc(imresize(frameIntFull,.5));axis image;
        % %                 figure(53453);
        % %                 imagesc(imresize(frameIntFull,.25));axis image;
        % %                 figure(53454);
        % %                 imagesc(imresize(frameIntFull,1/8));axis image;
        % %
        % %         imagesc(frameInt);axis image;
        
        % downsample the data by two
        frameArray(:,:,:,iFrame) =  imresize(frameInt,0.5);
        if iFrame>1
            dRgbFrameArray(:,:,:,iFrame) = single(frameArray(:,:,:,iFrame)) - single(frameArray(:,:,:,iFrame-1));
            dGreyScaleArray(:,:,iFrame)  = imresize(vecnorm(single(frameArray(:,:,:,iFrame)),2,3) -   vecnorm(single(frameArray(:,:,:,iFrame-1)),2,3),0.5);
        end
    end
    
    disp("Frame differencing finished...")
    %     figure(423423); histogram(dRgbFrameArray(1e4:1e5));
    %     yLog;
    
    disp("Start 3D convolution...")
    %%%%  3D convolution
    sigma        = 10;
    Tau          = 36;
    delArray     = 1:nDel;
    %waggleFilt   = exp(-delArray/Tau).*sin(4/(2*pi)*(delArray+6.5));  % slowestOne
    %waggleFilt   = exp(-delArray/Tau).*sin(6.5/(2*pi)*(delArray+4));  % too fast
    waggleFilt1   = exp(-delArray/Tau).*sin(5.75/(2*pi)*(delArray+5.2));
    waggleFilt4d1 = single(reshape(waggleFilt1,[1,1,1,nDel]));
    
    waggleMap1 = convn(dRgbFrameArray,waggleFilt4d1,'full');
    waggleMap1 = waggleMap1(:,:,:,1:nFrame);
    
    waggleFilt2   = exp(-delArray/Tau).*sin(5/(2*pi)*(delArray+5.2));
    waggleFilt4d2 = single(reshape(waggleFilt2,[1,1,1,nDel]));
    waggleMap2 = convn(dRgbFrameArray,waggleFilt4d2,'full');
    waggleMap2 = waggleMap2(:,:,:,1:nFrame);
    
    waggleFilt3   =exp(-delArray/Tau).*sin(4/(2*pi)*(delArray+6.5));
    waggleFilt4d3 = single(reshape(waggleFilt3,[1,1,1,nDel]));
    waggleMap3 = convn(dRgbFrameArray,waggleFilt4d3,'full');
    waggleMap3 = waggleMap3(:,:,:,1:nFrame);
    disp("Finish 3D convolution...")
    
    %%% display convolution kernels
    if SHOW
        figure(2);
        subplot(2,2,1)
        plot(waggleFilt1,'*-','LineWidth',2); grid on;
        xlabel("Kernel Size");
        ylabel("Kernel value (0-1)");
        title("Convolution kernel layer 1");
        eqtext = '$$waggleFilt1={e^{\frac{-a}{\tau}} \ast sin(\frac{5.75}{2\pi(a+5.2)})}$$';
        ylim([-1 1.5]);
        text(0.5, 1.2, eqtext, 'Interpreter', 'Latex', 'FontSize', 18, 'Color', 'k')
        
        subplot(2,2,2)
        plot(waggleFilt2,'*-','LineWidth',2); grid on;
        xlabel("Kernel Size");
        ylabel("Kernel value (0-1)");
        title("Convolution kernel layer 2");
        eqtext = '$$waggleFilt2={e^{\frac{-a}{\tau}} \ast sin(\frac{5}{2\pi(a+5.2)})}$$';
        ylim([-1 1.5]);
        text(0.5, 1.2, eqtext, 'Interpreter', 'Latex', 'FontSize', 18, 'Color', 'k')
        
        subplot(2,2,[3 4])
        plot(waggleFilt3,'*-','LineWidth',2); grid on;
        xlabel("Kernel Size");
        ylabel("Kernel value (0-1)");
        title("Convolution kernel layer 3");
        eqtext = '$$waggleFilt3={e^{\frac{-a}{\tau}} \ast sin(\frac{4}{2\pi(a+6.5)})}$$';
        ylim([-1 1.5]);
        text(0.5, 1.2, eqtext, 'Interpreter', 'Latex', 'FontSize', 18, 'Color', 'k')
        set(gcf, 'Name', 'Convolution Kernel Layer 1');
    end
    
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
    
    if SHOW
        %%%% flattening each frame and take the max and average
        for iFrame = 1:nFrame
            maxWaggleVal(iFrame)  = max(mat2vec(waggleMapMaxed(:,:,iFrame)));
            meanWaggleVal(iFrame) = mean(mat2vec(waggleMapMaxed(:,:,iFrame)));
        end
        figure(3); clf;
        subplot(2,2,1)
        plot(1:nFrame,maxWaggleVal,"LineWidth",2); box on; grid on;
        xlabel("frames");
        ylabel("Max");
        title("Maximum waggle per frame");
        subplot(2,2,2)
        plot(1:nFrame,meanWaggleVal,"LineWidth",2); box on; grid on;
        xlabel("frames");
        ylabel("Mean");
        title("Average waggle per frame");
        subplot(2,2,[3 4])
        plot(1:nFrame,maxWaggleVal./meanWaggleVal,"LineWidth",2); box on; grid on;
        xlabel("frames");
        ylabel("Max/Mean");
        title("Max/Mean waggle per frame");
        set(gcf, 'Name', 'Waggle frequency');
    end
    
    %     waggleMapMaxedHalf =  zeros(imageHeight/2,imageWidth/2,nFrame,'single');
    %     for iFrame = 1:nFrame
    %         waggleMapMaxedHalf(:,:,iFrame) = imresize(waggleMapMaxed(:,:,iFrame),0.5);
    %     end
    
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
%         iFrame
        meanWaggleMapFrame = mean(waggleMapMaxedHalf(:,:,max(iFrame-avgFrameDepth,1):iFrame),3);
        
        %%% 2D convolution between every meanWaggleMapFrame and each
        %%% template
        for iTemplate = 1:nTemplate
            waggleConvResult(:,:,iTemplate) = conv2(meanWaggleMapFrame,waggleTemplate25(:,:,iTemplate),'same');
        end
        
        if SHOW
            for iTemplate = 1:nTemplate
                figure(567570);
                subplot(4,4,iTemplate)
                imagesc(waggleConvResult(:,:,iTemplate))
                set(gcf, 'Name', '2D Convolution on the frame differencing');
            end
        end
        
        [waggleConvResultMaxedVal, waggleTemplateIdx ]= max(waggleConvResult,[],3);
        
        resizedWaggleMatch          = imresize(waggleConvResultMaxedVal,.5);
        waggleConvThreshed          = resizedWaggleMatch>convMapThreshold;
        waggleConvThreshedMaxed     = waggleConvThreshed & resizedWaggleMatch==max(resizedWaggleMatch(:));% only one waggle detection per frame
        waggleDetectionMap          = (waggleConvThreshedMaxed & ~waggleConvInh);
       
%         if SHOW
%            figure(567);
%            subplot(2,2,1)
%            imagesc(resizedWaggleMatch);colorbar;
%            subplot(2,2,2)
%            imagesc(waggleConvThreshed);colorbar;
%            subplot(2,2,3)
%            imagesc(waggleConvThreshedMaxed);colorbar;
%            subplot(2,2,4)
%            imagesc(waggleDetectionMap);colorbar;
%         end
        
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
%                 if SHOW%rand>5
% %                     figure(345646);
% %                     imagesc(imrotate(dGreyScaleArray(r-rad25:r+rad25,c-rad25:c+rad25,iFrame),bestAngle));
% %                     colorbar;
%                     
% %                     figure(345646);
% %                     imagesc(imrotate(dGreyScaleArray(r-rad25:r+rad25,c-rad25:c+rad25,iFrame),bestAngle));
% %                     colorbar;
% %                     if (c-rad25>0) && (c+rad25<size(dGreyScaleArray,2)) && (r-rad25>0) && (r+rad25<size(dGreyScaleArray,1))
% %                         for jFrame = -20:20
% %                             
% %                             rotatedWaggleSection = imrotate(dGreyScaleArray(r-rad25:r+rad25,c-rad25:c+rad25,iFrame+jFrame),bestAngle);
% %                             figure(4523543)
% %                             imagesc(rotatedWaggleSection);
% %                             colorbar
% %                             caxis([-100 100])
% %                             drawnow 
% %                         end
% %                     end
%                     jdx = 0;
%                     for jFrame = -20:10
%                         rotatedWaggleSection = imrotate(dGreyScaleArray(r-rad25:r+rad25,c-rad25:c+rad25,iFrame+jFrame),bestAngle);
%                         [jRow, jCol, jVal] = find(abs(rotatedWaggleSection)>jThresh);
%                         for jj = 1:numel(jRow)
%                             jdx = jdx + 1;
%                             jr = jRow(jj);
%                             jc = jCol(jj);
%                             jTd(idxInh).x(jdx)= jc;
%                             jTd(idxInh).y(jdx)= jr;
%                             jTd(idxInh).ts(jdx)  = jFrame+startFrame-1;
%                             jTd(idxInh).p(jdx) = sign(rotatedWaggleSection(jr,jc));%fix(jVal(jj)/jThresh);
%                             jTd(idxInh).val(jdx) = rotatedWaggleSection(jr,jc);%fix(jVal(jj)/jThresh);
%                         end
%                     end
%                     jTd(idxInh).ts = jTd(idxInh).ts - jTd(idxInh).ts(1);
%                     figure(534513); clf; hold on;
%                     for ccc = -6:1:6
%                         jThisVal = find(fix(jTd(idxInh).val/10)==ccc);
%                         CCC = (ccc+6)/12;
%                         plot3(jTd(idxInh).x(jThisVal),jTd(idxInh).y(jThisVal),jTd(idxInh).ts(jThisVal),'.','color',[CCC 0 1-CCC ],'markersize',abs(ccc)*5+10)
%                     end
%                     box on; grid on;
%                     drawnow
%                 end
            end
        end
        waggleConvInh = min(max(waggleConvInh + single(conv2(waggleConvThreshed,inhDisc,'same')) - .1,0),1);
        
%         figure(567567);
%         subplot(2,1,1)
%         imh2 =  imagesc(uint8((dRgbFrameArray(:,:,:,iFrame))+127));axis image;
%         subplot(2,1,2)
%         imh3 =  imagesc(waggleMapMaxed(:,:,iFrame));axis image;
%         caxis([0 127])
%         colorbar;
        
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
                random_variable = 1;
%                 disp("nothing")
                
%                 set(titleHandle,'String',num2str(iFrame))
%                 set(imh1,'CDATA',   uint8(frameArray(:,:,:,iFrame)));
%                 set(imh2,'CDATA',   uint8((dRgbFrameArray(:,:,:,iFrame))))
%                 set(imh3,'CDATA',  waggleMapMaxed(:,:,iFrame) )
%                 set(imh4,'CDATA',  meanWaggleMapFrame)
%                 set(imh5,'CDATA',  waggleConvResultMaxedVal)
%                 set(imh6,'CDATA',  waggleConvThreshed)
%                 set(imh7,'CDATA',  waggleConvInh)
%                 set(imh8,'CDATA',  waggleDetectionMap)
%                 set(imh9,'CDATA', waggleTemplateIdx)
%                 set(imh10,'CDATA', dGreyScaleArray(:,:,iFrame))
                
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

% if SHOW
%     for idx = 1:size(tdAugmented.context,1)
%         figure(24346)
%         imshow(tdAugmented.context{idx, 1}), hold on, preview(v0) ;
%     end
% end

% figure(5234534)
% tdCleaned.meanAngle*360/2/pi;
% for idx = 1:numel(tdCleaned.x)
% end

% save([videoFileName '_tds.mat'],'td','tdWithInh','tdCleaned','tdAugmented','limits','videoFileName')



