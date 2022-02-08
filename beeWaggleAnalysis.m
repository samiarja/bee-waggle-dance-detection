clear
tic;
addpath("../DeepGreen/greenhouseCode")

%%%%%%%%%%%%%%%% LOAD VIDEO
% videoName = "20210803t1727d200m";
videoName = "input_videos/20210803t1259d050m_cropped";
videoFileName = "./input_videos/" + videoName + ".MP4";
% limits.rowStart    = 401;    limits.rowEnd      = 1000;    limits.colStart    = 201;    limits.colEnd      = 1700;
limits.rowStart    = 1;    limits.rowEnd      = 800;    limits.colStart    = 1;    limits.colEnd      = 800;
v0 = VideoReader(videoName + ".MP4");
disp("Loading data...")

%%%%%%%%%%%%%%%% LOAD TEMPLATE
load('waggle16Templates_v1.mat')
disp("Template loaded...")

td                          = [];
tdWithInh                   = [];
jTd                         = [];
iWaggleInh                  = 0;
iWaggleEvent                = 0;
avgFrameDepth               = 6;
convMapThreshold            = 60;
nEventsForWaggleThreshold   = 10;
nDel                        = 18;
nTemplate                   = size(waggleTemplate25,3);
nFrameTotal                 = round(v0.FrameRate *v0.Duration);
numberofSegment             = 5;
framesPerSegment            = nFrameTotal/numberofSegment;
nSegment                    = ceil(nFrameTotal/framesPerSegment);
waggleLabellingROI          = [984 390 284 190];

writerObj = VideoWriter('./output_videos/simulation_20210803t1259d050m_bee_waggles_10fps.avi');
writerObj.FrameRate = 30;
open(writerObj);


segFrame = 0;
for iSegment = 1:numberofSegment
    if iSegment < 3
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
        imageWidth  = 800;
        imageHeight = 800;
        
        
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
            
            frameInt = frameIntFull(limits.rowStart:limits.rowEnd,limits.colStart:limits.colEnd,:);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            downsamplingFactor = 0.5;
            %         frameInt = imcrop(frameInt,waggleLabellingROI);
            frameArray(:,:,:,iFrame) =  imresize(frameInt,downsamplingFactor);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if iFrame>1
                dRgbFrameArray(:,:,:,iFrame) = single(frameArray(:,:,:,iFrame)) - single(frameArray(:,:,:,iFrame-1));
                dGreyScaleArray(:,:,iFrame) =  vecnorm(single(frameArray(:,:,:,iFrame)),2,3) -   vecnorm(single(frameArray(:,:,:,iFrame-1)),2,3);
            end
        end
        
        
        %     for iFrame = 1:nFrame
        %         figure(567);
        %         imh1 = imagesc(uint8(frameArray(:,:,:,iFrame)));axis image;
        %     end
        
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
        waggleMap1 = waggleMap1(:,:,:,1:nFrame);
        
        waggleMap2 = convn(dGreyScaleArray,waggleFilt4d2,'full');
        waggleMap2 = waggleMap2(:,:,:,1:nFrame);
        
        
        waggleMap3 = convn(dGreyScaleArray,waggleFilt4d3,'full');
        waggleMap3 = waggleMap3(:,:,:,1:nFrame);
        
        disp("Finish 3D convolution...")
        
        waggleMapMaxed =  zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        maxMat = waggleMap1(:,:,:,1) +nan;
        
        for iFrame = 1:nFrame
            maxMat(:,:,1) = vecnorm(waggleMap1(:,:,iFrame),2,3);
            maxMat(:,:,2) = vecnorm(waggleMap2(:,:,iFrame),2,3);
            maxMat(:,:,3) = vecnorm(waggleMap3(:,:,iFrame),2,3);
            waggleMapMaxed(:,:,iFrame) = max( maxMat,[], 3);
            
            %         figure(567);
            %         subplot(1,3,1)
            %         imagesc(maxMat(:,:,1));axis image;
            %         subplot(1,3,2)
            %         imagesc(maxMat(:,:,1));axis image;
            %         subplot(1,3,3)
            %         imagesc(maxMat(:,:,1));axis image;
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
            
            
            %%% 2D convolution between every meanWaggleMapFrame and each
            %%% template
            for iTemplate = 1:nTemplate
                waggleConvResult(:,:,iTemplate) = conv2(meanWaggleMapFrame,waggleTemplate25(:,:,iTemplate),'same');
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
            
            %         %%%%% show ROI on the detected bee
            %         if ~isempty(r) && (c-rad25>0) && (c+rad25<size(dGreyScaleArray,2)) && (r-rad25>0) && (r+rad25<size(dGreyScaleArray,1))
            %             iWaggleEvent = iWaggleEvent + 1;
            %             waggleRegion  = frameArray(r-rad25:r+rad25,c-rad25:c+rad25,:,iFrame);
            %             dwaggleRegion = dGreyScaleArray(r-rad25:r+rad25,c-rad25:c+rad25,iFrame);
            %
            %             figure(567);
            %             imh1 = imagesc(uint8(frameArray(:,:,:,iFrame))); hold on;
            %             rectangle('Position', [c, r, rad25*2, rad25*2], 'EdgeColor', 'r', 'LineWidth', 2);
            % %             imagesc(uint8(waggleRegion));axis image;
            %         end
            
            figure(567);
            subtightplot(3,3,1);
            imh1 =  imagesc(uint8(frameArray(:,:,:,iFrame)));axis image;colorbar; title("Input frames");colormap('gray');
            title(num2str(iFrame),'color','r','fontSize',14);
            subtightplot(3,3,2);
            imhx =  imagesc(dGreyScaleArray(:,:,iFrame));axis image;  colorbar; title("Normalised framediff");colormap('gray');caxis([-150 200] )
%                     caxis([0 512] )
            subtightplot(3,3,3);
            imh3 =  imagesc(waggleMapMaxed(:,:,iFrame));axis image;colorbar;title("1D convolution");colormap('gray');
                    caxis([-200 200] )
            subtightplot(3,3,4);
            imh4 =  imagesc( meanWaggleMapFrame  );axis image;  colorbar;title("Moving average");colormap('gray');
                    caxis([-250 250] )
            subtightplot(3,3,5);
            imh5 =  imagesc(waggleConvResultMaxedVal );axis image;  colorbar;title("2D convolution waggle map");colormap('gray');
                    caxis([-250 250] )
            subtightplot(3,3,6);
            imh6 =  imagesc(waggleConvThreshed );axis image;  colorbar;title("Threshold waggle map");colormap('gray');
                    caxis([-1 1] )
            subtightplot(3,3,8);
            imh7 =  imagesc(waggleDetectionMap);axis image;  colorbar;title("Detected waggle map");colormap('gray');
                    caxis([-1 1] )
            set(gcf,'Position',[100 100 1100 1000]);
            
            F = getframe(gcf) ;
            writeVideo(writerObj, F);
            
        end
    end
end
close(writerObj);
fprintf('Sucessfully generated the video\n')
toc;
%% plot all bees trajectory
figure(657);
Files=dir('/media/sami/Samsung_T5/MPhil/Code/bee_tracking/data/integration/*.mat');
for k=1:length(Files)
    FileNames=Files(k).name;
    
    load( "/media/sami/Samsung_T5/MPhil/Code/bee_tracking/data/integration/" + FileNames)
    time = 1:numel(beeTrajectory(1,:));
    scatter3(beeTrajectory(1,:),beeTrajectory(2,:),time,'.b'); 
    xlabel("X [px]");
    ylabel("Y [px]");
    zlabel("#Frame");hold on
end

figure(767);
scatter3(td.x,td.y,td.ts,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frame");


time = 1:numel(beeTrajectory(1,:));
scatter3(beeTrajectory(1,:),beeTrajectory(2,:),time,'.');
plot3(beeTrajectory(1,:),beeTrajectory(2,:),time,'-')
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frame");
zlim([0 400])
grid on;
