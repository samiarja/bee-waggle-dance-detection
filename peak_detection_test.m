clear
tic;
addpath("../DeepGreen/greenhouseCode")

% % %        ['20210803t1727d200m_cropped';  --> Done
% % %         '20210803t1301d050m_cropped';  --> Done
% % %         '20210803t1508d100m_cropped';  --> Done
% % %         '20210803t1517d100m_cropped';  --> Done
% % %         '20210803t1719d200m_cropped';  --> Done
% % %         '20210803t1259d050m_cropped';] --> Done


% videoFileName = "/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1301d050m_cropped/20210803t1301d050m_cropped.MP4";
% load('/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1301d050m_cropped/20210803t1301d050m_cropped_ground_truth.');

videoFileName = "/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1727d200m_cropped/20210803t1727d200m_cropped.MP4";
load("/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1727d200m_cropped/20210803t1727d200m_cropped_ground_truth.mat");

% videoName = {'20210803t1301d050m_cropped';
%         '20210803t1508d100m_cropped'; '20210803t1719d200m_cropped'; '20210803t1517d100m_cropped';
%         '20210803t1732d200m_cropped_W1'};

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
iRow                        = [];
iCol                        = [];
iWaggleInh                  = 0;
iWaggleEvent                = 0;
frameID                     = 0;
avgFrameDepth               = 6;
convMapThreshold            = 15;
nEventsForWaggleThreshold   = 6;
nDel                        = 18;
nTemplate                   = size(waggleTemplate,3);
nFrameTotal                 = round(v0.FrameRate *v0.Duration);
numberofSegment             = 40;
framesPerSegment            = nFrameTotal/numberofSegment;
nSegment                    = ceil(nFrameTotal/framesPerSegment);
AllFrames = 0;

% writerObj = VideoWriter('./output_videos/peak_detector_working_algorithm2.avi');
% writerObj.FrameRate = 10;
% open(writerObj);

segFrame = 0;
for iSegment = 1:numberofSegment
    if iSegment < numberofSegment
        iSegment
        
        iFrame = 0;
        startFrame = max((iSegment-1)*framesPerSegment+1-nDel,1);
        endFrame = min(iSegment*framesPerSegment,nFrameTotal);
        nFrame =  endFrame - startFrame;
        iFrameWithWaggle = 0;waggleStats = {};
        imageWidth  = v0.Width;
        imageHeight = v0.Height;
        
        downsamplingFactorframeArray            = 2; %2;
        frameArray                              = zeros(round(imageHeight/downsamplingFactorframeArray),round(imageWidth/downsamplingFactorframeArray),3,nFrame,'uint8');
        downsamplingFactorRGB                   = 2; %2;
        dRgbFrameArray                          = zeros(round(imageHeight/downsamplingFactorRGB),round(imageWidth/downsamplingFactorRGB),3,nFrame,'single');
        downsamplingFactorGreyScale             = 4; %4;
        dGreyScaleArray                         = zeros(round(imageHeight/downsamplingFactorRGB),round(imageWidth/downsamplingFactorRGB),nFrame,'single');
        
        for iFrameOriginal = startFrame:endFrame
            iFrame = iFrame + 1;
            
            frameIntFull = read(v0,iFrameOriginal);
            frameInt = frameIntFull;
            downsamplingFactor = 0.5;
            frameArray(:,:,:,iFrame) =  imresize(frameInt,downsamplingFactor);
            
            if iFrame>1
                dRgbFrameArray(:,:,:,iFrame) = single(frameArray(:,:,:,iFrame)) - single(frameArray(:,:,:,iFrame-1));
                dGreyScaleArray(:,:,iFrame) =  vecnorm(single(frameArray(:,:,:,iFrame)),2,3) -   vecnorm(single(frameArray(:,:,:,iFrame-1)),2,3);
            end
        end
        
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
        
        %%%%% run the conv on the actual frame and the actual greyscale
        %%%%% representation
        waggleConvResult = zeros(round(imageHeight/2),round(imageWidth/2),nTemplate,'single');
        waggleConvFinalMaxed =  zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        waggleConvFinalMaxed_LPF =  zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        
        for iFrame = 1:nFrame
            meanWaggleMapFrame = mean(waggleMapMaxedHalf(:,:,max(iFrame-avgFrameDepth,1):iFrame),3);
            segFrame = segFrame + 1;
            AllFrames = AllFrames + 1;
            
            %%% 2D convolution between every meanWaggleMapFrame and each
            %%% template
            newframe = rgb2gray(frameArray(:,:,:,iFrame));
            for iTemplate = 1:nTemplate
%                 waggleConvResult(:,:,iTemplate) = conv2(dGreyScaleArray(:,:,iFrame),waggleTemplate(:,:,iTemplate),'same');
                waggleConvResult(:,:,iTemplate) = conv2(meanWaggleMapFrame,waggleTemplate(:,:,iTemplate),'same');
            end
            
            [waggleConvResultMaxedVal, waggleTemplateIdx ]= max(waggleConvResult,[],3);
            
            waggleConvFinalMaxed(:,:,iFrame) = max(waggleConvResult,[],3);
            waggleConvFinalMaxed_LPF(:,:,iFrame)=imgaussfilt(waggleConvFinalMaxed(:,:,iFrame),4);
            
%             figure(676);
%             thres = 3e4; %value for binarisation, my image is uint coded and has maximum values of about 200
%             filt = (fspecial('gaussian', 70, 2)); %my peaks are nearly gaussiian with an amplitude of 200 and a sigma of about 15
%             edg=2; %I dont know exactly, but I thingk this is the distance to the edgo of the image zo exclude from search
%             res=2; %1: Local maximum 20%faster than 2: weighted centroid with subpixel polation
%             p=FastPeakFind(waggleConvResultMaxedVal,thres, filt,edg,res);
%             imagesc(waggleConvResultMaxedVal);colorbar; hold on
%             plot(p(1:2:end),p(2:2:end),'rs');
            
            new_waggleConvResultMaxedVal = waggleConvResultMaxedVal;
            new_waggleConvResultMaxedVal(:,1:40)=0;
            new_waggleConvResultMaxedVal(1:40,:)=0;
            new_waggleConvResultMaxedVal(:,260:end)=0;
            new_waggleConvResultMaxedVal(260:end,:)=0;
            
            summation = nan(size(new_waggleConvResultMaxedVal,1),2);
            for iteration = 1:size(new_waggleConvResultMaxedVal,1)
                summation(iteration,1) = sum(new_waggleConvResultMaxedVal(iteration,:));
                summation(iteration,2) = sum(new_waggleConvResultMaxedVal(:,iteration));
            end            
            
            MinPeakDistance = 50;
            % Improve the estimate of the cycle duration by ignoring peaks that are very close to each other
            % Restrict the acceptable peak-to-peak separations to values greater than 200 pixels
            [value_row,location_row]=findpeaks(summation(:,1),'SortStr','descend','MinPeakDistance',MinPeakDistance);
            [value_column,location_column]=findpeaks(summation(:,2),'SortStr','descend','MinPeakDistance',MinPeakDistance);
            
%             figure(47567);
%             subplot(2,3,[1 3])
%             % chooses the tallest peak in the signal and eliminates all peaks within 200 pixels of it
%             plot(summation(:,1));hold on
%             findpeaks(summation(:,1),'MinPeakDistance',MinPeakDistance,'Annotate','extents');
%             text(location_row+.02,value_row,num2str((1:numel(value_row))'))
%             plot(summation(:,2));
%             findpeaks(summation(:,2),'MinPeakDistance',MinPeakDistance,'Annotate','extents');
%             text(location_column+.02,value_column,num2str((1:numel(value_column))'))
%             xlim([40 260]);

%             subplot(2,3,4)
%             imagesc(uint8(frameArray(:,:,:,iFrame)));

%             subplot(2,3,5)
%             imagesc(waggleConvResultMaxedVal);colorbar;hold on
%             if ~isempty(location_column) && ~isempty(location_row)
%                 plot(location_column(1), location_row(1),"ro",'MarkerSize',30,'LineWidth',2);
%             end

%             %%%%%%%%%%%%%% example
%             figure(670);
%             subplot(3,1,1);
%             plot(summation(:,1));hold on;
%             findpeaks(summation(:,1),'MinPeakDistance',MinPeakDistance,'Annotate','extents','SortStr','descend');
%             text(location_row+.02,value_row,num2str((1:numel(value_row))'));xlim([40 260]);
%             plot(summation(:,2));hold on;
%             findpeaks(summation(:,2),'MinPeakDistance',MinPeakDistance,'Annotate','extents','SortStr','descend');
%             text(location_column+.02,value_column,num2str((1:numel(value_column))'));xlim([40 260]);
%             subplot(3,1,2);
% 
%             imagesc(waggleConvResultMaxedVal);colorbar;hold on
%             if ~isempty(location_column) && ~isempty(location_row)
%                 plot(location_column(1), location_row(1),"ro",'MarkerSize',30,'LineWidth',2);
%             end
%             subplot(3,1,3)
%             plot(movmean(waggleConvResultMaxedVal(:),20));grid on
%             [value_row_new,location_row_new]=findpeaks(movmean(waggleConvResultMaxedVal(:),20),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');
%             findpeaks(movmean(waggleConvResultMaxedVal(:),20),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');
%             text(location_row_new+.02,double(value_row_new),num2str((1:numel(double(value_row_new)))'));
%             yline(8.5e4,"LineWidth",2);
              
              
%             subplot(2,3,6)
%             x_coor = 0:size(waggleConvResultMaxedVal,2)-1;
%             y_coor = 0:size(waggleConvResultMaxedVal,1)-1;
%             [X,Y] = meshgrid(x_coor,y_coor);
%             meshc(X, Y, waggleConvResultMaxedVal);


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             figure(6756);
            pixelBoundries = 40;
            peakThreshold  = 9e4;
%             subplot(2,4,[1 4])
%             plot(movmean(waggleConvResultMaxedVal(:),20));grid on
            [value_row_new,location_row_new]=findpeaks(waggleConvResultMaxedVal(:),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');
%             findpeaks(waggleConvResultMaxedVal(:),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');
%             text(location_row_new+.02,double(value_row_new),num2str((1:numel(double(value_row_new)))'));
%             if ~isempty(value_row_new)
% % % % %                 yline(value_row_new(1) - peakThreshold,"LineWidth",2);
%                 yline(peakThreshold,"LineWidth",2);
%             end
%             subplot(2,4,5)
%             imagesc(uint8(frameArray(:,:,:,iFrame)));
%             subplot(2,4,6)
            if ~isempty(value_row_new)
                if value_row_new(1) > peakThreshold %value_row_new(1) - peakThreshold
                    waggleConvThreshed = waggleConvResultMaxedVal> peakThreshold;%value_row_new(1)-peakThreshold;
                    [iRow, iCol, ~] = find(waggleConvThreshed>0);
%                     imagesc(waggleConvThreshed);colorbar
                else
                    waggleConvThreshed = 0;
%                     imagesc(waggleConvThreshed);colorbar;
                end
            else
                waggleConvThreshed = 0;
%                 imagesc(waggleConvThreshed);colorbar;
            end
%             subplot(2,4,7)
%             x_coor = 0:size(waggleConvResultMaxedVal,2)-1;
%             y_coor = 0:size(waggleConvResultMaxedVal,1)-1;
%             [X,Y] = meshgrid(x_coor,y_coor);
%             meshc(X, Y, waggleConvResultMaxedVal);
%             subplot(2,4,8)
%             imagesc(waggleConvResultMaxedVal);colorbar;hold on
            if ~isempty(iRow) && ~isempty(iCol)
                coordinate_points = find(iCol-pixelBoundries>0 & iCol+pixelBoundries<size(waggleConvThreshed,2) & iRow-pixelBoundries>0 & iRow+pixelBoundries<size(waggleConvThreshed,2));
%                 plot(iCol(coordinate_points),iRow(coordinate_points),'+r');
%                 plot(round(mean(iCol(coordinate_points))),round(mean(iRow(coordinate_points))),'og','MarkerSize',14);
                td.x(AllFrames)  = round(mean(iCol(coordinate_points)));
                td.y(AllFrames)  = round(mean(iRow(coordinate_points)));
                td.waggleregionx{AllFrames} = iCol(coordinate_points);
                td.waggleregiony{AllFrames} = iRow(coordinate_points);
                td.ts(AllFrames) = AllFrames;
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
%             if ~isempty(location_column) && ~isempty(location_row)
%                 td.x(AllFrames) = location_column(1);
%                 td.y(AllFrames) = location_row(1);
%             else
%                 td.x(AllFrames) = 0;
%                 td.y(AllFrames) = 0;
%             end
            
%             F = getframe(gcf) ;
%             writeVideo(writerObj, F);

%             figure(657);
%             subplot(2,3,1)
%             imagesc(newframe);colorbar;
%             subplot(2,3,2)
%             imagesc(waggleConvFinalMaxed(:,:,iFrame));caxis([0 6e5]);colorbar;
%             subplot(2,3,3)
%             imagesc(waggleConvFinalMaxed_LPF(:,:,iFrame));caxis([0 6e5]);colorbar;
%             subplot(2,3,[4 6])
%             newGaussianFrame = waggleConvFinalMaxed_LPF(:,:,iFrame);
%             findpeaks(sum(newGaussianFrame(1:end,:),1)','MinPeakProminence',200)
        end
    end
end

% close(writerObj);
% fprintf('Sucessfully generated the video\n');

figure(6565);
% scatter3(td.x*2,td.y*2,td.ts,'.r');hold on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");

figure(758);
subplot(1,3,1)
plot(td_gt.x,'.b');hold on
plot(td.x*2,'.r');hold off;grid on;
subplot(1,3,2)
plot(td_gt.y,'.b');hold on
plot(td.y*2,'.r');hold off;grid on;
subplot(1,3,3)
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');hold on
scatter3(td.x*2,td.y*2,td.ts,'.r');grid on;


% pts = linspace(0, 800, numel(td.y));
% N = histcounts2(td.y,td.ts, pts, pts);
% [xG, yG] = meshgrid(-25:25);
% sigma = 15;
% g = exp(-xG.^2./(2.*sigma.^2)-yG.^2./(2.*sigma.^2));
% g = g./sum(g(:));
% figure(6786);
% subplot(1, 2, 1);
% plot(td.y, 'r.');grid on
% subplot(1, 2, 2);
% imagesc(pts, pts, conv2(N, g, 'same'));
% set(gca, 'XLim', pts([1 2000]), 'YLim', pts([1 2000]), 'YDir', 'normal');
% 
% 
% % Normally distributed sample points:
% x = randn(1, 100);
% y = randn(1, 100);
% 
% % Bin the data:
% pts = linspace(-3, 3, 101);
% N = histcounts2(y(:), x(:), pts, pts);
% 
% % Create Gaussian filter matrix:
% [xG, yG] = meshgrid(-5:5);
% sigma = 2.5;
% g = exp(-xG.^2./(2.*sigma.^2)-yG.^2./(2.*sigma.^2));
% g = g./sum(g(:));
% 
% % Plot scattered data (for comparison):
% subplot(1, 2, 1);
% scatter(x, y, 'r.');
% axis equal;
% set(gca, 'XLim', pts([1 end]), 'YLim', pts([1 end]));
% 
% % Plot heatmap:
% subplot(1, 2, 2);
% imagesc(pts, pts, conv2(N, g, 'same'));
% axis equal;
% set(gca, 'XLim', pts([1 end]), 'YLim', pts([1 end]), 'YDir', 'normal');

%% Additional image filtering and pre-processing/ plot ellipse
tic;
addpath("../DeepGreen/greenhouseCode")

%        ['20210803t1727d200m_cropped';
%         '20210803t1301d050m_cropped';
%         '20210803t1508d100m_cropped';
%         '20210803t1517d100m_cropped';
%         '20210803t1719d200m_cropped';
%         '20210803t1259d050m_cropped';]


%%%%%%%%%%%%%%%% LOAD VIDEO
% videoName = "20210803t1727d200m";
% dataPATH = "input_videos/20210803t1259d050m_cropped";

videoFileName = "/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/input_videos/20210803t1259d050m_cropped.MP4";
load('/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_ground_truth.mat');

% figure(6767);
% scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.r');hold on
% plot3(td_gt.x,td_gt.y,td_gt.frameID,'-r');


% load('/media/sam/Samsung_T5/PhD/Code/bee_tracking/data/integration/track_id_all.mat')
% load('./td_out/20210803t1727d200m_cropped_PeakDetection2.mat');
% load('waggle36Templates_25x25_HQ.mat');

% td.x(isnan(td.x))=0;
% td.y(isnan(td.y))=0;
% findNewCoordinate = find(td.x*4>500 & td.x*4 < 800 & td.y*4>600 & td.y*4 < 900);
% td.x = td.x(findNewCoordinate);
% td.y = td.y(findNewCoordinate);
% td.waggleregionx = td.waggleregionx(findNewCoordinate);
% td.waggleregiony = td.waggleregiony(findNewCoordinate);
% td.ts = td.ts(findNewCoordinate);

v0 = VideoReader(videoFileName);

SHOW                        = 0;
RECORD_VIDEO                = SHOW && 1;
tdWithInh                   = [];
jTd                         = [];
iRow                        = [];
iCol                        = [];
iWaggleInh                  = 0;
iWaggleEvent                = 0;
frameID                     = 0;
avgFrameDepth               = 6;
convMapThreshold            = 15;
nTemplate                   = size(waggleTemplate,3);
nEventsForWaggleThreshold   = 6;
nDel                        = 18;
nFrameTotal                 = round(v0.FrameRate *v0.Duration);
numberofSegment             = 10;
framesPerSegment            = nFrameTotal/numberofSegment;
nSegment                    = ceil(nFrameTotal/framesPerSegment);
radius                      = 15;

% writerObj = VideoWriter('./output_videos/20210803t1259d050m_cropped_convolution_on_the_original_frame.avi');
% writerObj.FrameRate = 30;
% open(writerObj);
PATH = "/media/sam/Samsung_T5/PhD/Code/bee_tracking/data/frames/";

segFrame = 0;
for iSegment = 1:numberofSegment
%     if iSegment > 4 && iSegment < 6
    if iSegment < numberofSegment
        iSegment
        iFrame = 0;
        
        startFrame = max((iSegment-1)*framesPerSegment+1-nDel,1);
        endFrame = min(iSegment*framesPerSegment,nFrameTotal);
        nFrame =  endFrame - startFrame;
        iFrameWithWaggle = 0;waggleStats = {};
        imageWidth  = v0.Width;
        imageHeight = v0.Height;
        downsamplingFactorframeArray            = 2; %2;
        frameArray                              = zeros(round(imageHeight/downsamplingFactorframeArray),round(imageWidth/downsamplingFactorframeArray),3,nFrame,'uint8');
        downsamplingFactorRGB                   = 2; %2;
        dRgbFrameArray                          = zeros(round(imageHeight/downsamplingFactorRGB),round(imageWidth/downsamplingFactorRGB),3,nFrame,'single');
        downsamplingFactorGreyScale             = 4; %4;
        dGreyScaleArray                         = zeros(round(imageHeight/downsamplingFactorRGB),round(imageWidth/downsamplingFactorRGB),nFrame,'single');
        
        for iFrameOriginal = startFrame:endFrame
            iFrame = iFrame + 1;
            frameIntFull = read(v0,iFrameOriginal);
            frameInt = frameIntFull;
            downsamplingFactor = 0.5;
            frameArray(:,:,:,iFrame) =  imresize(frameInt,downsamplingFactor);
            
            if iFrame>1
                dRgbFrameArray(:,:,:,iFrame) = single(frameArray(:,:,:,iFrame)) - single(frameArray(:,:,:,iFrame-1));
                dGreyScaleArray(:,:,iFrame) =  vecnorm(single(frameArray(:,:,:,iFrame)),2,3) -   vecnorm(single(frameArray(:,:,:,iFrame-1)),2,3);
            end
        end
        
        waggleConvResult = zeros(round(imageHeight/2),round(imageWidth/2),nTemplate,'single');
        waggleConvFinalMaxed =  zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        
        newframe =  zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        blurredImage =  zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        meanWaggleMapFrame = zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        newImagePixel = zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        final_frame = zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
        
        for iFrame = 1:nFrame
            newframe(:,:,iFrame)            = rgb2gray(frameArray(:,:,:,iFrame));
        end
        
        avgFrameDepth = 10;
        for iFrame = 1:nFrame
%             newframe(:,:,iFrame)            = rgb2gray(frameArray(:,:,:,iFrame));
            blurredImage(:,:,iFrame)        = imgaussfilt(newframe(:,:,iFrame),5);
            meanWaggleMapFrame(:,:,iFrame)  = mean(blurredImage(:,:,max(iFrame-avgFrameDepth,1):iFrame),3);
            newImagePixel(:,:,iFrame)       = 0.9*meanWaggleMapFrame(:,:,iFrame) + 0.1*blurredImage(:,:,iFrame);
            newImagePixelDecayed            = 0.9*newImagePixel(:,:,iFrame);
            final_frame(:,:,iFrame)         = newframe(:,:,iFrame)./(newImagePixelDecayed+1);
            
            figure(67868);
%             imagesc(uint8(frameInt));axis image;hold on
            imshow(PATH + "png/" + FrameNames);axis on; hold on
            for idx = 1:numel(td.beeTrajectory)
                [X,Y] = calculateEllipse(td.beeTrajectory{idx,1}(iFrame,1),td.beeTrajectory{idx,1}(iFrame,2),25,10,round(td.beeTrajectory{idx,1}(iFrame,3)));
                plot(X, Y,'LineWidth',2);
            end
          
%             plot(beeTrajectory(1,iFrame),beeTrajectory(2,iFrame),'or','MarkerSize',35);
            
%             figure(6768);
%             subplot(4,3,1);imagesc(newframe(:,:,iFrame));title("Greyscale");colorbar;
%             subplot(4,3,2);imagesc(blurredImage(:,:,iFrame));title("f: Blurred");colorbar;
%             subplot(4,3,3);imagesc(meanWaggleMapFrame(:,:,iFrame));title("M: Average map");colorbar;
%             subplot(4,3,4);imagesc(newImagePixel(:,:,iFrame));title("M: 0.9*M + 0.1f");colorbar;
%             subplot(4,3,5);imagesc(newImagePixelDecayed);title("M_n: 0.99*M");colorbar;
%             subplot(4,3,6);imagesc(final_frame(:,:,iFrame));caxis([0,5]);colorbar;title("greyscale/M_n");
%             subplot(4,3,[7 9]);blurredImage_new = blurredImage(:,:,iFrame);findpeaks(blurredImage_new(:),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');
%             [value_row_new,location_row_new]=findpeaks(blurredImage_new(:),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');title("Flattened before");
%             text(location_row_new+.02,double(value_row_new),num2str((1:numel(double(value_row_new)))'));
%             subplot(4,3,[10 12]);
%             findpeaks(newImagePixelDecayed(:),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');
%             [value_row_decayed,location_row_decayed]=findpeaks(newImagePixelDecayed(:),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');title("Flattened before");
%             text(location_row_decayed+.02,double(value_row_decayed),num2str((1:numel(double(value_row_decayed)))'));
            
%             for iTemplate = 1:nTemplate
%                 waggleConvResult(:,:,iTemplate) = conv2(final_frame(:,:,iFrame),waggleTemplate(:,:,iTemplate),'same');
%             end
%             [waggleConvResultMaxedVal, waggleTemplateIdx ]= max(waggleConvResult,[],3);
%             waggleConvFinalMaxed(:,:,iFrame) = max(waggleConvResult,[],3);
%             
%             figure(5670);
%             subplot(1,2,1)
%             imagesc(final_frame(:,:,iFrame));caxis([0,5]);colorbar;
%             subplot(1,2,2)
%             imagesc(waggleConvFinalMaxed(:,:,iFrame));
%             colorbar;caxis([0,50000]);axis off;
            
%             figure(2);
%             imagesc(uint8(frameArray(:,:,:,iFrame)));axis image;hold on
%             title( num2str(iFrame),'color','r','fontSize',14);
%             if ~isempty(td.waggleregionx) && ~isempty(td.waggleregiony)
%                 plot(td.waggleregionx{1,iFrame}(:),td.waggleregiony{1,iFrame}(:),'+r');
%             end
            
%             F = getframe(gcf);
%             writeVideo(writerObj, F);
%             
        end
    end
end

% close(writerObj);
% fprintf('Sucessfully generated the video\n')

%% Evaluate all recordings for the peak detection algorithm
% LOAD VIDEOS
data = ['20210803t1727d200m_cropped';
        '20210803t1301d050m_cropped';
        '20210803t1508d100m_cropped';
        '20210803t1517d100m_cropped';
        '20210803t1719d200m_cropped';
        '20210803t1259d050m_cropped'];

test_record = 1;
sensitivity     = nan(test_record,1);
specificity     = nan(test_record,1);
precision       = nan(test_record,1);
recall          = nan(test_record,1);
F1              = nan(test_record,1);
informedness    = nan(test_record,1);
accuracy        = nan(test_record,1);

ANGLE_DETECTION             = 0;
beePixelSize                = 150;
angleRange                  = 20;

for recording_index = 1%:30
    human_labelled_waggles      = [];
    algorithm_detected_waggles  = [];
    
    for filedata = 1:numel(data(:,1))
        dataName = data(filedata,:)
        %     filedata
        load("td_out/" + data(filedata,:) + "_oldTemplate_" + recording_index + ".mat")
        load("./final_labels/" + data(filedata,:) + "/" + data(filedata,:) + "_ground_truth.mat")
        
%         load("td_out/" + data(filedata,:) + "_PeakDetection.mat")
%         load("./final_labels/" + data(filedata,:) + "/" + data(filedata,:) + "_ground_truth.mat")
        
        if numel(td) == 0 || numel(td.x) < 2
            recording_index = recording_index + 1;
%             load("td_out/" + data(filedata,:) + "_PeakDetection.mat")
%             load("./final_labels/" + data(filedata,:) + "/" + data(filedata,:) + "_ground_truth.mat")

            load("td_out/" + data(filedata,:) + "_oldTemplate_" + recording_index + ".mat")
            load("./final_labels/" + data(filedata,:) + "/" + data(filedata,:) + "_ground_truth.mat")
        end
        %         if numel(td.x) < 2
        %             sensitivity(recording_index,1) = 0;
        %             specificity(recording_index,1) = 0;
        %             precision(recording_index,1) = 0;
        %             recall(recording_index,1) = 0;
        %             F1(recording_index,1) = 0;
        %             informedness(recording_index,1) = 0;
        %             accuracy(recording_index,1) = 0;
        %         else
        
        counter = 0;
        groundtruth = [];
        frame_index_Gt = 0;
        frameID = td_gt.frameID(1);
        missclassified            = 0;
        correctlyclassified       = 0;
        td_missclassified         = [];
        td_correctlyclassified    = [];
        
        for idx = 1:td_gt.frameID(end)
            frame_index_Gt = frame_index_Gt + 1;
            if frameID > idx
                groundtruth.x(idx)      = 0;
                groundtruth.y(idx)      = 0;
%                 groundtruth.angle(idx)  = 0;
                groundtruth.c(idx)      = 0;
                groundtruth.frameID(idx)      = frame_index_Gt;
            else
                counter = counter + 1;
                groundtruth.x(idx)      = td_gt.x(counter);
                groundtruth.y(idx)      = td_gt.y(counter);
%                 groundtruth.angle(idx)  = td_gt.angle(counter);
                groundtruth.c(idx)      = 1;
                groundtruth.frameID(idx)      = frame_index_Gt;
            end
        end
        td_gt = groundtruth;
        
        if numel(td.x) > numel(td_gt.x)
            td.x = td.x(:,1:td_gt.frameID(end));
            td.y = td.y(:,1:td_gt.frameID(end));
%             td.angle = td.angle(:,1:td_gt.frameID(end));
            td.ts = td.ts(:,1:td_gt.frameID(end));
            td.ts = td.ts(:,1:td_gt.frameID(end));
        end
        
        % detected waggle setup
        d = diff(td.ts);
        d(d==0) = 1;
        ivec = cumsum([1 d]);
        y = nan(1,ivec(end));
        y(ivec) = td.x;td.x = y;
        td.x(isnan(td.x))=0;
        y(ivec) = td.y;td.y = y;
        td.y(isnan(td.y))=0;
        y(ivec) = td.ts;td.ts = y;
        td.ts(isnan(td.ts))=0;
%         y(ivec) = td.angle;td.angle = y;
%         td.angle(isnan(td.angle))=0;
%         y(ivec) = td.ts;td.ts = y;
%         td.ts(isnan(td.ts))=0;
        
        counter = 0;
        frame_index_pred = 0;
        pred_waggle = [];
        pred_waggleLabel = [];
        frameID = td.ts(1);
        pred_waggleLabel(find(td.ts==0))=0;
        pred_waggleLabel(find(td.ts~=0))=1;
        
        for idx = 1:td.ts(end)
            frame_index_pred = frame_index_pred + 1;
            if frameID > idx
                pred_waggle.x                   = 0;
                pred_waggle.y(idx)              = 0;
%                 pred_waggle.angle(idx)          = 0;
                pred_waggle.c(idx)              = 0;
                pred_waggle.frameID(idx)        = frame_index_pred;
            else
                counter = counter + 1;
                pred_waggle.x(idx)      = td.x(counter);
                pred_waggle.y(idx)      = td.y(counter);
%                 pred_waggle.angle(idx)  = td.angle(counter);
                pred_waggle.c(idx)      = pred_waggleLabel(counter);
                pred_waggle.ts(idx)      = frame_index_pred;
            end
        end
        
        pred_waggle.x(td.ts(end):td_gt.frameID(end)) = 0;
        pred_waggle.y(td.ts(end):td_gt.frameID(end)) = 0;
%         pred_waggle.angle(td.ts(end):td_gt.frameID(end)) = 0;
        pred_waggle.ts(td.ts(end):td_gt.frameID(end)) = td.ts(end):td_gt.frameID(end);
        pred_waggle.c(td.ts(end):td_gt.frameID(end)) = 0;
        
        if numel(td.x) > numel(td_gt.x)
            pred_waggle.x(td_gt.frameID(end)+1:td.ts(end))         = [];
            pred_waggle.y(td_gt.frameID(end)+1:td.ts(end))         = [];
%             pred_waggle.angle(td_gt.frameID(end)+1:td.ts(end))     = [];
            pred_waggle.ts(td_gt.frameID(end)+1:td.ts(end))   = [];
            pred_waggle.c(td_gt.frameID(end)+1:td.ts(end))         = [];
        end
        td = pred_waggle;
        predicted_labels = nan(numel(td.x),1);
        for idx = 1:numel(td.x)
            if ANGLE_DETECTION
                if ((td.x(idx)*4 - td_gt.x(idx))^2 + (td.y(idx)*4 - td_gt.y(idx))^2)<beePixelSize^2 && td.angle(idx) > td_gt.angle(idx)-angleRange && td.angle(idx) < td_gt.angle(idx)+angleRange
                    correctlyclassified = correctlyclassified +1;
                    td_correctlyclassified.x(correctlyclassified) = td.x(idx)*4;
                    td_correctlyclassified.y(correctlyclassified) = td.y(idx)*4;
%                     td_correctlyclassified.angle(correctlyclassified) = td.angle(idx);
                    td_correctlyclassified.ts(correctlyclassified) = td.ts(idx);
                    predicted_labels(idx) = 1;
                else
                    missclassified = missclassified +1;
                    td_missclassified.x(missclassified) = td.x(idx)*4;
                    td_missclassified.y(missclassified) = td.y(idx)*4;
%                     td_missclassified.angle(missclassified) = td.angle(idx);
                    predicted_labels(idx) = 0;
                    td_missclassified.ts(missclassified) = td.ts(idx);
                end
            else
                %                     if numel(td.x) > numel(td_gt.x)
                %                         pred_waggle.x(td_gt.frameID(end)+1:td.frameID(end))         = [];
                %                         pred_waggle.y(td_gt.frameID(end)+1:td.frameID(end))         = [];
                %                         pred_waggle.angle(td_gt.frameID(end)+1:td.frameID(end))     = [];
                %                         pred_waggle.frameID(td_gt.frameID(end)+1:td.frameID(end))   = [];
                %                         pred_waggle.c(td_gt.frameID(end)+1:td.frameID(end))         = [];
                %                     end
                if ((td.x(idx)*4 - td_gt.x(idx))^2 + (td.y(idx)*4 - td_gt.y(idx))^2)<beePixelSize^2
                    correctlyclassified = correctlyclassified +1;
                    td_correctlyclassified.x(correctlyclassified) = td.x(idx)*4;
                    td_correctlyclassified.y(correctlyclassified) = td.y(idx)*4;
%                     td_correctlyclassified.angle(correctlyclassified) = td.angle(idx);
                    td_correctlyclassified.ts(correctlyclassified) = td.ts(idx);
                    predicted_labels(idx) = 1;
                else
                    missclassified = missclassified +1;
                    td_missclassified.x(missclassified) = td.x(idx)*4;
                    td_missclassified.y(missclassified) = td.y(idx)*4;
%                     td_missclassified.angle(missclassified) = td.angle(idx);
                    predicted_labels(idx) = 0;
                    td_missclassified.ts(missclassified) = td.ts(idx);
                end
            end
        end
        
        human_labelled_waggles      = [human_labelled_waggles;td_gt.c'];
        algorithm_detected_waggles  = [algorithm_detected_waggles;predicted_labels];
        %         end
    end
    label_postprocessing = [human_labelled_waggles algorithm_detected_waggles];
    
    [testingPresentations,numLabels]   = size(predicted_labels);
    [allObservation channels]          = size(predicted_labels);
    
    true_positive = sum(and(label_postprocessing(:,1),label_postprocessing(:,2)),1);  % TP
    true_negative = sum(and(~label_postprocessing(:,1),~label_postprocessing(:,2)),1); % TN
    
    false_positive = sum((label_postprocessing(:,1) == 1) & (label_postprocessing(:,2) == 0)); % FP
    false_negative = sum((label_postprocessing(:,1) == 0) & (label_postprocessing(:,2) == 1)); % FN
    
    groundtruth = human_labelled_waggles;
    actualpositives = numel(groundtruth(groundtruth > 0,1));  % TP+FN
    actualnegatives = numel(groundtruth(groundtruth <  1,1)); % TN+FP
    
    confusion_matrix = [true_positive false_positive ; false_negative true_negative]
    
    sensitivity(recording_index,1) = true_positive ./ actualpositives;
    specificity(recording_index,1) = true_negative ./ actualnegatives;
    precision(recording_index,1) = true_positive / (true_positive + false_positive);
    recall(recording_index,1) = true_positive / (true_positive + false_negative);
    F1(recording_index,1) = (2 * precision(recording_index,1) * recall(recording_index,1)) / (precision(recording_index,1) + recall(recording_index,1));
    informedness(recording_index,1) = sensitivity(recording_index,1) + specificity(recording_index,1) - 1;
    accuracy(recording_index,1) = (1 - missclassified/testingPresentations)*100;
end

figure(676723);confusionchart(confusion_matrix,'RowSummary','row-normalized','ColumnSummary','column-normalized');

figure(66);
% subplot(2,2,1);
plot(F1,"*-","LineWidth",2);grid on;hold on;
plot(informedness,"s-","LineWidth",2);
plot(sensitivity,"<-","LineWidth",2);
plot(specificity,">-","LineWidth",2);
title("Evaluation over all waggles recordings");
xlabel("Threshold");
% ylim([0 1.25])
% xline(7,'-',{'Acceptable','Threshold'},"LineWidth",4);
legend("F1-Score","Informedness","Sensitivity","Specificity");
set(gca,'fontsize', 18);

% subplot(2,2,2);
% plot(informedness,"LineWidth",2);grid on;
% title("Informedness");
% xlabel("Threshold");
% ylim([0 1.25])
% hold on;
% xline(7,'-',{'Acceptable','Threshold'},"LineWidth",4);
% set(gca,'fontsize', 18);
% 
% subplot(2,2,3);
% plot(sensitivity,"LineWidth",2);grid on;
% title("Sensitivity");
% xlabel("Threshold");
% ylim([0 1.35])
% hold on;
% xline(7,'-',{'Acceptable','Threshold'},"LineWidth",4);
% set(gca,'fontsize', 18);
% 
% subplot(2,2,4);
% plot(specificity,"LineWidth",2);grid on;
% color = [0, 0, 1];
% title("Specificity");
% xlabel("Threshold");
% hold on;
% xline(7,'-',{'Acceptable','Threshold'},"LineWidth",4);
% set(gca,'fontsize', 18);

%%
addpath("../DeepGreen/greenhouseCode")
% data = ['20210803t1727d200m_cropped'; --> done
%         '20210803t1301d050m_cropped'; --> done
%         '20210803t1508d100m_cropped'; --> done
%         '20210803t1517d100m_cropped'; --> done
%         '20210803t1719d200m_cropped'; --> done
%         '20210803t1259d050m_cropped']; --> done

%load peak detection output data
peak = load('td_out/20210803t1259d050m_cropped_PeakDetection.mat');

%load frame differencing output data
frameDiff = load('td_out/20210803t1259d050m_cropped_oldTemplate_2.mat');

%load ground truth
load('final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_ground_truth.mat');

counter = 0;
detected_peak = [];
beePixelSize = 100;
for idx = 1:numel(td_gt.x)
    if ((peak.td.x(idx)*2 - td_gt.x(idx))^2 + (peak.td.y(idx)*2 - td_gt.y(idx))^2)<beePixelSize^2
        counter = counter + 1;
        detected_peak.x(counter)  = peak.td.x(idx);
        detected_peak.y(counter)  = peak.td.y(idx);
        detected_peak.ts(counter) = peak.td.ts(idx);
    end
end

counter = 0;
detected_framdiff = [];
for idx = 1:numel(td_gt.x)
    if ((frameDiff.td.x(idx)*4 - td_gt.x(idx))^2 + (frameDiff.td.y(idx)*4 - td_gt.y(idx))^2)<beePixelSize^2
        counter = counter + 1;
        detected_framdiff.x(counter)  = frameDiff.td.x(idx);
        detected_framdiff.y(counter)  = frameDiff.td.y(idx);
        detected_framdiff.ts(counter) = frameDiff.td.ts(idx);
    end
end

figure(66);
subtightplot(2,2,1)
scatter3(peak.td.x*2,peak.td.y*2,peak.td.ts,'.r');hold on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
title("Peak detection algorithm");
subtightplot(2,2,2);
scatter3(detected_peak.x,detected_peak.y,detected_peak.ts,'.r');
title("Peak detection algorithm");
subtightplot(2,2,3);
scatter3(frameDiff.td.x*4,frameDiff.td.y*4,frameDiff.td.ts,'.r');hold on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
title("Frame differencing algorithm");
subtightplot(2,2,4);
scatter3(detected_framdiff.x,detected_framdiff.y,detected_framdiff.ts,'.r');
title("Frame differencing algorithm");
