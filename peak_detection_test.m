clear
tic;
addpath("../DeepGreen/greenhouseCode")

videoFileName = "./final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped.MP4";
load('./final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_ground_truth.mat');

% videoFileName = "./input_videos/20210803t1301d050m_cropped.MP4";
% load('./final_labels/20210803t1301d050m_cropped/20210803t1301d050m_cropped_ground_truth.mat');

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
numberofSegment             = 20;
framesPerSegment            = nFrameTotal/numberofSegment;
nSegment                    = ceil(nFrameTotal/framesPerSegment);
AllFrames = 0;

writerObj = VideoWriter('./output_videos/peak_detector_working_algorithm2.avi');
writerObj.FrameRate = 10;
open(writerObj);

segFrame = 0;
for iSegment = 1:numberofSegment
    if iSegment < 4
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
%             res=2; %1: Local maximum 20%faster than 2: weighted centroid with subpixel interpolation
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
            figure(6756);
            pixelBoundries = 40;
            peakThreshold  = 7e2;
            subplot(2,4,[1 4])
            plot(movmean(waggleConvResultMaxedVal(:),20));grid on
            [value_row_new,location_row_new]=findpeaks(movmean(waggleConvResultMaxedVal(:),20),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');
            findpeaks(movmean(waggleConvResultMaxedVal(:),20),'MinPeakDistance',4000,'Annotate','extents','SortStr','descend');
            text(location_row_new+.02,double(value_row_new),num2str((1:numel(double(value_row_new)))'));
            if ~isempty(value_row_new)
                yline(value_row_new(1) - peakThreshold,"LineWidth",2);
            end
            subplot(2,4,5)
            imagesc(uint8(frameArray(:,:,:,iFrame)));
            subplot(2,4,6)
            if ~isempty(value_row_new)
                if value_row_new(1) > value_row_new(1) - peakThreshold
                    waggleConvThreshed = waggleConvResultMaxedVal>value_row_new(1)-peakThreshold;
                    [iRow, iCol, ~] = find(waggleConvThreshed>0);
                    imagesc(waggleConvThreshed);colorbar
                else
                    waggleConvThreshed = 0;
                    imagesc(waggleConvThreshed);colorbar;
                end
            else
                waggleConvThreshed = 0;
                imagesc(waggleConvThreshed);colorbar;
            end
            subplot(2,4,7)
            x_coor = 0:size(waggleConvResultMaxedVal,2)-1;
            y_coor = 0:size(waggleConvResultMaxedVal,1)-1;
            [X,Y] = meshgrid(x_coor,y_coor);
            meshc(X, Y, waggleConvResultMaxedVal);
            subplot(2,4,8)
            imagesc(waggleConvResultMaxedVal);colorbar;hold on
            if ~isempty(iRow) && ~isempty(iCol)
                coordinate_points = find(iCol-pixelBoundries>0 & iCol+pixelBoundries<size(waggleConvThreshed,2) & iRow-pixelBoundries>0 & iRow+pixelBoundries<size(waggleConvThreshed,2));
                plot(iCol(coordinate_points),iRow(coordinate_points),'+r');
                plot(round(mean(iCol(coordinate_points))),round(mean(iRow(coordinate_points))),'og','MarkerSize',14);
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
            
            F = getframe(gcf) ;
            writeVideo(writerObj, F);

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

figure(6565);
scatter3(td.x*4,td.y*4,td.ts,'.r');hold on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");

% figure(670);
% findNewCoordinate = find(td.x*4>500 & td.x*4 < 800 & td.y*4>600 & td.y*4 < 900);
% scatter3(td.x(findNewCoordinate)*4,td.y(findNewCoordinate)*4,td.ts(findNewCoordinate),'.r');
% xlabel("X [px]");
% ylabel("Y [px]");
% zlabel("#Frames");
% xlim([0 max(td.x)*4]);
% ylim([0 max(td.y)*4]);

close(writerObj);
fprintf('Sucessfully generated the video\n')

%%

tic;
addpath("../DeepGreen/greenhouseCode")

%%%%%%%%%%%%%%%% LOAD VIDEO
% videoName = "20210803t1727d200m";
% dataPATH = "input_videos/20210803t1259d050m_cropped";

videoFileName = "./input_videos/20210803t1727d200m_cropped.MP4";
load('./final_labels/20210803t1727d200m_cropped/20210803t1727d200m_cropped_ground_truth.mat');
load('./td_out/20210803t1727d200m_cropped_PeakDetection2.mat');
td.x(isnan(td.x))=0;
td.y(isnan(td.y))=0;
findNewCoordinate = find(td.x*4>500 & td.x*4 < 800 & td.y*4>600 & td.y*4 < 900);
td.x = td.x(findNewCoordinate);
td.y = td.y(findNewCoordinate);
td.waggleregionx = td.waggleregionx(findNewCoordinate);
td.waggleregiony = td.waggleregiony(findNewCoordinate);
td.ts = td.ts(findNewCoordinate);

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
nEventsForWaggleThreshold   = 6;
nDel                        = 18;
nFrameTotal                 = round(v0.FrameRate *v0.Duration);
numberofSegment             = 10;
framesPerSegment            = nFrameTotal/numberofSegment;
nSegment                    = ceil(nFrameTotal/framesPerSegment);
radius                      = 15;

writerObj = VideoWriter('./output_videos/peak_detection_output.avi');
writerObj.FrameRate = 10;
open(writerObj);

segFrame = 0;
for iSegment = 1:numberofSegment
    if iSegment > 4 && iSegment < 7
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
        
        for iFrame = 1:nFrame
            figure(2);
            imagesc(uint8(frameArray(:,:,:,iFrame)));axis image;hold on
            title( num2str(iFrame),'color','r','fontSize',14);
            if ~isempty(td.waggleregionx) && ~isempty(td.waggleregiony)
                plot(td.waggleregionx{1,iFrame}(:),td.waggleregiony{1,iFrame}(:),'+r');
            end
%             rectangle('Position',[td.x(iFrame) td.y(iFrame) radius radius],'Curvature',[1 1],'FaceColor',[1 0 0],'EdgeColor','k',...
%                 'LineWidth',1);axis equal
            
            F = getframe(gcf);
            writeVideo(writerObj, F);
            
            %             for i=1:100
            %             subplot(1,2,1)
            
            %             subplot(1,2,2)
            %                 scatter3(td.x*4,td.y*4,td.ts,'.r');
            %                 p=plot3(td.x*4,td.y*4,td.ts,'.r');
            %                 xlabel("X [px]");
            %                 ylabel("Y [px]");
            %                 zlabel("#Frames");
            %
            %                 direction = [0 0 1];
            %                 rotate(p,direction,10);
            %                 x = p.XData;
            %                 y = p.YData;
            %                 z = p.ZData;
            %                 figure(2);
            %                 scatter3(x,y,z,'.r');
            %                 pause(0.1);
            
                            
            %             end
        end
    end
end

close(writerObj);
fprintf('Sucessfully generated the video\n')
%%
figure(2);
[X,Y,Z] = sphere(16);
x = [0.5*X(:); 0.75*X(:); X(:)];
y = [0.0*Y(:); 0.0*Y(:); Y(:)];
z = [0.0*Z(:); 0.0*Z(:); Z(:)];

for i=1:100
    scatter3(x,y,z);
    p = plot3(x,y,z);
    direction = [0 0 1];
    rotate(p,direction,10);
    x = p.XData;
    y = p.YData;
    z = p.ZData;
    figure(2);
    scatter3(x,y,z);
    pause(0.1);
end