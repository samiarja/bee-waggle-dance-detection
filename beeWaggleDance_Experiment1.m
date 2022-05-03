%% construct a new template to detect bee angle
% crop a real bee from one of the recording
% remove background
% rotate the bee 360 degree, 10 degree at a time
% rerun the algorithm again

rect = [244.51 430.51 42.98 60.98];
I = imread('final_labels/20210803t1727d200m_cropped/png/000001.png');
RGB_bee = imcrop(I,rect);
grey_bee = rgb2gray(RGB_bee);
grey_bee_straight = imrotate(grey_bee,300);

grey_bee_straight_flattened = grey_bee_straight(:);
zeroPixel = find(grey_bee_straight_flattened==0);
padded_column = 25 + (30-25).*rand(numel(zeroPixel),1);
for idx = 1:numel(padded_column)
    grey_bee_straight_flattened(zeroPixel(idx),:) = padded_column(idx);
end
grey_bee_straight_flattened_reshaped = reshape(grey_bee_straight_flattened,[68,75]);

waggleTemplate = [];
figure(567);
imagesc(grey_bee_straight_flattened_reshaped);
counter = 0;
resize_matrix = [70 70];
for angle = 1:10:360
    counter = counter + 1;
    subtightplot(6,6,counter)
    rotated_image = imrotate(grey_bee_straight,angle);
    rotated_image_resized = imresize(rotated_image,resize_matrix);
    grey_bee_straight_flattened = rotated_image_resized(:);
    zeroPixel = find(grey_bee_straight_flattened==0);
    padded_column = 20 + (30-20).*rand(numel(zeroPixel),1);
    for idx = 1:numel(padded_column)
        grey_bee_straight_flattened(zeroPixel(idx),:) = padded_column(idx);
    end
    grey_bee_straight_flattened_reshaped = reshape(grey_bee_straight_flattened,resize_matrix);
    grey_bee_straight_flattening = grey_bee_straight_flattened_reshaped(:);
    find_low_pixel_value = find(grey_bee_straight_flattening<17);
    padded_column_borders = 20 + (30-20).*rand(numel(find_low_pixel_value),1);
    for idx = 1:numel(padded_column_borders)
        grey_bee_straight_flattening(find_low_pixel_value(idx),:) = padded_column_borders(idx);
    end
    segment_bee_low_pixel = find(grey_bee_straight_flattening <= 43);
    segment_bee_high_pixel = find(grey_bee_straight_flattening >= 70);
    
    grey_bee_straight_flattening(segment_bee_low_pixel,:) = 0;
    grey_bee_straight_flattening(segment_bee_high_pixel,:) = 0;
    
    grey_bee_straight_flattened_reshaped = reshape(grey_bee_straight_flattening,resize_matrix);
    imagesc(grey_bee_straight_flattened_reshaped);axis off
    waggleTemplate = [waggleTemplate;grey_bee_straight_flattening];
end

waggleTemplate = reshape(waggleTemplate,[70,70,36]);
nTemplate = size(waggleTemplate,3);
[nSubplotRows,nSubplotCols] = goodSubPlotRowCols(nTemplate);
for iTemplate = 1:nTemplate
    figure(65);
    subplot(nSubplotRows,nSubplotCols,iTemplate);imagesc(waggleTemplate(:,:,iTemplate)); axis image; colorbar;
end

%% construct another template
I = imread('fig/bee_top_view_shopped.png');
grey_bee = rgb2gray(I);

% grey_bee_straight = imresize(grey_bee,[60,60]);
% figure(6780);imshow(grey_bee_straight);

% grey_bee_straight_flattened = grey_bee_straight(:);
% zeroPixel = find(grey_bee_straight_flattened==0);
% padded_column = 25 + (30-25).*rand(numel(zeroPixel),1);
% for idx = 1:numel(padded_column)
%     grey_bee_straight_flattened(zeroPixel(idx),:) = padded_column(idx);
% end
% grey_bee_straight_flattened_reshaped = reshape(grey_bee_straight_flattened,[68,75]);

waggleTemplate = [];
% figure(567);
% imagesc(grey_bee_straight_flattened_reshaped);

counter = 0;
resize_matrix = [25 25];
for angle = 1:10:360
    counter = counter + 1;
    subtightplot(6,6,counter)
    rotated_image = imrotate(grey_bee,angle);
    rotated_image_resized = imresize(rotated_image,resize_matrix);
    
    grey_bee_straight_flattening = rotated_image_resized(:);
    segment_bee_low_pixel = find(grey_bee_straight_flattening <= 30);
    segment_bee_high_pixel = find(grey_bee_straight_flattening >= 200);
    
    grey_bee_straight_flattening(segment_bee_low_pixel,:) = 0;
    grey_bee_straight_flattening(segment_bee_high_pixel,:) = 0;
    
    grey_bee_straight_flattened_reshaped = reshape(grey_bee_straight_flattening,resize_matrix);
    imagesc(grey_bee_straight_flattened_reshaped);axis off
    
    %     zeroPixel = find(grey_bee_straight_flattened==0);
%     padded_column = 20 + (30-20).*rand(numel(zeroPixel),1);
%     for idx = 1:numel(padded_column)
%         grey_bee_straight_flattened(zeroPixel(idx),:) = padded_column(idx);
%     end
%     grey_bee_straight_flattened_reshaped = reshape(grey_bee_straight_flattened,resize_matrix);
%     grey_bee_straight_flattening = grey_bee_straight_flattened_reshaped(:);
%     find_low_pixel_value = find(grey_bee_straight_flattening<17);
%     padded_column_borders = 20 + (30-20).*rand(numel(find_low_pixel_value),1);
%     for idx = 1:numel(padded_column_borders)
%         grey_bee_straight_flattening(find_low_pixel_value(idx),:) = padded_column_borders(idx);
%     end
%     
%     
%     grey_bee_straight_flattened_reshaped = reshape(grey_bee_straight_flattening,resize_matrix);
%     imagesc(grey_bee_straight_flattened_reshaped);axis off
    waggleTemplate = [waggleTemplate;grey_bee_straight_flattening];
end

waggleTemplate = reshape(waggleTemplate,[25,25,36]);
nTemplate = size(waggleTemplate,3);
[nSubplotRows,nSubplotCols] = goodSubPlotRowCols(nTemplate);
for iTemplate = 1:nTemplate
    figure(65);
    subplot(nSubplotRows,nSubplotCols,iTemplate);imagesc(waggleTemplate(:,:,iTemplate)); axis image; colorbar;
end

%%
clear
addpath("../DeepGreen/greenhouseCode")
tic;
%%%%%%%%%%%%%%%% LOAD VIDEO
% videoName = "20210803t1727d200m";
% dataPATH = "input_videos/20210803t1259d050m_cropped";

dataPATH = "/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1259d050m_cropped/";
load(dataPATH + '20210803t1259d050m_cropped_ground_truth.mat')
videoFileName = dataPATH + "20210803t1259d050m_cropped.MP4";

% limits.rowStart    = 401;    limits.rowEnd      = 1000;    limits.colStart    = 201;    limits.colEnd      = 1700;
% limits.rowStart    = 1;    limits.rowEnd      = 800;    limits.colStart    = 1;    limits.colEnd      = 800;
v0 = VideoReader(videoFileName);
disp("Loading data...")

%%%%%%%%%%%%%%%% LOAD TEMPLATE
% load('waggle16Templates_v1.mat')
load('waggle36Templates_25x25_HQ.mat')

disp("Template loaded...")

rad25                       = 12; % for 25x25 waggle template
% rad25                       = 35; % for 70x70 waggle template
SHOW                        = 0;
RECORD_VIDEO                = SHOW && 1;
td                          = [];
tdWithInh                   = [];
jTd                         = [];
iWaggleInh                  = 0;
iWaggleEvent                = 0;
AllFrames                   = 0;
avgFrameDepth               = 6;
convMapThreshold            = 30;
nEventsForWaggleThreshold   = 6;
nDel                        = 18;
nTemplate                   = size(waggleTemplate,3);
nFrameTotal                 = round(v0.FrameRate *v0.Duration);
numberofSegment             = 10;
framesPerSegment            = nFrameTotal/numberofSegment;
nSegment                    = ceil(nFrameTotal/framesPerSegment);

writerObj = VideoWriter('./output_videos/20210803t1259d050m_cropped_conv2D.avi');
writerObj.FrameRate = 10;
open(writerObj);

% [nSubplotRows,nSubplotCols] = goodSubPlotRowCols(nTemplate);
% bee_angle_rotation = 0;
% for iTemplate = 1:nTemplate
%     bee_angle_rotation = bee_angle_rotation + 10;
%     figure(65);
%     subplot(nSubplotRows,nSubplotCols,iTemplate);
%     imagesc(waggleTemplate(:,:,iTemplate));
%     title("Index: " + iTemplate + " - Angle: "+ bee_angle_rotation)
%     axis off;colormap('gray');
% end

segFrame = 0;
for iSegment = 1:numberofSegment
    if iSegment < 6
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
%                 figure(47);
%                 subtightplot(6,6,iTemplate);
%                 imagesc(waggleConvResult(:,:,iTemplate));axis off;caxis([0 5e5]);
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
            
            %%%%%%%%%%%%%%%%%%%%%%%%%% FIND WAGGLE ANGLE %%%%%%%%%%%%%%%%%%%%%%%%%%
            convolutionMapThreshold = 1e5;
            [row_coor, col_coor, ~] = find(waggleConvResultMaxedVal>convolutionMapThreshold);
            
            
            figure(5656);
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
            imh5 =  imagesc(waggleConvResultMaxedVal );axis image;  colorbar;colormap('gray');%title("2D convolution waggle map");
            caxis([0 5e5]);
            
            %                         subtightplot(3,3,6);
            %                         imh6 =  imagesc(waggleConvThreshed );axis image;  colorbar;title("Threshold waggle map");colormap('gray');
            %             caxis([0 1] )
            %             subtightplot(3,3,8);
            %                         imh7 =  imagesc(waggleDetectionMap);axis image;  colorbar;title("Detected waggle map");colormap('gray');
            %             %             caxis([0 1] )
            %             set(gcf,'Position',[100 100 1000 1000]);
            
            F = getframe(gcf) ;
            writeVideo(writerObj, F);
            
            
            if ~isempty(r) && (c-rad25>0) && (c+rad25<size(dGreyScaleArray,2)) && (r-rad25>0) && (r+rad25<size(dGreyScaleArray,1))
                iWaggleEvent = iWaggleEvent + 1;
                waggleRegion  = frameArray(r-rad25:r+rad25,c-rad25:c+rad25,:,iFrame);
                dwaggleRegion = dGreyScaleArray(r-rad25:r+rad25,c-rad25:c+rad25,iFrame);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%% FIND WAGGLE ANGLE METHOD 1 %%%%%%%%%%%%%%%%%%%%%%%%%%
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
                if ~isempty(row_coor) && (col_coor(ceil(end/2), :)-rad25>0) && (col_coor(ceil(end/2), :)+rad25<size(waggleConvResultMaxedVal,2)) && (row_coor(ceil(end/2), :)-rad25>0) && (row_coor(ceil(end/2), :)+rad25<size(waggleConvResultMaxedVal,1))
                    %%%%%%%%%%%%%%%%%%%%%%%%%% FIND WAGGLE ANGLE METHOD 2 %%%%%%%%%%%%%%%%%%%%%%%%%%
%                     wagglePatch = waggleConvResultMaxedVal(row_coor(ceil(end/2))-rad25:row_coor(ceil(end/2))+rad25-1,col_coor(ceil(end/2))-rad25:col_coor(ceil(end/2))+rad25-1);
                    wagglePatch = waggleConvResultMaxedVal(row_coor(ceil(end/2))-rad25:row_coor(ceil(end/2))+rad25,col_coor(ceil(end/2))-rad25:col_coor(ceil(end/2))+rad25);
                    newWaggleTemplate = waggleTemplate(:);newWaggleTemplate = single(reshape(newWaggleTemplate,[size(waggleTemplate,1)*size(waggleTemplate,1),nTemplate]));
                    newWagglePatch = wagglePatch(:);
                    dotProds = sum(bsxfun(@times,newWaggleTemplate,newWagglePatch),1);
                    [value_max,index_max] = max(dotProds);
                    rotationAngle = index_max * 10;
                    td.newAngel(iWaggleEvent) = 360 - rotationAngle;
                    
%                     figure(56756);
%                     imagesc(wagglePatch);colormap('gray');axis off;
                    
%                     figure(678);
%                     subtightplot(1,3,1);
%                     imagesc(waggleConvResultMaxedVal);colormap('gray');caxis([0 4.5e5]);axis off;
% %                     rectangle('Position',[col_coor(ceil(end/2))-15 row_coor(ceil(end/2))-15 40 40], ...
% %                     'Curvature',[1 1],'EdgeColor','r',...
% %                     'LineWidth',2);
%                     subtightplot(1,3,2)
%                     imagesc(wagglePatch);colormap('gray');axis off;
%                     subtightplot(1,3,3)
%                     imagesc(uint8(frameArray(:,:,:,iFrame)));colormap('gray');hold on;
%                     rectangle('Position',[col_coor(ceil(end/2))-15 row_coor(ceil(end/2))-15 33 33], ...
%                         'Curvature',[1 1],'EdgeColor','r',...
%                         'LineWidth',2);axis off;
%                     F = getframe(gcf) ;
%                     writeVideo(writerObj, F);
                    
                else
                    td.newAngel(iWaggleEvent) = 0;
                end
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
            
        end
    end
end

close(writerObj);
fprintf('Sucessfully generated the video\n');

figure(6565);
scatter3(td.x*4,td.y*4,td.ts,'.r');hold on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
plot3(td_gt.x,td_gt.y,td_gt.frameID,'-b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
toc;

%% Ground truth and predicted waggle post-processing
tic;
load('./final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_TD.mat')
load('./final_labels/20210803t1259d050m_cropped/20210803t1259d050m_ground_truth.mat')

% td.frameID =td.ts;

% td.x = td.x(:,1:td_gt.frameID(end));
% td.y = td.y(:,1:td_gt.frameID(end));
% td.angle = td.angle(:,1:td_gt.frameID(end));
% td.c = td.c(:,1:td_gt.frameID(end));
% td.frameID = td.frameID(:,1:td_gt.frameID(end));

% ground truth setup
beePixelSize = 100;
angleRange   = 20;
missclassified = 0;
correctlyclassified = 0;
td_missclassified = [];
td_correctlyclassified = [];


counter = 0;
groundtruth = [];
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

if numel(td.x) > numel(td_gt.x)
    td.x = td.x(:,1:td_gt.frameID(end));
    td.y = td.y(:,1:td_gt.frameID(end));
    td.angle = td.angle(:,1:td_gt.frameID(end));
    td.ts = td.ts(:,1:td_gt.frameID(end));
    td.frameID = td.frameID(:,1:td_gt.frameID(end));
end

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

if numel(td.x) > numel(td_gt.x)
    pred_waggle.x(td_gt.frameID(end)+1:td.frameID(end)) = [];
    pred_waggle.y(td_gt.frameID(end)+1:td.frameID(end)) = [];
    pred_waggle.angle(td_gt.frameID(end)+1:td.frameID(end)) = [];
    pred_waggle.frameID(td_gt.frameID(end)+1:td.frameID(end)) = [];
    pred_waggle.c(td_gt.frameID(end)+1:td.frameID(end)) = [];
end
td = pred_waggle;

predicted_labels = nan(numel(td.x),1);
for idx = 1:numel(td.x)
    if ((td.x(idx)*4 - td_gt.x(idx))^2 + (td.y(idx)*4 - td_gt.y(idx))^2)<beePixelSize^2 && td.angle(idx) > td_gt.angle(idx)-angleRange && td.angle(idx) < td_gt.angle(idx)+angleRange
        correctlyclassified = correctlyclassified +1;
        td_correctlyclassified.x(correctlyclassified) = td.x(idx)*4;
        td_correctlyclassified.y(correctlyclassified) = td.y(idx)*4;
        td_correctlyclassified.angle(correctlyclassified) = td.angle(idx);
        td_correctlyclassified.frameID(correctlyclassified) = td.frameID(idx);
        predicted_labels(idx) = 1;
    else
        missclassified = missclassified +1;
        td_missclassified.x(missclassified) = td.x(idx)*4;
        td_missclassified.y(missclassified) = td.y(idx)*4;
        td_missclassified.angle(missclassified) = td.angle(idx);
        predicted_labels(idx) = 0;
        td_missclassified.frameID(missclassified) = td.frameID(idx);
    end
end

label_postprocessing = [td_gt.c' predicted_labels];
[testingPresentations,numLabels]   = size(predicted_labels);
[allObservation channels]          = size(predicted_labels);

true_positive = sum(and(label_postprocessing(:,1),label_postprocessing(:,2)),1);  % TP
true_negative = sum(and(~label_postprocessing(:,1),~label_postprocessing(:,2)),1); % TN

false_positive = sum((label_postprocessing(:,1) == 1) & (label_postprocessing(:,2) == 0)); % FP
false_negative = sum((label_postprocessing(:,1) == 0) & (label_postprocessing(:,2) == 1)); % FN

groundtruth = td_gt.c';
actualpositives = numel(groundtruth(groundtruth > 0,1));  % TP+FN
actualnegatives = numel(groundtruth(groundtruth <  1,1)); % TN+FP

confusion_matrix = [true_positive false_positive ; false_negative true_negative]

sensitivity = true_positive ./ actualpositives % TP/TP+FN
specificity = true_negative ./ actualnegatives % TN/TN+FP
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
F1 = (2 * precision * recall) / (precision + recall)
informedness = sensitivity + specificity - 1
accuracy = (1 - missclassified/testingPresentations)*100

% informedness = (true_positive/(true_positive+false_negative)) + (true_negative/(true_negative+false_positive)) - 1
% accuracy = (1 - missclassified/(missclassified + correctlyclassified))*100

td.x(td.x==0) = NaN;
td.y(td.y==0) = NaN;
td.angle(td.angle==0) = NaN;
td.c(td.c==0) = NaN;
td.frameID(td.frameID==0) = NaN;

td_gt.x(td_gt.x==0) = NaN;
td_gt.y(td_gt.y==0) = NaN;
td_gt.angle(td_gt.angle==0) = NaN;
td_gt.c(td_gt.c==0) = NaN;
td_gt.frameID(td_gt.frameID==0) = NaN;


figure(66);
subplot(4,2,[1 3])
scatter3(td.x*4,td.y*4,td.frameID,'.r');hold on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
plot3(td_gt.x,td_gt.y,td_gt.frameID, 'bo', 'MarkerSize', 2);
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
title("Detected waggles");
xlim([min(td_gt.x)-50 max(td_gt.x)+50]);
ylim([min(td_gt.y)-50 max(td_gt.y)+50]);
zlim([0 numel(td_gt.y)]);
legend([{'Detected Waggles'},{'Labelled Waggles'}]);
set(gca,'fontsize', 14)
subplot(4,2,[2 4])
scatter3(td_correctlyclassified.x,td_correctlyclassified.y,td_correctlyclassified.frameID,'.r');hold on
scatter3(td_missclassified.x,td_missclassified.y,td_missclassified.frameID,'.k');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
xlim([min(td_gt.x)-50 max(td_gt.x)+50]);
ylim([min(td_gt.y)-50 max(td_gt.y)+50]);
zlim([0 numel(td_gt.y)]);
title("F1-Score: " + F1);
legend([{'Correctly Detected Waggles'},{'Wrongly Detected Waggles'}]);
set(gca,'fontsize', 14);
subplot(4,2,[5 6]);
plot(td_correctlyclassified.frameID,td_correctlyclassified.x,'-r');hold on;grid on
plot(td_gt.x,'-b');
xlabel("#Frames");
ylabel("X [px]");
title("Across X");
legend("Correctly Detected Waggle","Ground truth");
subplot(4,2,[7 8]);
plot(td_correctlyclassified.frameID,td_correctlyclassified.y/1.05,'-r');hold on;grid on
plot(td_gt.y,'-b');
xlabel("#Frames");
ylabel("Y [px]");
title("Across Y");
legend("Correctly Detected Waggle","Ground truth");


% figure(67);
% scatter3(td_correctlyclassified.x,td_correctlyclassified.y,td_correctlyclassified.frameID,'.r');hold on;grid on
% scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
% xlim([100 800]);
% ylim([200 900]);
% zlim([0 3000]);
% xlabel("X [px]");
% ylabel("Y [px]");
% zlabel("#Frames");
% set(gca,'fontsize', 14);
toc;