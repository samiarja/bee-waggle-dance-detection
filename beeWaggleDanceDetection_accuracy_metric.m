clear
addpath("../DeepGreen/greenhouseCode")
tic;

td                          = [];
tdWithInh                   = [];
jTd                         = [];
iWaggleInh                  = 0;
iWaggleEvent                = 0;
avgFrameDepth               = 6;
convMapThreshold            = 20;
nEventsForWaggleThreshold   = 10;
nDel                        = 18;
AllFrames                   = 0;
videoName       = ["final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped" "final_labels/20210803t1727d200m_cropped/20210803t1727d200m_croppedv2"];
groundtruthName = ["final_labels/20210803t1259d050m_cropped/20210803t1259d050m_ground_truth.mat" "final_labels/20210803t1727d200m_cropped/20210803t1727d200m_ground_truth.mat"];

%%%%%%%%%%%%%%%% LOAD TEMPLATE %%%%%%%%%%%%%%%%
load('waggle16Templates_v1.mat')
disp("Template loaded...")

for dataset = 1:2
    load(groundtruthName(dataset));
    videoFileName = "./input_videos/" + videoName(dataset) + ".MP4";
    v0 = VideoReader(videoName(dataset) + ".MP4");
    disp("Loading data...")
    
    nTemplate                   = size(waggleTemplate25,3);
    nFrameTotal                 = round(v0.FrameRate *v0.Duration);
    numberofSegment             = 5;
    framesPerSegment            = nFrameTotal/numberofSegment;
    nSegment                    = ceil(nFrameTotal/framesPerSegment);
    
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
            
            %%%%%%%% for 20210803t1259d050m_cropped
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
                AllFrames = AllFrames + 1;
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
                    td.dataID(iWaggleEvent)= dataset;
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
                        tdWithInh.frameID(iWaggleInh)= iFrame;
                        tdWithInh.dataID(iWaggleInh)= dataset;
                        %
                    end
                end
                waggleConvInh = min(max(waggleConvInh + single(conv2(waggleConvThreshed,inhDisc,'same')) - .1,0),1);
            end
        end
    end
end

%%
% LOAD VIDEOS
VIDS = ['final_labels/20210803t1727d200m_cropped';
        'final_labels/20210803t1301d050m_cropped';
        'final_labels/20210803t1508d100m_cropped';
        'final_labels/20210803t1517d100m_cropped';
        'final_labels/20210803t1719d200m_cropped';
        'final_labels/20210803t1259d050m_cropped'];
% LOAD DETECTED WAGGLES
TD =   ['final_labels/20210803t1727d200m_cropped/20210803t1727d200m_cropped';
        'final_labels/20210803t1301d050m_cropped/20210803t1301d050m_cropped';
        'final_labels/20210803t1508d100m_cropped/20210803t1508d100m_cropped';
        'final_labels/20210803t1517d100m_cropped/20210803t1517d100m_cropped';
        'final_labels/20210803t1719d200m_cropped/20210803t1719d200m_cropped';
        'final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped'];
% LOAD GROUND TRUTH
GT = ['final_labels/20210803t1727d200m_cropped/20210803t1727d200m_ground_truth.mat';
      'final_labels/20210803t1301d050m_cropped/20210803t1301d050m_ground_truth.mat';
      'final_labels/20210803t1508d100m_cropped/20210803t1508d100m_ground_truth.mat';
      'final_labels/20210803t1517d100m_cropped/20210803t1517d100m_ground_truth.mat';
      'final_labels/20210803t1719d200m_cropped/20210803t1719d200m_ground_truth.mat';
      'final_labels/20210803t1259d050m_cropped/20210803t1259d050m_ground_truth.mat'];

% convMapThresholdRange   = 24;
% sensitivity             = nan(convMapThresholdRange,1);
% specificity             = nan(convMapThresholdRange,1);
% precision               = nan(convMapThresholdRange,1);
% recall                  = nan(convMapThresholdRange,1);
% F1                      = nan(convMapThresholdRange,1);
% informedness            = nan(convMapThresholdRange,1);
% accuracy                = nan(convMapThresholdRange,1);

nFrameTotalperData = [];
for filedata = 1:numel(VIDS(:,1))
    video=dir(VIDS(filedata,:) + "/*.MP4");
    v0 = VideoReader(VIDS(filedata,:) + "/" + video.name);
    nFrameTotalperData = [nFrameTotalperData;round(v0.FrameRate *v0.Duration)];
end
nFrameTotal = sum(nFrameTotalperData);

beePixelSize              = 150;
missclassified            = 0;
correctlyclassified       = 0;

td_missclassified         = [];
td_correctlyclassified    = [];
oneHotEncodedLabels_td    = nan(nFrameTotal,1);
oneHotEncodedLabels_td_gt = nan(nFrameTotal,1);

% for file_index = 7 %1:convMapThresholdRange
for filedata = 1:numel(VIDS(:,1))
    filedata
    load(TD(filedata,:) + "_TD.mat")
    %     load(VIDS(filedata,:) + "td_with_threshold_" + num2str(file_index) + ".mat");
    load(GT(filedata,:))
    
    %         if isempty(td)
    %             sensitivity(file_index)     = 0; % TP/TP+FN
    %             specificity(file_index)     = 0; % TN/TN+FP
    %             precision(file_index)       = 0;
    %             recall(file_index)          = 0;
    %             F1(file_index)              = 0;
    %             informedness(file_index)    = 0;
    %             accuracy(file_index)        = 0;
    %         else
    
    if numel(td.x) > numel(td_gt.x)
        sequence = numel(td_gt.x);
    else
        sequence = numel(td.x);
    end
    for idx = 1:sequence
        if ((td.x(idx)*4 - td_gt.x(idx))^2 + (td.y(idx)*4 - td_gt.y(idx))^2)<beePixelSize^2
            correctlyclassified = correctlyclassified +1;
            td_correctlyclassified.x(correctlyclassified) = td.x(idx)*4;
            td_correctlyclassified.y(correctlyclassified) = td.y(idx)*4;
            td_correctlyclassified.ts(correctlyclassified) = td.ts(idx);
        else
            missclassified = missclassified +1;
            td_missclassified.x(missclassified) = td.x(idx)*4;
            td_missclassified.y(missclassified) = td.y(idx)*4;
            td_missclassified.ts(missclassified) = td.ts(idx);
        end
    end
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
end

correctlyLabeledPositives_TP = sum(and(oneHotEncodedLabels_td_gt,oneHotEncodedLabels_td),1);   % TP
correctlyLabeledNegatives_TN = sum(and(~oneHotEncodedLabels_td_gt,~oneHotEncodedLabels_td),1); % TN
wronglyLabeledPositives_FP = sum((oneHotEncodedLabels_td_gt == 1) & (oneHotEncodedLabels_td == 0)); % FP
wronglyLabeledNegatives_FP = sum((oneHotEncodedLabels_td_gt == 0) & (oneHotEncodedLabels_td == 1)); % FN
actualpositives = numel(oneHotEncodedLabels_td_gt(oneHotEncodedLabels_td_gt(:,1) > 0,1)); % TP+FN
actualnegatives = numel(oneHotEncodedLabels_td_gt(oneHotEncodedLabels_td_gt(:,1) <  1,1)); % TN+FP

%%%%%%%%%% FINAL EVALUATION %%%%%%%%%%
%         sensitivity(file_index) = correctlyLabeledPositives_TP ./ actualpositives; % TP/TP+FN
%         specificity(file_index) = correctlyLabeledNegatives_TN ./ actualnegatives; % TN/TN+FP
%         precision(file_index) = correctlyLabeledPositives_TP / (correctlyLabeledPositives_TP + wronglyLabeledPositives_FP);
%         recall(file_index) = correctlyLabeledPositives_TP / (correctlyLabeledPositives_TP + wronglyLabeledNegatives_FP);
%         F1(file_index) = (2 * precision(file_index) * recall(file_index)) / (precision(file_index) + recall(file_index));
%         informedness(file_index) = sensitivity(file_index) + specificity(file_index) - 1;
%         accuracy(file_index) = (1 - missclassified/(missclassified + correctlyclassified))*100;

sensitivity     = correctlyLabeledPositives_TP ./ actualpositives % TP/TP+FN
specificity     = correctlyLabeledNegatives_TN ./ actualnegatives % TN/TN+FP
precision       = correctlyLabeledPositives_TP / (correctlyLabeledPositives_TP + wronglyLabeledPositives_FP)
recall          = correctlyLabeledPositives_TP / (correctlyLabeledPositives_TP + wronglyLabeledNegatives_FP)
F1              = (2 * precision * recall) / (precision + recall)
informedness    = sensitivity + specificity - 1
accuracy        = (1 - missclassified/(missclassified + correctlyclassified))*100
%     end
% end

figure(66);
% subplot(2,2,1);
plot(F1,"*-","LineWidth",2);grid on;hold on;
plot(informedness,"s-","LineWidth",2);
plot(sensitivity,"<-","LineWidth",2);
plot(specificity,">-","LineWidth",2);
title("Evaluation over all waggles recordings");
xlabel("Threshold"); 
ylim([0 1.25])
xline(7,'-',{'Acceptable','Threshold'},"LineWidth",4);
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
