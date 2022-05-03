%%%%% start here and make a code to process all recording all together and
%%%%% save td for each threshold value and each datasets
clear
tic;

%%%%%%%%%%%%%%%% LOAD VIDEO %%%%%%%%%%%%%%%%
% videoName = "20210803t1727d200m";
% dataPATH = "input_videos/20210803t1259d050m_cropped";
% load('./final_labels/20210803t1301d050m_cropped/20210803t1301d050m_ground_truth.mat');
videoName = {'20210803t1259d050m_cropped'; '20210803t1727d200m_cropped'; '20210803t1301d050m_cropped';
        '20210803t1508d100m_cropped'; '20210803t1719d200m_cropped'; '20210803t1517d100m_cropped';
        '20210803t1732d200m_cropped_W1'};

for video_index = 1:numel(videoName)

    addpath("../DeepGreen/greenhouseCode")
    videoFileName = "./final_labels/" + videoName(video_index) + "/" + videoName(video_index) + ".MP4";
    v0 = VideoReader(videoFileName);
    disp("Loading data...")
    
    %%%%%%%%%%%%%%%% LOAD TEMPLATE
    load('waggle36Templates_25x25_HQ.mat')
%     load('waggle16Templates_v1.mat')
    disp("Template loaded...")
    
    for convThres = 100:50:1000
        td                          = [];
        tdWithInh                   = [];
        jTd                         = [];
        iWaggleInh                  = 0;
        iWaggleEvent                = 0;
        avgFrameDepth               = 6;
        %     convMapThreshold            = 15;
        nEventsForWaggleThreshold   = 6;
        nDel                        = 18;
        nTemplate                   = size(waggleTemplate,3);
        nFrameTotal                 = round(v0.FrameRate *v0.Duration);
        
        divisor                     = alldivisors(nFrameTotal);
        numberofSegment             = divisor(:, ceil(end/2)+1);
        
        %     numberofSegment             = 40;
        framesPerSegment            = nFrameTotal/numberofSegment;
        nSegment                    = ceil(nFrameTotal/framesPerSegment);
        AllFrames = 0;
        
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
                
                waggleFilt3   = exp(-delArray/Tau).*sin(4/(2*pi)*(delArray+6.5));
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
                    waggleConvThreshed          = resizedWaggleMatch>convThres;%convMapThreshold;
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
        % save file
        save("./td_out/" + videoName(video_index) + "_newTemplate_" + convThres, "td");
    end
end

%% Evaluate multiple recording together

% LOAD VIDEOS
data = ['20210803t1727d200m_cropped';
        '20210803t1301d050m_cropped';
        '20210803t1508d100m_cropped';
        '20210803t1517d100m_cropped';
        '20210803t1719d200m_cropped';
        '20210803t1259d050m_cropped'];

sensitivity     = nan(30,1);
specificity     = nan(30,1);
precision       = nan(30,1);
recall          = nan(30,1);
F1              = nan(30,1);
informedness    = nan(30,1);
accuracy        = nan(30,1);

ANGLE_DETECTION             = 0;
beePixelSize                = 150;
angleRange                  = 20;

for recording_index = 1%:30
    human_labelled_waggles      = [];
    algorithm_detected_waggles  = [];
    
    for filedata = 6%1:numel(data(:,1))
%         load("/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/td_out/" + data(filedata,:) + "_oldTemplate_" + recording_index + ".mat")
%         load("/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/" + data(filedata,:) + "/" + data(filedata,:) + "_ground_truth.mat")
        
        load('/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_TD.mat')
        load('/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_ground_truth.mat')

        if numel(td) == 0 || numel(td.x) < 2
            recording_index = recording_index + 1;
            load('/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_TD.mat')
            load('/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_ground_truth.mat')

%             load("/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/td_out/" + data(filedata,:) + "_oldTemplate_" + recording_index + ".mat")
%             load("/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/final_labels/" + data(filedata,:) + "/" + data(filedata,:) + "_ground_truth.mat")
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
            pred_waggle.x(td_gt.frameID(end)+1:td.frameID(end))         = [];
            pred_waggle.y(td_gt.frameID(end)+1:td.frameID(end))         = [];
            pred_waggle.angle(td_gt.frameID(end)+1:td.frameID(end))     = [];
            pred_waggle.frameID(td_gt.frameID(end)+1:td.frameID(end))   = [];
            pred_waggle.c(td_gt.frameID(end)+1:td.frameID(end))         = [];
        end
        td = pred_waggle;
        predicted_labels = nan(numel(td.x),1);
        for idx = 1:numel(td.x)
            if ANGLE_DETECTION
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

% figure(676723);confusionchart(confusion_matrix,'RowSummary','row-normalized','ColumnSummary','column-normalized');

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

%% Evaluate angle detection
videoFileName = "final_labels/20210803t1727d200m_cropped/20210803t1727d200m_cropped.MP4";
load('final_labels/20210803t1727d200m_cropped/20210803t1727d200m_ground_truth.mat')
load('final_labels/20210803t1727d200m_cropped/20210803t1727d200m_cropped_TD_25x25Template.mat')
v0 = VideoReader(videoFileName);
nFrameTotal = round(v0.FrameRate *v0.Duration);

% nFrameTotal = 2400;
beePixelSize            = 70;
angleRange              = 20;
missclassified          = 0;
correctlyclassified     = 0;
td_missclassified       = [];
td_correctlyclassified  = [];

for idx = 1:numel(td.x)
    if ((td.x(idx)*4 - td_gt.x(idx))^2 + (td.y(idx)*4 - td_gt.y(idx))^2)<beePixelSize^2 && td.angle(idx) > td_gt.angle(idx)-angleRange && td.angle(idx) < td_gt.angle(idx)+angleRange
        correctlyclassified = correctlyclassified +1;
        td_correctlyclassified.x(correctlyclassified) = td.x(idx)*4;
        td_correctlyclassified.y(correctlyclassified) = td.y(idx)*4;
        td_correctlyclassified.ts(correctlyclassified) = td.ts(idx);
        td_correctlyclassified.angle(correctlyclassified) = td.angle(idx);
        td_correctlyclassified.frameID(correctlyclassified) = td.frameID(idx);
    else
        missclassified = missclassified +1;
        td_missclassified.x(missclassified) = td.x(idx)*4;
        td_missclassified.y(missclassified) = td.y(idx)*4;
        td_missclassified.angle(missclassified) = td.y(idx)*4;
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


figure(66);
subplot(4,2,[1 3])
scatter3(td.x*4,td.y*4,td.ts,'.r');hold on
scatter3(td_gt.x,td_gt.y,td_gt.frameID,'.b');
plot3(td_gt.x,td_gt.y,td_gt.frameID, 'bo', 'MarkerSize', 2);
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
title("Detected waggles");
xlim([min(td_gt.x)-150 max(td_gt.x)+150]);
ylim([min(td_gt.y)-150 max(td_gt.y)+150]);
zlim([0 numel(td_gt.y)]);
legend([{'Detected Waggles'},{'Labelled Waggles'}]);
set(gca,'fontsize', 14)
subplot(4,2,[2 4])
scatter3(td_correctlyclassified.x,td_correctlyclassified.y,td_correctlyclassified.ts,'.r');hold on
scatter3(td_missclassified.x,td_missclassified.y,td_missclassified.ts,'.k');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames");
xlim([min(td_gt.x)-150 max(td_gt.x)+150]);
ylim([min(td_gt.y)-150 max(td_gt.y)+150]);
zlim([0 numel(td_gt.y)]);
title("F1-Score: " + F1 + "%");
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

