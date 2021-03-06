%% make a tracking id file
clear
addpath("/media/sam/Samsung_T5/PhD/Code/bee_tracking/data/")
load('final_labels/20210803t1719d200m_cropped/20210803t1719d200m_cropped_ground_truth.mat')
total_frame = numel(td_gt.x);
track = [];
trajectory_path = [];

PATH = "/media/sam/Samsung_T5/PhD/Code/bee_tracking/data/trajectories";
trajectories = dir(PATH + "/*.txt");

for idx = 1:size(trajectories,1)-1
    trajectory = readcell("/media/sam/Samsung_T5/PhD/Code/bee_tracking/data/trajectories/" + trajectories(idx).name);
    frameid = trajectory(:,1);frameid = [frameid{:}]';
    x       = trajectory(:,2);x = [x{:}]';
    y       = trajectory(:,3);y = [y{:}]';
    angle   = trajectory(:,5);angle = [angle{:}]';
    
    trajectory_path.x       = nan(total_frame,1);
    trajectory_path.y       = nan(total_frame,1);
    trajectory_path.id      = nan(total_frame,1);
    trajectory_path.angle   = nan(total_frame,1);
    
    if frameid(1) ~=0
        trajectory_path.x(1:frameid(1)) = 0;
        trajectory_path.y(1:frameid(1)) = 0;
        trajectory_path.id(1:frameid(1)) = 0;
        trajectory_path.angle(1:frameid(1)) = 0;
        
        trajectory_path.x(frameid(1)+1:frameid(1)+numel(frameid)) = x;
        trajectory_path.y(frameid(1)+1:frameid(1)+numel(frameid)) = y;
        trajectory_path.id(frameid(1)+1:frameid(1)+numel(frameid)) = frameid;
        trajectory_path.angle(frameid(1)+1:frameid(1)+numel(frameid)) = angle;
        
        trajectory_path.x(frameid(1)+numel(frameid)+1:end) = 0;
        trajectory_path.y(frameid(1)+numel(frameid)+1:end) = 0;
        trajectory_path.id(frameid(1)+numel(frameid)+1:end) = 0;
        trajectory_path.angle(frameid(1)+numel(frameid)+1:end) = 0;
        
    elseif frameid(1) == 0 && frameid(end) ~= total_frame
        trajectory_path.x(frameid(1)+1:frameid(1)+numel(frameid)) = x;
        trajectory_path.y(frameid(1)+1:frameid(1)+numel(frameid)) = y;
        trajectory_path.id(frameid(1)+1:frameid(1)+numel(frameid)) = frameid;
        trajectory_path.angle(frameid(1)+1:frameid(1)+numel(frameid)) = angle;
        
        trajectory_path.x(frameid(1)+numel(frameid)+1:end) = 0;
        trajectory_path.y(frameid(1)+numel(frameid)+1:end) = 0;
        trajectory_path.id(frameid(1)+numel(frameid)+1:end) = 0;
        trajectory_path.angle(frameid(1)+numel(frameid)+1:end) = 0;
        
    elseif frameid(1) ~=0 && frameid(end) == total_frame
        trajectory_path.x(1:frameid(1)) = 0;
        trajectory_path.y(1:frameid(1)) = 0;
        trajectory_path.id(1:frameid(1)) = 0;
        trajectory_path.angle(1:frameid(1)) = 0;
        
        trajectory_path.x(frameid(1)+1:frameid(1)+numel(frameid)) = x;
        trajectory_path.y(frameid(1)+1:frameid(1)+numel(frameid)) = y;
        trajectory_path.id(frameid(1)+1:frameid(1)+numel(frameid)) = frameid;
        trajectory_path.angle(frameid(1)+1:frameid(1)+numel(frameid)) = angle;
    end
    
    track{idx} = trajectory_path;
   
end
track = track';
% track(1) = [];


% figure(6565);
% tiledlayout(1,2)
% ax1 = nexttile;
% % subplot(1,2,1)
% for idx = 1:numel(track)-1
%     scatter3(track{idx, 1}.x,track{idx, 1}.y,track{idx, 1}.id,'.');
%     xlabel("X [px]");
%     ylabel("Y [px]");
%     zlabel("#Frames"); grid on; hold on
% end
% % subplot(1,2,2)
% ax2 = nexttile;
% 
% load('final_labels/20210803t1719d200m_cropped/20210803t1719d200m_cropped_ground_truth.mat')
% load('final_labels/20210803t1719d200m_cropped/20210803t1719d200m_cropped_PeakDetection.mat')
% 
% td.x(end:numel(td_gt.x)) = 0;
% td.y(end:numel(td_gt.x)) = 0;
% 
% peak_output = [];
% beePixelSize = 100;
% for idx = 1:numel(td_gt.x)
%     if ((td.x(idx)*2 - td_gt.x(idx))^2 + (td.y(idx)*2 - td_gt.y(idx))^2)<beePixelSize^2
%         peak_output.x(idx) = td.x(idx)*2;
%         peak_output.y(idx) = td.y(idx)*2;
%         peak_output.ts(idx) = td.ts(idx);
%     end
% end
% time = 1:numel(peak_output.ts);
% scatter3(peak_output.x,peak_output.y,time,'.r');
% xlabel("X [px]");
% ylabel("Y [px]");
% zlabel("#Frames"); grid on;
% linkaxes([ax1 ax2],'xy')

%% Extract correctly detected waggles from the track id
%%%% load data path
% dataPath = "final_labels/20210803t1259d050m_cropped";
dataPath = "final_labels/20210803t1719d200m_cropped";

%%%% load segmentation-tracker output "tracks"
% load(dataPath + '/20210803t1259d050m_Tracker_outputV3.mat')
load(dataPath + '/20210803t1719d200m_cropped_Tracker_OutputV2.mat')

%%%% load ground truth
% load('final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_ground_truth.mat')
load(dataPath + '/20210803t1719d200m_cropped_ground_truth.mat')

%%%% load peak detection output
% load(dataPath + '/20210803t1259d050m_peak_output.mat')
load('td_out/20210803t1719d200m_peak_output_corrected')

peak_output.x(end:numel(td_gt.x)) = 0;
peak_output.y(end:numel(td_gt.y)) = 0;
peak_output.ts(end:numel(td_gt.frameID)) = numel(peak_output.ts):numel(td_gt.frameID);

beePixelSize = 50;
correct_detected_track_coordinate = [];

for idx = 1:numel(td_gt.x)
    for idx2 = 1:numel(track)
        if ((track{idx2, 1}.x(idx) - td_gt.x(idx))^2 + (track{idx2, 1}.y(idx) - td_gt.y(idx))^2)<beePixelSize^2
            correct_detected_track_coordinate.x(idx)        = track{idx2, 1}.x(idx);
            correct_detected_track_coordinate.y(idx)        = track{idx2, 1}.y(idx);
            correct_detected_track_coordinate.angle(idx)    = track{idx2, 1}.angle(idx);
            correct_detected_track_coordinate.ts(idx)       = track{idx2, 1}.id(idx);
            correct_detected_track_coordinate.trackid(idx)  = idx2;
        end
    end
end

findZerosx = find(correct_detected_track_coordinate.x==0);
correct_detected_track_coordinate.x(findZerosx) = NaN;
findZerosy = find(correct_detected_track_coordinate.y==0);
correct_detected_track_coordinate.y(findZerosy) = NaN;
findZerosangle = find(correct_detected_track_coordinate.angle==0);
correct_detected_track_coordinate.angle(findZerosangle) = NaN;
findZerosts = find(correct_detected_track_coordinate.ts==0);
correct_detected_track_coordinate.ts(findZerosts) = NaN;

%%%%% interpolate datapoint
correct_detected_track_coordinate.interx = correct_detected_track_coordinate.x;
nanx = isnan(correct_detected_track_coordinate.interx);
tx    = 1:numel(correct_detected_track_coordinate.interx);
correct_detected_track_coordinate.interx(nanx) = interp1(tx(~nanx), correct_detected_track_coordinate.interx(~nanx), tx(nanx));
correct_detected_track_coordinate.interx = round(correct_detected_track_coordinate.interx);

correct_detected_track_coordinate.intery = correct_detected_track_coordinate.y;
nany = isnan(correct_detected_track_coordinate.intery);
ty    = 1:numel(correct_detected_track_coordinate.intery);
correct_detected_track_coordinate.intery(nany) = interp1(ty(~nany), correct_detected_track_coordinate.intery(~nany), ty(nany));
correct_detected_track_coordinate.intery = round(correct_detected_track_coordinate.intery);

correct_detected_track_coordinate.interangle = correct_detected_track_coordinate.angle;
nanangle = isnan(correct_detected_track_coordinate.interangle);
tangle    = 1:numel(correct_detected_track_coordinate.interangle);
correct_detected_track_coordinate.interangle(nanangle) = interp1(tangle(~nanangle), correct_detected_track_coordinate.interangle(~nanangle), tangle(nanangle));
correct_detected_track_coordinate.interangle = round(correct_detected_track_coordinate.interangle);

correct_detected_track_coordinate.interts = correct_detected_track_coordinate.ts;
nants = isnan(correct_detected_track_coordinate.interts);
tts    = 1:numel(correct_detected_track_coordinate.interts);
correct_detected_track_coordinate.interts(nants) = interp1(tts(~nants), correct_detected_track_coordinate.interts(~nants), tts(nants));
correct_detected_track_coordinate.interts = round(correct_detected_track_coordinate.interts);


figure(786);
tiledlayout(1,5)
ax1 = nexttile;
scatter3(td_gt.x, td_gt.y,td_gt.frameID, '.b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames"); grid on;
xlim([0 800]);
ylim([0 800]);
title("Ground truth");

ax2 = nexttile;
scatter3(peak_output.x,peak_output.y,peak_output.ts,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames"); grid on;
xlim([0 800]);
ylim([0 800]);
title("Peak detector output");

ax3 = nexttile;
C = unique(correct_detected_track_coordinate.trackid);
for idx = 1:numel(C)
    values = find(correct_detected_track_coordinate.trackid==C(idx));
    scatter3(correct_detected_track_coordinate.x(values),correct_detected_track_coordinate.y(values),td_gt.frameID(values),'.'); hold on
end
xlabel("X [px]");
ylabel("Y [px]");
xlim([0 800]);
ylim([0 800]);
zlabel("#Frames"); grid on;
title("Tracker output");

ax4 = nexttile;
scatter3(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,correct_detected_track_coordinate.interts,'.r');hold on
plot3(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,correct_detected_track_coordinate.interts,'-r');
xlabel("X [px]");
ylabel("Y [px]");
xlim([0 800]);
ylim([0 800]);
zlabel("#Frames"); grid on;
title("Tracker output - Linear Interpolation");

ax5 = nexttile;
scatter3(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,correct_detected_track_coordinate.interts,'.r');hold on
plot3(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,correct_detected_track_coordinate.interts,'-r');
scatter3(td_gt.x, td_gt.y,td_gt.frameID, '.b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames"); grid on;
title("Tracker output/Ground truth overlay");
linkaxes([ax1 ax2 ax3 ax4 ax5],'xy')

%%%%% evaluate algorithm
% evaluate this "correct_detected_track_coordinate" with this "td_gt"
% evaluate this "peak_output" with this "td_gt"
% evaluate "td_gt" with the combined output

correct_detected_track_coordinate.c = nan(1,numel(correct_detected_track_coordinate.x));
findNaN = isnan(correct_detected_track_coordinate.interts);
findNonNaN = ~isnan(correct_detected_track_coordinate.interts);

td_gt.c = ones(1,numel(td_gt.x));
correct_detected_track_coordinate.c(findNaN) = 0;
correct_detected_track_coordinate.c(findNonNaN) = 1;

if numel(correct_detected_track_coordinate.c) < numel(td_gt.c)
    correct_detected_track_coordinate.c(end:numel(td_gt.c)) = 0;
    correct_detected_track_coordinate.interx(end:numel(td_gt.c)) = NaN;
    correct_detected_track_coordinate.intery(end:numel(td_gt.c)) = NaN;
    correct_detected_track_coordinate.interangle(end:numel(td_gt.c)) = NaN;
    correct_detected_track_coordinate.trackid(end:numel(td_gt.c)) = NaN;
    correct_detected_track_coordinate.interts(end:numel(td_gt.c)) = numel(correct_detected_track_coordinate.c):numel(td_gt.c);
end

label_postprocessing = [td_gt.c; correct_detected_track_coordinate.c]';

true_positive = sum(and(label_postprocessing(:,1),label_postprocessing(:,2)),1);  % TP
true_negative = sum(and(~label_postprocessing(:,1),~label_postprocessing(:,2)),1); % TN

false_positive = sum((label_postprocessing(:,1) == 1) & (label_postprocessing(:,2) == 0)); % FP
false_negative = sum((label_postprocessing(:,1) == 0) & (label_postprocessing(:,2) == 1)); % FN

groundtruth = td_gt.c';
actualpositives = numel(groundtruth(groundtruth > 0,1));  % TP+FN
actualnegatives = numel(groundtruth(groundtruth <  1,1)); % TN+FP

confusion_matrix = [true_positive false_positive ; false_negative true_negative]

testingPresentations = numel(td_gt.x);
missclassified = numel(find(correct_detected_track_coordinate.c==0));
sensitivity     = true_positive ./ actualpositives;
specificity     = true_negative ./ actualnegatives;
precision       = true_positive / (true_positive + false_positive);
recall          = true_positive / (true_positive + false_negative);
F1              = (2 * precision * recall) / (precision + recall);
informedness    = sensitivity + specificity - 1;
accuracy        = (1 - missclassified/testingPresentations)*100

% evaluate angle
correct_angle   = 0;
wrong_angle     = 0;
angleRange      = 90;
for idx = 1:numel(td_gt.angle)
   if correct_detected_track_coordinate.interangle(idx) > td_gt.angle(idx)-angleRange && correct_detected_track_coordinate.interangle(idx) < td_gt.angle(idx)+angleRange
       correct_angle = correct_angle + 1;
   else
       wrong_angle = wrong_angle + 1;
   end
end
angle_accuracy_percentage = (1 - wrong_angle/testingPresentations)*100

positionOffset = 50;
radius = 75;
overlapRatio = nan(1,numel(td_gt.x));
for k=1:numel(td_gt.x)
    if ~isnan(correct_detected_track_coordinate.interx(k)-positionOffset)
        bbox_correct_detected_track = [correct_detected_track_coordinate.interx(k)-positionOffset, correct_detected_track_coordinate.intery(k)-positionOffset, radius, radius];
        bbox_groundtruth = [td_gt.x(k)-positionOffset, td_gt.y(k)-positionOffset, radius, radius];
        overlapRatio(k) = bboxOverlapRatio(bbox_correct_detected_track,bbox_groundtruth);
    end
end
figure(678);
subplot(3,1,1);
plot(overlapRatio,'r');
xlabel("# Frame");
ylabel("IoU");
title("Intersection over Union (IoU)");grid on;
subplot(3,1,2);
plot(correct_detected_track_coordinate.interx,'b');
xlabel("# Frame");
ylabel("X [px]");
title("Waggle trajectory over X");grid on;
subplot(3,1,3);
plot(correct_detected_track_coordinate.intery,'b');
xlabel("# Frame");
ylabel("Y [px]");
title("Waggle trajectory over Y");grid on;
findNonNaN = ~isnan(overlapRatio);
average_IoU = mean(overlapRatio(findNonNaN))

figure(577);
subplot(1,2,1)
scatter(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,'.r');hold on
plot(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,'-r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames"); grid on;
subplot(1,2,2)
scatter(td_gt.x, td_gt.y, '.b');
plot(td_gt.x,td_gt.y,'-b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames"); grid on;

X_smooth = movmedian(td_gt.x,50);
Y_smooth = movmedian(td_gt.y,50);

figure(670);
subplot(2,3,[1 2]);
plot(X_smooth,'LineWidth',2);title("X");grid on;
subplot(2,3,[4 5]);
plot(Y_smooth,'LineWidth',2);title("Y");grid on;
subplot(2,3,[3 6]);
scatter3(X_smooth, Y_smooth, 1:numel(Y_smooth),'.r');hold on;
plot3(X_smooth, Y_smooth, 1:numel(Y_smooth),'.r');

% X_smooth_output = movmedian(correct_detected_track_coordinate.interx,50);
% Y_smooth_output = movmedian(correct_detected_track_coordinate.intery,50);
%% plot the correctly detected waggle events over frames without animation
frames=dir("../bee_tracking/data/frames/*.png");

radius = 75;
positionOffset = 50;
firstFrameLabel = 240;

% writerObj = VideoWriter('output_videos/20210803t1719d200m_cropped_tracking_video_overlay_interpolated_animated.avi');
% writerObj.FrameRate = 15;
% open(writerObj);

figure(56);
overlapRatio = nan(1,numel(td_gt.x));

subtightplot(3,3,[1 4])
scatter3(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,correct_detected_track_coordinate.interts,'.r');hold on
plot3(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,correct_detected_track_coordinate.interts,'-r');
scatter3(td_gt.x, td_gt.y,td_gt.frameID, '.b');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames"); grid on;
xlim([0 1000]);
ylim([0 1000]);
zlim([240 numel(td_gt.x)]);
title("Tracker output/Ground truth overlay");

for k=td_gt.frameID(1):650%numel(td_gt.x)
    k
%     subtightplot(3,3,[1 4])
%     scatter3(correct_detected_track_coordinate.interx(k),correct_detected_track_coordinate.intery(k),td_gt.frameID(k),'.r');hold on
%     plot3(correct_detected_track_coordinate.interx(k),correct_detected_track_coordinate.intery(k),td_gt.frameID(k),'-r');
%     scatter3(td_gt.x(k), td_gt.y(k),td_gt.frameID(k), '.b');
%     xlabel("X [px]");
%     ylabel("Y [px]");
%     zlabel("#Frames"); grid on;
%     xlim([0 1000]);
%     ylim([0 1000]);
%     zlim([240 numel(td_gt.x)]);
%     title("Tracker output/Ground truth overlay");
    FrameNames=frames(firstFrameLabel+k).name;    
    subtightplot(3,3,[2 6])
    I = imread("../bee_tracking/data/frames/" + FrameNames);
    imshow("../bee_tracking/data/frames/" + FrameNames);hold on;
    
    text(50, 20, '___', 'Color', 'g', 'FontSize',20);
    text(105, 35, 'Ground truth', 'Color', 'g', 'FontSize',16);
    text(50, 50, '___', 'Color', 'r', 'FontSize',20);
    text(105, 65, 'Detected Waggle', 'Color', 'r', 'FontSize',16);
    
    if ~isnan(correct_detected_track_coordinate.interx(k)) && ~isnan(correct_detected_track_coordinate.intery(k)) && correct_detected_track_coordinate.interx(k) ~= 0 && correct_detected_track_coordinate.intery(k) ~= 0
        rectangle('Position',[correct_detected_track_coordinate.interx(k)-positionOffset correct_detected_track_coordinate.intery(k)-positionOffset radius radius],'EdgeColor','r','LineWidth',2);
    end
    rectangle('Position',[td_gt.x(k)-positionOffset td_gt.y(k)-positionOffset radius radius],'EdgeColor','g','LineWidth',2);
    title("Waggle trajectory overlayed on original frames");
    
    if ~isnan(correct_detected_track_coordinate.interx(k)-positionOffset)
        bbox_correct_detected_track = [correct_detected_track_coordinate.interx(k)-positionOffset, correct_detected_track_coordinate.intery(k)-positionOffset, radius, radius];
        bbox_groundtruth = [td_gt.x(k)-positionOffset, td_gt.y(k)-positionOffset, radius, radius];
        overlapRatio(k) = bboxOverlapRatio(bbox_correct_detected_track,bbox_groundtruth);
        text(600, 40, "IoU= " + num2str(overlapRatio(k)), 'Color', 'y', 'FontSize',16);
    end
    
    subtightplot(3,3,[7 9])
    X_Axis = 1:numel(td_gt.x);
    plot(X_Axis,overlapRatio,'r','LineWidth',2);grid on;hold on
    ylabel("IoU");
    xlabel("# Frames");
    drawnow
    
%     F = getframe(gcf) ;
%     writeVideo(writerObj, F);
end

% close(writerObj);
% fprintf('Sucessfully generated the video\n');
%% plot the correctly detected waggle events over frames with animation
frames=dir("DenseObjectAnnotation/static/png/*.png");

radius = 75;
firstFrameLabel = 240;
figure(1);
x = correct_detected_track_coordinate.x;
y = correct_detected_track_coordinate.y;
z = correct_detected_track_coordinate.ts;
p = plot3(x,y,z);

writerObj = VideoWriter('output_videos/tracking_video_overlay_test.avi');
writerObj.FrameRate = 15;
open(writerObj);

for k=20:70%numel(td_gt.x)
    k
    FrameNames=frames(firstFrameLabel+k).name;
    figure(56);
    subplot(1,2,1)
    imshow("DenseObjectAnnotation/static/png/" + FrameNames);
    if ~isnan(correct_detected_track_coordinate.x(k)) && ~isnan(correct_detected_track_coordinate.y(k)) && correct_detected_track_coordinate.x(k) ~= 0 && correct_detected_track_coordinate.y(k) ~= 0
        rectangle('Position',[correct_detected_track_coordinate.x(k)-20 correct_detected_track_coordinate.y(k)-20 radius radius],'EdgeColor','r','LineWidth',2);
    end
    subplot(1,2,2)
    direction = [0 0 1];
    rotate(p,direction,25);
    x = p.XData;
    y = p.YData;
    z = p.ZData;
    scatter3(x,y,z,'.r');
    xlabel("X [px]");
    ylabel("Y [px]");
    zlabel("#Frames");
    xlim([0 1000]);
    ylim([0 1000]);
    zlim([0 1200]);
    pause(2);
    drawnow
    
    F = getframe(gcf) ;
    writeVideo(writerObj, F);
end

close(writerObj);
fprintf('Sucessfully generated the video\n');
%% plot correctly detect tracks from each algorithm along with the ground truth
standard = load('/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/td_out/20210803t1259d050m_non_causal_segmentation_output_standard.mat');
high_scalefactor = load('/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/td_out/20210803t1259d050m_non_causal_segmentation_output_high_scaleFactor.mat');
high_scalefactor_retrained = load('/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/td_out/20210803t1259d050m_non_causal_segmentation_output_high_scaleFactor_retrained.mat');
peak_output = load("/media/sam/Samsung_T5/PhD/Code/bee-waggle-dance-detection/td_out/20210803t1259d050m_non_causal_peak_output.mat");

beePixelSize = 30;
correct_detected_track_coordinate = [];


figure(786);
tiledlayout(1,3)
ax1 = nexttile;
for idx = 1:numel(td_gt.x)
    for idx2 = 1:numel(standard.track)
        if ((standard.track{idx2, 1}.x(idx) - td_gt.x(idx))^2 + (standard.track{idx2, 1}.y(idx) - td_gt.y(idx))^2)<beePixelSize^2
            correct_detected_track_coordinate.x(idx) = standard.track{idx2, 1}.x(idx);
            correct_detected_track_coordinate.y(idx) = standard.track{idx2, 1}.y(idx);
            correct_detected_track_coordinate.ts(idx) = standard.track{idx2, 1}.id(idx);
            correct_detected_track_coordinate.trackid(idx) = idx2;
        end
    end
end
C = unique(correct_detected_track_coordinate.trackid);
for idx = 1:numel(C)
    values = find(correct_detected_track_coordinate.trackid==C(idx));
    scatter3(correct_detected_track_coordinate.x(values),correct_detected_track_coordinate.y(values),correct_detected_track_coordinate.ts(values),'.'); hold on
end
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames"); grid on;
xlim([0 600]);
ylim([0 600]);
title("Standard");

ax2 = nexttile;
for idx = 1:numel(td_gt.x)
    for idx2 = 1:numel(high_scalefactor.track)
        if ((high_scalefactor.track{idx2, 1}.x(idx) - td_gt.x(idx))^2 + (high_scalefactor.track{idx2, 1}.y(idx) - td_gt.y(idx))^2)<beePixelSize^2
            correct_detected_track_coordinate.x(idx) = high_scalefactor.track{idx2, 1}.x(idx);
            correct_detected_track_coordinate.y(idx) = high_scalefactor.track{idx2, 1}.y(idx);
            correct_detected_track_coordinate.ts(idx) = high_scalefactor.track{idx2, 1}.id(idx);
            correct_detected_track_coordinate.trackid(idx) = idx2;
        end
    end
end
C = unique(correct_detected_track_coordinate.trackid);
for idx = 1:numel(C)
    values = find(correct_detected_track_coordinate.trackid==C(idx));
    scatter3(correct_detected_track_coordinate.x(values),correct_detected_track_coordinate.y(values),correct_detected_track_coordinate.ts(values),'.'); hold on
end
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames"); grid on;
xlim([0 600]);
ylim([0 600]);
title("Higher scale factor");

ax3 = nexttile;
for idx = 1:numel(td_gt.x)
    for idx2 = 1:numel(high_scalefactor_retrained.track)
        if ((high_scalefactor_retrained.track{idx2, 1}.x(idx) - td_gt.x(idx))^2 + (high_scalefactor_retrained.track{idx2, 1}.y(idx) - td_gt.y(idx))^2)<beePixelSize^2
            correct_detected_track_coordinate.x(idx) = high_scalefactor_retrained.track{idx2, 1}.x(idx);
            correct_detected_track_coordinate.y(idx) = high_scalefactor_retrained.track{idx2, 1}.y(idx);
            correct_detected_track_coordinate.ts(idx) = high_scalefactor_retrained.track{idx2, 1}.id(idx);
            correct_detected_track_coordinate.trackid(idx) = idx2;
        end
    end
end
C = unique(correct_detected_track_coordinate.trackid);
for idx = 1:numel(C)
    values = find(correct_detected_track_coordinate.trackid==C(idx));
    scatter3(correct_detected_track_coordinate.x(values),correct_detected_track_coordinate.y(values),correct_detected_track_coordinate.ts(values),'.'); hold on
end
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frames"); grid on;
xlim([0 600]);
ylim([0 600]);
title("Higher scale factor retrained");
linkaxes([ax1 ax2 ax3],'xy')

%% test frame differencing wario dataset

% writerObj = VideoWriter('output_videos/wario_Dataset.avi');
% writerObj.FrameRate = 15;
% open(writerObj);

frames=dir("~/PhD/bee_waggle_run/GroundTruthData/GTRuns/20160814_1001_1/6/*.png");
nFrame = numel(frames)-1;
for iFrame = 1:numel(frames)-1
    FrameNames=frames(iFrame).name;
    frameIntFull = imread("~/PhD/bee_waggle_run/GroundTruthData/GTRuns/20160814_1001_1/1/" + FrameNames);
    frameInt = frameIntFull;
    imageHeight = 50;
    imageWidth = imageHeight;
    
    frameArray = zeros(round(imageHeight/1),round(imageWidth/1),3,nFrame,'uint8');
    dRgbFrameArray = zeros(round(imageHeight/1),round(imageWidth/1),3,nFrame,'single');
    dGreyScaleArray = zeros(round(imageHeight/1),round(imageWidth/1),nFrame,'single');
    
    frameArray(:,:,:,iFrame) =  imresize(frameInt,1);
    
    if iFrame>1
        dRgbFrameArray(:,:,:,iFrame) = single(frameArray(:,:,:,iFrame)) - single(frameArray(:,:,:,iFrame-1));
        dGreyScaleArray(:,:,iFrame) =  vecnorm(single(frameArray(:,:,:,iFrame)),2,3) -   vecnorm(single(frameArray(:,:,:,iFrame-1)),2,3);
    end
    
 
    figure(678);
    imagesc(dGreyScaleArray(:,:,iFrame));

    
    %     figure(68);
    %     for plot = 1:6
    %         subplot(3,2,plot)
    %         imshow("~/PhD/bee_waggle_run/GroundTruthData/GTRuns/20160814_1001_1/" + num2str(plot) + "/" + FrameNames);
    %     end
    %     F = getframe(gcf) ;
    %     writeVideo(writerObj, F);
end

% close(writerObj);
% fprintf('Sucessfully generated the video\n');
