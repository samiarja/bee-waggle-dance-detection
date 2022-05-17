%% Interpolate ground truth
PATH = "final_labels/20210803t1727d200m_cropped/"; % no linear interpolation
% PATH = "final_labels/20210803t1259d050m_cropped/"; % no linear interpolation
% PATH = "final_labels/20210803t1508d100m_cropped/";   % with linear interpolation
% PATH = "final_labels/20210803t1719d200m_cropped/";   % with linear interpolation
% PATH = "final_labels/20210803t1732d200m_cropped_W1/";   % with linear interpolation
% PATH = "DenseObjectAnnotation/static/"; % with linear interpolation

INTERPOLATE = 0;
frames=dir(PATH + "png/*.png");
labels=dir(PATH + "txt/*.txt");
radius = 15;
beeTrajectory = nan(1000,2);
td_gt = [];
iFrame = 0;
numberofpoints = 5;
iWaggleEvent = 0;
firstFrameLabel = str2double(labels(1).name(4:6))-1;
% frameID = 214;

% writerObj = VideoWriter('./output_videos/20210803t1259d050m_cropped_ground_truth.avi');
% writerObj.FrameRate = 30;
% open(writerObj);

if INTERPOLATE
    beeTrajectory_x = nan(5,3000);
    beeTrajectory_y = nan(5,3000);
    beeTrajectory_angle = nan(5,3000);
    for k=1:numel(labels)-1
        FrameNames=frames(firstFrameLabel+k+numberofpoints).name;
        LabelNames_before=labels(k).name;
        LabelNames_after=labels(k+1).name;
        iWaggleEvent = iWaggleEvent + 1;
        labelID_before = fileread(PATH + "txt/" + LabelNames_before);
        datacell_before = textscan(labelID_before, '%f%f%f%f', 'Delimiter',',', 'CollectOutput', 1);
        
        labelID_after = fileread(PATH + "txt/" + LabelNames_after);
        datacell_after = textscan(labelID_after, '%f%f%f%f', 'Delimiter',',', 'CollectOutput', 1);
        
        x_coor_before = datacell_before{1}(1,1);
        y_coor_before = datacell_before{1}(1,2);
        angle_before  = datacell_before{1}(1,4);
        
        x_coor_after = datacell_after{1}(1,1);
        y_coor_after = datacell_after{1}(1,2);
        angle_after  = datacell_after{1}(1,4);
        
        x_coor = linspace(x_coor_before, x_coor_after, numberofpoints);
        y_coor = linspace(y_coor_before, y_coor_after, numberofpoints);
        angle  = linspace(angle_before, angle_after, numberofpoints);

        beeTrajectory_x(:,iWaggleEvent) = x_coor';
        beeTrajectory_y(:,iWaggleEvent) = y_coor';
        beeTrajectory_angle(:,iWaggleEvent) = angle';
    end
beeTrajectory_x = reshape(beeTrajectory_x,[],1);
beeTrajectory_y = reshape(beeTrajectory_y,[],1);
beeTrajectory_angle = reshape(beeTrajectory_angle,[],1);
td_gt.x = beeTrajectory_x;
td_gt.y = beeTrajectory_y;
td_gt.angle = beeTrajectory_angle;

frameIDNAN = find(isnan(td_gt.x));
td_gt.x(frameIDNAN(1):end) = [];
td_gt.y(frameIDNAN(1):end,:) = [];
td_gt.angle(frameIDNAN(1):end,:) = [];
td_gt.frameID = (firstFrameLabel+1:firstFrameLabel+frameIDNAN(1)-1)';

for k=1:numel(td_gt.x)
    FrameNames=frames(firstFrameLabel+k).name;
    hf = figure(56);
    imshow(PATH + "png/" + FrameNames);axis on; hold on
    rectangle('Position',[td_gt.x(k)-5 td_gt.y(k)-5 radius radius],'Curvature',[1 1],'FaceColor',[1 0 0],'EdgeColor','k',...
        'LineWidth',2);axis equal
    drawnow
% %     F = getframe(gcf);
% %     writeVideo(writerObj, F);
end
% % close(writerObj);
% % fprintf('Sucessfully generated the video\n')

else
    for k=1:numel(labels)-1
        FrameNames=frames(firstFrameLabel+k).name;
        LabelNames=labels(k).name;
        iWaggleEvent = iWaggleEvent + 1;
        labelID = fileread(PATH + "txt/" + LabelNames);
        datacell = textscan(labelID, '%f%f', 'Delimiter',',', 'CollectOutput', 1);
        x_coor = datacell{1}(1);
        y_coor = datacell{1}(3);
        angle  = datacell{1}(4);
        beeTrajectory(k,1) = x_coor;beeTrajectory(k,2) = y_coor;
        td_gt.x(iWaggleEvent) = x_coor;
        td_gt.y(iWaggleEvent) = y_coor;
        td_gt.angle(iWaggleEvent) = angle;
        td_gt.frameID(iWaggleEvent) = iFrame + firstFrameLabel;
        iFrame = iFrame + 1;
        %     hf = figure(56);
        %     imshow(PATH + "png/" + FrameNames);axis on; hold on
        %     rectangle('Position',[x_coor-5 y_coor-5 radius radius],'Curvature',[1 1],'FaceColor',[1 0 0],'EdgeColor','k',...
        %         'LineWidth',2);axis equal
        %     drawnow
        %     F = getframe(gcf);
        %     writeVideo(writerObj, F);
    end
    % close(writerObj);
    % fprintf('Sucessfully generated the video\n')
end

length_of_video     = numel(frames);
first_label_frame   = firstFrameLabel + 1;

td_gt2 = [];
td_gt2.x(1:first_label_frame)       = 0;
td_gt2.y(1:first_label_frame)       = 0;
td_gt2.angle(1:first_label_frame)   = 0;
td_gt2.frameID(1:first_label_frame) = 0;

td_gt2.x(first_label_frame:numel(td_gt.x)+first_label_frame-1)          = td_gt.x;
td_gt2.y(first_label_frame:numel(td_gt.x)+first_label_frame-1)          = td_gt.y;
td_gt2.angle(first_label_frame:numel(td_gt.x)+first_label_frame-1)      = td_gt.angle;
td_gt2.frameID(first_label_frame:numel(td_gt.x)+first_label_frame-1)    = td_gt.frameID;

td_gt2.x(numel(td_gt.x)+first_label_frame:length_of_video)          = 0;
td_gt2.y(numel(td_gt.x)+first_label_frame:length_of_video)          = 0;
td_gt2.angle(numel(td_gt.x)+first_label_frame:length_of_video)      = 0;
td_gt2.frameID(numel(td_gt.x)+first_label_frame:length_of_video)    = 0;

td_gt = td_gt2;


figure(60);
timestamp = 1:numel(td_gt.x);
scatter3(td_gt.x,td_gt.y,timestamp,'.');
% plot3(td_gt.x,td_gt.y,timestamp,'-','LineWidth',2)
xlabel("X [px]");
ylabel("Y [px]");
zlabel("#Frame");
title("Bee Waggle Trajectory - Ground Truth");
grid on;
set(gca,'fontsize', 16);

%% Extract tracks ID from GNNT 
addpath("/media/sam/Samsung_T5/PhD/Code/bee_tracking/data/")

load('final_labels/20210803t1248d050m_cropped_W2/20210803t1248d050m_cropped_W2_ground_truth_fixed.mat')
% load('final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_ground_truth.mat')

total_frame = numel(dir("/home/sam/PhD/bee_waggle_run/20210803t1248d050m_cropped_W2/png/*.png"));
track = [];
trajectory_path = [];

PATH = "/media/sam/Samsung_T5/PhD/Code/bee_tracking/data/trajectories/";
trajectories = dir(PATH + "*.txt");

for idx = 1:size(trajectories,1)-1
    trajectory = readcell(PATH + trajectories(idx).name);
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

%% Extract correctly detected waggles from the track id
%%%% load data path
dataPath = "final_labels/20210803t1248d050m_cropped_W2";
% dataPath = "final_labels/20210803t1727d200m_cropped";

%%%% load segmentation-tracker output "tracks"
% load(dataPath + '/20210803t1259d050m_Tracker_outputV3.mat')
% load(dataPath + '/20210803t1719d200m_cropped_Tracker_Output_aligned.mat')
load(dataPath + '/WaggleNet/20210803t1248d050m_cropped_W2_tracks.mat')

%%%% load ground truth
% load('final_labels/20210803t1259d050m_cropped/20210803t1259d050m_cropped_ground_truth.mat')
% load(dataPath + '/20210803t1719d200m_cropped_ground_truth_fixed.mat')
load(dataPath + '/WaggleNet/20210803t1248d050m_cropped_W2_ground_truth_fixed.mat')

%%%% load peak detection output
% load(dataPath + '/20210803t1259d050m_peak_output.mat')
% load(dataPath + '/20210803t1719d200m_peak_output_corrected_fixed.mat')
% load(dataPath + '/WaggleNet/20210803t1259d050m_peak_output.mat')

% peak_output.x(end:numel(td_gt.x)) = 0;
% peak_output.y(end:numel(td_gt.y)) = 0;
% peak_output.ts(end:numel(td_gt.frameID)) = numel(peak_output.ts):numel(td_gt.frameID);

beePixelSize = 40;
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
% xlim([0 800]);
% ylim([0 800]);
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
timestamp = 1:numel(td_gt.frameID);
scatter3(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,correct_detected_track_coordinate.interts,'.r');hold on
plot3(correct_detected_track_coordinate.interx,correct_detected_track_coordinate.intery,correct_detected_track_coordinate.interts,'-r');
scatter3(td_gt.x, td_gt.y,timestamp, '.b');
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

%% interpolate angle data between ground truth and detected angles and evaluate event_gt and event_det
dataNAME = "20210803t1248d050m_cropped_W2";
load("final_labels/" + dataNAME + "/WaggleNet/" + dataNAME + "_event_det.mat")
load("final_labels/" + dataNAME + "/WaggleNet/" + dataNAME + "_event_gt.mat")


event_det = struct('x',single(event_det(:,1)),'y',single(event_det(:,2)),'angle',single(event_det(:,3)),'ts',single(event_det(:,4)));
event_gt  = struct('x',single(event_gt(:,1)),'y',single(event_gt(:,2)),'angle',single(event_gt(:,3)),'ts',single(event_gt(:,4)));

event_det.interx = event_det.x;
event_det.interx(event_det.interx==-1) = NaN;
nanx = isnan(event_det.interx);
tx    = 1:numel(event_det.interx);
event_det.interx(nanx) = interp1(tx(~nanx), event_det.interx(~nanx), tx(nanx));
event_det.interx = round(event_det.interx);

event_det.intery = event_det.y;
event_det.intery(event_det.intery==-1) = NaN;
nany = isnan(event_det.intery);
ty    = 1:numel(event_det.intery);
event_det.intery(nany) = interp1(ty(~nany), event_det.intery(~nany), ty(nany));
event_det.intery = round(event_det.intery);

event_det.interangle = event_det.angle;
event_det.interangle(event_det.interangle==-1) = NaN;
nanangle = isnan(event_det.interangle);
tangle    = 1:numel(event_det.interangle);
event_det.interangle(nanangle) = interp1(tangle(~nanangle), event_det.interangle(~nanangle), tx(nanangle));
event_det.interangle = round(event_det.interangle);

event_det.interts = event_det.ts;
event_det.interts(event_det.interts==-1) = NaN;
nants = isnan(event_det.interts);
tts    = 1:numel(event_det.interts);
event_det.interts(nants) = interp1(tts(~nants), event_det.interts(~nants), tx(nants));
event_det.interts = round(event_det.interts);

figure(67678);
scatter3(event_det.interx,event_det.intery,event_det.interts,'.r');hold on
scatter3(event_gt.x,event_gt.y,event_gt.ts,'.b');

% evaluate position
beePixelSize = 50;
correct_position   = 0;
wrong_position     = 0;
testingPresentations = numel(event_gt.angle);
for idx = 130:2648
    if ((event_det.interx(idx) - event_gt.x(idx))^2 + (event_det.intery(idx) - event_gt.y(idx))^2)<beePixelSize^2
        correct_position = correct_position + 1;
    else
        wrong_position = wrong_position + 1;
    end
end
position_accuracy_percentage = (1 - wrong_position/testingPresentations)*100

% evaluate angle
correct_angle   = 0;
wrong_angle     = 0;
angleRange      = 90;
testingPresentations = numel(event_gt.angle);
for idx = 130:2648%numel(event_gt.angle)
   if event_det.interangle(idx) > event_gt.angle(idx)-angleRange && event_det.interangle(idx) < event_gt.angle(idx)+angleRange
       correct_angle = correct_angle + 1;
   else
       wrong_angle = wrong_angle + 1;
   end
end
angle_accuracy_percentage = (1 - wrong_angle/testingPresentations)*100
count_ground_truth = numel(find(event_gt.x~=-1))
%%
load('final_labels/20210803t1259d050m_cropped/WaggleNet/20210803t1259d050m_cropped_ground_truth_fixed.mat')
load('final_labels/20210803t1259d050m_cropped/WaggleNet/20210803t1259d050m_correctly_detected_waggle.mat')

figure(67089);
scatter3(correct_detected_track_coordinate.interx,1:numel(correct_detected_track_coordinate.interts),correct_detected_track_coordinate.intery,'.r');hold on;
scatter3(td_gt.x,1:numel(td_gt.x),td_gt.y,'.b');
xlabel("X (px)")
ylabel("# Frames")
zlabel("Y (px)")

figure(688799);
plot(td_gt.x)