% PATH = "final_labels/20210803t1727d200m_cropped/"; % no linear interpolation
% PATH = "final_labels/20210803t1259d050m_cropped/"; % no linear interpolation
% PATH = "final_labels/20210803t1508d100m_cropped/";   % with linear interpolation
% PATH = "final_labels/20210803t1719d200m_cropped/";   % with linear interpolation
PATH = "final_labels/20210803t1732d200m_cropped_W1/";   % with linear interpolation
% PATH = "DenseObjectAnnotation/static/"; % with linear interpolation

INTERPOLATE = 1;
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
td_gt.frameID = (firstFrameLabel:firstFrameLabel+frameIDNAN(1)-2)';

% for k=1:numel(td_gt.x)
%     FrameNames=frames(firstFrameLabel+k).name;
%     hf = figure(56);
%     imshow(PATH + "png/" + FrameNames);axis on; hold on
%     rectangle('Position',[td_gt.x(k)-5 td_gt.y(k)-5 radius radius],'Curvature',[1 1],'FaceColor',[1 0 0],'EdgeColor','k',...
%         'LineWidth',2);axis equal
%     drawnow
%     F = getframe(gcf);
%     writeVideo(writerObj, F);
% end
% close(writerObj);
% fprintf('Sucessfully generated the video\n')

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
