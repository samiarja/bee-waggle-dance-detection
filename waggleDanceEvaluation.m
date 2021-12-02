%% evaluation
videoIndex = "20210803t1727d200m";
load("./data/" + videoIndex + "_labels.mat") % load ground truth
load("./data/" + videoIndex + ".MP4_tds.mat") % load predicted labels

objectEventIndex = 0;objectTd = [];valid = 0;
x_coor = real(userInputArray(:,1));x_coor(x_coor==0) = NaN;
y_coor = -real(1i*userInputArray(:,1));y_coor(y_coor==0) = NaN;
timeStamp = 1:numel(userInputArray);

figure(56757);
subplot(2,2,1)
scatter3(td.x,td.y,td.ts,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
title("Detected waggle dance");
subplot(2,2,2)
scatter3(x_coor,y_coor,timeStamp,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
title("Detected waggle ground truth");

v0 = VideoReader(videoIndex + ".MP4");
nFrameTotal = round(v0.FrameRate *v0.Duration);
outputLabel = nan(numel(y_coor),1);

for iFrame = 1:nFrameTotal
    valid = valid + 1;
    
    if iFrame < numel(td.y) && iFrame < numel(y_coor)
        if ((td.x(iFrame) - x_coor(iFrame))^2 + (td.y(iFrame) - y_coor(iFrame))^2)<userInputRadiusArray(iFrame)^2
            objectEventIndex = objectEventIndex +1;
            objectTd.x(objectEventIndex)     = td.x(iFrame);
            objectTd.y(objectEventIndex)     = td.y(iFrame);
            objectTd.ts(objectEventIndex)    = td.ts(iFrame);
            outputLabel(valid) = 1;
        else
            outputLabel(valid) = 0;
        end
    end
end

Accuracy = 1 - numel(find(outputLabel==0))/numel(td.x);
% Accuracy = numel(find(outputLabel==1))/numel(objectTd.x);

subplot(2,2,[3 4])
scatter3(objectTd.x,objectTd.y,objectTd.ts,'.r');
xlabel("X [px]");
ylabel("Y [px]");
zlabel("Time [frame]");
title("Correctly detected waggle dance \color{magenta}[Accuracy: " + num2str(Accuracy) + "%]");