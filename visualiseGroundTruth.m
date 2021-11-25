%% label data
run beeWaggleLabeller.m

%% visualise labels
load('./data/20210803t1727d200m_labels.mat')
iFrame = 1; objectTd = []; objectEventIndex = 0;
if ~isempty(userInputArray)
    [nFrame, nObj]      = size(userInputArray);
end

for iFrame = 1:nFrame
    while iFrame < nFrame
        for iObj = 1:nObj
            objx(iObj) = real(userInputArray(iFrame,iObj));
            objy(iObj) = -real(1i*userInputArray(iFrame,iObj));
            objRadSquared(iObj) = userInputRadiusArray(iFrame,iObj);
        end
        iFrame = iFrame + 1;
        [objx; objy; objRadSquared;]
        
        for iObj = 1:nObj
            [objx; objy; objRadSquared;]
            objectEventIndex = objectEventIndex +1;
            objectTd.x(objectEventIndex) = objx();
            objectTd.y(objectEventIndex) = objy;
            objectTd.ts(objectEventIndex) = iFrame;
            objectTd.radius(objectEventIndex) = objRadSquared;
            objectTd.objID(objectEventIndex) = iObj;
        end
    end
end