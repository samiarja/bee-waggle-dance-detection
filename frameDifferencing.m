addpath("/media/sami/Samsung_T5/MPhil/Code/DeepGreen/greenhouseCode")
tic;
%if ~exist('FULL_IMAGE')
clear

td                          = [];
tdWithInh                   = [];
jTd                         = [];
FULL_IMAGE                  = 0;
iWaggleInh                  = 0;
iWaggleEvent                = 0;
SHOW                        = 0;%rand>5;
convMapThreshold            = 140;
nEventsForWaggleThreshold   = 10;
nDel                        = 18;
framesPerSegment            = 500;
RECORD_VIDEO                = SHOW && 0;


load('waggle16Templates_v1.mat')
disp("Template loaded...")
nTemplate = size(waggleTemplate25,3);
if SHOW
    [nSubplotRows,nSubplotCols] = goodSubPlotRowCols(nTemplate);
    % % % %% Show waggleTemplates
    for iTemplate = 1:nTemplate
        
        figure(1);
        subplot(nSubplotRows,nSubplotCols,iTemplate);imagesc(waggleTemplate25(:,:,iTemplate)); axis image; colorbar;
        set(gcf, 'Name', 'Waggle Dance Spatial Template')
    end
end
%v0 = VideoReader('E:\googleDrive\MATLAB\ML\data\bees\waggleDance\darwinExperiments\sixHiveExp\Honeybees01\20210803t1243d050m.MP4');
videoFileName = './data/20210803t1727d200m.MP4';
limits.rowStart    = 401;    limits.rowEnd      = 1000;    limits.colStart    = 201;    limits.colEnd      = 1700;

v0 = VideoReader(videoFileName);
%rowStart    = 1;    rowEnd      = 100;    colStart    = 1;    colEnd      = 100;
disp("Video data loaded...")

if SHOW
    figNum = 536451;
    figure(figNum);clf;
end

if RECORD_VIDEO
    set(gcf,'color','k')
    globalFrameRate                 = 60;
    vid1           = VideoWriter('beeWaggle4Panels_v2.avi');
    vid1.FrameRate = globalFrameRate; %// Change the desired frame rate here.
    vid1.Quality   = 95;
    open(vid1);
end

nFrameTotal = round(v0.FrameRate *v0.Duration);
nSegment = ceil(nFrameTotal/framesPerSegment);
%nDel         = 36;

for iSegment = 1:5
    iSegment
    %startFrame = 1;%
    startFrame = max((iSegment-1)*framesPerSegment+1-nDel,1);
    
    %endFrame = nFrameTotal;%
    endFrame = min(iSegment*framesPerSegment,nFrameTotal);%500;
    
    nFrame =  endFrame -   startFrame ;
    
    avgFrameDepth = 6;
    iFrameWithWaggle = 0;waggleStats = {};
    if FULL_IMAGE
        imageWidth  = v0.Width;
        imageHeight = v0.Height;
    else
        
        imageWidth  = limits.colEnd - limits.colStart +1;
        imageHeight = limits.rowEnd - limits.rowStart +1;
    end
    
    frameArray     =  zeros(imageHeight/2,imageWidth/2,3,nFrame,'uint8');
    dRgbFrameArray =  zeros(imageHeight/2,imageWidth/2,3,nFrame,'single');
    dGreyScaleArray    = zeros(imageHeight/4,imageWidth/4,nFrame,'single');
    
    iFrame = 0;
    for iFrameOriginal = startFrame:  endFrame
        iFrame = iFrame + 1;
        frameIntFull = read(v0,iFrameOriginal);
        if FULL_IMAGE
            frameInt = frameIntFull;
        else
            frameInt = frameIntFull(limits.rowStart:limits.rowEnd,limits.colStart:limits.colEnd,:);
        end
        % %                 figure(53451);
        % %                 imagesc(frameIntFull);axis image;
        % %                 figure(53452);
        % %                 imagesc(imresize(frameIntFull,.5));axis image;
        % %                 figure(53453);
        % %                 imagesc(imresize(frameIntFull,.25));axis image;
        % %                 figure(53454);
        % %                 imagesc(imresize(frameIntFull,1/8));axis image;
        % %
        % %         imagesc(frameInt);axis image;
        
        % downsample the data by two
        frameArray(:,:,:,iFrame) =  imresize(frameInt,0.5);
        if iFrame>1
            dRgbFrameArray(:,:,:,iFrame) = single(frameArray(:,:,:,iFrame)) - single(frameArray(:,:,:,iFrame-1));
            dGreyScaleArray(:,:,iFrame)  =     imresize(vecnorm(single(frameArray(:,:,:,iFrame)),2,3) -   vecnorm(single(frameArray(:,:,:,iFrame-1)),2,3),0.5);
        end
    end
end
toc;
%%
for idx = 1:size(dRgbFrameArray,4)
    figure(567);
    imagesc(dRgbFrameArray(:,:,:,idx));
    pause(0.1)
end
