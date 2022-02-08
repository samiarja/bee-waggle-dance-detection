clear
addpath("../DeepGreen/greenhouseCode")

%%%%%%%%%%%%%%%% LOAD VIDEO
videoName = "20210803t1727d200m";
videoFileName = "./input_videos/" + videoName + ".MP4";
limits.rowStart    = 401;    limits.rowEnd      = 1000;    limits.colStart    = 201;    limits.colEnd      = 1700;
v0 = VideoReader(videoName + ".MP4");
disp("Loading data...")

%%%%%%%%%%%%%%%% LOAD TEMPLATE
load('waggle16Templates_v1.mat')
disp("Template loaded...")

td                          = [];
tdWithInh                   = [];
jTd                         = [];
iWaggleInh                  = 0;
iWaggleEvent                = 0;
avgFrameDepth               = 6;
convMapThreshold            = 40;
nEventsForWaggleThreshold   = 10;
nDel                        = 18;
framesPerSegment            = 500;
nTemplate                   = size(waggleTemplate25,3);
nFrameTotal                 = round(v0.FrameRate *v0.Duration);
nSegment                    = ceil(nFrameTotal/framesPerSegment);
waggleLabellingROI          = [984 390 284 190];

segFrame = 0;
for iSegment = 1:5
    iSegment
    
    iFrame = 0;
    %startFrame = 1;%
    startFrame = max((iSegment-1)*framesPerSegment+1-nDel,1);
    endFrame = min(iSegment*framesPerSegment,nFrameTotal);%500;
    nFrame =  endFrame - startFrame;
    iFrameWithWaggle = 0;waggleStats = {};
    
    imageWidth  = 286;
    imageHeight = 192;
    
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
        
        frameInt = frameIntFull(limits.rowStart:limits.rowEnd,limits.colStart:limits.colEnd,:);
        frameInt = imcrop(frameInt,waggleLabellingROI);
        
        % downsample the data by two
        downsamplingFactor = 0.5;
        frameArray(:,:,:,iFrame) =  imresize(frameInt,downsamplingFactor);
        if iFrame>1
            dRgbFrameArray(:,:,:,iFrame) = single(frameArray(:,:,:,iFrame)) - single(frameArray(:,:,:,iFrame-1));
            dGreyScaleArray(:,:,iFrame) =  vecnorm(single(frameArray(:,:,:,iFrame)),2,3) -   vecnorm(single(frameArray(:,:,:,iFrame-1)),2,3);
        end
    end
    
    %%%%  3D convolution
    sigma        = 10;
    Tau          = 40;
    delArray     = 1:nDel;
    %waggleFilt   = exp(-delArray/Tau).*sin(4/(2*pi)*(delArray+6.5));  % slowestOne
    %waggleFilt   = exp(-delArray/Tau).*sin(6.5/(2*pi)*(delArray+4));  % too fast
    
    %%% make the waggle filter
    waggleFilt1   = exp(-delArray/Tau).*sin(5/(2*pi)*(delArray+5));%exp(-delArray/Tau).*sin(5.75/(2*pi)*(delArray+5.2));
    waggleFilt2   = exp(-delArray/Tau).*sin(4/(2*pi)*(delArray+5.2));
    waggleFilt3   = exp(-delArray/Tau).*sin(3.5/(2*pi)*(delArray+6.5));
    waggleFilt4   = exp(-delArray/Tau).*sin(3/(2*pi)*(delArray+7));%exp(-delArray/Tau).*sin(5.75/(2*pi)*(delArray+5.2));
    waggleFilt5   = exp(-delArray/Tau).*sin(2.5/(2*pi)*(delArray+8));
    waggleFilt6   = exp(-delArray/Tau).*sin(2/(2*pi)*(delArray+10));
    
    %%% reshape the waggle filter to 4D to suit FrameArray
    waggleFiltd1 = single(reshape(waggleFilt1,[1,1,1,nDel]));
    waggleFiltd2 = single(reshape(waggleFilt2,[1,1,1,nDel]));
    waggleFiltd3 = single(reshape(waggleFilt3,[1,1,1,nDel]));
    waggleFiltd4 = single(reshape(waggleFilt1,[1,1,1,nDel]));
    waggleFiltd5 = single(reshape(waggleFilt2,[1,1,1,nDel]));
    waggleFiltd6 = single(reshape(waggleFilt3,[1,1,1,nDel]));
    
    %%% convolve filter with frame differencing
    waggleMap1 = convn(dRgbFrameArray,waggleFiltd1,'full');waggleMap1 = waggleMap1(:,:,:,1:nFrame);
    waggleMap2 = convn(dRgbFrameArray,waggleFiltd2,'full');waggleMap2 = waggleMap2(:,:,:,1:nFrame);
    waggleMap3 = convn(dRgbFrameArray,waggleFiltd3,'full');waggleMap3 = waggleMap3(:,:,:,1:nFrame);
    waggleMap4 = convn(dRgbFrameArray,waggleFiltd1,'full');waggleMap1 = waggleMap1(:,:,:,1:nFrame);
    waggleMap5 = convn(dRgbFrameArray,waggleFiltd2,'full');waggleMap2 = waggleMap2(:,:,:,1:nFrame);
    waggleMap6 = convn(dRgbFrameArray,waggleFiltd3,'full');waggleMap3 = waggleMap3(:,:,:,1:nFrame);
    
    waggleMapMaxed =  zeros(round(imageHeight/2),round(imageWidth/2),nFrame,'single');
    maxMat = waggleMap1(:,:,:,1) +nan;
    
    for iFrame = 1:nFrame
        maxMat(:,:,1) = vecnorm(waggleMap1(:,:,:,iFrame),2,3);
        maxMat(:,:,2) = vecnorm(waggleMap2(:,:,:,iFrame),2,3);
        maxMat(:,:,3) = vecnorm(waggleMap3(:,:,:,iFrame),2,3);
        maxMat(:,:,4) = vecnorm(waggleMap4(:,:,:,iFrame),2,3);
        maxMat(:,:,5) = vecnorm(waggleMap5(:,:,:,iFrame),2,3);
        maxMat(:,:,6) = vecnorm(waggleMap6(:,:,:,iFrame),2,3);
        waggleMapMaxed(:,:,iFrame) = max( maxMat,[], 3);
        
        figure(517);
        subplot(3,3,1)
        imagesc(maxMat(:,:,1));axis image;colormap('gray');
        subplot(3,3,2)
        imagesc(maxMat(:,:,1));axis image;colormap('gray');
        subplot(3,3,3)
        imagesc(maxMat(:,:,1));axis image;colormap('gray');
        subplot(3,3,4)
        imagesc(maxMat(:,:,1));axis image;colormap('gray');
        subplot(3,3,5)
        imagesc(maxMat(:,:,1));axis image;colormap('gray');
        subplot(3,3,6)
        imagesc(maxMat(:,:,1));axis image;colormap('gray');
        subplot(3,3,7)
        imagesc(waggleMapMaxed(:,:,iFrame));axis image;colormap('gray');
        set(gcf,'Position',[100 100 1000 1000])        
    end
end