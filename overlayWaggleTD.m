for iSegment = 1:5
    startFrame = max((iSegment-1)*framesPerSegment+1-nDel,1);
    endFrame = min(iSegment*framesPerSegment,nFrameTotal);%500;
    nFrame =  endFrame -   startFrame ;
    frameIntFull = read(v0,iFrameOriginal);
    frameArray(:,:,:,iFrame) =  imresize(frameInt,0.5);
    for iFrame = 1:nFrame
        figure(567);
        imagesc(uint8(frameArray(:,:,:,iFrame)));axis image;hold on
%         scatter3(td.x(iFrame),td.y(iFrame),td.ts(iFrame),'.r')
        imagesc(uint8(td.context{1, iFrame}(:,:,1)));axis image;
        drawnow
    end
end