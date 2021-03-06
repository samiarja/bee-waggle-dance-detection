function VideoInCustomGUIExample()

%%
% Initialize the video reader.
% only modify framerate and waggleLabellingROI
downsampling = 1;
iFrame  = 1;
iObject = 1;
framerate = 20;
waggleLabellingROI = [1200 800 300 300]; % [0 0 1920 1080]
userInputArray = zeros(500,1);
userInputRadiusArray = zeros(500,1);
videoName = "20210803t1727d200m.MP4";
videoFileName = "./data/" + videoName;
% videoSrc = VideoReader(videoFileName);
videoSrc = vision.VideoFileReader(videoName,'ImageColorSpace','RGB','VideoOutputDataType','uint8');
% videoSrc1 = VideoReader(videoFileName);
%%
% Create a figure window and two axes to display the input video and the
% processed video.
[hFig, hAxes] = createFigureAndAxes();

%%
% Add buttons to control video playback.
insertButtons(hFig, hAxes, videoSrc);

%% Interact with the New User Interface
% Now that the GUI is constructed, we can press the play button to trigger
% the main video processing loop defined in the |getAndProcessFrame| function
% listed below.

% Initialize the display with the first frame of the video
frame = getAndProcessFrame(videoSrc);
% Display input video frame on axis
% showFrameOnAxis(hAxes.axis1, frame);
a = size(frame,1);
% showFrameOnAxis(hAxes.axis2, zeros(a,b,3,'uint8'));

%%
% Note that each video frame is centered in the axis box. If the axis size
% is bigger than the frame size, video frame borders are padded with
% background color. If axis size is smaller than the frame size scroll bars
% are added.

%% Create Figure, Axes, Titles
% Create a figure window and two axes with titles to display two videos.
    function [hFig, hAxes] = createFigureAndAxes()
        
        % Close figure opened by last run
        figTag = 'CVST_VideoOnAxis_9804532';
        close(findobj('tag',figTag));
        
        % Create new figure
        hFig = figure('numbertitle', 'off', ...
            'name', 'Bee Waggle Dance Labeller', ...
            'menubar','none', ...
            'toolbar','none', ...
            'resize', 'on', ...
            'tag',figTag, ...
            'renderer','painters', ...
            'position',[50 150 1500/downsampling 700/downsampling],... % [X Y W H]
            'HandleVisibility','callback'); % hide the handle to prevent unintended modifications of our custom UI
        
        % Create axes and titles
        hAxes.axis1 = createPanelAxisTitle(hFig,[0.1 0.17 0.8 0.7],'Bee Waggle Dance'); % [X Y W H]
    end

%% Create Axis and Title
% Axis is created on uipanel container object. This allows more control
% over the layout of the GUI. Video title is created using uicontrol.
    function hAxis = createPanelAxisTitle(hFig, pos, axisTitle)
        
        % Create panel
        hPanel = uipanel('parent',hFig,'Position',pos,'Units','Normalized');
        
        % Create axis
        hAxis = axes('position',[0 0 1 1],'Parent',hPanel);
        hAxis.XTick = [];
        hAxis.YTick = [];
        hAxis.XColor = [1 1 1];
        hAxis.YColor = [1 1 1];
        % Set video title using uicontrol. uicontrol is used so that text
        % can be positioned in the context of the figure, not the axis.
        titlePos = [pos(1)+0.2 pos(2)+pos(3)+4 1 1];
        uicontrol('style','text',...
            'String', axisTitle,...
            'Units','Normalized',...
            'Parent',hFig,'Position', titlePos,...
            'BackgroundColor',hFig.Color);
    end

%% Insert Buttons
% Insert buttons to play, pause the videos.
    function insertButtons(hFig,hAxes,videoSrc)
        
        % Start labelling button
        uicontrol(hFig,'unit','pixel','style','pushbutton','string','Start labelling',...
            'position',[150 80 150 25],'callback', ...
            {@labelCallback,videoSrc,hFig});
        
        % Next bee button
        uicontrol(hFig,'unit','pixel','style','pushbutton','string','Previous Bee',...
            'position',[320 80 100 25],'callback', ...
            {@previousbeeCallback,videoSrc,hFig});
        
        % previous bee button
        uicontrol(hFig,'unit','pixel','style','pushbutton','string','Next Bee',...
            'position',[440 80 100 25],'callback', ...
            {@beeCallback,videoSrc,hFig});
        
        % Play button with text Start/Pause/Continue
        uicontrol(hFig,'unit','pixel','style','pushbutton','string','Play',...
            'position',[580 80 75 25], 'tag','PBButton123','callback',...
            {@playCallback,videoSrc,hAxes});
        
        % Save labels button
        uicontrol(hFig,'unit','pixel','style','pushbutton','string','Save labels',...
            'position',[680 80 100 25],'callback', ...
            {@savelabelCallback,videoSrc,hFig});
        
        % Exit button with text Exit
        uicontrol(hFig,'unit','pixel','style','pushbutton','string','Exit',...
            'position',[800 80 50 25],'callback', ...
            {@exitCallback,videoSrc,hFig});
    end

%% Next bee button callback
    function beeCallback(hObject,~,videoSrc,hFig)
        try
            iObject = iObject + 1;
        catch ME
            % Re-throw error message if it is not related to invalid handle
            if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
                rethrow(ME);
            end
        end
    end

%% Save labels button callback
    function savelabelCallback(hObject,~,videoSrc,hFig)
        try
            save("data/20210803t1727d200m_labels","userInputArray","userInputRadiusArray");
            disp("Labels saved...")
        catch ME
            % Re-throw error message if it is not related to invalid handle
            if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
                rethrow(ME);
            end
        end
    end
%% Previous bee button callback
    function previousbeeCallback(hObject,~,videoSrc,hFig)
        try
            if iObject > 0
                iObject = iObject - 1;
            end
        catch ME
            % Re-throw error message if it is not related to invalid handle
            if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
                rethrow(ME);
            end
        end
    end
%% Start Labelling button callback
    function labelCallback(hObject,~,videoSrc,hFig)
        try
            [frame1] = getAndProcessFrame(videoSrc);
            showFrameOnAxis(hAxes.axis1, imresize(imcrop(frame1,waggleLabellingROI),1/downsampling));
            hObject.String = 'Labelling Started...';
            while strcmp(hObject.String, 'Labelling Started...') && ~isDone(videoSrc)
                [y,x,button] = ginput(1);
                if button == 1 || button == 2  % mouse clicks
                    if x<1080 && x>0 && y<1920 && y>0
                        if button == 2
                            [y2,x2] = ginput(1);
                            % use x2 and y2 as well as x and y to find the radius of this
                            % object
                            thisObjectsRadiusNow = sqrt((x2-x)^2+(y2-y)^2);
                            framerateinc = 0;
                            while framerateinc < framerate
                                framerateinc = framerateinc + 1;
                                [frame1] = getAndProcessFrame(videoSrc);
                                showFrameOnAxis(hAxes.axis1, imresize(imcrop(frame1,waggleLabellingROI),1/downsampling));
                                userInputArray(iFrame,iObject) = x + y*1i;
                                userInputRadiusArray(iFrame,iObject) = thisObjectsRadiusNow;
                                iFrame  = iFrame + 1;
                            end
                        end           
                        [x y thisObjectsRadiusNow]
                        
                    end
                end
            end
        catch ME
            % Re-throw error message if it is not related to invalid handle
            if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
                rethrow(ME);
            end
        end
    end

%% Play Button Callback
% This callback function rotates input video frame and displays original
% input video frame and rotated frame on axes. The function
% |showFrameOnAxis| is responsible for displaying a frame of the video on
% user-defined axis. This function is defined in the file
% <matlab:edit(fullfile(matlabroot,'examples','vision','main','showFrameOnAxis.m')) showFrameOnAxis.m>
    function playCallback(hObject,~,videoSrc, hAxes)
        try
            % Check the status of play button
            isTextStart = strcmp(hObject.String,'Start');
            isTextCont  = strcmp(hObject.String,'Continue');
            if isTextStart
                % Two cases: (1) starting first time, or (2) restarting
                % Start from first frame
                if isDone(videoSrc)
                    reset(videoSrc);
                end
            end
            if (isTextStart || isTextCont)
                hObject.String = 'Pause';
            else
                hObject.String = 'Continue';
            end
            
            
            while strcmp(hObject.String, 'Pause') && ~isDone(videoSrc)
                % Get input video frame and rotated frame
                [frame1] = getAndProcessFrame(videoSrc);
                % Display input video frame on axis
                showFrameOnAxis(hAxes.axis1, imresize(imcrop(frame1,waggleLabellingROI),1/downsampling));
                iFrame  = iFrame + 1;
                % Display rotated video frame on axis
                %                 showFrameOnAxis(hAxes.axis2, frame2);
            end
            
            % When video reaches the end of file, display "Start" on the
            % play button.
            if isDone(videoSrc)
                hObject.String = 'Start';
            end
        catch ME
            % Re-throw error message if it is not related to invalid handle
            if ~strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
                rethrow(ME);
            end
        end
    end
%% Video Processing Algorithm
% This function defines the main algorithm that is invoked when play button
% is activated.
    function [frame1] = getAndProcessFrame(videoSrc)
        
        % Read input video frame
        frame1 = step(videoSrc);
%         frame1 = padarray(frame1, [100 100], 1, 'both');
        %         frame2 = padarray(frame2, [30 30], 0, 'both');
        
    end

%% Exit Button Callback
% This callback function releases system objects and closes figure window.
    function exitCallback(~,~,videoSrc,hFig)
        
        % Close the video file
        release(videoSrc);
        
        % Close the figure window
        close(hFig);
    end
end