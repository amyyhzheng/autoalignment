 % % % B.Madruga script to generate TIFFs more effeciently (I can't do this no more)
% % % Notes:
% % % v0.0.1
% % % The code relies on reading the .ndv file generated from the microscope. It will
% % % grab that file and calculate the various parameters for the images based on that
% % % such as image dimensions and so on. If you want to prevent unmixing or do bi-di 
% % % phase correction, those parameters are set in the first few lines of the script. 
                                                                    
clear all, close all 
Data = struct(); % Create main data structure 
Data.bidi_correction = false; % Set this to true if you want to do bidi - set pixels in first if statement 
Data.unmix = true; % Set to true if you want to unmix


if Data.bidi_correction == true
    Data.bidi = 54; % Number of pixels to shift, bidirectional phase
else
    Data.bidi = 0; % Or just don't do it 
end

if Data.unmix == true
        Data.m = [0.622325,0.199198,0.06555; % This is from Joe - (new scope, caculated in Feb-March 2022 and added to the code 3/21/26)
                  0.34619,0.707837,0.1612;
                  0.031483,0.092964,0.77324];
else
    Data.m = eye(3); % Eyedentity matrix - basically don't do anything 
end

%% Configure 
[ndvFile, path] = uigetfile({'*.ndv', 'Pls Select the .ndv File (*.ndv)'}, 'Select ndv file');
if isequal(ndvFile,0)
    error('File selection cancelled by user.');
end

tic
ndvFullPath = fullfile(path, ndvFile); % Combine directory and file name into a full path
dataDir = path; % Use the same folder for the .I16 files
fprintf('Selected NDV file: %s\n', ndvFullPath);
fprintf('Data directory: %s\n', dataDir);

% Get info from .NDV
txt = fileread(ndvFullPath);
Data.nx = str2double(regexp(txt, 'ResolutionX\s*=\s*(\d+)', 'tokens', 'once'));
Data.ny = str2double(regexp(txt, 'ResolutionY\s*=\s*(\d+)', 'tokens', 'once'));

if isempty(Data.nx) || isempty(Data.ny)
    error('Could not read ResolutionX/ResolutionY from NDV header.'); % Freak out if .NDV is broken
end

pixelsPerPlane = Data.nx * Data.ny; % This is used for channel no. calculation

% List .I16 files
I16files = dir(fullfile(dataDir, '*.I16'));
if isempty(I16files)
    error('No .I16 files found in directory: %s', dataDir);
end

% Extract numeric indices: basically get the I16 files in the right order 
% This was a bit of a faf but 
N = numel(I16files);
frameIndex = nan(N,1);


for k = 1:N
    name = I16files(k).name;
    token = regexp(name, '(?:File|Frame)\s*[_-]?\s*(\d+)', 'tokens', 'once', 'ignorecase');
    if ~isempty(token)
        frameIndex(k) = str2double(token{1});
        continue;
    end
    token = regexp(name, '(?:File|Frame).*?(\d+)', 'tokens', 'once', 'ignorecase');
    if ~isempty(token)
        frameIndex(k) = str2double(token{1});
        continue;
    end
    tokens = regexp(name, 'Z[_\s-]*(\d+)', 'tokens', 'ignorecase');
    if ~isempty(tokens)
        token = tokens{end};  % pick the last Z-number match
        frameIndex(k) = str2double(token{1});
        continue;
    end
    tokensAll = regexp(name, '(\d+)', 'tokens');
    if ~isempty(tokensAll)
        frameIndex(k) = str2double(tokensAll{end}{1});
    else
        frameIndex(k) = NaN;  % no numeric token found
    end
end

% Place any NaNs at the end by mapping them to +Inf for sorting
sortKeys = frameIndex;
sortKeys(isnan(sortKeys)) = Inf;

% Sort numerically by the extracted index
[~, order] = sort(sortKeys, 'ascend');
I16files = I16files(order);
frameIndex = frameIndex(order);  % sorted numeric indices (NaN remain NaN)

nFrames = numel(I16files);

% Detect number of channels using the first file (after sorting)
firstFile = fullfile(dataDir, I16files(1).name);
fid = fopen(firstFile, 'r');
if fid < 0, error('Cannot open %s', firstFile); end
rawTest = fread(fid, inf, 'int16=>int16');
fclose(fid);

numChannels = numel(rawTest) / pixelsPerPlane;
if mod(numChannels,1) ~= 0
    warning('First file size not integer multiple of pixelsPerPlane; truncating channels to floor.');
    numChannels = floor(numChannels);
end
Data.numChannels = numChannels;
fprintf('Detected %d channels per frame.\n', numChannels);

% Preallocate Data.chX stacks
for ch = 1:numChannels
    Data.(['ch' num2str(ch)]) = zeros(Data.ny, Data.nx, nFrames, 'int16');
end

% Load all frames in sorted order
for f = 1:nFrames
    fname = fullfile(dataDir, I16files(f).name);
    fid = fopen(fname,'r');
    if fid < 0
        warning('Skipping file (cannot open): %s', fname);
        continue;
    end
    raw = fread(fid, inf, 'int16=>int16');
    fclose(fid);

    expectedSize = pixelsPerPlane * numChannels;
    if numel(raw) < expectedSize
        warning('File %s shorter than expected; padding with zeros.', I16files(f).name);
        raw(end+1:expectedSize) = 0;
    elseif numel(raw) > expectedSize
        raw = raw(1:expectedSize);
    end

    raw = reshape(raw, Data.ny, Data.nx, numChannels);  % note: ny first, then nx
    for ch = 1:numChannels
        Data.(['ch' num2str(ch)])(:,:,f) = raw(:,:,ch);  % already [ny x nx]
    end

    % progress print every 100 frames
    if mod(f,100) == 0 || f == nFrames
        fprintf('Loaded frame %d / %d\n', f, nFrames);
    end
end

fprintf('Done Loading %d frames, %d channels (%dx%d)\n', nFrames, numChannels, Data.nx, Data.ny);

%% Now process data 

% Bidi Correction for all channels 
if Data.bidi_correction == true
    Data.ch1(2:2:end, :) = circshift(Data.ch1(2:2:end, :), [0, Data.bidi]); % correct phase 
    Data.ch2(2:2:end, :) = circshift(Data.ch2(2:2:end, :), [0, Data.bidi]); % correct phase 
    Data.ch3(2:2:end, :) = circshift(Data.ch3(2:2:end, :), [0, Data.bidi]); % correct phase 
end

% Unmixing, Lets work on some unmixing to replicate what is being done currently 
fprintf('Now Unmixing %d frames, %d channels (%dx%d)\n', nFrames, numChannels, Data.nx, Data.ny);

for frames = 1:size(Data.ch1,3) % For every frame in the stack 
    for k = 1:Data.ny %for each column in a single frame 

        um = Data.m \ [double(Data.ch1(k,:,frames));
                       double(Data.ch2(k,:,frames));
                       double(Data.ch3(k,:,frames))]; % lifted from existing code

        Data.ch1_unmixed(k,:,frames) = um(1,:);
        Data.ch2_unmixed(k,:,frames) = um(2,:);
        Data.ch3_unmixed(k,:,frames) = um(3,:);

    end 

        if mod(frames,25) == 0 || frames == size(Data.ch1,3)
            if Data.unmix == true
                fprintf('Unmixed frame %d / %d\n', frames, size(Data.ch1,3));
            else
                fprintf('Frame Not Unmixed %d / %d\n', frames, size(Data.ch1,3));
            end
        end

end
 
if Data.unmix == true % Set to true if you want to unmix
    disp(['Linearly Unmixing Completed in ',num2str(toc), ' Seconds - Now Saving']);
else
    disp(['Data not Unmixed, in ',num2str(toc), ' Seconds - Now Saving'])
end

% Saving
Data.filename = ndvFile(1:end-4); 
Data.output.ch1.unmixed_path = fullfile(path, strcat(ndvFile(1:end-4),'_Ch1_unmixed.tif'));
Data.output.ch2.unmixed_path = fullfile(path, strcat(ndvFile(1:end-4),'_Ch2_unmixed.tif'));
Data.output.ch3.unmixed_path = fullfile(path, strcat(ndvFile(1:end-4),'_Ch3_unmixed.tif'));

Data.output.ch1.unmixed = Tiff(Data.output.ch1.unmixed_path, 'w');
Data.output.ch2.unmixed = Tiff(Data.output.ch2.unmixed_path, 'w');
Data.output.ch3.unmixed = Tiff(Data.output.ch3.unmixed_path, 'w');
    
% Set common tag values (for grayscale images)
tagstruct.ImageLength = Data.ny;
tagstruct.ImageWidth = Data.nx;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 16;             
tagstruct.SamplesPerPixel = 1;
tagstruct.RowsPerStrip = Data.ny;    
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Compression = Tiff.Compression.None;
tagstruct.Software = 'MATLAB';
    
% Loop through slices
for k = 1:size(Data.ch1, 3)
    Data.output.ch1.unmixed.setTag(tagstruct);
    Data.output.ch2.unmixed.setTag(tagstruct);
    Data.output.ch3.unmixed.setTag(tagstruct);

    % Set write mode of each of the Tiff objects  
    Data.output.ch1.unmixed.write(uint16(Data.ch1_unmixed(:,:,k))); % Saves the unmixed data (CH1)
    Data.output.ch2.unmixed.write(uint16(Data.ch2_unmixed(:,:,k))); % Saves the unmixed data (CH2)
    Data.output.ch3.unmixed.write(uint16(Data.ch3_unmixed(:,:,k))); % Saves the unmixed data (CH3)

    if k < size(Data.ch1, 3) 
        Data.output.ch1.unmixed.writeDirectory(); 
        Data.output.ch2.unmixed.writeDirectory(); 
        Data.output.ch3.unmixed.writeDirectory(); 

    end
end

% Close the Tiff file object 
Data.output.ch1.unmixed.close();
Data.output.ch2.unmixed.close();
Data.output.ch3.unmixed.close();

if Data.unmix == true % Set to true if you want to unmix
    disp(['Raw data formatted into frames, linearly unmixed and saved in ',num2str(toc), ' seconds']);
else
    disp(['Raw data formatted into frames, NOT linearly unmixed and saved in ',num2str(toc), ' seconds'])
end
disp('Ready for next file!')

clear all

% Note: looks like the data is mirrored about the horizontal axis (i.e. the
% bottom of the frame is displayed at the top, etc.) 