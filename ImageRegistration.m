% Specify the main folder containing multiple subfolders with video files
mainVideoFolder = 'E:\Lab\Side Eye project\Dataset exp\Pixel 3a glass\85dB floor\user-04';
outputMainFolder = 'E:\Lab\Side Eye project\Dataset exp\Pixel 3a glass\85dB floor\user-04\Processed';

% Get a list of all subfolders in the main folder
subfolders = dir(mainVideoFolder);
subfolders = subfolders([subfolders.isdir]); % Keep only directories
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Remove '.' and '..'

% Start parallel pool
parpool;

% Parameters for imregdemons
sigma = 3;
iterations = 50;  % Reduced iterations for speed

% Loop through each subfolder
for subfolderIdx = 1:length(subfolders)
    videoFolder = fullfile(mainVideoFolder, subfolders(subfolderIdx).name);
    outputFolder = fullfile(outputMainFolder, subfolders(subfolderIdx).name);
    
    % Create output folder if it doesn't exist
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    % Get a list of all MP4 files in the folder
    videoFiles = dir(fullfile(videoFolder, '*.mp4'));
    
    % Process each video file using parallel processing
    parfor fileIdx = 1:length(videoFiles)
        videoFile = fullfile(videoFolder, videoFiles(fileIdx).name);
        
        % Check if the output file already exists
        outputFileName = fullfile(outputFolder, [videoFiles(fileIdx).name(1:end-4), '.wav']);
        if isfile(outputFileName)
            disp(['Skipping existing file: ', outputFileName]);
            continue;
        end
        
        videoReader = VideoReader(videoFile);
        
        % Read the first frame as the reference image
        referenceFrame = im2gray(readFrame(videoReader));
        referenceFrame = referenceFrame.';  % Transpose the frame

        % Initialize displacement fields
        numFrames = videoReader.NumFrames;
        [M, N] = size(referenceFrame);
        n_g = round(2 * N / M);  % Number of groups (4 in this case)

        % Initialize arrays to store reduced displacement fields for X and Y
        ConcateXchannel = [];
        ConcateYchannel = [];

        % Process each frame and concatenate displacements group-wise
        while hasFrame(videoReader)
            % Read the current frame
            currentFrame = im2gray(readFrame(videoReader));
            currentFrame = currentFrame.';  % Transpose the frame

            % Register the current frame to the reference frame using imregdemons
            [x, ~] = imregdemons(currentFrame, referenceFrame, iterations, ...
                'AccumulatedFieldSmoothing', sigma, 'PyramidLevels', 2, ...
                'DisplayWaitBar', false);
            referenceFrame = currentFrame;

            xDisplacement = x(:,:,1);  % X displacement
            yDisplacement = x(:,:,2);  % Y displacement

            % Initialize arrays to store the current frame's displacement
            frameXchannel = zeros(M, n_g);
            frameYchannel = zeros(M, n_g);

            % Combined group-wise average for X and Y displacement
            for groupIdx = 1:n_g
                startCol = 1 + (groupIdx - 1) * (N / n_g);
                endCol = groupIdx * (N / n_g);
                
                % X displacement: Calculate the reduced displacement for the group
                reducedXDisplacement = mean(xDisplacement(:, startCol:endCol), 2);
                frameXchannel(:, groupIdx) = reducedXDisplacement;

                % Y displacement: Calculate the reduced displacement for the group
                reducedYDisplacement = mean(yDisplacement(:, startCol:endCol), 2);
                frameYchannel(:, groupIdx) = reducedYDisplacement;
            end

            % Concatenate the current frame's channels with previous frames
            ConcateXchannel = [ConcateXchannel; frameXchannel];
            ConcateYchannel = [ConcateYchannel; frameYchannel];
        end

        % Preparing the matrix to hold 8 channels
        numChannels = 8;  % Total number of channels for the 8-channel audio file
        audioMatrix = zeros(size(ConcateXchannel, 1), numChannels);

        % Normalizing and assigning X and Y displacement to channels 1-8
        for i = 1:4
            % Normalize X displacement
            max_val_X = max(abs(ConcateXchannel(:, i)));
            audioMatrix(:, i) = ConcateXchannel(:, i) / max_val_X;  % Normalize to [-1, 1]

            % Normalize Y displacement
            max_val_Y = max(abs(ConcateYchannel(:, i)));
            audioMatrix(:, i + 4) = ConcateYchannel(:, i) / max_val_Y;  % Normalize to [-1, 1]
        end

        % Write the combined audio to an 8-channel WAV file
        audiowrite(outputFileName, audioMatrix, 30000);
        disp(['8-channel audio file created: ', outputFileName]);
    end
end

% Close parallel pool
delete(gcp);
