clc
clear
close all

data_dir = 'data';
img_dir = 'images';
json_file = 'info.json';
csv_file = 'sensor.csv';
img_csv_file = 'images.csv';

%% create data directory if not present
if ~exist(data_dir, 'dir')
    mkdir(data_dir)
end

%% read bag files location
train_text = fopen('train_bags.txt','r');
test_text = fopen('test_bag.txt','r');
train_bags = textscan(train_text,'%s','delimiter','\n'); 
train_bags = train_bags{1};
test_bags = textscan(test_text,'%s','delimiter','\n'); 
test_bags = test_bags{1};
if(length(train_bags)<1)
    disp('Error! At Least one bag file must be selected for training')
    return
end
if(length(test_bags)~=1)
    disp('Error! Only one bag file must be selected for testing')
    return
end
% concate test bag to train bag list
train_bags{length(train_bags)+1} = test_bags{1};
%% loop over train(+test) bag files and extract
for index = 1:length(train_bags)
    bag_filepath = train_bags{index};
    [bag_dir,bag_filename,ext] = fileparts(bag_filepath);
    
    if ~isfile(bag_filepath)
        disp('***Error***')
        text = sprintf("Invalid path: %s", bag_filepath);
        disp(text);
        continue
    end
    
    
    output_dir = fullfile(data_dir,bag_filename);
    output_img_dir = fullfile(output_dir,img_dir);
    output_json = fullfile(output_dir,json_file);
    output_csv = fullfile(output_dir,csv_file);
    image_csv = fullfile(output_dir,img_csv_file);
    
    if isfile(output_json)
        %saved_info = jsondecode(fileread(output_json));
        disp('***Warning***')
        text = sprintf("%s already extracted. skipping %s", bag_filename ,bag_filepath);
        disp(text);
        text = sprintf("If you think that you should re-extract, delete the folder and re-run this script: %s", output_dir);
        disp(text);
        continue
    end
    
    if ~exist(output_dir,'dir')
        mkdir(output_dir);
    end
    
    if ~exist(output_img_dir,'dir')
        mkdir(output_img_dir);
    end
    
    bagInfo = rosbag('info',bag_filepath);
    timeOffset = bagInfo.Start.Time;
    
    bag = rosbag(bag_filepath);
    sensorValueCount = extractImuData(bag, output_csv, timeOffset);
    imageCount = saveAllImages(bag, output_img_dir, image_csv, timeOffset);
    
    info.filename = bag_filename; 
    info.fullpath = strrep(bag_filepath,'\','/'); 
    info.timeOffset = timeOffset;
    info.sensorValues = sensorValueCount;
    info.images = imageCount;
    info.imageDir = strrep(output_img_dir,'\','/');
	saveJSONfile(info, output_json);
    
    text = sprintf("Extracted %d IMU Sendor Data and %d images for %s",...
        sensorValueCount, imageCount, bag_filename);
    disp(text);
end

%% 
function count = extractImuData(bag, output_csv, timeOffset)
    count = extractTypeImu(bag, '/mavros/imu/data', output_csv, timeOffset);
end

function count = extractTypeImu(bag, topic, output_csv, timeOffset)
    bagImuInfos = select(bag, 'Topic', topic);
    allMsgsPropTable = bagImuInfos.MessageList;
    allMsgsDataTable = readMessages(bagImuInfos);

    count = bagImuInfos.NumMessages;
    angVelocityX = zeros(1,count); angVelocityY = zeros(1,count); angVelocityZ = zeros(1,count);
    linAccX = zeros(1,count); linAccY = zeros(1,count); linAccZ = zeros(1,count);
    orientationX = zeros(1,count); orientationY = zeros(1,count); orientationZ = zeros(1,count);
    orientationW = zeros(1,count);
    time = zeros(1, count);
    for i = 1: count
        msgPropTable = allMsgsPropTable(i,:);
        msgData = allMsgsDataTable{i,:};
        sampleMessage = MyMsgObj(msgPropTable, msgData);

        p = sampleMessage.Time ;
        time(i) = p -timeOffset;
        angVelocityX(i) = sampleMessage.Data.AngularVelocity.X;
        angVelocityY(i) = sampleMessage.Data.AngularVelocity.Y;
        angVelocityZ(i) = sampleMessage.Data.AngularVelocity.Z;
        linAccX(i) = sampleMessage.Data.LinearAcceleration.X;
        linAccY(i) = sampleMessage.Data.LinearAcceleration.Y;
        linAccZ(i) = sampleMessage.Data.LinearAcceleration.Z;
        orientationX(i) = sampleMessage.Data.Orientation.X;
        orientationY(i) = sampleMessage.Data.Orientation.Y;
        orientationZ(i) = sampleMessage.Data.Orientation.Z;
        orientationW(i) = sampleMessage.Data.Orientation.W;
    end
    cHeader = {'time' ...
        'angVelocityX' 'angVelocityY' 'angVelocityZ'...
        'linAccX' 'linAccY' 'linAccZ' ...
        'orientationX' 'orientationY' 'orientationZ' 'orientationW'...
        }; %dummy header
    textHeader = strjoin(cHeader, ',');

    %write header to file
    fid = fopen(output_csv,'w');
    fprintf(fid,'%s\n',textHeader);
    fclose(fid);
    %write data to end of file
    time = time';angVelocityX = angVelocityX';angVelocityY = angVelocityY';angVelocityZ=angVelocityZ';
    linAccX = linAccX';linAccY = linAccY';linAccZ = linAccZ';
    orientationX = orientationX';orientationY=orientationY';orientationZ=orientationZ';
    orientationW=orientationW';
    data = [time,angVelocityX,angVelocityY,angVelocityZ,linAccX,linAccY,linAccZ,orientationX,orientationY,orientationZ,orientationW];
    dlmwrite(output_csv,data,'-append');
    %csvwrite('sensor.csv' ,data);
end

function count = saveAllImages(bag, output_img_dir, image_csv, timeOffset)
    bagImages = select(bag, 'Topic', '/pylon_camera_node/image_raw');
    mList = bagImages.MessageList; %table rows: time, topic, messageType, fileoffset
    count = bagImages.NumMessages;
    allMsgs = readMessages(bagImages);
    cells = cell(count,2);
    for i = 1: count
        singleMessage = allMsgs{i,1};
        img = readImage(singleMessage);
        outputFileName = char(fullfile(output_img_dir,sprintf("%d.jpg" ,i)));
        imwrite(img, outputFileName);
        p = mList{i,1} - timeOffset;
        cells{i,1} = p;
        cells{i,2} = outputFileName;
    end
    table = cell2table(cells, 'VariableNames',{'time' 'image_path'});
    writetable(table,image_csv,'Delimiter',',','QuoteStrings',true)
end
