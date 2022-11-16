clc
clear all
close all

dir = 'C:\Users\shafi\Downloads\Sp Cup 2020 Resources\Sp ar koto\codes\sp-cup-2020\'

%% Normal file
sensors_combined = []
for i=2:4
    
    sensorFileN = strcat(dir, 'Normal File Scores\N' , num2str(i), '_sensor_scores.txt')
    sensorFileA = strcat(dir, 'Abnormal File Scores\A' , num2str(i), '_sensor_scores.txt')
    sensorIDN = fopen(sensorFileN,'r');
    sensorIDA = fopen(sensorFileA,'r');
    formatSpec = '%f';
    sensor_newn = fscanf(sensorIDN,formatSpec);
    sensor_newa = fscanf(sensorIDA,formatSpec);
    
    if i==2
        sensors_combinedN = sensor_newn;
        axis_holderN = ones(size(sensor_newn));
        sensors_combinedA = sensor_newa;
        axis_holderA = 4*ones(size(sensor_newa));
    elseif i==3
        sensors_combinedN = vertcat(sensors_combinedN, sensor_newn);
        axis_holderN = vertcat(axis_holderN, 2*ones(size(sensor_newn)));
        sensors_combinedA = vertcat(sensors_combinedA, sensor_newa);
        axis_holderA = vertcat(axis_holderA, 5*ones(size(sensor_newa)));
    else
        sensors_combinedN = vertcat(sensors_combinedN, sensor_newn);
        axis_holderN = vertcat(axis_holderN, 3*ones(size(sensor_newn)));
        sensors_combinedA = vertcat(sensors_combinedA, sensor_newa);
        axis_holderA = vertcat(axis_holderA, 6*ones(size(sensor_newa)));
    end 
end
axis_holder = vertcat(axis_holderN,axis_holderA);
sensors_combined = vertcat(sensors_combinedN,sensors_combinedA);
subplot(1,2,1), boxplot(sensors_combined, axis_holder)
title({'Abnormality Scores Predicted by Sensor Model'})
xlabel('Datasets')
ylabel('Score')
ylim([0 1]);
set(gca,'xticklabel',{'N3','N4','N5','A3','A4','A5'})

sensors_combined = []
for i=2:4
    
    sensorFileN = strcat(dir, 'Normal File Scores\N' , num2str(i), '_image_scores.txt')
    sensorFileA = strcat(dir, 'Abnormal File Scores\A' , num2str(i), '_image_scores.txt')
    sensorIDN = fopen(sensorFileN,'r');
    sensorIDA = fopen(sensorFileA,'r');
    formatSpec = '%f';
    sensor_newn = fscanf(sensorIDN,formatSpec);
    sensor_newa = fscanf(sensorIDA,formatSpec);
    
    if i==2
        sensors_combinedN = sensor_newn;
        axis_holderN = ones(size(sensor_newn));
        sensors_combinedA = sensor_newa;
        axis_holderA = 4*ones(size(sensor_newa));
    elseif i==3
        sensors_combinedN = vertcat(sensors_combinedN, sensor_newn);
        axis_holderN = vertcat(axis_holderN, 2*ones(size(sensor_newn)));
        sensors_combinedA = vertcat(sensors_combinedA, sensor_newa);
        axis_holderA = vertcat(axis_holderA, 5*ones(size(sensor_newa)));
    else
        sensors_combinedN = vertcat(sensors_combinedN, sensor_newn);
        axis_holderN = vertcat(axis_holderN, 3*ones(size(sensor_newn)));
        sensors_combinedA = vertcat(sensors_combinedA, sensor_newa);
        axis_holderA = vertcat(axis_holderA, 6*ones(size(sensor_newa)));
    end 
end
axis_holder = vertcat(axis_holderN,axis_holderA);
sensors_combined = vertcat(sensors_combinedN,sensors_combinedA);
subplot(1,2,2), boxplot(sensors_combined, axis_holder)
title({'Abnormality Scores Predicted by Image Model'})
xlabel('Datasets')
ylabel('Score')
ylim([0 1]);
set(gca,'xticklabel',{'N3','N4','N5','A3','A4','A5'})

% sensors_combined = []
% for i=2:4
%     sensorFile = strcat(dir, 'Abnormal File Scores\A' , num2str(i), '_sensor_scores.txt')
%     sensorID = fopen(sensorFile,'r');
%     formatSpec = '%f';
%     sensor_new = fscanf(sensorID,formatSpec);
%     
%     if i==2
%         sensors_combined = sensor_new;
%         axis_holder = ones(size(sensor_new)); 
%     else
%         sensors_combined = vertcat(sensors_combined, sensor_new);
%         axis_holder = vertcat(axis_holder, (i+1)*ones(size(sensor_new))); 
%     end 
% end
% 
% subplot(2,2,2), boxplot(sensors_combined, axis_holder)
% boxplot(sensors_combined, axis_holder)
% title({'Abnormality Scores on Abnormal Data', 'Predicted by Sensor Model'})
% xlabel('Datasets')
% ylabel('Score')
% ylim([0 1]);
% set(gca,'xticklabel',{'N3','N4','N5'})
% 
% 
% sensors_combined = []
% for i=2:4
%     sensorFile = strcat(dir, 'Normal File Scores\N' , num2str(i), '_image_scores.txt');
%     sensorID = fopen(sensorFile,'r');
%     formatSpec = '%f';
%     sensor_new = fscanf(sensorID,formatSpec);
%     
%     if i==2
%         sensors_combined = sensor_new;
%         axis_holder = ones(size(sensor_new)); 
%     else
%         sensors_combined = vertcat(sensors_combined, sensor_new);
%         axis_holder = vertcat(axis_holder, (i+1)*ones(size(sensor_new))); 
%     end 
% end
% subplot(2,2,3), boxplot(sensors_combined, axis_holder)
% title({'Abnormality Scores on Normal Data', 'Predicted by Image Model'})
% xlabel('Datasets')
% ylabel('Score')
% 
% sensors_combined = []
% for i=2:4
%     sensorFile = strcat(dir, 'Abnormal File Scores\A' , num2str(i), '_image_scores.txt')
%     sensorID = fopen(sensorFile,'r');
%     formatSpec = '%f';
%     sensor_new = fscanf(sensorID,formatSpec);
%     
%     if i==2
%         sensors_combined = sensor_new;
%         axis_holder = ones(size(sensor_new)); 
%     else
%         sensors_combined = vertcat(sensors_combined, sensor_new);
%         axis_holder = vertcat(axis_holder, (i+1)*ones(size(sensor_new))); 
%     end 
% end
% 
% subplot(2,2,4), boxplot(sensors_combined, axis_holder)
% boxplot(sensors_combined, axis_holder)
% title({'Abnormality Scores on abnormal Data', 'Predicted by Image Model'})
% xlabel('Datasets')
% ylabel('Score')
% 
suptitle('Summary of Results For Multiple Bag Files')
% 
% 
% 
% 
% 
