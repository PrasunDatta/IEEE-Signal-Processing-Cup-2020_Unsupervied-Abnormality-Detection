clc
clear all
close all

%%

dir = 'C:\Users\shafi\Downloads\Sp Cup 2020 Resources\Sp ar koto\codes\sp-cup-2020\'
cpun = strcat(dir, 'CpuTime_norm.txt');
gpun = strcat(dir, 'GpuTime_norm.txt');
cpua = strcat(dir, 'CpuTime_anom.txt');
gpua = strcat(dir, 'GpuTime_anom.txt');

cpunID = fopen(cpun,'r');
formatSpec = '%f';
cpu_normal = fscanf(cpunID,formatSpec);
cpun_mean = mean(cpu_normal)*ones(length(cpu_normal),1);

gpunID = fopen(gpun,'r');
formatSpec = '%f';
gpu_normal = fscanf(gpunID,formatSpec);
gpun_mean = mean(gpu_normal)*ones(length(gpu_normal),1);

cpuaID = fopen(cpua,'r');
formatSpec = '%f';
cpu_abnormal = fscanf(cpuaID,formatSpec);
cpua_mean = mean(cpu_abnormal)*ones(length(cpu_abnormal),1);

gpuaID = fopen(gpua,'r');
formatSpec = '%f';
gpu_abnormal = fscanf(gpuaID,formatSpec);
gpua_mean = mean(gpu_abnormal)*ones(length(gpu_abnormal),1);

figure()
subplot(1,2,1), plot(cpu_normal)
hold on
plot(gpu_normal)
plot(cpun_mean,'--')
plot(gpun_mean,'--')
%yline(cpun_mean,':')
title('Normal Data')
xlabel('Frame Number')
ylabel('Seconds Required')
%legend('CPU','GPU','CPU MEAN', 'GPU MEAN')
ylim([0 6]);
xlim([0 50]);

subplot(1,2,2), plot(cpu_abnormal)
hold on
plot(gpu_abnormal)
plot(cpua_mean,'--')
plot(gpua_mean,'--')
title('Abnormal Data')
xlabel('Frame Number')
ylabel('Seconds Required')
%legend('CPU','GPU', 'CPU MEAN', 'GPU MEAN')
ylim([0 6]);
xlim([0 50]);

suptitle('CPU vs GPU Perfromance For Optical Flow Calculation')






