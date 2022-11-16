clc
clear all
close all

%%
dir = 'C:\Users\shafi\Downloads\Sp Cup 2020 Resources\Sp ar koto\codes\sp-cup-2020';
file = strcat(dir, '\Flow_losses.txt');

fileID = fopen(file,'r');
formatSpec = '%f';
losses = fscanf(fileID,formatSpec);

losses_new = [];

for i=1:length(losses)
    if losses(i) < 10
        losses_new = horzcat(losses_new,losses(i));
    end
end


% histogram(losses_new, 100)
% hold on
% xlabel('Reconstruction Loss')
% ylabel('Frequency')

edges = linspace(0,0.3,18);
edges2 = linspace(0.3,0.5,10);
edges2 = edges2([2:end]);
edges3 = linspace(0.5,10,30);
edges3 = edges3([2:end]);
edges = horzcat(edges,edges2,edges3);

a = histcounts(losses_new, edges);

% pd = fitdist(a','Normal')
% 
% %// Plot curve
% x = 0: 0.01: 10.5;  %// Plotting range
% y = exp(- 0.5 * ((x - pd.mu) / pd.sigma) .^ 2) / (pd.sigma * sqrt(2 * pi));
% plot(x, y*500)
% xlim([0,10.5])
bar(a)
l1 = length(edges2)
set(gca,'xticklabel',{0, num2str(edges2(1)),0.52222,num2str(edges3(1)), 1.1379, 2.3174})
xlabel('Reconstruction Loss For Optical Flow Vectors')
ylabel('Frequency')



