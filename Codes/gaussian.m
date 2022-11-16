clc
clear all
close all

%%
dir = 'C:\Users\shafi\Downloads\Sp Cup 2020 Resources\LSTM Autoencoder Sequence wise Anomaly Detection';
file = strcat(dir, '\plot_data.txt');
%image_file = strcat(dir, '\B', num2str(i), '_image_scores.txt');

fileID = fopen(file,'r');
formatSpec = '%f';
losses = fscanf(fileID,formatSpec);

a = reshape(losses,[10,664])

b = size(a)

figure()
for i=1:10
    new_data = a(i,:);
%     if i==4
%         continue;
%     end
%     %     if i<4
%     %     subplot(3,3,i), histogram(new_data,36)
%     %     end
%     %     if i>4
%     %         subplot(3,3,i-1), histogram(new_data,36)
%     %     end
%     if i==1
%         subplot(3,3,i), histogram(new_data,36)
%         ylabel('Ori-x')
%     end
%     if i==2
%         subplot(3,3,i), histogram(new_data,36)
%         ylabel('Ori-y')
%     end
%     if i==3
%         subplot(3,3,i), histogram(new_data,36)
%         ylabel('Ori-z')
%     end
%     if i==5
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('AngVel-x')
%     end
%     if i==6
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('AngVel-y')
%     end
%     if i==7
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('AngVel-z')
%     end
%     if i==8
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('Acc-x')
%     end
%     if i==9
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('Acc-y')
%     end
    if i==10
        histogram(new_data,36)
        ylabel('Frequency')
        xlabel('Reconstruction Loss For Acceleration (z-axis)')
    end
end

%      if i==1
%         subplot(3,3,i), histogram(new_data,36)
%         ylabel('Ori_x')
%     end
%     if i==2
%         subplot(3,3,i), histogram(new_data,36)
%         ylabel('Ori_y')
%     end
%     if i==3
%         subplot(3,3,i), histogram(new_data,36)
%         ylabel('Ori_z')
%     end
%     if i==5
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('AngVel_x')
%     end
%     if i==6
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('AngVel_y')
%     end
%     if i==7
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('AngVel_z')
%     end
%     if i==8
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('Acc_x')
%     end
%     if i==9
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('Acc_y')
%     end
%     if i==10
%         subplot(3,3,i-1), histogram(new_data,36)
%         ylabel('Acc_z')
%     end















