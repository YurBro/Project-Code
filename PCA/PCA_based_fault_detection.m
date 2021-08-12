% PCA based Fault Detection
% 
clear;clc;close all;
load('MPD2000.mat')
%% �������ݽ���PCA���ģ��
dataRow = MPD0;
[numSample dim] = size(dataRow);
% ��ģ���ݹ�һ��
meanData = mean(dataRow);
stdData = std(dataRow);
dataNorm = (dataRow - repmat(meanData,numSample,1))./repmat(stdData,numSample,1);
% Э�����������ֵ�ֽ�
covData = dataNorm'*dataNorm./(numSample - 1);
[U lambda] = eig(covData);
[lambda index] = sort(diag(lambda),'descend'); % ����ֵ���մӴ�С����
U = U(:,index); % ��ԭ��������
% ѡ�����ɷָ���
threshold = 0.80;
for k = 1:length(lambda)
    percentage = sum(lambda(1:k))/sum(lambda);
    if percentage > threshold
        numPC = k;
        break
    end
end
% ��ԭP��lambda_d
P = U(:,1:numPC);
lambda_d = diag(lambda(1:numPC));
% ����ͳ����
alpha = 0.01; % ������ˮƷ
% T2 ͳ����
T2_lim = numPC*(numSample^2 - 1)*finv(1-alpha,numPC,numSample - numPC)/(numSample*(numSample - numPC));
% SPE ͳ����
for k = 1:3
    theta(k) = sum(lambda(numPC + 1:end).^(k));
end
h0 = 1 - 2*theta(1)*theta(3)/(3*theta(2)^2);
SPE_lim = theta(1)*(norminv(1-alpha,0,1)*sqrt(2*theta(2)*h0^2)/theta(1) + 1 + theta(2)*h0*(h0-1)/theta(1)^2)^(1/h0);

%% ��������
dataFault = MPD4;
[numSample dim] = size(dataFault);
% �������ݹ�һ��
dataFaultNorm = (dataFault - repmat(meanData,numSample,1))./repmat(stdData,numSample,1);
% T2_index��SPE_ndex����
for k = 1:numSample
    T2_index(k) = dataFaultNorm(k,:)*P*inv(lambda_d)*P'*dataFaultNorm(k,:)';
end
for k = 1:numSample
    SPE_index(k) = dataFaultNorm(k,:)*(eye(dim,dim) - P*P')*dataFaultNorm(k,:)';
end
figure
subplot(2,1,1)
plot([1 numSample],[T2_lim T2_lim],'r--','LineWidth',2);xlabel('Sample Number');ylabel('T^2');hold on;
plot(1:numSample,T2_index,'LineWidth',2);
subplot(2,1,2)
plot([1 numSample],[SPE_lim SPE_lim],'r--','LineWidth',2);xlabel('Sample Numble');ylabel('SPE');hold on;
plot(1:numSample,SPE_index,'LineWidth',2);
