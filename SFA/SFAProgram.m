%SFA Multivariate Statistical Process Monitoring base on Slow Feature Analysis
% Input data:
%           Format: n samples, m variables.
% Output data:
% NOTE: This script will call the function named SFA,
% Althor:
% Data:
clear,close all,clc;
load('MPD2000.mat');% �������ݣ����ݸ�ʽ���д���������д������
%% ������ʷ���ݽ�ģ
% ��ģ���ݡ��������ݵ���
ModelData = MPD0;
% DFSA ��ʷ������չ��d=1,��������ʷ���ݡ�d>1,������ʷ����
dataExtend = 1;
% ��ģ���� ��չ����׼��
ExtX = ExtendData(ModelData, dataExtend);
[N, num_slowVar] = size(ExtX);
meanStand = mean(ExtX);stdStand = std(ExtX);
NorX = NormalizeData(ExtX, meanStand, stdStand);
% ��ģ����ǰ����
DotNorX = NorX(2:end,:)-NorX(1:end-1,:);
CovNorX = NorX'*NorX/(size(NorX,1));
CovDotNorX = DotNorX'*DotNorX/(size(DotNorX,1));
[W, Omega] = LinearSFA(CovNorX, CovDotNorX); 

% ����ԭʼ�����仯������ȷ��dominant SFs and residual SFs
slowNorX = diag(cov(DotNorX));
% slowNorX_order = sort(slowNorX, 'descend');
% num = ceil(0.1*size(NorX,2));
% maxslow = slowNorX_order(num);
% Me = sum(diag(Omega) > maxslow); % residual SFs
% M = num_slowVar-Me; % dominant SFs

%--------------------------------------------------------------------------
% ʹ��quantile����ȷ��0.9��λ��
quan_element = quantile(slowNorX,0.9);
Me = sum(diag(Omega) >= quan_element);
M = num_slowVar-Me;
Omega_d1 = Omega(end-M+1:end,end-M+1:end);
Omega_e1 = Omega(1:Me,1:Me);
Omega_dInv = inv(Omega_d1);
Omega_eInv = inv(Omega_e1);

% ͳ�����趨��������ˮƽalpha=0.01������ˮƽ1-alpha=0.99
alpha = 0.01;
T2 = chi2inv(1-alpha, M);
T2e = chi2inv(1-alpha, Me);
S2 = finv(1-alpha, M, N-M-1)*M*(N^2-2*N)/(N-1)/(N-M-1);
S2e = finv(1-alpha, Me, N-Me-1)*Me*(N^2-2*N)/(N-1)/(N-Me-1);

%% ���ò������ݲ���ģ��
% �������ݵ���
TestData = MPD0;
% �������� ��չ����׼��
ExtTX = ExtendData(TestData, dataExtend);
NorTX = NormalizeData(ExtTX, meanStand, stdStand);


Feature = NorX*W;
testFeature = NorTX*W;


%% ���ù������ݽ��м�� 
FaultData = MPD4;
% ������չ
ExtFD = ExtendData(FaultData,dataExtend);
NorFD = NormalizeData(ExtFD,meanStand,stdStand);
DotNorFD = NorFD(2:end,:) - NorFD(1:end-1,:);
% ��������
FeatureFD = NorFD*W;
DominantSF = FeatureFD(:,end-M+1:end);
ResidualSF = FeatureFD(:,1:Me);

DotFeatureFD = DotNorFD*W;
DotDominantSF = DotFeatureFD(:,end-M+1:end);
DotResidualSF = DotFeatureFD(:,1:Me);

T2_index = sum(DominantSF.^2, 2);
T2e_index = sum(ResidualSF.^2, 2);

S2_index = [];
for k = 1:size(DotDominantSF,1)
    temp_index = DotDominantSF(k,:)*Omega_dInv*DotDominantSF(k,:)';
    S2_index = [S2_index; temp_index];
end

S2e_index = [];
for k = 1:size(DotDominantSF,1)
    temp_index = DotResidualSF(k,:)*Omega_eInv*DotResidualSF(k,:)';
    S2e_index = [S2e_index; temp_index];
end

% ��ͼ
index = [T2_index T2e_index [S2_index;0] [S2e_index;0]];
for k=1:4
    subplot(4,1,k)
%     plot(1:size(index,1),index(:,k),'-b','LineWidth',2),hold on;    
    plot(1:size(index,1),index(:,k),'-b','LineWidth',2),hold on;
    switch k
        case 1
            plot([0 size(index,1)],[T2 T2],'--r','LineWidth',2);
            ylabel('T^2');
        case 2
            plot([0 size(index,1)],[T2e T2e],'--r','LineWidth',2);
            ylabel('T^2_e');
        case 3
            plot([0 size(index,1)],[S2 S2],'--r','LineWidth',2);
            ylabel('S^2');
        case 4
            plot([0 size(index,1)],[S2e S2e],'--r','LineWidth',2);
            ylabel('S^2_e');
            xlabel('Sample sequence')
        otherwise
            disp('��ͼ�������');break;
    end
end
