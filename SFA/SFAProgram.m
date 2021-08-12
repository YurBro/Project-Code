%SFA Multivariate Statistical Process Monitoring base on Slow Feature Analysis
% Input data:
%           Format: n samples, m variables.
% Output data:
% NOTE: This script will call the function named SFA,
% Althor:
% Data:
clear,close all,clc;
load('MPD2000.mat');% 载入数据，数据格式：行代表采样，列代表变量
%% 利用历史数据建模
% 建模数据、测试数据导入
ModelData = MPD0;
% DFSA 历史数据扩展。d=1,不包含历史数据。d>1,包含历史数据
dataExtend = 1;
% 建模数据 扩展、标准化
ExtX = ExtendData(ModelData, dataExtend);
[N, num_slowVar] = size(ExtX);
meanStand = mean(ExtX);stdStand = std(ExtX);
NorX = NormalizeData(ExtX, meanStand, stdStand);
% 建模数据前向差分
DotNorX = NorX(2:end,:)-NorX(1:end-1,:);
CovNorX = NorX'*NorX/(size(NorX,1));
CovDotNorX = DotNorX'*DotNorX/(size(DotNorX,1));
[W, Omega] = LinearSFA(CovNorX, CovDotNorX); 

% 计算原始变量变化快慢，确定dominant SFs and residual SFs
slowNorX = diag(cov(DotNorX));
% slowNorX_order = sort(slowNorX, 'descend');
% num = ceil(0.1*size(NorX,2));
% maxslow = slowNorX_order(num);
% Me = sum(diag(Omega) > maxslow); % residual SFs
% M = num_slowVar-Me; % dominant SFs

%--------------------------------------------------------------------------
% 使用quantile函数确定0.9分位数
quan_element = quantile(slowNorX,0.9);
Me = sum(diag(Omega) >= quan_element);
M = num_slowVar-Me;
Omega_d1 = Omega(end-M+1:end,end-M+1:end);
Omega_e1 = Omega(1:Me,1:Me);
Omega_dInv = inv(Omega_d1);
Omega_eInv = inv(Omega_e1);

% 统计限设定，显著性水平alpha=0.01，置信水平1-alpha=0.99
alpha = 0.01;
T2 = chi2inv(1-alpha, M);
T2e = chi2inv(1-alpha, Me);
S2 = finv(1-alpha, M, N-M-1)*M*(N^2-2*N)/(N-1)/(N-M-1);
S2e = finv(1-alpha, Me, N-Me-1)*Me*(N^2-2*N)/(N-1)/(N-Me-1);

%% 利用测试数据测试模型
% 测试数据导入
TestData = MPD0;
% 测试数据 扩展、标准化
ExtTX = ExtendData(TestData, dataExtend);
NorTX = NormalizeData(ExtTX, meanStand, stdStand);


Feature = NorX*W;
testFeature = NorTX*W;


%% 利用故障数据进行检测 
FaultData = MPD4;
% 数据扩展
ExtFD = ExtendData(FaultData,dataExtend);
NorFD = NormalizeData(ExtFD,meanStand,stdStand);
DotNorFD = NorFD(2:end,:) - NorFD(1:end-1,:);
% 求慢特征
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

% 作图
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
            disp('作图程序出错');break;
    end
end
