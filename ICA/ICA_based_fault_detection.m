% 基于ICA的故障诊断
clear;clc;close all;
load('MPD2000.mat');

Xnormal = MPD0';
% 数据归一化
[dim, numSample] = size(Xnormal);
XnormalMean = mean(Xnormal, 2);
XnormalStd = std(Xnormal, 0, 2);
XnormalNorm = normalization(Xnormal, XnormalMean, XnormalStd);

% 正常数据计算 解混矩阵W
% P = rand(dim,dim)*100;
load('P.mat');
[S, Q, P] = FastICA(XnormalNorm, P);
W = P'*Q;
% ------------------------利用2范数大小对W的行重新排列---------------------
Wnorm = zeros(dim,1);
for k = 1:dim
    Wnorm(k) = norm(W(k,:));
end
[Wnorm, indices] = sort(Wnorm, 'descend');

% -------------------------确定主导成分Sd与参与成分Se----------------------
threshold = 0.80;
percentage = cumsum(Wnorm)./sum(Wnorm);
for k = 1:dim
    if percentage(k) > threshold
        break;
    end
end
% --------------------------------生成Wd与We------------------------------
Wnew = W(indices,:);
Wd = Wnew(1:k,:);
We = Wnew(k+1:end,:);

Xtest = MPD0';
XtestNorm = normalization(Xtest, XnormalMean, XnormalStd);
% 测试数据计算I2,Ie2,SPE控制限，置信水平1-alpha=0.99
Sd = Wd*XtestNorm; I2 = sum(Sd.*Sd);
Se = We*XtestNorm; Ie2 = sum(Se.*Se);
E = XtestNorm - inv(Q)*(Wd*inv(Q))'*Wd*XtestNorm; SPE = sum(E.*E);
I2_lim=ksdensity(I2,0.99,'function','icdf');
Ie2_lim=ksdensity(Ie2,0.99,'function','icdf');
SPE_lim=ksdensity(SPE,0.99,'function','icdf');

Xfault = MPD4';
XfaultNorm = normalization(Xfault, XnormalMean, XnormalStd);
% 在线故障检测
[dim, numSample] = size(Xfault);
Sd = Wd*XfaultNorm; I2 = sum(Sd.*Sd);
Se = We*XfaultNorm; Ie2 = sum(Se.*Se);
E = XfaultNorm - inv(Q)*(Wd*inv(Q))'*Wd*XfaultNorm; SPE = sum(E.*E);

% I2，Ie2,SPE监控图
figure
subplot(3,1,1);
plot([1 numSample],[I2_lim I2_lim],'r--','LineWidth',2);hold on
plot(1:numSample,I2,'LineWidth',2);ylabel('I^2');
subplot(3,1,2);
plot([1 numSample],[Ie2_lim Ie2_lim],'r--','LineWidth',2);hold on
plot(1:numSample,Ie2,'LineWidth',2);ylabel('Ie^2');
subplot(3,1,3);
plot([1 numSample],[SPE_lim SPE_lim],'r--','LineWidth',2);hold on
plot(1:numSample,SPE,'LineWidth',2);ylabel('SPE');xlabel('Sample Number');
