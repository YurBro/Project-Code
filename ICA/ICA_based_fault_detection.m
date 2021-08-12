% ����ICA�Ĺ������
clear;clc;close all;
load('MPD2000.mat');

Xnormal = MPD0';
% ���ݹ�һ��
[dim, numSample] = size(Xnormal);
XnormalMean = mean(Xnormal, 2);
XnormalStd = std(Xnormal, 0, 2);
XnormalNorm = normalization(Xnormal, XnormalMean, XnormalStd);

% �������ݼ��� ������W
% P = rand(dim,dim)*100;
load('P.mat');
[S, Q, P] = FastICA(XnormalNorm, P);
W = P'*Q;
% ------------------------����2������С��W������������---------------------
Wnorm = zeros(dim,1);
for k = 1:dim
    Wnorm(k) = norm(W(k,:));
end
[Wnorm, indices] = sort(Wnorm, 'descend');

% -------------------------ȷ�������ɷ�Sd�����ɷ�Se----------------------
threshold = 0.80;
percentage = cumsum(Wnorm)./sum(Wnorm);
for k = 1:dim
    if percentage(k) > threshold
        break;
    end
end
% --------------------------------����Wd��We------------------------------
Wnew = W(indices,:);
Wd = Wnew(1:k,:);
We = Wnew(k+1:end,:);

Xtest = MPD0';
XtestNorm = normalization(Xtest, XnormalMean, XnormalStd);
% �������ݼ���I2,Ie2,SPE�����ޣ�����ˮƽ1-alpha=0.99
Sd = Wd*XtestNorm; I2 = sum(Sd.*Sd);
Se = We*XtestNorm; Ie2 = sum(Se.*Se);
E = XtestNorm - inv(Q)*(Wd*inv(Q))'*Wd*XtestNorm; SPE = sum(E.*E);
I2_lim=ksdensity(I2,0.99,'function','icdf');
Ie2_lim=ksdensity(Ie2,0.99,'function','icdf');
SPE_lim=ksdensity(SPE,0.99,'function','icdf');

Xfault = MPD4';
XfaultNorm = normalization(Xfault, XnormalMean, XnormalStd);
% ���߹��ϼ��
[dim, numSample] = size(Xfault);
Sd = Wd*XfaultNorm; I2 = sum(Sd.*Sd);
Se = We*XfaultNorm; Ie2 = sum(Se.*Se);
E = XfaultNorm - inv(Q)*(Wd*inv(Q))'*Wd*XfaultNorm; SPE = sum(E.*E);

% I2��Ie2,SPE���ͼ
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
