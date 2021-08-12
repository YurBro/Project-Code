% Input：X 行变量维数，列采样个数；需要对原始矩阵转置
% Output：Sources重构的原信号, Q白化矩阵, P白化信号解混矩阵
function [Sources, Q, P] = FastICA(X, P)
% 白化处理
[dim, numSample] = size(X);
Xcov = cov(X');
[U, lambda] = eig(Xcov);
Q = lambda^(-1/2)*U';
Z = Q*X;
% FastICA
maxiteration = 10000; %最大迭代次数
error = 1e-5; % 收敛误差
% P = randn(dim,dim); % 随机初始化P，并按照列更新

for k = 1:dim
    Pk = P(:,k);
    Pk = Pk./norm(Pk); % 向量归一化
    lastPk = zeros(dim,1); % 0不需要再归一化
    count = 0;
    while abs(Pk - lastPk)&abs(Pk + lastPk) > error
        count = count + 1;
        lastPk = Pk;
        g = tanh(lastPk'*Z); % g(y)函数
        dg = 1 - g.^2; % g(y)的一阶导函数
%-------------------------------核心公式------------------------------------        
        Pk = mean(Z.*repmat(g,dim,1), 2) - repmat(mean(dg),dim,1).*lastPk;
        Pk = Pk - sum(repmat(Pk'*P(:,1:k-1),dim,1).*P(:,1:k-1),2);
        Pk = Pk./norm(Pk);
%--------------------------------------------------------------------------       
        if count == maxiteration
            fprintf('第%d个分量在%d次迭代内不收敛！\n',k,maxiteration);
            break;
        end
    end
    P(:,k) = Pk;
end
    Sources = P'*Z;
% end
        
        