% Input��X �б���ά�����в�����������Ҫ��ԭʼ����ת��
% Output��Sources�ع���ԭ�ź�, Q�׻�����, P�׻��źŽ�����
function [Sources, Q, P] = FastICA(X, P)
% �׻�����
[dim, numSample] = size(X);
Xcov = cov(X');
[U, lambda] = eig(Xcov);
Q = lambda^(-1/2)*U';
Z = Q*X;
% FastICA
maxiteration = 10000; %����������
error = 1e-5; % �������
% P = randn(dim,dim); % �����ʼ��P���������и���

for k = 1:dim
    Pk = P(:,k);
    Pk = Pk./norm(Pk); % ������һ��
    lastPk = zeros(dim,1); % 0����Ҫ�ٹ�һ��
    count = 0;
    while abs(Pk - lastPk)&abs(Pk + lastPk) > error
        count = count + 1;
        lastPk = Pk;
        g = tanh(lastPk'*Z); % g(y)����
        dg = 1 - g.^2; % g(y)��һ�׵�����
%-------------------------------���Ĺ�ʽ------------------------------------        
        Pk = mean(Z.*repmat(g,dim,1), 2) - repmat(mean(dg),dim,1).*lastPk;
        Pk = Pk - sum(repmat(Pk'*P(:,1:k-1),dim,1).*P(:,1:k-1),2);
        Pk = Pk./norm(Pk);
%--------------------------------------------------------------------------       
        if count == maxiteration
            fprintf('��%d��������%d�ε����ڲ�������\n',k,maxiteration);
            break;
        end
    end
    P(:,k) = Pk;
end
    Sources = P'*Z;
% end
        
        