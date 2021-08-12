function [ W, Omega ] = LinearSFA( CovX, CovDotX )
%LINEARSFA Summary of this function goes here
%   Detailed explanation goes here
[U_T, lambda,~] = svd(CovX);
% ʹ��/����\��������inv(),������߼��㾫�ȡ�
sqrt_lambda = sqrt(lambda);
CovDotZ = sqrt_lambda\U_T'*CovDotX*U_T/sqrt_lambda;
[P, Omega, ~] = svd(CovDotZ);
W = U_T/sqrt_lambda*P;
%--------------------------------------------------------------------------
% ʹ��inv()���㡣
% Q = U_T*sqrt(inv(lambda));
% CovDotZ = Q'*CovDotX*Q;
%[P, Omega, ~] = svd(CovDotZ);
% W = Q*P;
%--------------------------------------------------------------------------
% P = fliplr(P);
% Omega = diag(sort(diag(Omega),'ascend'));
end

