function [ norData] = NormalizeData( Data, meanData, stdData  )
%NORMALIZDDATA �����ݽ��б�׼������ת�����þ�ֵ�������ɺ���ָ����
%  Input: ���ݼ�������ֵ����׼��
%  Output: ��׼�������ݼ�
%  NOTE�����ݱ�׼�����õľ�ֵ�������Ϊ��ģ���ݵľ�ֵ����
[rowData, ~] = size(Data);
norData = (Data - repmat(meanData,rowData,1))./repmat(stdData,rowData,1);
end

