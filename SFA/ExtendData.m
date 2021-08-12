function [ extendData ] = ExtendData( Data, d  )
%EXTENDDATA �����ݼ���չʹ�����d-1����ʷ���ݣ���d�����ݡ�
%  Input: ���ݼ�����չ����
%  Output: ��չ֮�����ݼ�
[rowData, colData] = size(Data);
extendData = zeros(rowData-d+1,d*colData);
    for k = 1:(rowData-d+1)
        TEMP = Data(k:k+d-1,:);
        extendData(k,:) = reshape(TEMP,1,[]);
    end
end

