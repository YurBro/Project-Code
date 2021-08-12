function [ extendData ] = ExtendData( Data, d  )
%EXTENDDATA 对数据集扩展使其包含d-1个历史数据，共d个数据。
%  Input: 数据集，扩展长度
%  Output: 扩展之后数据集
[rowData, colData] = size(Data);
extendData = zeros(rowData-d+1,d*colData);
    for k = 1:(rowData-d+1)
        TEMP = Data(k:k+d-1,:);
        extendData(k,:) = reshape(TEMP,1,[]);
    end
end

