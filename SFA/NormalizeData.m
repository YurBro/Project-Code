function [ norData] = NormalizeData( Data, meanData, stdData  )
%NORMALIZDDATA 对数据进行标准化，表转化所用均值、方差由函数指定。
%  Input: 数据集、、均值、标准差
%  Output: 标准化后数据集
%  NOTE：数据标准化所用的均值、方差均为建模数据的均值方差
[rowData, ~] = size(Data);
norData = (Data - repmat(meanData,rowData,1))./repmat(stdData,rowData,1);
end

