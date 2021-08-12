% data 行变量维数，列采样个数；dataMean 列均值向量；dataStd 列方差向量
function dataNorm = normalization(data, dataMean, dataStd)
numSample = size(data, 2);
dataNorm = (data - repmat(dataMean, 1, numSample))./repmat(dataStd, 1, numSample);
end
