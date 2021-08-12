% data �б���ά�����в���������dataMean �о�ֵ������dataStd �з�������
function dataNorm = normalization(data, dataMean, dataStd)
numSample = size(data, 2);
dataNorm = (data - repmat(dataMean, 1, numSample))./repmat(dataStd, 1, numSample);
end
