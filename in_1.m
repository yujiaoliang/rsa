rng default
dataIn = randi([0 1],288000,1);  % Generate vector of binary data
dataInMatrix = reshape(dataIn,48000,6)
dataInMatrix1 = reshape(dataIn,6000,48)   % Reshape data into binary k-tuples, k = log2(M)
csvwrite("C:\\Users\\yujiaoliang\\Desktop\\one.csv",dataInMatrix1)