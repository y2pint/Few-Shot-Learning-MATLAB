function [supportData,queryData] = episodemakerTrain(allImgs, allLabels,auger, N, K, Q)
    supportCell = cell(N*K, 1);
    queryCell = cell(N*Q, 1);
    supportLabels = zeros(N*K, 1);
    queryLabels = zeros(N*Q, 1);
    
    baseClassNames = categories(allLabels); 
    selectedClasses = datasample(baseClassNames, N, 'Replace', false); 
    
    for i = 1:N 
        currentClass = selectedClasses(i);
        idxCClass = find(allLabels == currentClass); 
        numNeeded = K+Q; 
        rngSampleIdx = datasample(idxCClass, numNeeded, 'Replace', false); 
        
        for j = 1:K   
            img = allImgs{rngSampleIdx(j)}; 
            img = augment(auger, img);      
            supportCell{(i-1)*K+j} = img;
            supportLabels((i-1)*K+j) = i; 
        end
           

        for j = 1:Q
            img = allImgs{rngSampleIdx(K+j)};
            img = augment(auger, img);
            queryCell{(i-1)*Q+j} = img;
            queryLabels((i-1)*Q+j) = i;
        end
    end
    supportData = struct('Images', {supportCell}, 'Labels', supportLabels);
    queryData = struct('Images', {queryCell}, 'Labels', queryLabels);
end
