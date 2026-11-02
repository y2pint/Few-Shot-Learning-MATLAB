function [sIdx,qIdx,sLabels,qLabels] = episodemakerTest(allLabels,N,K,Q)

allLabels = removecats(allLabels); % Clean the categories first
baseClassNames = categories(allLabels);

classes=unique(baseClassNames);
selectedClasses=classes(randperm(length(classes),N));


sIdx=zeros(N*K,1);
qIdx=zeros(N*Q,1);
sLabels=zeros(N*K,1);
qLabels=zeros(N*Q,1);



    for i=1:N
    
    currentClass=selectedClasses(i);    
    idxCClass=find(allLabels == currentClass); %uses ram
    %cClassFiles=imdsBase.Files(idxCClass);
    numNeeded = K+Q;
    rngSampleIdx=idxCClass(randperm(length(idxCClass),numNeeded));
    sIdx((i-1)*K+(1:K))=rngSampleIdx(1:K);
    qIdx((i-1)*Q+(1:Q))=rngSampleIdx(K+1:end);
    sLabels((i-1)*K + (1:K)) = i;
    qLabels((i-1)*Q + (1:Q))=i;
   
   
    end

end
