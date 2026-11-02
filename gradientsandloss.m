function [loss,gradient,accuracy]=gradientsandloss(net,dlSupport,dlQuery)

   
    sEmbedRaw=forward(net,dlSupport.Images);
    qEmbedRaw=forward(net,dlQuery.Images);
    
    
    sEmbed=stripdims(sEmbedRaw)';
    qEmbed=stripdims(qEmbedRaw)';
    
 
    sLabels=dlSupport.Labels;
    qLabels=dlQuery.Labels;
    
   
    classes=unique(sLabels);
    
    numclass=numel(classes);
    
    
   
    prototypez=cell(numclass,1);
    
    for i=1:numclass
        cls=classes(i); 
        idx=(sLabels==cls);
      
        classEmb=sEmbed(idx,:);
     
        prototypez{i}=mean(classEmb,1);
      
    end

    prototypes=cat(1,prototypez{:}); %put all the prototypes into array
    
    
    q_s=sum(qEmbed.^2,2);  %      [M x 1]
    p_s=sum(prototypes.^2,2); %   [N x 1]
    dotProd=qEmbed*prototypes'; % [M x N]
    distance=q_s+p_s'-2*dotProd;
    
    
    logits=-1*distance;
    
    slogits=logits-max(logits,[],2);
    logProb=slogits-log(sum(exp(slogits),2));
    
    gtTable=(qLabels==classes');
    
    correctClassLP=sum(logProb.*gtTable,2); 
    
    loss=-1*mean(correctClassLP); 
    gradient=dlgradient(loss,net.Learnables);
    
    
    [~,predIdx]=max(logits,[],2);
    [~,targIdx]=max(gtTable,[],2);
    
    accuracy=mean(predIdx==targIdx,'all');
    
end
