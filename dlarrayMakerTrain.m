function [dlSupport,dlQuery]=dlarrayMakerTrain(supportData,queryData)
    
    supportImages=cat(4,supportData.Images{:});
    dlSupport.Images=dlarray(gpuArray(single(supportImages)),'SSCB');
    dlSupport.Labels=supportData.Labels;
    
    queryImages=cat(4,queryData.Images{:});
    dlQuery.Images=dlarray(gpuArray(single(queryImages)),'SSCB');
    dlQuery.Labels=queryData.Labels;

end
