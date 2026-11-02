function [dlSupport,dlQuery] = dlarrayMakerTest(allImages,sIdx,qIdx,sLabels,qLabels)

sImages=allImages(:,:,:,sIdx);
sImages=single(sImages);
dlSupport.Images=dlarray(gpuArray(sImages),'SSCB');
dlSupport.Labels=sLabels;


qImages=allImages(:,:,:,qIdx);
qImages=single(qImages);
dlQuery.Images=dlarray(gpuArray(qImages),'SSCB');
dlQuery.Labels=qLabels;

end
