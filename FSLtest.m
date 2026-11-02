%fsl with novel modular
clc;
clear all;
close all;



prompt='would you like to test googlenet, resnet, or bestnet?';
%disp(prompt);
input1=input(prompt,'s');


if strcmpi(input1,'rf')
    load('TrainedFSLNet_resnet18f.mat','fslNet');
elseif strcmpi(input1,'r')
      load('TrainedFSLNet_resnet18.mat','fslNet');
elseif strcmpi(input1,'rc')
      load('resnetfsl_cosine.m','fslNet');
elseif strcmpi(input1,'g')
       load('TrainedFSLNet_googlenet.mat','fslNet');
elseif strcmpi(input1,'gf')
       load('TrainedFSLNet_googlenetf.mat','fslNet');
else
     load('hybridNET.mat','fardnet');
     fslNet=fardnet;
end    


% load("hybridNET.mat",'fardnet');
% fslNet=fardnet;

imdsRaw=imageDatastore("novel\","IncludeSubfolders",true,"LabelSource","foldernames");
  

count=1;
numepoch=100;

availableClasses=unique(imdsRaw.Labels);
numClass=numel(availableClasses);


nWay=5;
kShot=5; % 2 img TO LEARN
qQuery=10; % 5 IMAGES TO TEST   
numEpisode=5;
ipSize=[224 224 3];
totalepisode=numEpisode*numepoch;
allacc=zeros(totalepisode,1);

accArray=zeros(numEpisode,1);
fprintf('Pre-loading and resizing novel dataset...\n');
numImages = numel(imdsRaw.Files);
allImages = zeros(224, 224, 3, numImages, 'single'); % Pre-allocate 4D array
allLabels = imdsRaw.Labels;

for i = 1:numepoch
    img = readimage(imdsRaw, i);
    % Resize immediately to 224x224 to save RAM
    imgResized = imresize(img, [224 224]);
    if size(imgResized, 3) == 1
        imgResized = cat(3, imgResized, imgResized, imgResized);
    end
    allImages(:,:,:,i) = single(imgResized);
end
fprintf('Load complete. Testing %d images...\n', numImages);

for slime=1:100

for i=1:numEpisode

    [sIdx,qIdx,sLabels,qLabels]=episodemakerTest(allLabels,nWay,kShot,qQuery);
    [dlSupport,dlQuery]=dlarrayMakerTest(allImages,sIdx,qIdx,sLabels,qLabels);

    sEmbedRaw=predict(fslNet,dlSupport.Images);
    qEmbedRaw=predict(fslNet,dlQuery.Images);

    % etract data makes math doable

    sEmbed=extractdata(sEmbedRaw)';
    qEmbed=extractdata(qEmbedRaw)';

    % proto type
    [~,dim]=size(sEmbed);
    sEmbedrs=reshape(sEmbed,kShot,nWay,dim);
    prototypes=reshape(mean(sEmbedrs,1),nWay,dim);

    dist=pdist2(qEmbed,prototypes,'euclidean');

    % p2dist can be used because NN is not updating based on these
    % distances

    [~,preds]=min(dist,[],2);

    trueLabels=repmat(1:nWay,qQuery,1);
    trueLabels=trueLabels(:);

    allacc(count)=mean(preds==trueLabels);
    % Calculate accuracy for the current episode
    accArray(i)=mean(preds==trueLabels);
    count=count+1;

end


%fprintf('final accuracy %.2f\n',mean(accArray)*100);

end



meanAcc = mean(allacc) * 100;
stdDev = std(allacc) * 100;

fprintf('\n--- TEST RESULTS ---\n');
fprintf('Total Episodes: %d\n', totalepisode);
fprintf('Mean Accuracy:  %.2f%%\n', meanAcc);
fprintf('Standard Dev:   %.2f%%\n', stdDev);
fprintf('--------------------\n');
