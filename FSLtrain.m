clc;
clear all;
close all;
                                                                        %
 %                           FSL: TRAINING                               
                                                                        
                                                                     
prompt='googlenet or resnet18';
input1=input(prompt,'s');


if strcmpi(input1,'r')
    net = resnet18;
    lgraph=layerGraph(net);
    replacingLayers={'fc1000', 'prob', 'ClassificationLayer_predictions'};
    lgraph=removeLayers(lgraph,replacingLayers);
    embeddingSize=128;
    fcEmbedding=fullyConnectedLayer(embeddingSize,Name="FC_embedding");
    dropLayer=dropoutLayer(0.4,Name="dropout_layer"); 
    lgraph=addLayers(lgraph, dropLayer);
    lgraph=addLayers(lgraph,fcEmbedding);
    lgraph=connectLayers(lgraph,'pool5','dropout_layer');
    lgraph=connectLayers(lgraph,'dropout_layer','FC_embedding');
    fslNet=dlnetwork(lgraph);
 
    learnRate=0.0001;
    LRdecay=0.3; % the percentage at which the learn rate is multiplied
    lrdecayfreq=15;

elseif strcmpi(input1,'rf')
    net = resnet18;
    lgraph=layerGraph(net);
    lgraph=freezeFirstPercent(lgraph,0.5);
    replacingLayers={'fc1000', 'prob', 'ClassificationLayer_predictions'};
    lgraph=removeLayers(lgraph,replacingLayers);
    embeddingSize=128;
    fcEmbedding=fullyConnectedLayer(embeddingSize,Name="FC_embedding"); 
    dropLayer=dropoutLayer(0.4,name="dropout_layer"); 
    lgraph=addLayers(lgraph, dropLayer);
    lgraph=addLayers(lgraph,fcEmbedding);
    lgraph=connectLayers(lgraph,'pool5','dropout_layer');
    lgraph=connectLayers(lgraph,'dropout_layer','FC_embedding');
    fslNet=dlnetwork(lgraph);


    learnRate=0.0002;
    LRdecay=0.2; % the percentage at which the learn rate is multiplied
    lrdecayfreq=25;

elseif strcmpi(input1,'g')
    net = googlenet; % loads pre-trained googlenet architecture
    lgraph=layerGraph(net); % converts to layer graph so can edit
    replacingLayers={'loss3-classifier','prob','output'}; % defines layers to be dleted
    lgraph=removeLayers(lgraph,replacingLayers); % remove layers
    embeddingSize=128;
    fcEmbedding=fullyConnectedLayer(embeddingSize,Name="FC_embedding");
    lgraph=addLayers(lgraph,fcEmbedding);
    lgraph=connectLayers(lgraph,'pool5-drop_7x7_s1','FC_embedding');
    fslNet=dlnetwork(lgraph);
    
    learnRate=0.00005;
    LRdecay=0.5; % the percentage at which the learn rate is multiplied
    lrdecayfreq=40;
   % warmupLearnRate=0.000005;

elseif strcmpi(input1,'gf')
    net = googlenet; % loads pre-trained googlenet architecture
    lgraph=layerGraph(net); % converts to layer graph so can edit
    lgraph=freezeFirstPercent(lgraph,0.5);
    replacingLayers={'loss3-classifier','prob','output'}; % defines layers to be dleted
    lgraph=removeLayers(lgraph,replacingLayers); % remove layers
    embeddingSize=128;
    fcEmbedding=fullyConnectedLayer(embeddingSize,Name="FC_embedding");
    lgraph=addLayers(lgraph,fcEmbedding);
    lgraph=connectLayers(lgraph,'pool5-drop_7x7_s1','FC_embedding');
    fslNet=dlnetwork(lgraph);
    
    learnRate=0.0001;
    LRdecay=0.5; % the percentage at which the learn rate is multiplied
    lrdecayfreq=20;
else
    pause(1)
    fprintf('\nno networks created\n')
    pause(2)
    fprintf('initiating backup protocol\n')
    pause(1)
     g=3;
     A=1000;
     pause(1);
    for i=1:3
    n=1;
    fprintf('\n%f',g);
    g=g-1;  
    pause(n);
    end
    fprintf('\n\n its over for you :)')
    pause(3)
    while true
    l=67
    x=sillyfunction(A);
    end
end    

fprintf('network architecture has been constructed\n\n');



imdsAll=imageDatastore('base\',IncludeSubfolders=true,LabelSource='foldernames');                             

augOptions=imageDataAugmenter("RandXReflection",true,"RandYReflection",true,"RandRotation",[-90,90]);

className=categories(imdsAll.Labels);
baseClass=(className(:));

idxBase=ismember(imdsAll.Labels,baseClass);  
imdsBase=subset(imdsAll, idxBase);

fprintf('have to preload images into ram rq\n');
numImages=numel(imdsBase.Files);
baseIMG=cell(numImages, 1);

for i = 1:numImages
    img=readimage(imdsBase, i);
    baseIMG{i}=imresize(img, [224 224]); 
end
fprintf('pre-load complete. starting training...\n');
inputSize=[224 224 3];


baseLabels=imdsBase.Labels;
baseLabels=removecats(baseLabels);                                      


                       


%                      training parameters + monitering


nWay=7; % The number of classes
kShot=5; % The number of support images per class
qQuery=10;% The number of query images per class


numEpochs=50; 
iterationsPerEpoch=10;



trailAvg=[];
trailAvgsq=[];

monitor=trainingProgressMonitor( ...
    Metrics=["Loss","Accuracy"], ... 
    Info=["Epoch","LearnRate","StepTime"], ... 
    XLabel="Iteration");

iteration=0; 


%                        training loop


for epoch=1:numEpochs
   
    if mod(epoch,lrdecayfreq)==0
        learnRate=learnRate*LRdecay; 
   end                           

    for iter=1:iterationsPerEpoch
        start=tic; 
        iteration=iteration+1;
        
  
        [supportData,queryData]=episodemakerTrain(baseIMG,baseLabels,augOptions,nWay,kShot,qQuery);
        [dlSupport,dlQuery]=dlarrayMakerTrain(supportData,queryData);

        [loss,gradient,acc]=dlfeval(@gradientsandloss,fslNet,dlSupport,dlQuery);

        [fslNet,trailAvg,trailAvgsq]=adamupdate(fslNet,gradient,trailAvg,trailAvgsq,iteration,learnRate);
    
        stop=toc(start);
        recordMetrics(monitor, iteration, ...
                Loss=extractdata(loss), ...
                Accuracy=extractdata(acc));
         
            updateInfo(monitor, ...
                Epoch=epoch, ...
                StepTime=string(stop) + "s");
        monitor.Progress=(iteration/(numEpochs*iterationsPerEpoch))*100;
    end
end

finalLoss = double(extractdata(gather(loss)));
finalAcc = double(extractdata(gather(acc))) * 100; 


fprintf('\nTRAINING COMPLETE\n');
fprintf('Final Training Loss:     %.4f\n', finalLoss);
fprintf('Final Training Accuracy: %.2f%%\n', finalAcc);


prompt='would you like to save the network (Y/N)';
input2=input(prompt,'s');

if strcmpi(input2,'y') && strcmpi(input1,'g')
    save('TrainedFSLNet_googlenet.mat', 'fslNet');
    fprintf('\nNetwork Saved\n')
elseif strcmpi(input2,'y') && strcmpi(input1,'gf')
    save('TrainedFSLNet_googlenetf.mat', 'fslNet');
    fprintf('\nNetwork Saved\n')
elseif strcmpi(input2,'y') && strcmpi(input1,'r')
    save('TrainedFSLNet_resnet18.mat', 'fslNet');
    fprintf('\nNetwork Saved\n')
elseif strcmpi(input2,'y') && strcmpi(input1,'rf')
    save('TrainedFSLNet_resnet18f.mat', 'fslNet');
    fprintf('\nNetwork Saved\n')
else
    fprintf('\nNetwork not saved\n');
end
