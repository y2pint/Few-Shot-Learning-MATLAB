function lgraph=freezeFirstPercent(lgraph,percent)
    layers=lgraph.Layers;
    numLayers=numel(layers);
    freezeIdx=round(numLayers * percent);
    
    for i=1:freezeIdx
        layer=layers(i);
     
        % Check if the layer is a "Learnable" type (Weights/Biases)
        if isa(layer, 'nnet.cnn.layer.LearnableLayer')
            layer.WeightLearnRateFactor=0;
            layer.BiasLearnRateFactor=0 ;
            
            lgraph=replaceLayer(lgraph,layer.Name,layer);
        end
    end
end
