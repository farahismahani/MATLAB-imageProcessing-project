dataDir = tempdir;
downloadIAPRTC12Data(dataDir);

trainImagesDir = fullfile(dataDir,"iaprtc12","images","39");
exts = [".jpg",".bmp",".png"];
pristineImages = imageDatastore(trainImagesDir,FileExtensions=exts);

numel(pristineImages.Files)

preprocessDataDir = fullfile(trainImagesDir,"preprocessedDataset");

bilateralFilterDataset(pristineImages,preprocessDataDir);

bilatFilteredImages = imageDatastore(preprocessDataDir,FileExtensions=exts);

miniBatchSize = 1;
patchSize = [256 256];
dsTrain = randomPatchExtractionDatastore(pristineImages,bilatFilteredImages, ...
    patchSize,PatchesPerImage=1);
dsTrain.MiniBatchSize = miniBatchSize;

inputBatch = read(dsTrain);
disp(inputBatch)

networkDepth = 10;
numberOfFilters = 32;
firstLayer = imageInputLayer([256 256 3],Name="InputLayer",Normalization="none");

Wgts = zeros(3,3,3,numberOfFilters); 
for ii = 1:3
    Wgts(2,2,ii,ii) = 1;
end
convolutionLayer = convolution2dLayer(3,numberOfFilters,Padding=1, ...
    Weights=Wgts,Name="Conv1");

batchNorm = batchNormalizationLayer(Name="BN1");
adaptiveMu = adaptiveNormalizationMu(numberOfFilters,"Mu1");
addLayer = additionLayer(2,Name="add1");
leakyrelLayer = leakyReluLayer(0.2,Name="Leaky1");

midLayers = [convolutionLayer batchNorm adaptiveMu addLayer leakyrelLayer];
    
Wgts = zeros(3,3,numberOfFilters,numberOfFilters);
for ii = 1:numberOfFilters
    Wgts(2,2,ii,ii) = 1;
end
    
for layerNumber = 2:networkDepth-2
    dilationFactor = 2^(layerNumber-1);
    padding = dilationFactor;
    conv2dLayer = convolution2dLayer(3,numberOfFilters, ...
        Padding=padding,DilationFactor=dilationFactor, ...
        Weights=Wgts,Name="Conv"+num2str(layerNumber));
    batchNorm = batchNormalizationLayer(Name="BN"+num2str(layerNumber));
    adaptiveMu = adaptiveNormalizationMu(numberOfFilters,"Mu"+num2str(layerNumber));
    addLayer = additionLayer(2,Name="add"+num2str(layerNumber));
    leakyrelLayer = leakyReluLayer(0.2,Name="Leaky"+num2str(layerNumber));
    midLayers = [midLayers conv2dLayer batchNorm adaptiveMu addLayer leakyrelLayer];    
end

conv2dLayer = convolution2dLayer(3,numberOfFilters, ...
    Padding=1,Weights=Wgts,Name="Conv9");

batchNorm = batchNormalizationLayer(Name="AN9");
adaptiveMu = adaptiveNormalizationMu(numberOfFilters,"Mu9");
addLayer = additionLayer(2,Name="add9");
leakyrelLayer = leakyReluLayer(0.2,Name="Leaky9");
midLayers = [midLayers conv2dLayer batchNorm adaptiveMu addLayer leakyrelLayer];

Wgts = sqrt(2/(9*numberOfFilters))*randn(1,1,numberOfFilters,3);
finalLayer = convolution2dLayer(1,3,NumChannels=numberOfFilters, ...
    Weights=Wgts,Name="Conv10");

layers = [firstLayer midLayers finalLayer];

opNet = dlnetwork;
opNet = addLayers(opNet,layers);

skipConv1 = adaptiveNormalizationLambda(numberOfFilters,"Lambda1");
skipConv2 = adaptiveNormalizationLambda(numberOfFilters,"Lambda2");
skipConv3 = adaptiveNormalizationLambda(numberOfFilters,"Lambda3");
skipConv4 = adaptiveNormalizationLambda(numberOfFilters,"Lambda4");
skipConv5 = adaptiveNormalizationLambda(numberOfFilters,"Lambda5");
skipConv6 = adaptiveNormalizationLambda(numberOfFilters,"Lambda6");
skipConv7 = adaptiveNormalizationLambda(numberOfFilters,"Lambda7");
skipConv8 = adaptiveNormalizationLambda(numberOfFilters,"Lambda8");
skipConv9 = adaptiveNormalizationLambda(numberOfFilters,"Lambda9");

opNet = addLayers(opNet,skipConv1);
opNet = connectLayers(opNet,"Conv1","Lambda1");
opNet = connectLayers(opNet,"Lambda1","add1/in2");

opNet = addLayers(opNet,skipConv2);
opNet = connectLayers(opNet,"Conv2","Lambda2");
opNet = connectLayers(opNet,"Lambda2","add2/in2");

opNet = addLayers(opNet,skipConv3);
opNet = connectLayers(opNet,"Conv3","Lambda3");
opNet = connectLayers(opNet,"Lambda3","add3/in2");

opNet = addLayers(opNet,skipConv4);
opNet = connectLayers(opNet,"Conv4","Lambda4");
opNet = connectLayers(opNet,"Lambda4","add4/in2");

opNet = addLayers(opNet,skipConv5);
opNet = connectLayers(opNet,"Conv5","Lambda5");
opNet = connectLayers(opNet,"Lambda5","add5/in2");

opNet = addLayers(opNet,skipConv6);
opNet = connectLayers(opNet,"Conv6","Lambda6");
opNet = connectLayers(opNet,"Lambda6","add6/in2");

opNet = addLayers(opNet,skipConv7);
opNet = connectLayers(opNet,"Conv7","Lambda7");
opNet = connectLayers(opNet,"Lambda7","add7/in2");

opNet = addLayers(opNet,skipConv8);
opNet = connectLayers(opNet,"Conv8","Lambda8");
opNet = connectLayers(opNet,"Lambda8","add8/in2");

opNet = addLayers(opNet,skipConv9);
opNet = connectLayers(opNet,"Conv9","Lambda9");
opNet = connectLayers(opNet,"Lambda9","add9/in2");

opNet = initialize(opNet,dlarray(single(inputBatch.InputImage{1}),"SSCB"));

deepNetworkDesigner(opNet)

maxEpochs = 181;
initLearningRate = 0.0001;
miniBatchSize = 1;

options = trainingOptions("adam", ...
    InitialLearnRate=initLearningRate, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    Plots="training-progress", ...
    Verbose=false);




doTraining = true;
if doTraining
    net = trainnet(dsTrain,opNet,"mean-squared-error",options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save("trainedBilateralFilterNet_v2"+modelDateTime+".mat","net");
else
    load("trainedBilateralFilterNet_v2.mat");
end