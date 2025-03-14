% load the pretrained CAN
load("trainedBilateralFilterNet_v2.mat");
% TestImages
testImage = "image.png";
testImage1 = "image1.png";

Ireference = imread(testImage);
Ireference = im2uint8(Ireference);
Inoisy = imnoise(Ireference,"gaussian",0.00001); 
InoisyDL = dlarray(single(Inoisy),"SSCB");
IapproxDL = predict(net,InoisyDL);
Iapprox = extractdata(IapproxDL);
Iapprox = rescale(Iapprox);
Iapprox = im2uint8(Iapprox);
% Display result
figure;
subplot(1, 2, 1), imshow(Inoisy), title('Noisy Image');
subplot(1, 2, 2), imshow(Iapprox), title('Denoised Image with Multiscale CAN');

% TestImage1
Ireference = imread(testImage1);
Ireference = im2uint8(Ireference);
Inoisy1 = imnoise(Ireference,"gaussian",0.00001); 

InoisyDL = dlarray(single(Inoisy1),"SSCB");
IapproxDL = predict(net,InoisyDL);
Iapprox1 = extractdata(IapproxDL);
Iapprox1 = rescale(Iapprox1);
Iapprox1 = im2uint8(Iapprox1);

% Display result
figure;
subplot(1, 2, 1), imshow(Inoisy1), title('Noisy Image');
subplot(1, 2, 2), imshow(Iapprox1), title('Denoised Image with Multiscale CAN');

