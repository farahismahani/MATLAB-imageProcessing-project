load("trainedBilateralFilterNet_v2.mat");
    testImage = "image.png";
Ireference = imread(testImage);
Ireference = im2uint8(Ireference);

imshow(Ireference)
title("Pristine Reference Image")

Inoisy = imnoise(Ireference,"gaussian",0.00001);
imshow(Inoisy)
title("Noisy Image")

degreeOfSmoothing = var(double(Inoisy(:)));
Ibilat = imbilatfilt(Inoisy,degreeOfSmoothing);
imshow(Ibilat)
title("Denoised Image Obtained Using Bilateral Filtering")

subplot(1, 3, 1), imshow(Ireference), title('Original Image');
subplot(1, 3, 2), imshow(Inoisy), title('Noisy Image');
subplot(1, 3, 3), imshow(Ibilat), title('Denoised Image with Bilateral Filtering');
