% Main for the experiment over 8 realizations of corner
% Intensity format and L=1 for simulated SAR images
% This demo shows results for kuan filter (5x5 window)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (c) 2013 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
% All rights reserved.
% this software should be used, reproduced and modified only for informational and nonprofit purposes.
% 
% By downloading and/or using any of these files, you implicitly agree to all the
% terms of the license, as specified in the document LICENSE.txt
% (included in this package) and online at
% http://www.grip.unina.it/download/LICENSE_OPEN.txt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

load corner_clean;
load corner_noisy;

for i=1:8, 
    y(:,:,i) = z(:,:,i);
    [C_NN(i), C_BG(i)] = corner_measures(x,y(:,:,i),z(:,:,i));
end

format compact;
disp(sprintf('C_NN = %0.3g', mean(C_NN)));
disp(sprintf('C_BG = %0.4g', mean(C_BG)));


% display
figure(2); 
subplot(1,3,1); imshow(sqrt(x),[0 4]); title('clean');
subplot(1,3,2); imshow(sqrt(y(:,:,1)),[0 4]); title('filtered');
subplot(1,3,3); imshow(sqrt(z(:,:,1)),[0 4]); title('noisy');


%Generating the images to process
maximum = max(max(max(y)));

I=(y/maximum)*255;
%imwrite(I(:,:,1),"Corner1.png")
%imwrite(I(:,:,2),"Corner2.png")
%imwrite(I(:,:,3),"Corner3.png")
%imwrite(I(:,:,4),"Corner4.png")
%imwrite(I(:,:,5),"Corner5.png")
%imwrite(I(:,:,6),"Corner6.png")
%imwrite(I(:,:,7),"Corner7.png")
%imwrite(I(:,:,8),"Corner8.png")

% this one is the real values of the image when saving the image values it scales to 255
% Reading the images
A = imread("CornerSimpleNLCNN.png");
A = double(A);
B = 255*ones(256,256);
X = A./B;
C = maximum*ones(256,256);
D =X.*C;
K = I(:,:,1);% To recover the integrity of the image because the imwrite is bad for doubles
KB = K <= 1;
KBDN= not(KB);
KBD = KB*255;
KBDT = KBD+KBDN;
Corrected = D./KBDT;
imshow(Corrected,[0 4]);
[C_n, C_b]= corner_measures(x,Corrected,z(:,:,1));
disp(sprintf('C_NN = %0.3g', mean(C_n)));
disp(sprintf('C_BG = %0.4g', mean(C_b)));


A = imread("CornerNLCNN.png");
A = double(A);
B = 255*ones(256,256);
X = A./B;
C = maximum*ones(256,256);
D =X.*C;
K = I(:,:,1);% To recover the integrity of the image because the imwrite is bad for doubles
KB = K <= 1;
KBDN= not(KB);
KBD = KB*255;
KBDT = KBD+KBDN;
Corrected = D./KBDT;
imshow(Corrected,[0 4]);
[C_n, C_b]= corner_measures(x,Corrected,z(:,:,1));
disp(sprintf('C_NN = %0.3g', mean(C_n)));
disp(sprintf('C_BG = %0.4g', mean(C_b)));


A = imread("CornerSARCNN/Corner1f_vh.png");
A = double(A);
B = 255*ones(256,256);
X = A./B;
C = maximum*ones(256,256);
D =X.*C;
K = I(:,:,1);% To recover the integrity of the image because the imwrite is bad for doubles
KB = K <= 1;
KBDN= not(KB);
KBD = KB*255;
KBDT = KBD+KBDN;
Corrected = D./KBDT;
imshow(Corrected,[0 4]);
[C_n, C_b]= corner_measures(x,Corrected,z(:,:,1));
disp(sprintf('C_NN = %0.3g', mean(C_n)));
disp(sprintf('C_BG = %0.4g', mean(C_b)));


A = imread("Lee/Corner1_Lee.png");
A = double(A);
B = 255*ones(256,256);
X = A./B;
C = maximum*ones(256,256);
D =X.*C;
K = I(:,:,1);% To recover the integrity of the image because the imwrite is bad for doubles
KB = K <= 1;
KBDN= not(KB);
KBD = KB*255;
KBDT = KBD+KBDN;
Corrected = D./KBDT;
imshow(Corrected,[0 4]);
[C_n, C_b]= corner_measures(x,Corrected,z(:,:,1));
disp(sprintf('C_NN = %0.3g', mean(C_n)));
disp(sprintf('C_BG = %0.4g', mean(C_b)));

A = imread("Leerefined/Corner1_LeeRefined.png");
A = double(A);
B = 255*ones(256,256);
X = A./B;
C = maximum*ones(256,256);
D =X.*C;
K = I(:,:,1);% To recover the integrity of the image because the imwrite is bad for doubles
KB = K <= 1;
KBDN= not(KB);
KBD = KB*255;
KBDT = KBD+KBDN;
Corrected = D./KBDT;
imshow(Corrected,[0 4]);
[C_n, C_b]= corner_measures(x,Corrected,z(:,:,1));
disp(sprintf('C_NN = %0.3g', mean(C_n)));
disp(sprintf('C_BG = %0.4g', mean(C_b)));

A = imread("LeeSigma/Corner1_LeeSigma.png");
A = double(A);
B = 255*ones(256,256);
X = A./B;
C = maximum*ones(256,256);
D =X.*C;
K = I(:,:,1);% To recover the integrity of the image because the imwrite is bad for doubles
KB = K <= 1;
KBDN= not(KB);
KBD = KB*255;
KBDT = KBD+KBDN;
Corrected = D./KBDT;
imshow(Corrected,[0 4]);
[C_n, C_b]= corner_measures(x,Corrected,z(:,:,1));
disp(sprintf('C_NN = %0.3g', mean(C_n)));
disp(sprintf('C_BG = %0.4g', mean(C_b)));