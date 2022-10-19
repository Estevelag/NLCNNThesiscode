% Main for the experiment over 8 realizations of squares
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

load squares_clean;
load squares_noisy;

%for i=1:8,
   %y(:,:,i) = z(:,:,i),1,2;
%   [ES_up(i), ES_down(i), Fmax(i)] = squares_measures(x,y(:,:,i),z(:,:,i),'fast');
%end

%format compact;
%disp(sprintf('ES_up = %0.2g', mean(ES_up)));
%disp(sprintf('ES_down = %0.3g', mean(ES_down)));
%disp(sprintf('FOM = %0.3g', mean(Fmax)));

figure(3);
subplot(1,3,1); imshow(sqrt(x(:,:,1)),[0 4]); title('clean');
subplot(1,3,2); imshow(sqrt(z(:,:,1)),[0 4]); title('filtered');
subplot(1,3,3); imshow(sqrt(z(:,:,1)),[0 4]); title('noisy');

%Generating the images to process
maximum = max(max(max(z)));
W = z(:,:,1);
I=(z/maximum);
%imwrite(I(:,:,1),"Squares1.png")
%imwrite(I(:,:,2),"Squares2.png")
%imwrite(I(:,:,3),"Squares3.png")
%imwrite(I(:,:,4),"Squares4.png")
%imwrite(I(:,:,5),"Squares5.png")
%imwrite(I(:,:,6),"Squares6.png")
%imwrite(I(:,:,7),"Squares7.png")
%imwrite(I(:,:,8),"Squares8.png")


A = imread("Squares1.png");
A = double(A);
B = 255.0*ones(512,512);
A = A./B;
C = maximum*ones(512,512);
Corrected = A.*C;


imshow(Corrected,[0 4]);
[ES_up, ES_down, Fmax] = squares_measures(x,Corrected,z(:,:,1),'fast');
format compact;
disp(sprintf('ES_up = %0.2g', ES_up));
disp(sprintf('ES_down = %0.3g', mean(ES_down)));
disp(sprintf('FOM = %0.3g', mean(Fmax)));

