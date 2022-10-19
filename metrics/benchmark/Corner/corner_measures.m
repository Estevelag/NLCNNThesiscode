function [C_NN, C_BG] = corner_measures(x,y,z);
%corner_measures is the version 0.1 to evaluate 
% performance on corner SAR simulated images as described in 
% "Benchmarking framework for SAR despeckling", written by
% G. Di Martino, M. Poderico, G. Poggi, D.Riccio and L. Verdoliva, 
% IEEE Trans. on Geoscience and Remote Sensing, in press, 2013.
% Please cite this paper when using these measures.
%
% Corner measures 
% [C_NN, C_BG] = corner_measures(x,y,z)
% x: clean image, y: filtered image; z: noisy image (all intensity format)
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

% Contrast measures
y_CF = y(129,129);
NN = y(128:130,128:130);
NN(129,129) = 0;
y_NN = sum(NN(:))/8;
C_NN = 10*log10(y_CF/y_NN);

section1 = y(1:101,1:101);     m1 = mean2(section1);
section2 = y(156:256,1:101);   m2 = mean2(section2);
section3 = y(1:101,156:256);   m3 = mean2(section3);
section4 = y(156:256,156:256); m4 = mean2(section4);

BG = (m1+m2+m3+m4)/4;
C_BG = 10*log10(y_CF/BG);

% display of results
a = 6;
W = 20;
figure(1); 
med = log(x(129,:)); plot([-W:1/a:W],interp1([128-W:128+W], med(128-W:128+W),[128-W:1/a:128+W],'spline'),'k-','LineWidth',2); 
hold on;
fil = log(y(129,:)); plot([-W:1/a:W],interp1([128-W:128+W], fil(128-W:128+W),[128-W:1/a:128+W],'spline'),'r-','LineWidth',2);
hold off;
axis([-20 +20 -1 9]); 
