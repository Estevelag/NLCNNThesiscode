function [ES_up, ES_down, Fmax] = squares_measures(x,y,z,config)
%squares_measures is the version 0.1 to evaluate 
% performance on squares SAR simulated images as described in 
% "Benchmarking framework for SAR despeckling", written by
% G. Di Martino, M. Poderico, G. Poggi, D.Riccio and L. Verdoliva, 
% IEEE Trans. on Geoscience and Remote Sensing, in press, 2013.
% Please cite this paper when using these measures.
%
% squares measures with normal configuration
% [ES_up, ES_down, Fmax] = squares_measures(x,y,z,'normal')
% use the following configuration to obtain exactly the same results as in the paper
%
% squares measures with fast configuration
% [ES_up, ES_down, Fmax] = squares_measures(x,y,z,'fast')
% use the following configuration to obtain results close to those of the paper in a much shorter time
%
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

W = 16;
figure(1); 
med1 = mean(x(1:253,:)); plot([-W:1/6:W],interp1([256-W:256+W],med1(256-W:256+W),[256-W:1/6:256+W],'spline'),'k-','LineWidth',2); 
hold on;
fil1 = mean(y(1:253,:)); plot([-W:1/6:W],interp1([256-W:256+W],fil1(256-W:256+W),[256-W:1/6:256+W],'spline'),'r-','LineWidth',2);  
hold off;
axis([-8 +8 0.4 1.2]);
title('edge profile');

med2 = mean(x(260:512,:)); 
fil2 = mean(y(260:512,:)); 

a = 2;
s = [1:512];
si = [1:1/a:512];
m = 255.5; sigma = 2; 
gauss_win = exp(-(s-m).^2/(2*(sigma^2)));
gauss_win_i = interp1(s,gauss_win,si,'linear');

% Edge Smearing up 
ES_up=sum(gauss_win_i.*((interp1(s,fil1,si,'linear')-interp1(s,med1,si,'linear')).^2))/a;
% Edge Smearing down
ES_down=sum(gauss_win_i.*((interp1(s,fil2,si,'linear')-interp1(s,med2,si,'linear')).^2))/a;

% FOM (Figure Of Merit)
% ideal edge-map
map_clean = zeros(size(x));
map_clean(255,:) = 1;
map_clean(:,255) = 1;
map_clean = uint8(map_clean);

% experiment varying threshold and sigma  
if nargin<4,
    thr   = [0.2:0.01:0.5];
    sigma = [7:22];
elseif strcmpi(config,'fast'),
    thr   = [0.2:0.1:0.5];
    sigma = [7:2:21];
else
    thr   = [0.2:0.01:0.5];
    sigma = [7:22];
end;

for i=1:length(sigma),
  for j=1:length(thr),
      map = edge(y,'canny',thr(j),sigma(i));
      map = uint8(map);
      F(i,j) = FOM(map_clean, map);
   end
end   
Fmax = max(F(:));
[r c] = find(F == Fmax);
tmax = thr(c(1));
smax = sigma(r(1));  

figure(2);
map = edge(y(:,:,1),'canny',tmax(1),smax(1));
imshow(map);

function FOM_index = FOM(Ref,Det)

% Sintax: FOM_index = FOM(Ref,Det)
%
% Computes Pratt's Figure Of Merit
% Det: detected edge map
% Ref: reference edge map
% map values are 0 (non-edge) or not 0 (edge)
gamma = 1/9;

if sum(Det(:))
   [mr nr] = find(Ref);    % coordinates of edge pixels in reference map
   [md nd] = find(Det);    % coordinates of edge pixels in detected map
   Nr = length(nr);        % number of true edge pixels
   Nd = length(nd);        % number of detected edge pixels
   % computes square distances between detected edge and nearest edge in reference map
   for i=1:Nd,
      d(i) = min((mr-md(i)).^2+(nr-nd(i)).^2);
   end
   % computes FOM: position errors are weighted by gamma
   Nm    = max(Nd,Nr);
   FOM_index = sum(1./(1+gamma*d))/Nm;   
else
   FOM_index = 0;
end