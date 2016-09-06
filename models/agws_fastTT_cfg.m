open models/agws_fastTT.slx

% Sampling frequency
fs = 200; % Hz
Ts = 1/fs;
% M1 update rate
M1_Ts = 0.1;

gs_zen = ones(1,3)*8*60;
gs_azi = (0:2)*120;

gs_calib = struct('M2TT',struct('mirror','M2','mode','segment tip-tilt','stroke',1e-6),...
    'AGWS_D',[...
    struct('mirror','M1','mode','Rxyz','stroke',1e-6),...
    struct('mirror','M2','mode','Rxyz','stroke',1e-6),...
    struct('mirror','M1','mode','Txyz','stroke',1e-6),...
    struct('mirror','M2','mode','Txyz','stroke',1e-6),...
    struct('mirror','M1','mode','bending modes','stroke',1e-6)]);
gs_calib_size = [14,20+20+20+21+42*7];

M1_n_mode = 42;

%% INITIAL CONDITIONS

M1_Txyz0      = randn(7,3)*75e-6;
M1_Txyz0(:,3) = randn(7,1)*160e-6;
M1_Txyz0(7,3) = 0; % M1 S1 Tz

M2_Txyz0      = randn(7,3)*75e-6;
M2_Txyz0(:,3) = randn(7,1)*170e-6;

arcs2rad = pi/180/3600;
M1_Rxyz0      = randn(7,3)*0.38*arcs2rad;
M1_Rxyz0(:,3) = randn(7,1)*40*arcs2rad;

M2_Rxyz0      = randn(7,3)*3.0*arcs2rad;
M2_Rxyz0(:,3) = randn(7,1)*330*arcs2rad;

M1_RiBo_d = [M1_Txyz0 M1_Rxyz0]; 
M2_RiBo_d = [M2_Txyz0 M2_Rxyz0];

radialOrders = cell2mat(arrayfun(@(x) ones(1,x+1)*x,0:8,'UniformOutput',false));
scale = 1.0./radialOrders(4:end);
M1_BeMo_d = 1e-6*bsxfun(@times,randn(7,M1_n_mode),scale);

figure(101)
clf

subplot('Position',[0.05,0.55,0.125,0.4])
imagesc(M1_Txyz0(:,1:2)*1e6)
colorbar()
set(gca,'xtick',[],'ytick',(1:7))
ylabel('M1 Segment #')
title('Txy')

subplot('Position',[0.175,0.55,0.1,0.4])
imagesc(M1_Txyz0(:,3)*1e6)
ylabel(colorbar(),'micron')
set(gca,'xtick',[],'ytick',[])
title('Tz')

subplot('Position',[0.3,0.55,0.125,0.4])
imagesc(M1_Rxyz0(:,1:2)/arcs2rad)
colorbar()
set(gca,'xtick',[],'ytick',[])
title('Rxy')

subplot('Position',[0.425,0.55,0.1,0.4])
imagesc(M1_Rxyz0(:,3)/arcs2rad)
ylabel(colorbar(),'arcsec')
set(gca,'xtick',[],'ytick',[])
title('Rz')

subplot('Position',[0.55,0.55,0.4,0.4])
imagesc(M1_BeMo_d*1e6)
ylabel(colorbar(),'micron')
set(gca,'xtick',[],'ytick',[])
title('Be.Mo.')

subplot('Position',[0.05,0.05,0.125,0.4])
imagesc(M2_Txyz0(:,1:2)*1e6)
colorbar()
set(gca,'xtick',[],'ytick',(1:7))
ylabel('M2 Segment #')
title('Txy')

subplot('Position',[0.175,0.05,0.1,0.4])
imagesc(M2_Txyz0(:,3)*1e6)
ylabel(colorbar(),'micron')
set(gca,'xtick',[],'ytick',[])
title('Tz')

subplot('Position',[0.3,0.05,0.125,0.4])
imagesc(M2_Rxyz0(:,1:2)/arcs2rad)
colorbar()
set(gca,'xtick',[],'ytick',[])
title('Rxy')

subplot('Position',[0.425,0.05,0.1,0.4])
imagesc(M2_Rxyz0(:,3)/arcs2rad)
ylabel(colorbar(),'arcsec')
set(gca,'xtick',[],'ytick',[])
title('Rz')

