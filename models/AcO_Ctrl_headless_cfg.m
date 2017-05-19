%open agws_diff_fastTT.slx
addpath StateSpace/V1.4/
%% CONFIG PARAMETERS
simulation_duration = 10;
STATE_SPACE_MDL     = 353;
WIND_LOAD_CASE      = 2;
N_M1_SEGMENT_PATCH  = 4;
INITIAL_CONDITION   = false;
metrics_file = sprintf('metrics_MDL%d_CASE%d_PATCH%d.mat',STATE_SPACE_MDL,WIND_LOAD_CASE,N_M1_SEGMENT_PATCH);
%% MODEL
%open_system('AcO_Ctrl_Geometric_R2016a.slx')
open_system('AcO_Ctrl_Geometric_headless_R2016a.slx','loadonly')
set_param('AcO_Ctrl_Geometric_headless_R2016a/On-axis imager/To File','Filename',metrics_file)
%% STATE SPACE SETUP
switch STATE_SPACE_MDL
    case 331
        load('StateSpace/Model_3pt3a_variables.mat')
    case 352
        load('StateSpace/Model3pt5pt02_variables.mat')
    case 353
        load('StateSpace/Model3pt5pt03_variables.mat')
end
if WIND_LOAD_CASE>0
    addpath StateSpace/windloads/
    M1CellLoads   = M1CellLoadTimeSeries01(    WIND_LOAD_CASE,simulation_duration);
    switch N_M1_SEGMENT_PATCH
        case 1
            M1MirrorLoads = M1LoadTimeSeries01(WIND_LOAD_CASE,simulation_duration);
        case 4
            M1MirrorLoads = M1LoadTimeSeries02(WIND_LOAD_CASE,simulation_duration);
    end
    TrussLoads    = TrussLoadTimeSeries02(     WIND_LOAD_CASE,simulation_duration);
    M2Loads       = M2LoadTimeSeries01(        WIND_LOAD_CASE,simulation_duration);
%     figure(102)
%     subplot(2,2,1)
%     plot(M1CellLoads.time,M1CellLoads.signals.values)
%     xlabel('Time [s]')
%     ylabel('Force [N]')
%     subplot(2,2,2)
%     plot(M1MirrorLoads.time,M1MirrorLoads.signals.values)
%     xlabel('Time [s]')
%     ylabel('Force [N]')
%     subplot(2,2,3)
%     plot(M2Loads.time,M2Loads.signals.values)
%     xlabel('Time [s]')
%     ylabel('Force [N]')
%     subplot(2,2,4)
%     plot(TrussLoads.time,TrussLoads.signals.values)
%     xlabel('Time [s]')
%     ylabel('Force [N]')
end
%% SIMCEO PARAMETERS
% Sampling frequency
fs = 500; % Hz
Ts = 1/fs;
% M1 update rate
M1_Ts = 0.1;
M1_gain = 0.5;

tt7_calibrations = struct('method_id','calibrate',...
                      'args',struct('mirror','M2','mode','segment tip-tilt','stroke',1e-6));
tt7_calib = struct('calibrations',struct('M2_TT',tt7_calibrations),...
                   'filename','TT7',...
                   'pseudo_inverse',struct('nThreshold',[]));

gs_zen = ones(1,3)*6*60;
gs_azi = (0:2)*120;

calibrations = struct('method_id','AGWS_calibrate',...
                      'args',struct('decoupled',true,'fluxThreshold',0.5));
gs_calib = struct('calibrations',struct('AGWS_D',calibrations),...
                  'filename','AcO_WFS',...
                  'pseudo_inverse',struct('nThreshold',[ones(1,6)*2,0],...
                                          'insertZeros',{{[],[],[],[],[],[],[2,4,6]}}));
gs_calib_size = 6*14 + 7*42;
M1_n_mode = 42;

on_axis_src_outputs = {{'wfe_rms',1},...
    {'segment_wfe_rms',7},...
    {'segment_piston',7},...
    {'tiptilt',2},...
    {'segment_tiptilt',14}};
%% INITIAL CONDITIONS

if INITIAL_CONDITION
    
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
    
%     figure(101)
%     clf
%     
%     subplot('Position',[0.05,0.55,0.125,0.4])
%     imagesc(M1_Txyz0(:,1:2)*1e6)
%     colorbar()
%     set(gca,'xtick',[],'ytick',(1:7))
%     ylabel('M1 Segment #')
%     title('Txy')
%     
%     subplot('Position',[0.175,0.55,0.1,0.4])
%     imagesc(M1_Txyz0(:,3)*1e6)
%     ylabel(colorbar(),'micron')
%     set(gca,'xtick',[],'ytick',[])
%     title('Tz')
%     
%     subplot('Position',[0.3,0.55,0.125,0.4])
%     imagesc(M1_Rxyz0(:,1:2)/arcs2rad)
%     colorbar()
%     set(gca,'xtick',[],'ytick',[])
%     title('Rxy')
%     
%     subplot('Position',[0.425,0.55,0.1,0.4])
%     imagesc(M1_Rxyz0(:,3)/arcs2rad)
%     ylabel(colorbar(),'arcsec')
%     set(gca,'xtick',[],'ytick',[])
%     title('Rz')
%     
%     subplot('Position',[0.55,0.55,0.4,0.4])
%     imagesc(M1_BeMo_d*1e6)
%     ylabel(colorbar(),'micron')
%     set(gca,'xtick',[],'ytick',[])
%     title('Be.Mo.')
%     
%     subplot('Position',[0.05,0.05,0.125,0.4])
%     imagesc(M2_Txyz0(:,1:2)*1e6)
%     colorbar()
%     set(gca,'xtick',[],'ytick',(1:7))
%     ylabel('M2 Segment #')
%     title('Txy')
%     
%     subplot('Position',[0.175,0.05,0.1,0.4])
%     imagesc(M2_Txyz0(:,3)*1e6)
%     ylabel(colorbar(),'micron')
%     set(gca,'xtick',[],'ytick',[])
%     title('Tz')
%     
%     subplot('Position',[0.3,0.05,0.125,0.4])
%     imagesc(M2_Rxyz0(:,1:2)/arcs2rad)
%     colorbar()
%     set(gca,'xtick',[],'ytick',[])
%     title('Rxy')
%     
%     subplot('Position',[0.425,0.05,0.1,0.4])
%     imagesc(M2_Rxyz0(:,3)/arcs2rad)
%     ylabel(colorbar(),'arcsec')
%     set(gca,'xtick',[],'ytick',[])
%     title('Rz')
    
else
    
    M1_RiBo_d = zeros(7,6);
    M2_RiBo_d = zeros(7,6);
    M1_BeMo_d = zeros(7,M1_n_mode);
    
end
