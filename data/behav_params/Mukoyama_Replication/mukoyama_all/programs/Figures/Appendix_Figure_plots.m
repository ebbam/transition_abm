
% Description: Creates charts for fitted values of the ATUS time v. methods regression
% Created by:     Christina Patterson
% Date:           5/2013        


clear;
clc;
set(0,'DefaultFigurePaperPosition', [0.25 0.25 10.5 8]);
set(0, 'DefaultFigurePaperOrientation', 'landscape');

%% Figure A1 

cd '../../int_data/ATUS';
nonemp_base = csvread('Fig2a_data.csv', 1, 0); %This goes from 0 to include the year
unemp_base = csvread('Fig2b_data.csv', 1, 1); 
year = nonemp_base(:,1);
nonemp_base = nonemp_base(:,2:3);
unemp_travel = csvread('FigA1a_data.csv', 1, 1); 

cd '../CPS';
data = csvread('Figure3b_data.csv', 1, 1); 
time_unemp = data(:,3);
data = csvread('FigureA1b_data.csv', 1, 1); 
time_create_travel	   = data(:,2);
[R,C] = size(time_create_travel);
for i=1:R
 date(i)=1994 +(i-1)/12;
end

cd '../../Figures';
figure(1);
ha1 = area([2007+11/12-.5 2009.5-.5], [80 80]); grid on; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85]); set(ha1,{'LineStyle'}, {'none'});         
hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(year(2:end), unemp_travel(:,1), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(year(2:end), unemp_base(2:end,2), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top'); hold off; 
axis([2004 2014 10 50]); 
set(gca, 'YTick', [10:10:50], 'fontsize', 12);
set(gca, 'XTick', [2003:1:2014], 'fontsize', 12); grid off; box off;
print('FigureA1a','-dpdf','-r0')

figure(2);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, time_unemp, 'LineWidth', 2, 'LineStyle', '-', 'Marker', 'none', 'Color', 'r');
plot(date, time_create_travel, 'LineWidth', 2, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 20 55]);
set(gca, 'YTick', [20:5:55], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('FigureA1b','-dpdf','-r0')


%% Figure A2 

cd '../int_data/ATUS';
nonemp_base = csvread('Fig2a_data.csv', 1, 0);
unemp_base = csvread('Fig2b_data.csv', 1, 1); 
nonemp_day = csvread('FigA2a_data.csv', 1, 1); 
unemp_day = csvread('FigA2b_data.csv', 1, 1); 
year = nonemp_base(:,1);
nonemp_base = nonemp_base(:,2:3);

cd '../../Figures';
figure(3);
ha1 = area([2007+11/12-.5 2009.5-.5], [80 80]); grid on; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85]); set(ha1,{'LineStyle'}, {'none'});         
hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(year, unemp_base(:,2), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(year, unemp_base(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
plot(year, unemp_day(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'k');
set(gca, 'Layer', 'top'); hold off; 
axis([2004 2014 10 50]); 
set(gca, 'YTick', [10:10:50], 'fontsize', 12);
set(gca, 'XTick', [2003:1:2014], 'fontsize', 12); grid off; box off;
legend('Actual', 'Imputed- Baseline', 'Imputed with Interview Controls')
print('FigureA2b','-dpdf','-r0') 

figure(4);
ha1 = area([2007+11/12-.5 2009.5-.5], [80 80]); grid on; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85]); set(ha1,{'LineStyle'}, {'none'});         
hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(year, nonemp_base(:,2), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(year, nonemp_base(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
plot(year, nonemp_day(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'k');
set(gca, 'Layer', 'top'); hold off; 
axis([2004 2014 0 10]); 
set(gca, 'YTick', [0:2:10], 'fontsize', 12);
set(gca, 'XTick', [2004:1:2014], 'fontsize', 12); grid off; box off;
legend('Actual', 'Imputed- Baseline', 'Imputed with Interview Controls')
print('FigureA2a','-dpdf','-r0') 


%% Figure A3 
cd '../int_data/ATUS';

ols = csvread('FigureA3_data_ols.csv', 1, 1);
unemp_step2 = csvread('FigureA3b_data.csv', 1, 1); 
nonemp_step2 = csvread('FigureA3a_data.csv', 1, 1); 
cd '../../Figures';


figure(5);
ha1 = area([2007+11/12-.5 2009.5-.5], [80 80]); grid on; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85]); set(ha1,{'LineStyle'}, {'none'});         
hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(year, ols(:,1), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(year, ols(:,3), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'k');
plot(year, unemp_step2(:,1), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top'); hold off; 
axis([2003 2014 10 50]); 
set(gca, 'YTick', [10:10:50], 'fontsize', 12);
set(gca, 'XTick', [2003:1:2014], 'fontsize', 12); grid off; box off;
print('FigureA3b','-dpdf','-r0')

figure(6);
ha1 = area([2007+11/12-.5 2009.5-.5], [80 80]); grid on; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85]); set(ha1,{'LineStyle'}, {'none'});         
hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(year, ols(:,2), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(year, ols(:,4), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'k');
plot(year, nonemp_step2(:,1), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top'); hold off; 
axis([2003 2014 0 10]); 
set(gca, 'YTick', [0:2:10], 'fontsize', 12);
set(gca, 'XTick', [2003:1:2014], 'fontsize', 12); grid off; box off;
print('FigureA3a','-dpdf','-r0')



%% Figure A4 
cd '../int_data/CPS';

data = csvread('FigureA4_data.csv', 1, 1); 
time_create_interaction   = data(:,2);

cd '../../Figures';
figure(7);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, time_create_interaction, 'LineWidth', 2, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
plot(date, time_unemp, 'LineWidth', 2, 'LineStyle', '-', 'Marker', 'none', 'Color', 'k');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 20 45]);
set(gca, 'YTick', [20:5:45], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('FigureA4','-dpdf','-r0')


%% Figure A5 
cd '../int_data/ATUS';
data_annual = csvread('FigureA5_data.csv', 1, 1); 
time_srch_annual = data_annual(:,2);
time_unemp_annual = data_annual(:,1);
methods_srch_annual = data_annual(:,5);
methods_unemp_annual = data_annual(:,4);
time_srch_travel_annual = data_annual(:,8);
time_unemp_travel_annual = data_annual(:,7);
year = 2003:1:2014;
cd '../../Figures';

figure(8);
ha1 = area([2007+11/12 2009.5], [80 80]);  grid on ; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on
ha2 = area([2001+3/12 2001+11/12], [80 80]); set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85])           
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
[AX, H1, H2] = plotyy(year,methods_srch_annual,year,time_srch_annual, 'plot');
set(get(AX(1),'Ylabel'),'String','Number', 'fontsize', 12, 'color', 'b') 
axes(AX(1)); set(gca,'ylim',[1.9 2.7], 'ytick',[1.9:0.1:2.7],'fontsize', 12,'xlim',[2003 2014], 'xtick',[2003:1:2014],'fontsize', 12)
set(AX(1),'XColor','k','YColor','b'); set(H1,'LineStyle','-','LineWidth', 3); set(H1,'color','b')
set(get(AX(2),'Ylabel'),'String','Time', 'interpreter', 'latex', 'fontsize', 12, 'color', 'r' ) 
set(AX(2),'XColor','k','YColor','r'); axes(AX(2)); set(gca,'ylim',[20 50], 'ytick',[20:5:50],'fontsize', 12,'xlim',[2003 2014], 'xtick',[2003:1:2014],'fontsize', 12)
set(H2,'LineStyle','--', 'LineWidth', 3); set(H2,'color','r')
saveas(gcf, 'FigureA5.fig');
print('FigureA5','-dpdf','-r0')


%% Figure A6 

cd '../int_data/ATUS';
nonemp_base = csvread('Fig2a_data.csv', 1, 0);
unemp_base = csvread('Fig2b_data.csv', 1, 1); 
nonemp_base = nonemp_base(:,2:3);

nonemp_post = csvread('FigureA6a_data_postrec.csv', 1, 1); 
unemp_post = csvread('FigureA6b_data_postrec.csv', 1, 1); 

nonemp_pre = csvread('FigureA6a_data_prerec.csv', 1, 1); 
unemp_pre = csvread('FigureA6b_data_prerec.csv', 1, 1); 

cd '../../Figures';
figure(9);
ha1 = area([2007+11/12-.5 2009.5-.5], [80 80]); grid on; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85]); set(ha1,{'LineStyle'}, {'none'});         
hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(year, unemp_pre(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'b');
plot(year, unemp_post(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'k');
plot(year, unemp_pre(:,2), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(year, unemp_base(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top'); hold off; 
axis([2003 2014 10 50]); 
set(gca, 'YTick', [10:10:50], 'fontsize', 12);
set(gca, 'XTick', [2003:1:2014], 'fontsize', 12); grid off; box off;
print('FigureA6b','-dpdf','-r0')
saveas(gcf, ['FigureA6b.fig']);

figure(10);
ha1 = area([2007+11/12-.5 2009.5-.5], [80 80]); grid on; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85]); set(ha1,{'LineStyle'}, {'none'});         
hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(year, nonemp_pre(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'b');
plot(year, nonemp_post(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'k');
plot(year, nonemp_pre(:,2), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(year, nonemp_base(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top'); hold off; 
axis([2003 2014 0 11]); 
set(gca, 'YTick', [0:2:11], 'fontsize', 12);
set(gca, 'XTick', [2003:1:2014], 'fontsize', 12); grid off; box off;
saveas(gcf, 'FigureA6a.fig');
print('FigureA6a','-dpdf','-r0')


%% Figure A7 

cd '../int_data/CPS';
data = csvread('FigureA7_data.csv', 1, 1); 
time_create_theta	   = data(:,2);
time_create_lgdp_cycle = data(:,3);
time_create_urate      = data(:,4);
 
cd '../../Figures';

figure(11);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, time_create_theta, 'LineWidth', 2, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
plot(date, time_create_urate, 'LineWidth', 2, 'LineStyle', '--', 'Marker', 'none', 'Color', 'm');
plot(date, time_create_lgdp_cycle, 'LineWidth', 2, 'LineStyle', '--', 'Marker', 'none', 'Color', 'b');
plot(date, time_unemp, 'LineWidth', 2, 'LineStyle', '-', 'Marker', 'none', 'Color', 'k');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 20 45]);
set(gca, 'YTick', [20:5:45], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('FigureA7','-dpdf','-r0')

%% Figure A8
cd '../int_data/CPS';
data = csvread('FigureA8_data.csv', 1, 1); 
methods_srch = data(:,2);
time_srch = data(:,3);

cd '../../Figures';

figure(12);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); 
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, methods_srch, 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 1.8 2.8]);
set(gca, 'YTick', [1.8:0.2:2.8], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
ylabel('Number of Methods','fontsize', 12);
print('FigureA8a','-dpdf','-r0')

figure(13);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); 
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, methods_srch./methods_srch(1), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'r');
plot(date, time_srch./time_srch(1), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0.8 1.4 ]);
set(gca, 'YTick', [0.8:0.1:1.4], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('FigureA8b','-dpdf','-r0')


%% Figure A9 
 
cd '../int_data/ATUS';
data_ATUS = csvread('FigureA5_data.csv', 1, 1); 
%year = data_ATUS(:,1);
ATUS_srch = data_ATUS(:,2);
ATUS_unemp = data_ATUS(:,1);
ATUS_method_srch = data_ATUS(:,5);
ATUS_method_unemp = data_ATUS(:,4);

cd '../CPS';
data = csvread('Figure3b_data.csv', 1, 1); 
methods_unemp = data(:,2);
time_unemp = data(:,3);
 
cd '../../Figures';

figure(14);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); 
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, time_unemp, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(year, ATUS_unemp, 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'k');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 15 50]);
set(gca, 'YTick', [15:5:50], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
saveas(gcf, 'FigureA9a.fig');
print('FigureA9a','-dpdf','-r0')


figure(15);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); 
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, methods_unemp, 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
plot(year, ATUS_method_unemp, 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'k');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 1.5 2.5]);
set(gca, 'YTick', [1.5:0.1:2.5], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
saveas(gcf, 'FigureA9b.fig');
print('FigureA9b','-dpdf','-r0')



%% Figure A10 

cd '../int_data/CPS';
data = csvread('FigureA10_data.csv', 1, 1); 
numsearch1=data(1:216,3);
numsearch2=data(217:end,3);
numsearch=data(:,3);
[R,C] = size(numsearch);
for i=1:R
 date(i)=1976 +(i-1)/12;
end
date1 = date(1:216);
date2 = date(217:end);

cd '../../Figures';

figure(16);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85])
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(date2, numsearch2, 'LineWidth', 2, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 1.3 1.8]);
set(gca, 'YTick', [1.3:.1:1.8], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('FigureA10b','-dpdf','-r0')

figure(17);
ha3 = area([1990+7/12 1991.25], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha3,'FaceColor',[.85 .85 .85])           
set(ha3,{'LineStyle'}, {'none'}); hAnnotation = get(ha3,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha4 = area([1981+7/12 1982 + 11/12], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha4,'FaceColor',[.85 .85 .85])           
set(ha4,{'LineStyle'}, {'none'}); hAnnotation = get(ha4,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha5 = area([1980 1980 + 7/12], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha5,'FaceColor',[.85 .85 .85])           
set(ha5,{'LineStyle'}, {'none'}); hAnnotation = get(ha5,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); 
plot(date1, numsearch1, 'LineWidth', 2, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1976 1993 1.2 1.6]);
set(gca, 'YTick', [1.2:.1:1.6], 'fontsize', 12);
set(gca, 'XTick', [1976:2:1993], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('FigureA10a','-dpdf','-r0')

%% Figure A11 
clear 
cd '../int_data/CPS';
data = csvread('FigureA11_data.csv', 1,2); 

count_empldir = data(:,1);
frac_empldir = data(:,2);

count_pubemkag  = data(:,3);
frac_pubemkag  = data(:,4);

count_PriEmpAg  = data(:,5);
frac_PriEmpAg  = data(:,6);

count_FrendRel = data(:,7);
frac_FrendRel = data(:,8);

count_SchEmpCt = data(:,9);
frac_SchEmpCt = data(:,10);

count_Unionpro= data(:,11);
frac_Unionpro= data(:,12);

count_Resumes = data(:,13);
frac_Resumes = data(:,14);

count_Plcdads = data(:,15);
frac_Plcdads = data(:,16);

count_Otheractve= data(:,17);
frac_Otheractve= data(:,18);

count_lkatads= data(:,19);
frac_lkatads= data(:,20);

count_Jbtrnprg = data(:,21);
frac_Jbtrnprg = data(:,22);

count_otherpas= data(:,23);
frac_otherpas= data(:,24);

[R,C] = size(count_otherpas);
for i=1:R
 date(i)=1994 +(i-1)/12;
end

cd '../../Figures';

figure(18);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_empldir, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0.3 1]);
set(gca, 'YTick', [0.3:0.2:1], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Contacted Employer Directly', 'fontsize', 30);
print('FigureA11a','-dpdf','-r0')

figure(19);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_pubemkag, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0.1 0.4]);
set(gca, 'YTick', [0.1:0.05:0.4], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Contacted a Public Employment Agency', 'fontsize', 30);
print('FigureA11b','-dpdf','-r0')

figure(20);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_PriEmpAg, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 0.3]);
set(gca, 'YTick', [0:0.05:0.3], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Checked a Private Employment Agency', 'fontsize', 30);
print('FigureA11c','-dpdf','-r0')


figure(21);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_FrendRel, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 0.5]);
set(gca, 'YTick', [0:0.1:0.5], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Contacted Friends or Relatives', 'fontsize', 30);
print('FigureA11d','-dpdf','-r0')

figure(22);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_SchEmpCt, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 0.1]);
set(gca, 'YTick', [0:0.02:0.1], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Contacted a School or University Employment Center', 'fontsize', 30);
print('FigureA11e','-dpdf','-r0')

figure(23);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_Unionpro, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 0.1]);
set(gca, 'YTick', [0:0.02:0.1], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Checked Union or Professional Registers', 'fontsize', 30);
print('FigureA11f','-dpdf','-r0')

figure(24);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_Resumes, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0.2 0.8]);
set(gca, 'YTick', [0.2:0.1:0.8], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Sent Out Resumes or Filled Out Applications', 'fontsize', 30);
print('FigureA11g','-dpdf','-r0')

figure(25);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_Plcdads, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 0.4]);
set(gca, 'YTick', [0:0.05:0.4], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Placed or Answered Advertisements', 'fontsize', 30);
print('FigureA11h','-dpdf','-r0')

figure(26);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_Otheractve, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 0.2]);
set(gca, 'YTick', [0:0.05:0.2], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Other Means of Active Job Search', 'fontsize', 30);
print('FigureA11i','-dpdf','-r0')

figure(27);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_lkatads, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0.1 0.6]);
set(gca, 'YTick', [0.1:0.1:0.6], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Read About Job Openings', 'fontsize', 30);
print('FigureA11j','-dpdf','-r0')

figure(28);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_Jbtrnprg, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 0.06]);
set(gca, 'YTick', [0:0.01:0.06], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Attended Job Training Program or Course', 'fontsize', 30);
print('FigureA11k','-dpdf','-r0')

figure(29);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_otherpas, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 0.06]);
set(gca, 'YTick', [0:0.01:0.06], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
title('Other Means of Passive Job Search', 'fontsize', 30);
print('FigureA11l','-dpdf','-r0')

%% Figure A12 
% hard code the coefficients from the regression in Column 2 of Table 5

age_int = 56.96;
age_int2 = -1.976;
age_int3 = 0.0302;
age_int4 = -0.0001728;

age_ext = 1;
age_ext2 = 1;
age_ext3 = 1;
age_ext4 = 1;

age = 25:1:75;
int = age.*age_int + age.^2.*age_int2 + age.^3.*age_int3 + age.^4.*age_int4;
ext = age.*age_ext + age.^2.*age_ext2 + age.^3.*age_ext3 + age.^4.*age_ext4;

% plot(age, ext, age, int)
% [AX, H1, H2] = plotyy(age,ext,age,int, 'plot');
% set(get(AX(1),'Ylabel'),'String','Extensive Margin', 'fontsize', 12, 'color', 'b') 
% axes(AX(1));
%  set(gca,'ylim',[-2 0], 'ytick',[-2:0.2:0],'fontsize', 12,'xlim',[25 75], 'xtick',[25:5:75],'fontsize', 12)
% set(AX(1),'XColor','k','YColor','b')
% set(H1,'LineStyle','-','LineWidth', 3)
% set(H1,'color','b')
% set(get(AX(2),'Ylabel'),'String','Intensive Margin', 'interpreter', 'latex', 'fontsize', 12, 'color', 'r' ) 
% set(AX(2),'XColor','k','YColor','r')
% axes(AX(2));
%  set(gca,'ylim',[100 250], 'ytick',[100:25:250],'fontsize', 12,'xlim',[25 75], 'xtick',[25:5:75],'fontsize', 12)
% set(H2,'LineStyle','--', 'LineWidth', 3)
% set(H2,'color','r')
% box off;

figure(30);
plot(age,int,  'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
box off;
print('FigureA12','-dpdf','-r0')

%% Figure A13 

undur = 0.83356;
undur2 = -0.0264;
undur3 = 0.000302;
undur4 = -0.00000115;

dur = 1:1:99;
search = dur.*undur + dur.^2.*undur2 + dur.^3.*undur3 + dur.^4.*undur4;

figure(31);
plot(dur,search,  'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
box off;
print('FigureA13','-dpdf','-r0')


%% Figure A14 
clear
cd '../int_data/CPS';
data = csvread('FigureA14_data.csv', 1, 1); 
frac_elig = data(:,2);

[R,C] = size(frac_elig);
for i=1:R
 date(i)=2000 +(i-1)/12;
end

cd '../../Figures';
figure(32);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); 
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_elig, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([2000 2015 0.3 0.8]);
set(gca, 'YTick', [0.3:0.1:0.8], 'fontsize', 12);
set(gca, 'XTick', [2000:2:2015], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('FigureA14','-dpdf','-r0')

close all 