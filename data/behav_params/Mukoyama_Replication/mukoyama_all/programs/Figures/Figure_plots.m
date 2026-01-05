
% Description: Creates stylized figures for Mukoyama, Patterson and Sahin (2017)

clear;
clc;
set(0,'DefaultFigurePaperPosition', [0.25 0.25 10.5 8]);
set(0, 'DefaultFigurePaperOrientation', 'landscape');

% ATTENTION: USER NEEDS TO MAKE SURE THEY ARE IN THE PROGRAM FOLDER.

%% Figure 2 

cd '../../int_data/ATUS';

nonemp_base = csvread('Fig2a_data.csv', 1, 0); %This goes from 0 to include the year
unemp_base = csvread('Fig2b_data.csv', 1, 1); 
year = nonemp_base(:,1);
nonemp_base = nonemp_base(:,2:3);

cd '../../Figures';
figure(1);
ha1 = area([2007+11/12-.5 2009.5-.5], [80 80]); grid on; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85]); set(ha1,{'LineStyle'}, {'none'});         
hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(year, unemp_base(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
plot(year, unemp_base(:,2), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top'); hold off; 
axis([2003 2014 10 50]); 
set(gca, 'YTick', [10:10:50], 'fontsize', 12);
set(gca, 'XTick', [2003:1:2014], 'fontsize', 12); grid off; box off;
print('Figure2b','-dpdf','-r0')

figure(2);
ha1 = area([2007+11/12-.5 2009.5-.5], [80 80]); grid on; set(gca,'Layer','top'); set(ha1,'FaceColor',[.85 .85 .85]); set(ha1,{'LineStyle'}, {'none'});         
hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off'); hold on;
plot(year, nonemp_base(:,1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
plot(year, nonemp_base(:,2), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top'); hold off; 
axis([2003 2014 0 10]); 
set(gca, 'YTick', [0:2:10], 'fontsize', 12);
set(gca, 'XTick', [2003:1:2014], 'fontsize', 12); grid off; box off;
print('Figure2a','-dpdf','-r0')


%% Figure 3 

cd '../int_data/CPS';
data_counts = csvread('Figure3a_data.csv',1,1); 
searchers = data_counts(:,2);
nonsearchers = data_counts(:,3);
unemp = data_counts(:,4);
nonpart = data_counts(:,5);
emp = data_counts(:,6);
frac_unemp  = unemp./(unemp + nonpart);
[R,C] = size(searchers);
for i=1:R
 date(i)=1994 +(i-1)/12;
end

data = csvread('Figure3b_data.csv', 1, 1); 
time_unemp = data(:,3);

cd '../../Figures';

figure(1);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); 
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
ha3 = area([2007+11/12 2009.5], [-60 -60]);  grid on; set(gca,'Layer','top') ;set(ha3,'FaceColor',[.85 .85 .85])           
set(ha3,{'LineStyle'}, {'none'}); hAnnotation = get(ha3,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha4 = area([2001+3/12 2001+11/12], [-60 -60]);set(gca,'Layer','top'); set(ha4,'FaceColor',[.85 .85 .85]); 
set(ha4,{'LineStyle'}, {'none'}); hAnnotation = get(ha4,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, frac_unemp , 'LineWidth', 2.5, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0.05 0.25]);
set(gca, 'YTick', [0.05:0.05:0.25], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('Figure3a','-dpdf','-r0')

figure(2);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, time_unemp, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 25 45]);
set(gca, 'YTick', [25:5:50], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('Figure3b','-dpdf','-r0')


%% Figure 4 

effort_unemp_UNE = time_unemp .* unemp ./ (unemp + nonpart + emp);
unemp_frac = unemp ./ (unemp + nonpart + emp);
[R,C] = size(unemp_frac);
for i=1:R
 date(i)=1994 +(i-1)/12;
end

figure(3);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, effort_unemp_UNE, 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 3]);
set(gca, 'YTick', [0:0.5:3], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('Figure4a','-dpdf','-r0')

% Normalized min/(U+N+E)  and U/(U+N+E)
figure(4);
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, effort_unemp_UNE / effort_unemp_UNE(1), 'LineWidth', 3, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(date, unemp_frac / unemp_frac(1), 'LineWidth', 3, 'LineStyle', '--', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 0 2.5]);
set(gca, 'YTick', [0:0.5:2.5], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
ylabel('','fontsize', 12);
grid off; box off;
print('Figure4b','-dpdf','-r0')

close all 

