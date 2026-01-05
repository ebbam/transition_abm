clear all;
close all;

% Setting parameter values
omega=2;
bta=(0.987)^(1/3);
rho=(0.95)^(1/3);
gama=0.72;
sigma=0.034;
chi=0.49;
%chi = .2514;
non = 0.7;
xi = -0.15; % This is the relationship between s and theta
s_theta;

[R,C] = size(s);
for i=1:R
 date(i)=1994 +(i-1)/12;
end
% Initializing vectors 
psivect=-1:(3.0-0.1)/399:4.0;
[d1,npsi]=size(psivect);
neta = 1;

psimat=zeros(npsi,neta);
etamat=zeros(npsi,neta);
kappamat=zeros(npsi,neta);
phimat=zeros(npsi,neta);
bmat=zeros(npsi,neta);
scoefmat=zeros(npsi,neta);
tcoefmat=zeros(npsi,neta);
alphavect = zeros(npsi,neta);

% Looping over possible alpha and psi values
for ieta = 1:neta
    for ipsi = 1:npsi
        
        % PART 1: Calculating the implied coefficients 
        psi=psivect(ipsi);
        %eta=etavect(ieta); CP: Changed to match the calibration in the paper
        eta = 1/psi;        
        alpha = 1 - ((1-xi*omega)/(psi - xi*psi)); % Needed to match xi
        alphavect(ipsi) = alpha;
        
        first=1/((1-gama)*bta*chi);
        second=-(1-sigma)/((1-gama)*chi);
        third=(gama)/(1-gama);

        kappamat(ipsi,ieta)=(1-non)/(first+second+third); % CP - changes 0.3 to 1-non (value of non-employment)
        kappa=kappamat(ipsi,ieta);
        
        phimat(ipsi,ieta)=(alpha*eta*gama*kappa*psi)/(1-gama);
        phi=phimat(ipsi,ieta);
        bmat(ipsi,ieta)=non+phi/omega; % Changed to /omega instead of /2
        b=bmat(ipsi,ieta);

        firstpart=(gama*kappa*eta*psi*alpha)/(1-gama);
        secondpart=(psi-1)+(psi*(eta-1)*alpha);

        secondorder=phi*(omega-1)-firstpart*secondpart;
        if secondorder<0 
            secondorder
            psi
            eta
        end 

        scoefmat(ipsi,ieta)=(alpha-(psi-1)*(1-alpha))/(omega*alpha+(omega-psi)*(1-alpha));
        xi=scoefmat(ipsi,ieta); % This is big phi

        fraction=alpha*psi*xi+(1-alpha)*psi;
        calApart=(1-eta*fraction);
        calA=(kappa/((1-gama)*bta*chi))*calApart;

        parenth=((1-sigma)*calApart)/chi-gama;
        calB=(kappa/(1-gama))*parenth+phi*xi;

        tcoefmat(ipsi,ieta)=rho/(calA-rho*calB);
    end
end


%plot(alphavect(11:end),psivect(11:end));

s=s./(mean(s));
theta=theta./(mean(theta));
ind = (alphavect<0);
ii = find(ind(1:end-1)-ind(2:end)==1) + 1;

for i=ii:npsi
    jf(:,i) =chi*(alphavect(i)*s.^psivect(i)+(1-alphavect(i))*theta.^psivect(i)).^(1./psivect(i));
    jf_c(:,i) = chi*(alphavect(i)*1.^psivect(i)+(1-alphavect(i))*theta.^psivect(i)).^(1./psivect(i));
    ctw(:,i) = chi*s;
end

F_data; % reading in the
% Intializing vectors for the 
dist1 = NaN(npsi,2); 
dist2 = NaN(npsi,2);
dist3 = NaN(npsi,2);
dist1_c = NaN(npsi,2);
dist2_c = NaN(npsi,2);
dist3_c = NaN(npsi,2);
dist1_ctw = NaN(npsi,2);
dist2_ctw = NaN(npsi,2);
dist3_ctw = NaN(npsi,2);
k = 168; % This is the end of 2007

% Finding the Parameters that are the best fit over the entire period and pre-recession only
for i=ii:npsi
    dist1(i,1) =  sum(abs(jf(:,i)-F(:,1)));
    dist2(i,1) =  sum(abs(jf(:,i)-F(:,2)));
    dist3(i,1) =  sum(abs(jf(:,i)-UE(:)));

    dist1_c(i,1) =  sum(abs(jf_c(:,i)-F(:,1)));
    dist2_c(i,1) =  sum(abs(jf_c(:,i)-F(:,2)));       
    dist3_c(i,1) =  sum(abs(jf_c(:,i)-UE(:)));  
    
    dist1_ctw(i,1) =  sum(abs(ctw(:,i)-F(:,1)));
    dist2_ctw(i,1) =  sum(abs(ctw(:,i)-F(:,2)));   
    dist3_ctw(i,1) =  sum(abs(ctw(:,i)-UE(:))); 
    
    dist1(i,2) =  sum(abs(jf(1:k,i)-F(1:k,1)));
    dist2(i,2) =  sum(abs(jf(1:k,i)-F(1:k,2)));
    dist3(i,2) =  sum(abs(jf(1:k,i)-UE(1:k)));
    
    dist1_c(i,2) =  sum(abs(jf_c(1:k,i)-F(1:k,1)));
    dist2_c(i,2) =  sum(abs(jf_c(1:k,i)-F(1:k,2)));       
    dist3_c(i,2) =  sum(abs(jf_c(1:k,i)-UE(1:k)));

    dist1_ctw(i,2) =  sum(abs(ctw(1:k,i)-F(1:k,1)));
    dist2_ctw(i,2) =  sum(abs(ctw(1:k,i)-F(1:k,2)));    
    dist3_ctw(i,2) =  sum(abs(ctw(1:k,i)-UE(1:k)));    
    
end

for i = 1:1:2  % Looping over full sample and pre-recession only to find the best fit parameters
    
    alpha_best(i,1) = alphavect(find(dist1(:,i)==min(dist1(:,i))));
    psi_best(i,1) = psivect(find(dist1(:,i)==min(dist1(:,i))));   
    jf_best(:,i,1) = jf(:,find(dist1(:,i)==min(dist1(:,i))));
    dist_min(i,1) = dist1(find(dist1(:,i)==min(dist1(:,i))));
   
    jf_c_best(:,i,1) = jf_c(:,find(dist1(:,i)==min(dist1(:,i))));
    jf_ctw_best(:,i,1) = ctw(:,ii+1); % always the same
    dist_ctw_min(i,1) = dist1_ctw(ii+1);
    
    alpha_best(i,3) = alphavect(find(dist3(:,i)==min(dist3(:,i))));
    psi_best(i,3) = psivect(find(dist3(:,i)==min(dist3(:,i))));   
    jf_best(:,i,3) = jf(:,find(dist3(:,i)==min(dist3(:,i))));
    dist_min(i,3) = dist1(find(dist3(:,i)==min(dist3(:,i))));
    
    jf_c_best(:,i,3) = jf_c(:,find(dist3(:,i)==min(dist3(:,i))));
    jf_ctw_best(:,i,3) = ctw(:,ii+1); % always the same
    dist_ctw_min(i,3) = dist3_ctw(ii+1);    
end

alpha_best 
psi_best

chi_t_base = 0.49* F(:,1)./ jf_best(:,2,1);
chi_t_c = 0.49* F(:,1)./ jf_c_best(:,2,1);

f_base = chi_t_base.*jf_best(:,2,1)./0.49;
f_c = chi_t_base.*jf_c_best(:,2,1)./0.49;
u_base = sep ./ (sep + f_base);
u_c = sep ./ (sep + f_c);

% Smoothing (Quarterly)
chi_t_base_q = (chi_t_base(1:end-2) + chi_t_base(2:end-1) + chi_t_base(3:end))/3;
chi_t_base_q = chi_t_base_q(1:3:end);
chi_t_c_q = (chi_t_c(1:end-2) + chi_t_c(2:end-1) + chi_t_c(3:end))/3;
chi_t_c_q = chi_t_c_q(1:3:end);
u_c_q = (u_c(1:end-2) + u_c(2:end-1) + u_c(3:end))/3;
u_c_q = u_c_q(1:3:end);
u_base_q = (u_base(1:end-2) + u_base(2:end-1) + u_base(3:end))/3;
u_base_q = u_base_q(1:3:end);
u_star_q = (u_star(1:end-2) + u_star(2:end-1) + u_star(3:end))/3;
u_star_q = u_star_q(1:3:end);


% Quarterly Average
jf_best(2:end-1,1,1) = (jf_best(1:end-2,1,1) + jf_best(2:end-1,1,1) + jf_best(3:end,1,1))/3;
jf_ctw_best(2:end-1,1,1) = (jf_ctw_best(1:end-2,1,1) + jf_ctw_best(2:end-1,1,1) + jf_ctw_best(3:end,1,1))/3;
F(2:end-1,1) = (F(1:end-2,1) + F(2:end-1,1) + F(3:end,1))/3;

cd '../../Figures';

figure(1)
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date(3:3:end), chi_t_base_q,'LineWidth', 1, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(date(3:3:end), chi_t_c_q,'LineWidth', 1, 'LineStyle', '-', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top');
hold off
axis([2008 2014 0.3 0.55]);
set(gca, 'YTick', [0.3:0.05:0.55], 'fontsize', 12);
set(gca, 'XTick', [2008:1:2014], 'fontsize', 12);
legend('Including Search Effort', 'Excluding Search Effort');
saveas(gcf, 'Figure5a.pdf', 'pdf');

figure(2)
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date(3:3:end), u_base_q,'LineWidth', 1, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(date(3:3:end), u_c_q,'LineWidth', 1, 'LineStyle', '-', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top');
hold off
axis([2008 2014 0.04 0.12]);
set(gca, 'YTick', [0.04:0.01:0.12], 'fontsize', 12);
set(gca, 'XTick', [2008:1:2014], 'fontsize', 12);
legend('Including Search Effort', 'Excluding Search Effort');
saveas(gcf, 'Figure5b.pdf', 'pdf');

figure(5)
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
hold on; ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation');         
hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date(3:end-3), F((3:end-3),1),'LineWidth', 2, 'LineStyle', '-', 'Marker', 'none', 'Color', 'k');
plot(date(3:end-3), jf_best((3:end-3),1,1),'LineWidth', 1, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
plot(date(3:end-3), jf_ctw_best((3:end-3),1,1),'LineWidth', 1, 'LineStyle', '-', 'Marker', 'none', 'Color', 'r');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 .2 0.9]);
set(gca, 'YTick', [0.2:0.1:0.9], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
legend('Data', 'General', 'Linear');
saveas(gcf, 'FigureC3.pdf', 'pdf');


%% ASIDE: Creating a figure that demonstrates the -0.15 target for Phi

log_s = log(s);
log_theta = log(theta);

cycle_log_s = log_s - hpfilter(log_s,129600);
cycle_log_theta = log_theta - hpfilter(log_theta,129600);

figure(8)
ha1 = area([2007+11/12 2009.5], [60 60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha2 = area([2001+3/12 2001+11/12], [60 60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); 
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
ha1 = area([2007+11/12 2009.5], [-60 -60]);  grid on; set(gca,'Layer','top') ;set(ha1,'FaceColor',[.85 .85 .85])           
set(ha1,{'LineStyle'}, {'none'}); hAnnotation = get(ha1,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');hold on; 
ha2 = area([2001+3/12 2001+11/12], [-60 -60]);set(gca,'Layer','top'); set(ha2,'FaceColor',[.85 .85 .85]); 
set(ha2,{'LineStyle'}, {'none'}); hAnnotation = get(ha2,'Annotation'); hLegendEntry = get(hAnnotation,'LegendInformation'); set(hLegendEntry,'IconDisplayStyle','off');
plot(date, cycle_log_s,'LineWidth', 2, 'LineStyle', '-', 'Marker', 'none', 'Color', 'k');
plot(date, cycle_log_theta,'LineWidth', 1, 'LineStyle', '-', 'Marker', 'none', 'Color', 'b');
set(gca, 'Layer', 'top');
hold off
axis([1994 2014 -0.6 0.6]);
set(gca, 'YTick', [-0.6:0.1:0.6], 'fontsize', 12);
set(gca, 'XTick', [1994:2:2014], 'fontsize', 12);
legend('Search Effort', '\theta', 'interpreter', 'latex');
saveas(gcf, 'FigureC2.pdf', 'pdf');

