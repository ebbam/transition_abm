clear all;

%%% with generalized matching function

tic %count time


%parameter values

bta = (0.988)^(1/3); %discount factor
gama = 0.72; %bargaining power parameter: Shimer
%gama = 0.052; %bargaining power parameter: HM
sigma = 0.033;%separation rate
chi = 0.49; %matching function 
omega = 2; 
alpha = 0.15; 
%alpha=0.2; %counterfactual
psi = 1.33; 
%psi=0.23; %counterfactual
eta = 1/psi;
totalb = 0.4; %UI: Shimer
%totalb = 0.955; %UI: HM

%shock process
rho = 0.949; 
stdeps = 0.0065; 


%ss calculation
uss = sigma/(sigma+chi); %ss unemployment
vss = 1.0; %ss vacancy
thetass = 1.0; %ss theta
pss = 1.0; %ss productivity
sss = 1.0; %ss s

%calibrate kappa
denomk = (1/(1-gama)-bta*(1-sigma-gama*chi)/(1-gama));
numk = bta*chi*(pss-totalb);
kappa= numk/denomk;

%calibrate phi
phi=(alpha*eta*psi*gama*kappa)/(1-gama);

%calibrate b
b=totalb+phi/omega;

%SOC
firstpart=(gama*kappa*eta*psi*alpha)/(1-gama);
secondpart=(psi-1)+(psi*(eta-1)*alpha);
secondorder=phi*(omega-1)-firstpart*secondpart;


%elasticity for s
%selas=0.0; %DMP
selas = (alpha-(psi-1)*(1-alpha))/(omega*alpha+(omega-psi)*(1-alpha)) %endogenous s
xi = selas;

%elasticity for theta
fraction=alpha*psi*xi+(1-alpha)*psi;
calApart=(1-eta*fraction);
calA=(kappa/((1-gama)*bta*chi))*calApart;

parenth=((1-sigma)*calApart)/chi-gama;
calB=(kappa/(1-gama))*parenth+phi*xi;

telas=rho/(calA-rho*calB)


%Simulation

Nperiods=5000; %number of periods
Nskip=100; %number of skip

Periods=zeros(Nperiods+1,1);
useries=zeros(Nperiods+1,1);
vseries=zeros(Nperiods+1,1);
sseries=zeros(Nperiods+1,1);
thetaseries=zeros(Nperiods+1,1);
jfprob=zeros(Nperiods+1,1);
pseries=zeros(Nperiods,1);


%log deviation
dthetaseries=zeros(Nperiods+1,1);
dpseries=zeros(Nperiods+1,1);
dsseries=zeros(Nperiods+1,1);

%initial condition
dpseries(1)=0; %first period 
pseries(1)=pss*exp(dpseries(1));
useries(1)=uss;
thetaseries(1)=thetass;
vseries(1)=thetaseries(1)*useries(1);
sseries(1)=sss;
dsseries(1)=0;
jfprob(1) = chi*(alpha*(sseries(1)^psi)+(1-alpha)*(thetaseries(1)^psi))^eta;
Periods(1)=1; %just counting periods 
rng('default')
shock=randn(Nperiods,1);

for i=1:Nperiods
    Periods(i+1)=i+1; %just counting
    useries(i+1)=sigma*(1-useries(i))+(1-jfprob(i))*useries(i); %unemployment transition
    dpseries(i+1)=rho*dpseries(i)+stdeps*shock(i,1); %AR(1) transition
    pseries(i)=pss*exp(dpseries(i)); % productivity level
    dthetaseries(i+1)=telas*dpseries(i+1); %theta determination
    thetaseries(i+1)=thetass*exp(dthetaseries(i)); %theta determination
    dsseries(i+1)=selas*dthetaseries(i+1); % s determination
    sseries(i+1) = sss*exp(dsseries(i)); % s determination
    jfprob(i+1) = chi*((alpha*(sseries(i+1)^psi)+(1-alpha)*(thetaseries(i+1)^psi))^eta); %job finding probability
    jfprob(i+1) = min(1,jfprob(i+1));
    jfprob(i+1) = min(thetaseries(i+1),jfprob(i+1));
    vseries(i+1)=thetaseries(i+1)*useries(i+1);
end

%convert to quarterly
j = 1;
for i=Nskip+1:3:Nperiods-2
  urate(j) = mean(useries(i:i+2));
  vacancy(j) = mean(vseries(i:i+2));
  theta(j) = mean(thetaseries(i:i+2));
  svect(j) = mean(sseries(i:i+2));
  prod(j) = mean(pseries(i:i+2));
  j = j+1;
end 

trends_u = hpfilter(log(urate), 1600);
cyclical_u = log(urate)-trends_u' ;
std_u = std(cyclical_u)
autocorr_u= corrcoef(cyclical_u(1:end-1),cyclical_u(2:end));

trends_v = hpfilter(log(vacancy), 1600);
cyclical_v = log(vacancy)-trends_v' ;
std_v = std(cyclical_v);
autocorr_v= corrcoef(cyclical_v(1:end-1),cyclical_v(2:end));

trends_s = hpfilter(log(svect), 1600);
cyclical_s = log(svect)-trends_s' ;
std_s = std(cyclical_s);
autocorr_s= corrcoef(cyclical_s(1:end-1),cyclical_s(2:end));

trends_th = hpfilter(log(theta), 1600);
cyclical_th = log(theta)-trends_th' ;
std_th = std(cyclical_th);
autocorr_th= corrcoef(cyclical_th(1:end-1),cyclical_th(2:end));

trends_p = hpfilter(log(prod), 1600);
cyclical_p = log(prod)-trends_p' ;
std_p = std(cyclical_p);
autocorr_p= corrcoef(cyclical_p(1:end-1),cyclical_p(2:end));

corrsu = corrcoef(cyclical_s,cyclical_u);
corrsv= corrcoef(cyclical_s,cyclical_v);
corrsth= corrcoef(cyclical_s,cyclical_th);
corrsp= corrcoef(cyclical_s,cyclical_p);
corruv= corrcoef(cyclical_u,cyclical_v);
corruth= corrcoef(cyclical_u,cyclical_th);
corrup= corrcoef(cyclical_u,cyclical_p);
corrvth= corrcoef(cyclical_v,cyclical_th);
corrvp= corrcoef(cyclical_v,cyclical_p);
corrthp= corrcoef(cyclical_th,cyclical_p);




t=toc
