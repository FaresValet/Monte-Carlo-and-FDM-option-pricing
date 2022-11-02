disp("Risque d'entreprise");
%IMPORTANT: remplacer les valeurs de var (travail 4 et 5) par celles des
%nouveaux codes var=-43 et -64 environ si je me souviens bien
% T1EtDens(); %remplacer fonctions de densité et de répartition pour
% B=100,50,36
% Travail3(); %Remettre les graphes de ce code là
% Travail4(); %Remettre les graphes de ce code là avec les B=100,50,36 et
% les seuils alpha=0.01 et 0.001
% Travail5(); %changer le code (mettre ce code là)
%  Travail6(); %graphe de la densité de la loi bêta
%Simulation1();
%Simulation2();
% Simulation3(); %Nouvelle simulation juste mettre le code et le disp 
% plotvar(); %nouvelle simulation mettre le plot
robbinsforsmallalphavalues()
function[] = T1EtDens()
% ----------------------------------------- 
% Valeurs initiales des variables

    S0 = 100;
    B=50; %on change cette valeur B=100,50,36
    r=0;
    s=0.4; 
    S(1)=S0;
 
    %Pour simuler la trajectoire de l'actif, discrétisons l'intervalle
    %[0,T] en N parties:
        T=1;
        N=100;
        delta_t = T / N;
        t=(0:N)*delta_t; %t=linspace(0,T,N+1);
    
    W(1)=0;
    Nmc = 10000;
    prob = 0;
    compt = 0;
  
% -----------------------------------------
    for k = 1:Nmc
        % boucle pour une trajectoire du mouvement brownien
        for i = 1:N
            W(i + 1) = W(i) + sqrt(delta_t) * randn;
            S(i+1)=S(i) + r*S(i)*delta_t + s*S(i)*(W(i+1)-W(i));
        end
         
         if S(N+1) < B 
             plot(t,S, 'r');
         else  
             plot(t,S, 'g');
         end

        hold on;
        %On stock la derniere valeur du MB dans un tableau
        last_value(k) = S(N + 1);
        X(k)=S(N+1) - B;
        %Compteur pour calculer la proba
        if (last_value(k) < B) 
            compt = compt + 1;
        end 
        
    end
%-----------------------------------------------------
%parametres du graphique (X,Y, Legend) hors de la boucle
%pour optimiser la rapidité d'exécution
        xlabel 't'
        ylabel 'S'
        title 'Trajectoires de S, S(T)<B en rouge et S(T)>=B en vert'
%----------------------------------------
%---------------------------------------------
%Calcul de la proba 
    prob = compt / Nmc;
    disp("probabilité " + prob);
    
   % B=100 -> proba dans [60%, 70%]
   % B=50  -> proba dans [3%, 10%]
   % B=36  -> proba dans [0, 4%]
%--------------------------(


%II- Simulation des fonctions de densité

a=-100;
b=200;
N_x=100;
delta_x=(b-a)/N_x;
%x=(0:Nmc)*delta_x; %t=linspace(0,T,Nmc+1);
P_d=linspace(0,delta_x,N_x);

for i =1:N_x+1
    
    x(i)= a + delta_x*(i-1);
    cont_d=0;
    cont_r=0;
    
    for n =1: Nmc
       %Fonction de densite empirique
       if x(i) < X(n) && X(n) <= x(i) + delta_x
           cont_d=cont_d+1;
       end
       %Fonction de reparition empirique
       if(X(n) <= x(i))
           cont_r=cont_r+1;
       end
    end
    disp(x);
    P_d(i)=cont_d/Nmc;
     repartition(i)=cont_r/Nmc;
    densite(i)=P_d(i)/delta_x;
   
end

 figure;
 plot(x,densite,'ro','MarkerSize',2,'MarkerFaceColor', 'g' );
 xlabel 'x'
 ylabel 'f_X(x)'
 title 'Fonctions de densité  '

 
 %Fonction de repartition
 tic;
 figure;
 plot(x,repartition,'ro','MarkerSize',2,'MarkerFaceColor', 'b' );
 xlabel 'x'
 ylabel 'F_X(x)'
 title 'Fonctions de repartition  '

 
end 


function[] = Travail3()

    function[res]=phi(z,y)
        if y>z
            res=0;
        else
            res=1;
        end
    end

       Z0=1;
    Z(1)=Z0;
    Nmc=1000;
    lambda=0.9;
    

 
    beta=100;
    
    for i=1: Nmc
        gamma(i)=beta/((i+1)^lambda);
        Z(i+1)=Z(i)-gamma(i)*(phi(Z(i),randn) - 1/2);
    end
     plot(Z)
    
end

function[] = Travail4()
    function[s] = psi(p,x)
       if(x<=p)
           s=1;
       else 
           s=0;
       end
    end
    r=0;
    sigma=0.4; 
    
    S0 = 100;
    Z0=0.1;
    Z(1)=Z0;
    Nmc=10000;
    lambda=0.9;
    b=100;
    
    % Valeurs a changer
    a = 0.1;
    T=1;
    B=100;
    
    for i=1:Nmc
        X(i)=S0*exp((r-sigma*sigma/2)*(T) + sigma*sqrt(T)*randn) - B;
        gamma(i)=b/((i+1)^lambda);
        Z(i+1) = Z(i) - gamma(i) * (psi(Z(i),X(i))-a);
    end
    disp(Z(Nmc+1));
    plot(Z);
    xlabel 'Nmc'
ylabel 'VaR'
title 'Value at Risk pour B et Alpha '
end

function[] = Travail5()



    S0 = 100;
    r=0;
    sigma=0.4; 
   
    S(1)=S0;
   
        T=1;
        N=100;
        delta_t = T / N;
%         t=(0:N)*delta_t; 
    
    W(1)=0;
    Nmc = 1000000; 
    a=0.1; 
   B=100;
   compt=0;

    for k = 1:Nmc
      
        for i = 1:N
            W(i + 1) = W(i) + sqrt(delta_t) * randn;
            S(i+1)=S(i) + r*S(i)*delta_t + sigma*S(i)*(W(i+1)-W(i));
        end
       
       X(k)=S(N+1) - B;

        
    end
    y=sort(X);
%      y=sort(last_value);
%      s=sort(S);
     v=y(floor(Nmc*a));
%      plot(last_value);
%     disp(+ y);
%     disp(+ s);
    disp(+ v);
    
end

function[P,x]=fonction_Emp_densite(X,a,delta)
N_x=100;
for i =1:N_x+1
    x(i)=a+delta*(i-1);
    cont=0;
    for n=1:length(X)
        if X(n)<=x(i)+delta && X(n)>x(i)
            cont=cont+1;
        end
    end
    P(i)=cont/(length(X)*delta);
    
end
end
function[]=densite_Emp_graphe(a,delta,X)
[P,x]=fonction_Emp_densite(X,a,delta);
figure;
plot(x,P);
xlabel 'x'
ylabel 'f_X(x)'
title 'Fonctions de densité empirique du rendement final RT'
end

function[]= Travail6()
    %Variables
    a=2;
    b=5;
    
    %Constantes
    x0=(a-1)/(a+b-2);
    c=(gamma(a+b)/(gamma(a)*gamma(b)))*(x0^(a-1))*((1-x0)^(b-1));
    k=1;
    Nmc=10000000;
    
    for n=1: Nmc
        y=rand;
        u=rand;
        f=(gamma(a+b)/(gamma(a)*gamma(b)))*y^(a-1)*(1-y)^(b-1);
        g= 1;
        if (u<= f/c*g)
           X(k) = y;
           k=k+1;
        end
        
    end
%     disp(X);
%     plot (X);
    densite_Emp_graphe(0,0.02,X);
end
function[]=Simulation1()
n5=0;
Nmc=1000000;
    for i=1:Nmc
   
    g=randn;
        if randn > 5
          n5=n5+1;
        end

    end
prob=n5/Nmc;
disp("probabilite" +prob);
end
function[]=Simulation2()
Nmc=10000;
u=5;
n5=0
Sum=0;
    for i=1:Nmc
        y=u+randn;
        if y>5
            n5=n5+1
            Sum=Sum+exp(-u*y+u*u*0.5);
        end
    end
    proba=n5/Nmc;
disp(+Sum/Nmc);

end
function[var]=Simulation3(theta)
T=1;
% theta=1;
Nmc=10000;
sum=0;
sum2=0;
for i=1:Nmc
    g=randn;
    W(i)=g*sqrt(T);
    sum=sum+(W(i)+theta*T)*exp(-W(i)*theta-theta*theta*T/2);
    sum2=sum2+((W(i)+theta*T)*exp(-W(i)*theta-theta*theta*T/2))^2;
end
Res=sum/Nmc;
Res2=sum2/Nmc;
var=Res2-Res^2;
disp(Res);
disp(var);
end
function[]=plotvar()
v=-5:0.1:5
for i=1:101
    M(i)=Simulation3(v(i));
end
plot(v,M);
end
function[]=robbinsforsmallalphavalues()
T=1;
theta=1;
function[s] = psi(p,x)
       if(x<=p)
           s=1*exp(-sqrt(T)*theta-theta*theta*T/2);
       else 
           s=0;
       end
    end
    r=0;
    sigma=0.4; 
    
    S0 = 100;
    Z0=0.1;
    Z(1)=Z0;
    Nmc=10000;
    lambda=0.9;
    b=100;
    
    % Valeurs a changer
    a = 0.0000001;
    T=1;
    B=100;
    
    for i=1:Nmc
        X(i)=S0*exp((r-sigma*sigma/2)*(T) + sigma*sqrt(T)*randn) - B;
        gamma(i)=b/((i+1)^lambda);
        Z(i+1) = Z(i) - gamma(i) * (psi(Z(i),X(i))-a);
    end
    disp(Z(Nmc+1));
    plot(Z);
    xlabel 'Nmc'
ylabel 'VaR'
title 'Value at Risk pour B et Alpha '
end