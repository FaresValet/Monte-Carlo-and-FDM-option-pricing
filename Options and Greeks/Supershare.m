% Dirichlet_Supershare();
% Newmann_Supershare();
% GAMMA();
volatilite();
function[f]=pay_off_supershare(S)
if S<8
    f=0;

elseif S>12 
    f=0;

else
    f=1;
end
end
function[]=Dirichlet_Supershare()
L=30;
T=0.5;
sigma=0.5;
N=99;
M=4999;
Dt=T/(M+1);
Ds=L/(N+1);
t=(1:M+2)*Dt;
S=(1:N+2)*Ds;
S(1)=0;
V=zeros(M+2,N+2);
r=0.1;
for i=1:N+2
    V(M+2,i)=pay_off_supershare(S(i)); %formule supershare
end
     for k=1:M+1
 V(k,1)=0;
 V(k,N+2)=0;
    end
    
for k=M+2:-1:2
   for i=2:N+1
    
      V(k-1,i)= V(k,i+1)*Dt*0.5*(sigma^2 *S(i)^2 /(Ds)^2 +r*(S(i))/(Ds)) +V(k,i)*(1-Dt*((sigma^2*S(i)^2 /(Ds)^2)+r))+V(k,i-1)*Dt*0.5 *((sigma^2*S(i)^2 )/(Ds)^2 -r*S(i)/Ds);
   end
end
figure;
plot(S,V(M+2,:));
title("Prix de l'option t=T difference finie ");
figure;
plot(S,V(1,:));

 title("Prix de l'option t=0 difference finie ");

figure;
mesh(S,t,V);
xlabel('prix de lactifS');
ylabel('temps t');
zlabel('prix de loption');
title('solution de lequation Black-Scholes');
disp(V(T/(4*Dt) +1,floor((N+1)/3)));
end
function[f]=Newmann_Supershare()
L=30;
T=0.5;
sigma=0.5;
N=99;
M=4999;
Dt=T/(M+1);
Ds=L/(N+1);
t=(1:M+2)*Dt;
S=(1:N+2)*Ds;
S(1)=0;
V=zeros(M+2,N+2);
r=0.1;
for i=1:N+2
    V(M+2,i)=pay_off_supershare(S(i)); %formule supershare
end
  
    
for k=M+2:-1:2
   for i=2:N+1
    
      V(k-1,i)= V(k,i+1)*Dt*0.5*(sigma^2 *S(i)^2 /(Ds)^2 +r*(S(i))/(Ds)) +V(k,i)*(1-Dt*((sigma^2*S(i)^2 /(Ds)^2)+r))+V(k,i-1)*Dt*0.5 *((sigma^2*S(i)^2 )/(Ds)^2 -r*S(i)/Ds);
   end
    V(k-1,1)=V(k-1,2);
 V(k-1,N+2)=V(k-1,N+1);
end
figure;
plot(S,V(M+2,:));
title("Prix de l'option t=T difference finie ");
figure;
plot(S,V(1,:));

 title("Prix de l'option t=0 difference finie ");

figure;
mesh(S,t,V);
xlabel('prix de lactifS');
ylabel('temps t');
zlabel('prix de loption');
title('solution de lequation Black-Scholes');
disp(V(T/(4*Dt) +1,floor((N+1)/3)));
f=V;
end
function[]=GAMMA()
V=Newmann_Supershare();
L=30;
T=0.5;
sigma=0.5;
N=99;
M=4999;
Dt=T/(M+1);
Ds=L/(N+1);
t=(1:M+2)*Dt;
S=(1:N+2)*Ds;

 r=0.1;


for n=1:M+2
    for k=1:N+1
        Gamma(n,k)=(V(n,k+1)+V(n,k-1)-2*V(n,k))/Ds^2;
    end
%   
    Gamma(n,1)=Gamma(n,2);
    Gamma(n,N+2)=Gamma(n,N+1);
end
plot(S,Gamma(1,:));
mesh(S,t,Gamma);
xlabel('prix de lactifS');
ylabel('temps t');
zlabel('prix de loption');
title('solution de lequation Black-Scholes');
mesh(S,t(1:floor((2*Dt))+1,Gamma(1:floor(T/(2*Dt))+1,:)));
end
function[prix]=Prix_St_Fixe_t_fixe(t,St)
L=30;
T=0.5;
sigma=0.5;

r=0.1;
Nmc=100;
for n=1:Nmc
    ST(n)=St*exp((r-(sigma^2)/2)*(T-t)+sigma*sqrt(T-t)/randn);
    gain(n)=pay_off_supershare(ST(n));
end
prix=exp(-r*(T-t))*mean(gain);

end
function[]=volatilite()
L=30;
T=0.5;
Ns=99;
Nt=99;
ds=L/Ns
for j=1:(Ns+1)
    for k=1:(Nt+1)
        St(j)=(L/Ns)*(j-1);
        t(k)=(T/Nt)*(k-1);
        V(k,j)=Prix_St_Fixe_t_fixe(t(k),St(j));
    end
end
plot(St,V(Nt+1,:));
hold on 
plot(St,V(1,:));
figure;
mesh(St,t,V);
disp(V(floor(T/(3*ds))+1,floor(6/(ds))+1));
disp(ST);
end