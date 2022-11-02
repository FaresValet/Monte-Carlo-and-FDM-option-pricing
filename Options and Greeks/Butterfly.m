function [m]= Butterfly()
L=50;
K=10;
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
    V(M+2,i)=max(S(i)-K,0)+max(S(i)-3*K,0)-2*max(S(i)-2*K,0);
end
   
for k=M+2:-1:2
   for i=2:N+1
    
      V(k-1,i)= V(k,i+1)*Dt*0.5*(sigma^2 *S(i)^2 /(Ds)^2 +r*(S(i))/(Ds)) +V(k,i)*(1-Dt*((sigma^2*S(i)^2 /(Ds)^2)+r))+V(k,i-1)*Dt/2 *((sigma^2*S(i)^2 )/(Ds)^2 -r*S(i)/Ds);
   end
   
 %V(k-1,1)=V(k-1,2);
 %V(k-1,N+2)=V(k-1,N+1)+ Ds;
   
    
end
figure;
plot(S,V(M+2,:));

hold on
plot(S,V(1,:));
hold on
 plot(S,V(T/(2*Dt)+1,:));
 title("option butterfly");
figure;
plot(t,V(:,1));
xlabel("temps t");
title("condition aux limites S0");
figure;
plot(t,V(:,N+2));
xlabel("temps t");
title("condition aux limites ST");


figure;
mesh(S,t,V);
xlabel('prix de lactifS');
ylabel('temps t');
zlabel('prix de loption');
title('solution de lequation Black-Scholes with butterfly payoff');


end