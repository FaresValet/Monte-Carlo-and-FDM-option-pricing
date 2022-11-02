

sigma = 0.3 ; 
L = 20 ; 
T = 1 ; 
r = 0.2 ; 
Sc = 8 ;
N = 100 ;
Nmc = 10000 ;

disp( AsianPriceFixedS0(8,Nmc,r,sigma,T,N,Sc) )
AsianOptionGraphic(Nmc,r,sigma,T,N,Sc)


function[f] = AsianPayoff(S0,r,sigma,T,N,Sc)
    deltat=T/N ; 
    Sum= 0 ; 
    S(1) = S0 ; 
    for i=1:N 
        S(i+1)=S(i)*exp((r - (1/2)*sigma^2)*deltat + sigma*sqrt(deltat) * randn) ;
        Sum = Sum + min(S(i+1),Sc) * deltat ;
    end
    A = Sum / T ; 
    f = max(A-S(N+1),0) ;
end

function[prix] = AsianPriceFixedS0(S0,Nmc,r,sigma,T,N,Sc)
    A = 0 ; 
    for i=1:Nmc 
        gain = AsianPayoff(S0,r,sigma,T,N,Sc) ;
        A = A + gain ;
    end
    prix = exp(-r*T) * A / Nmc ;
end
function[]=AsianOptionGraphic(Nmc,r,sigma,T,N,Sc)
    for j=1:45
        S0(j)=0.4 * (j-1) ; 
        prix(j) =  AsianPriceFixedS0(S0(j),Nmc,r,sigma,T,N,Sc) ;
    end
    plot(S0,prix) ;
    title('Asian option price at  t=0 ')
    legend('Asian option price using  MC')
end


function[f] =AsianOptionFixedTime(t,At,St,r,sigma,T,N,Sc)
    deltat=(T-t)/N ; 
    Somme = At*t ;
    S(1) = St ; 
    for i=1:N 
        S(i+1)=S(i)*exp((r - (1/2)*sigma^2)*deltat + sigma*sqrt(deltat) * randn) ;
        Somme = Somme + min(S(i+1),Sc) * deltat ;
    end
    AT = Somme / T ; 
    f = max(AT-S(N+1),0) ;
end
