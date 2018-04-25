%%
G=zpk([],[-1 -2 -5],[1000])
nyquist(G);
grid on;
%%
G1=tf([2812.5 2250 1800],[1 55.3 616.5 180 0 0])
G2=tf([1],[1])
G=feedback(G1,G2,-1)
pzmap(G)
grid on;
figure;
bode(G)
grid on;
[Gm,Pm,Wcg,Wcp]=margin(G)
%%
k=5;
Gzpk=zpk([],[0 -1 -10],[10*k])
H=tf([1],[1])
G=feedback(Gzpk,H,-1)
nyquist(G)
grid on;
figure;
step(G)
grid on;
figure;
bode(G)
grid on;
%%
%%
k=20;
Gzpk=zpk([],[0 -1 -10],[10*k])
H=tf([1],[1])
G=feedback(Gzpk,H,-1)
nyquist(G)
grid on;
figure;
step(G)
grid on;
figure;
bode(G)
grid on;
