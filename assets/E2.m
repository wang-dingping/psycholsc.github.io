%%
G1=tf([80],[1 2 0])
step(G1,2)
grid on;
%%
a=0.6;
b=5;
Gtf=tf([b*b],[1 2*a*b b*b])
step(Gtf)
% td=0.271s
% tr=0.371s
% tp=0.783s
% ts=1.19s
% sigma=9.48%
%%
a=0.707;
b=1;
Gtf=tf([b*b],[1 2*a*b b*b])
step(Gtf)
grid on;
figure;
a=1;
b=1;
Gtf=tf([b*b],[1 2*a*b b*b])
step(Gtf)
grid on;
figure;
a=2;
b=1;
Gtf=tf([b*b],[1 2*a*b b*b])
step(Gtf)
grid on;
%%
a=0.5;
b=1;
Gtf=tf([b*b],[1 2*a*b b*b])
step(Gtf)
grid on;
figure
a=0.5;
b=5;
Gtf=tf([b*b],[1 2*a*b b*b])
step(Gtf)
grid on;
figure
a=0.5;
b=10;
Gtf=tf([b*b],[1 2*a*b b*b])
step(Gtf)
grid on;