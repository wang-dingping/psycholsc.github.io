A=[-2 -1 1;1 0 1;-1 0 1]
B=[1;1;1]
P=[-1 -2 -3]
K=place(A,B,P)
%%
A=[0 -1;-3 -4]
B=[0;1]
C=[2 0]
D=[]
Gss=ss(A,B,C,D)
nyquist(Gss)
grid on;
figure;
bode(Gss)
grid on