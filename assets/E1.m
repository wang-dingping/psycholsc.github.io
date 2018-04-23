%%
num=[2 18 40]
den=[1 5 8 6]
Gtf=tf(num,den)
Gzpk=zpk(Gtf)
Gss=ss(Gtf)
pzmap(Gzpk)
grid on;
%%
A=[0 1 0 0;0 0 1 0;0 0 0 1;-1 -2 -3 -4]
B=[0;0;0;1]
C=[10 2 0 0]
D=[0]
Gss=ss(A,B,C,D)
Gtf=tf(Gss)
Gzpk=zpk(Gss)
pzmap(Gzpk)
grid on;
%%
num1=[2 6 5]
den1=[1 4 5 2]
Gtf1=tf(num1,den1)
num2=[1 4 1]
den2=[1 9 8 0]
Gtf2=tf(num2,den2)
z=[-3 -7]
p=[-1 -4 -6]
k=[5]
Gzpk=zpk(z,p,k)
G=Gtf1*Gtf2*Gzpk
%%
Gtf1=tf([1],[1 1])
Gtf2=tf([1],[0.5,1])
Gtf3=tf([3],[1 0])
Gtf4=Gtf2
G=feedback((Gtf1+Gtf2)*Gtf3,Gtf4,-1)
%%
G1=tf([10],[1 1])
G2=tf([2],[1 1 0])
G3=zpk([-3],[-2],[1])
G4=tf([5 0],[1 6 8])
G=feedback(G1*feedback(G2,G3,+1),G4,-1)
