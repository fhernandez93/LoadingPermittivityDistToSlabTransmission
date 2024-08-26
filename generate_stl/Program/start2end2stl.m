clc
clear all
r=load("1_sample_L18_lines_cut.mat");
scale = 18
r=r.r*scale;
l=sqrt(        sum((r(:,1:3)-r(:,4:6)).^2,2)           );
R=0.25; %%% radio de los cilindros
N=17; %%% numero de caras de los cilindros
name='1_sample_L18_lines_cut.stl';
r=unique(r,"rows");
a=length(r); 
max(r)
min(r)

l=sqrt(        sum((r(:,1:3)-r(:,4:6)).^2,2)           );
histogram(l)
% 
% 
% lx=12; ly=12; lz=12; 
% r=sortrows(r,1); ind = find(min(abs(r(:,1) - lx)) == abs(r(:,1) - lx)); r(ind(1):end,:)=[];
% r=sortrows(r,2); ind = find(min(abs(r(:,2) - ly)) == abs(r(:,2) - ly)); r(ind(1):end,:)=[];
% r=sortrows(r,3); ind = find(min(abs(r(:,3) - lz)) == abs(r(:,3) - lz)); r(ind(1):end,:)=[];
% 
% lx=0; ly=0; lz=0; 
% r=sortrows(r,1); ind = find(min(abs(r(:,1) - lx)) == abs(r(:,1) - lx)); r(1:ind(1),:)=[];
% r=sortrows(r,2); ind = find(min(abs(r(:,2) - ly)) == abs(r(:,2) - ly)); r(1:ind(1),:)=[];
% r=sortrows(r,3); ind = find(min(abs(r(:,3) - lz)) == abs(r(:,3) - lz)); r(1:ind(1),:)=[];

l=sqrt(        sum((r(:,1:3)-r(:,4:6)).^2,2)           );
histogram(l)

a=length(r);
max(r)
min(r)
% 
% 
% 
% l=sqrt(        sum((r(:,1:3)-r(:,4:6)).^2,2)           );

% r2=r; xxx=max(r2(:,3)); r3=r;
% r2(:,3)=r2(:,3)+xxx; 
% r2(:,6)=r2(:,6)+xxx; 
% 
% r=[r;r2];
% xxx=max(r(:,3)); 
% r3(:,3)=r3(:,3)+xxx; 
% r3(:,6)=r3(:,6)+xxx; 
% 
% 
% 
% r=[r;r3];
min(r)

l=sqrt(        sum((r(:,1:3)-r(:,4:6)).^2,2)           );
histogram(l)






r=sortrows(r,1);
r=sortrows(r,2);  
r=sortrows(r,3); 

% aspec_ratio=8;
 l=sqrt(        sum((r(:,1:3)-r(:,4:6)).^2,2)           );
 histogram(l)




X=[]; Y=[]; Z=[];



for q=1:1:length(r)
[Xs, Ys, Zs] = cylinder2P(R, N,r(q,1:3), r(q,4:6) );
%  [Xs, Ys, Zs] = cylinder2P_v2(R, N,[0 0 0], [0 1 0] , aspec_ratio);
q
for j=1:1:N
X(:,end+1)=Xs(:,j);
Y(:,end+1)=Ys(:,j);
Z(:,end+1)=Zs(:,j);
end
X(:,end+1)=[-200 -200];
Y(:,end+1)=[-200 -200];
Z(:,end+1)=[-200 -200];
end
s=surf(Xs,Ys,Zs);
axis equal
hold on
xlabel('x')
ylabel('y')
zlabel('z')


surf2stl(name,X,Y,Z)