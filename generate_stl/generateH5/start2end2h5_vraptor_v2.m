clc 
clear all

% data=single(randi(13,500,500,500));
resolution=[500 500 500]-1;
% h5create('test_unit8_epsilon.h5','/epsilon',resolution,'Datatype','uint8')
% h5write("test_unit8_epsilon.h5","/epsilon",data)
% h5disp("test_unit8_epsilon.h5")
file_name='1_sample_L18_lines_cut.h5'
epsilon=uint8(13);

r=readtable('1_sample_L18_lines_cut.dat'); 
r = table2array(r); 
l=sqrt(        sum((r(:,1:3)-r(:,4:6)).^2,2)           );
promedio=mean(l)
r=r/promedio;
l=sqrt(        sum((r(:,1:3)-r(:,4:6)).^2,2)           );
histogram(l)
pos_minxy=min(  [ min(r(:,1))   min(r(:,2))  min(r(:,4))  min(r(:,5))  ] ); 
r(:,1)=r(:,1)-pos_minxy; r(:,2)=r(:,2)-pos_minxy; r(:,4)=r(:,4)-pos_minxy; r(:,5)=r(:,5)-pos_minxy;
pos_minz=min(  [ min(r(:,3))   min(r(:,6))  ] )
r(:,3)=r(:,3)-pos_minz; r(:,6)=r(:,6)-pos_minz;
min(r)
max(r)
radius=0.3122; %%% here is important to put a consistent radius 0.55 este es el bueno
sizex=max(max(r(:,1)),max(r(:,4)))-min(min(r(:,1)),min(r(:,4))); 
sizey=max(max(r(:,2)),max(r(:,5)))-min(min(r(:,2)),min(r(:,5))); 
sizez=max(max(r(:,3)),max(r(:,6)))-min(min(r(:,3)),min(r(:,6)));
structure_size=[sizex,sizey,sizez];
%%% reshape of the structure
reshape_factor=mean(resolution./structure_size);
% r(:,1)=r(:,1)*reshape_factor(1); r(:,4)=r(:,4)*reshape_factor(1);
% r(:,2)=r(:,2)*reshape_factor(2); r(:,5)=r(:,5)*reshape_factor(2);
% r(:,3)=r(:,3)*reshape_factor(3); r(:,6)=r(:,6)*reshape_factor(3);
r=round(r*reshape_factor); 
R=round(radius*reshape_factor);
r=r+1;
l=sqrt(        sum((r(:,1:3)-r(:,4:6)).^2,2)           );
rmean=mean(l)
histogram(l)
maxsize=max(max(r));
resolution=[maxsize,maxsize,maxsize];
data=ones(resolution,'int8');
r=sortrows(r,1);
r=sortrows(r,2);  
r=sortrows(r,3);
min(r)
max(r)
XYZout=[];


for j=1:1:length(r)

XYZout=[];
for q=0:0.25:R

rstart=r(j,1:3);
rend=r(j,4:6);
Distance = round ( pdist2(rstart,rend));
N=1024; Radio=q*ones(1,Distance);
[Xout, Yout, Zout] = solidcylinder2P(Radio, N,rstart,rend);
XYZoutq=[Xout; Yout; Zout]';
XYZoutq=round((XYZoutq));
XYZoutq=unique(XYZoutq,"rows");
aux=size(XYZoutq);
for p=1:1:aux(1)
XYZout(:,end+1)=XYZoutq(p,:);
end
end
XYZout=XYZout';

for p=1:1:length(XYZout)
    rpoint=XYZout(p,:);
    if rpoint<=maxsize & rpoint>0
    data(rpoint(1),rpoint(2),rpoint(3))=epsilon;
    end 
end 


j
end

r_spheres=[r(:,1:3);r(:,4:6)]; 
r_spheres=unique(r_spheres,"rows");



for q=1:1:length(r_spheres)
    q
r_center=r_spheres(q,:); 

XYZ_cube = cartprod((r_center(1)-R:r_center(1)+R),(r_center(2)-R:r_center(2)+R),(r_center(3)-R:r_center(3)+R));
D=pdist2(r_center,XYZ_cube); 
XYZ_cube=cat(2,XYZ_cube,D');
XYZ_cube( XYZ_cube(:,4)>R,: )=[]; 
XYZ_cube=XYZ_cube(:,1:3);
for p=1:1:length(XYZ_cube)
    rpoint=XYZ_cube(p,:);
    if rpoint<=maxsize & rpoint>0
    data(rpoint(1),rpoint(2),rpoint(3))=epsilon;
    end 
end
end 


 phi=sum(sum(sum(data==epsilon)))/(resolution(1)*resolution(2)*resolution(3));

h5create(file_name,'/epsilon',size(data),'Datatype','uint8')
h5create(file_name,'/positions',size(r),'Datatype','double')
h5create(file_name,'/phi',size(phi),'Datatype','double')
h5write(file_name,"/positions",r)
h5write(file_name,"/epsilon",data)
h5write(file_name,"/epsilon",data)
h5disp(file_name)


image(data(:,:,500))

















