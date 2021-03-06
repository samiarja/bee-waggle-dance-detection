clear;

dataPath = "final_labels/20210803t1719d200m_cropped";
load(dataPath + '/20210803t1719d200m_cropped_ground_truth.mat');


kk = 10; 
jj = 200; 
%x = td_gt.x(1500:3000);
%y = td_gt.y(1500:3000);

x = td_gt.x;
y = td_gt.y;
nn = numel(x);

  
% sx = movmean(x,kk);
% sy = movmean(y,kk);
% szx = movmean(x,kk)-movmean(x,jj);
% szy = movmean(y,kk)-movmean(y,jj);

sx = movmedian(x,kk);
sy = movmedian(y,kk);
szx = movmedian(x,kk)-movmedian(x,jj);
szy = movmedian(y,kk)-movmedian(y,jj);



figure(525234); clf;
subplot(2,1,1);hold on;
plot(x); 
plot(sx,'.-'); 
grid on;
subplot(2,1,2);hold on;
plot(y); grid on;
plot(sy,'.-'); 

figure(234523)
plot3(sx,sy,1:nn);
grid on;


figure(525231); clf;
subplot(2,1,1);hold on;
plot(szx,'.-'); 
grid on;
subplot(2,1,2);hold on;
plot(szy,'.-'); 
grid on;


figure(2345231)
plot3(szx,szy,1:nn);
grid on;
xlim([-60 60])
ylim([-60 60])

thArray = 1:180;
nTh = numel(thArray);
figure(354534);clf

for iTh = thArray
    th = deg2rad(thArray(iTh));
    
    
    R = [cos(th) -sin(th); sin(th) cos(th)];
    gg = [szx szy]*R;
    
    szxR = gg(:,1);
    szyR = gg(:,2);
    
    
    subplot(2,1,1); hold on;
    plot(szxR); 
    grid on;
    subplot(2,1,2); hold on;
    plot(szyR);
    grid on;
    
    %     px(iTh) = sum(abs(szxR));
    %     py(iTh) = sum(abs(szyR));
    px(iTh) = sum(szxR.^2);
    py(iTh) = sum(szyR.^2);
end
figure(645645); clf;
plot(thArray,px); hold on;
plot(thArray,py); hold on;
grid on;
% 
[Vx,Ix]=max(px);
[Vy,Iy]=max(py);
if Vy>Vx
    I = Iy;
else
    I = Ix;
end

% 
% [Vx,Ix]=min(px);
% [Vy,Iy]=min(py);
% if Vy<Vx
%     I = Iy;
% else
%     I = Ix;
% end


th = deg2rad(thArray(I));
R = [cos(th) -sin(th); sin(th) cos(th)];
gg = [szx szy]*R;

szxRotBest = gg(:,1);
szyRotBest = gg(:,2);

t = 1:nn;
MinPeakDistance = 120;
figure(454398);clf;
subplot(2,3,[1 2]);
[px ll] = findpeaks(szxRotBest,'SortStr','descend','MinPeakDistance',MinPeakDistance);
plot(t,szxRotBest);hold on
plot(t(ll),px,'k^','MarkerFaceColor','g');
[pn ln] = findpeaks(-szxRotBest,'SortStr','descend','MinPeakDistance',MinPeakDistance);
plot(t(ln),-pn,'kv','MarkerFaceColor','r');title("X");
grid on;
subplot(2,3,[4 5]);
[py ly] = findpeaks(szyRotBest,'SortStr','descend','MinPeakDistance',MinPeakDistance);
plot(t,szxRotBest);hold on
plot(t(ly),py,'k^','MarkerFaceColor','g');
[pn ln] = findpeaks(-szxRotBest,'SortStr','descend','MinPeakDistance',MinPeakDistance);
plot(t(ln),-pn,'kv','MarkerFaceColor','r');title("Y");
grid on;
subplot(2,3,[3 6]);
plot3(szxRotBest,szyRotBest,t,'r');
xlabel("X");ylabel("Y");zlabel("Z")
grid on;
xlim([-60 60])
ylim([-60 60])

figure_8_motionts = find(t(t>1950 & t < 2350))';
figure_8_motionx = szxRotBest(figure_8_motionts);
figure_8_motiony = szyRotBest(figure_8_motionts);

figure(567560);
plot3(figure_8_motionx,figure_8_motiony,figure_8_motionts,'r');grid on;hold on
xlim([-60 60])
ylim([-60 60])

figure(5675350);
time=linspace(0,4*pi,40);
x=cos(time);y=sin(time);z=0.2*time;
p=[figure_8_motionx,figure_8_motiony,figure_8_motionts];
p1=p(1:end-1,:);
p2=p(2:end,:);
arrow3d(p1,p2,10,'cylinder',[0.5,0.5]);axis equal;grid on;

%%
p=[td_gt.x,td_gt.y,td_gt.angle,td_gt.frameID];
p1=p(1:end-1,:);
p2=p(2:end,:);
differences = nan(numel(p1),3);

thArray = 1:180;
for iTh = thArray
    th = deg2rad(thArray(iTh));
    
    R = [cos(th) -sin(th); sin(th) cos(th)];
    gg = [p(:,1) p(:,2)]*R;
    
    szxR = gg(:,1);
    szyR = gg(:,2);
    
    for num = 1:numel(szxR)
        u = [p1(num,1) p1(num,2) p1(num,4)];
        v = [p2(num,1) p2(num,2) p2(num,4)];
        C=cross(u,v);
        NC=norm(C);
        D=dot(u,v);
        ThetaInDegrees = atan2d(NC,D);
        Rep=5;
        uf = repmat(u,5,1);
        vf = repmat(v,5,1);
        vC=cross(uf,vf,2); % vectorized
        vNC=vecnorm(vC,2,2); % since only z-rotation is allowed anyway, this is equivalent to: vNC=vC(:,3)
        vD=dot(uf,vf,2);
        vThetaInDegrees = mean(atan2d(vNC,vD));
        angle =  [rad2deg(vThetaInDegrees) td_gt.angle(num+1)];
        differences(num,1) = rad2deg(vThetaInDegrees);
        differences(num,2) = td_gt.angle(num);
        differences(num,3) = td_gt.angle(num+1) - new_angle;
    end
end
%%

p=[td_gt.x,td_gt.y,td_gt.angle,td_gt.frameID];
p1=p(1:end-1,:);
p2=p(2:end,:);

final_angle = nan(numel(p(:,1))-1,2);
for num = 1:numel(p(:,1))-1
    P1 = [p1(num,1) p1(num,2) p1(num,4)];
    P2 = [p2(num,1) p2(num,2) p2(num,4)];
    
    angleradian = acos((P2(2)-P1(2)) / norm(P2-P1));
    angledegree = rad2deg(angleradian);
    angle_info = [angleradian angledegree];
    final_angle(num,1) = 360 - angledegree;
    final_angle(num,2) = td_gt.angle(num);
end
