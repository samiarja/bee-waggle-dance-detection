ts = 13;
tt = 0:12;
y = ((tt-ts)./6).^8;


figure(64536); clf;
plot(tt,y)
cnt = 13;
rs = 25; cs = 25;
for r = 1:rs
    for c = 1:cs
        d = sqrt((r-cnt)^2+(c-cnt)^2);
        yy(r,c) = (max(cnt-d,0)/6)^4;
    end
end

figure(3534); clf;
%imagesc(yy); 
surf(yy)
axis image;
colorbar