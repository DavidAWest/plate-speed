library(rgl)
dat = read.csv("lin_reg.csv", sep = ',')
colnames(dat) <- c("diff", "vel1", "vel2")

dat = dat[101:nrow(dat),]

mod1 = lm(vel2 ~ vel1 + I(vel1^2) + diff+ I(diff^2), data=dat)
summary(mod1)
plot(mod1)


newdat <- expand.grid(vel1=seq(0,17,by=0.2),diff=seq(-1400,1400,by=5))
newdat$pp  <- predict(mod1, newdata=newdat)
with(dat,plot3d(vel1, diff, vel2, col="blue", size=0.5, type="s", main="3D Linear Model Fit"))
with(newdat,surface3d(unique(vel1),unique(diff),pp,
                      alpha=0.3,front="line", back="line"))