library(rgl)
library(zoo) ## for rollmean


dat15 = read.csv("15_lin_reg_div.csv", sep = ',')
dat14 = read.csv("14lin_reg_div.csv", sep = ',')
colnames(dat14) <- c("diff", "vel1", "vel2")
colnames(dat15) <- c("diff", "vel1", "vel2")

dat14 = dat14[1100:nrow(dat14),]
dat15 = dat15[485:nrow(dat15),]

dat <- rbind(dat14, dat15)

dat_r <- as.data.frame(rollmean(dat, 120, fill=NA))

boxplot(dat$diff)

mod1 = lm(vel2 ~ vel1 + I(vel1^2) + diff+ I(diff^2), data=dat_r)

summary(mod1)
plot(mod1)

##3d plot
newdat <- expand.grid(vel1=seq(0,26,by=0.2),diff=seq(-2,2,by=0.1))
newdat$pp  <- predict(mod1, newdata=newdat)
with(dat,plot3d(vel1, diff, vel2, col="blue", size=0.5, type="s", main="3D Linear Model Fit"))
with(newdat,surface3d(unique(vel1),unique(diff), pp,alpha=0.3,front="line", back="line"))


