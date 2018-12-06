
library(readr)
library(forecast)
library(TSA)
library(quantmod)

df <- read_csv("~/Desktop/energy-prices-forecasting/dane.csv")

weekends=c(20180407,20180408,20180414,20180415,20180421,20180422,20180428,20180429,
           20180501,20180503,20180505,20180506,20180512,20180513,20180519,20180520,
           20180526,20180527,20180602,20180603,20180609,20180610,20180617,20180618)


# adding flag to data if day is weekend or not

weekeneds=cbind(weekends,rep(1,length(weekends)))
colnames(weekeneds)=c('Data','weekend')

df2=merge(df,weekeneds,by='Data',all.x = TRUE)
df2$weekend[is.na(df2$weekend)==T]=0

# sort by dates and hours

df2=df2[order(df2$Data,df2$godz),]


# creating time-series features - lags, moving averages, moving standard deviations, ratios etc.

cro2=Lag(df2$cro,72)
cro3=Lag(df2$cro,73)
cro4=Lag(df2$cro,96)
cro5=Lag(df2$cro,168)

ma_cro2=(cro2+cro4)/2
ma_cro3=(cro2+cro4+cro5)/3

sd_cro1=apply(cbind(cro2,cro4),1,sd)
sd_cro2=apply(cbind(cro2,cro4,cro5),1,sd)

ratio_cro1=(ma_cro2-ma_cro3)/(sd_cro2+1)
ratio_cro2=(sd_cro1-sd_cro2)/(sd_cro2+1)

fix2=Lag(df2$rdn,24)
fix3=Lag(df2$rdn,25)
fix4=Lag(df2$rdn,48)
fix5=Lag(df2$rdn,72)
fix6=Lag(df2$rdn,168)

ma_fix2=(fix2+fix4)/2
ma_fix3=(fix2+fix4+fix5)/3
ma_fix4=(fix2+fix4+fix5+fix6)/4

sd_fix1=apply(cbind(fix2,fix4),1,sd)
sd_fix2=apply(cbind(cro2,cro4,cro5),1,sd)

se2=lag(df2$SE4,24)
se3=lag(df2$SE4,48)
se4=lag(df2$SE4,72)

z=cbind(cro2,cro3,cro4,cro5,ma_cro2,ma_cro3,sd_cro1,sd_cro2,ratio_cro1,ratio_cro2,
        fix2,fix3,fix4,fix5,fix6,ma_fix2,ma_fix3,ma_fix4,sd_fix1,sd_fix2,
        se2,se3,se4)

colnames(z)=cbind('cro2','cro3','cro4','cro5','ma_cro2','ma_cro3','sd_cro1','sd_cro2','ratio_cro1','ratio_cro2',
        'fix2','fix3','fix4','fix5','fix6','ma_fix2','ma_fix3','ma_fix4','sd_fix1','sd_fix2',
        'se2','se3','se4')

z_col=colnames(z)


# fundamental features

x1=df2[c('godz','deman','supply','wind_prod','reserve','cro','rdn','weekend')]

x=cbind(x1,z)

n=nrow(x)


# train-test division and saving data

x_train=x[169:(n-144),]
x_test=x[(n-143):n,]

write_csv(x_train,'~/Desktop/energy/train_data.csv')
write_csv(x_test,'~/Desktop/energy/test_data.csv')

eee=prcomp(z[200:nrow(z),])
summary(eee)

# linear regression - logarithm of prices dependent on fundamental features

x=x_train[c('deman','supply','wind_prod','reserve','cro2','ma_cro2','se2')]
y=x_train$rdn

model1=lm(log(y+1)~.,x)
summary(model1)


# calculate residuals

e=model1$residuals


# transforming residuals into time-series 

rdn_ts=ts(e,frequency=24)


# checking optimal ARIMA parameters of residuals

model01=auto.arima(rdn_ts)
summary(model01)

x2=x[c('deman','supply','wind_prod','reserve','cro2','ma_cro2','se2')]
x3=x_test[c('deman','supply','wind_prod','reserve','cro2','ma_cro2','se2')]
y_test=x_test$rdn

# arimax models - forecasts in a loop

iters=nrow(x3)/24

predictions=c()

for (i in 1:iters){

    model3=arimax(log(y+1),order=c(0,1,1),seasonal=c(2,0,0),xreg=x2)
    
    # forecasts
    x_t=x3[(1+24*(i-1)):(24*i),]
    pred_t=as.vector(stats::predict(model3,newx=x_t)$pred)
    
    predictions=c(predictions,pred_t)
    x2=rbind(x2,x_t)
    y=c(y,y_test[(1+24*(i-1)):(24*i)])
}

# because logarithm of price was forecasted, we have to invert transformation back to original price

inv_pred=exp(predictions)-1
write_csv(as.data.frame(inv_pred),'~/Desktop/energy/arimax_forecasts.csv')

