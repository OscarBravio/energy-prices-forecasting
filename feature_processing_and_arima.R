
library(readr)
library(forecast)
library(TSA)
library(stringr)
library(quantmod)
library(dplyr)

df <- read_csv("~/Oskar/energy-prices-forecasting/new_df.csv")
#test_df <- read_csv("~/Oskar/energy-prices-forecasting/test_df.csv")


# exploratory analysis - checking stationarity of time-series

week_ma=df[c('ma_rdn_168','ma_cro_168')]

z=scale(week_ma)

km1=kmeans(z,4)

kol1=km1$cluster+1

grouped_feat=group_by(as.data.frame(cbind(df$ma_cro_168,df$ma_rdn_168,kol1)),kol1)
summarise(grouped_feat,mean(V1),mean(V2))

plot(df$ma_cro_168,col=kol1)
plot(df$ma_rdn_168,col=kol1)


# train-test split

df2=subset(df, day>9)

train_split=round((max(df2$day)-min(df2$day))*0.67)+min(df2$day)

train_df=subset(df, day>10)
test_df=df[481:nrow(df),]


# building ARIMA models

x_cols=c('demand','supply1','supply2','wind','reserve')

rdn_ts=ts(train_df$lag_rdn_24, frequency=24)
cro_ts=ts(train_df$lag_cro_72, frequency=24)

model1=arimax(log(rdn_ts+1),order=c(1,1,1),seasonal=c(1,0,0),xreg=train_df[x_cols])

model2=arimax(log(cro_ts+1),order=c(1,1,1),seasonal=c(1,0,0),xreg=train_df[x_cols])

arima_rdn=as.vector(exp(stats::predict(model1,newx=train_df[x_cols])$pred)-1)
arima_cro=as.vector(exp(stats::predict(model2,newx=train_df[x_cols])$pred)-1)

x_train2=cbind(train_df,arima_rdn,arima_cro)

write_csv(x_train2,'~/Oskar/energy-prices-forecasting/train_df.csv')

# arimax models - forecasts in a loop

iters=nrow(test_df)/24

pred_rdn=c()
pred_cro=c()

x=train_df[x_cols]
xt=test_df[x_cols]

y1=rdn_ts
y2=cro_ts

y_test1=test_df$rdn
y_test2=test_df$cro

for (i in 1:iters){

    model1=arimax(log(y1+1),order=c(2,1,1),seasonal=c(1,0,0),xreg=x)
    model2=arimax(log(y2+1),order=c(2,1,1),seasonal=c(1,0,0),xreg=x)
    
    # forecasts
    x_i=xt[(1+24*(i-1)):(24*i),]
    pred_t1=as.vector(exp(stats::predict(model1,newx=x_i)$pred)-1)
    pred_t2=as.vector(exp(stats::predict(model2,newx=x_i)$pred)-1)
    
    pred_rdn=c(pred_rdn,pred_t1)
    pred_cro=c(pred_cro,pred_t2)
    
    x=rbind(x,x_i)
    
    y1=ts(c(as.vector(y1),y_test1[(1+24*(i-1)):(24*i)]),frequency=24)
    y2=ts(c(as.vector(y2),y_test2[(1+24*(i-1)):(24*i)]),frequency=24)
    
}

# because logarithm of price was forecasted, we have to invert transformation back to original price

arima_rdn=pred_rdn
arima_cro=pred_cro

x_test2=cbind(test_df,arima_rdn,arima_cro)
write_csv(x_test2,'~/Oskar/energy-prices-forecasting/test_df.csv')

