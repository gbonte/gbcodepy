### "Statistical foundations of machine learning" software
## R package gbcode 
## Author: G. Bontempi


rm(list=ls())




corDC<-function(X,Y){
  ## correlation continuous matrix and discrete vector
  ## NB: the notion of sign has no meaning in this case. Mean of absolute values is taken
  ## 14/11/2011

  if (!is.factor(Y))
    stop("This is not the right function. Y is not a factor !!")

  N<-NROW(X)
  L<-levels(Y)

  if( length(L)==2)
    lL<-1
  else
    lL<-length(L)

  cxy<-NULL
  for (i in 1:lL){
    yy<-numeric(N)
    ind1<-which(Y==L[i])
    ind2<-setdiff(1:N,ind1)
    yy[ind1]<-1
    cxy<-cbind(cxy,abs(cor(X,yy)))
  }

  apply(cxy,1,mean)
}

corXY<-function(X,Y){
  ## correlation continuous matrix and continuous/discrete vectormatrix
  ## 14/11/2011

  n<-NCOL(X)
  N<-NROW(X)
  m<-NCOL(Y)

  cXY<-array(NA,c(n,m))

  for (i in 1:m){
    if (m==1)
      YY<-Y
    else
      YY<-Y[,i]
    if (is.numeric(YY)){
      cXY[,i]<-cor(X,YY,use="pairwise.complete.obs")
    } else {
      cXY[,i]<-corDC(X,YY)
    }
  }
  cXY
}


cor2I2<- function(rho){
  rho<-pmin(rho,1-1e-5)
  rho<-pmax(rho,-1+1e-5)
  -1/2*log(1-rho^2)
}

rankrho<-function(X,Y,nmax=5,regr=FALSE){
  
  m<-NCOL(Y)
  ## number of outputs
  
  X<-scale(X)
  
  Iy<-cor2I2(corXY(X,Y))
  
  if (m>1)
    Iy<-apply(Iy,1,mean)
  
  return(sort(c(Iy), decreasing=T, index.return=T)$ix[1:nmax])
  
}

KNN<- function(X,Y,k,q ){
  l<-levels(Y)
  N<-nrow(X)
  
  d<-sqrt(apply((X-array(1,c(N,1))%*%q)^2,1,sum)) ## Euclidean metric
  ## d<-sqrt(apply(abs(X-array(1,c(N,1))%*%q),1,sum)) ## Manhattan metric
  ##  d<-1/cor(t(X),q)           ## correlation metric
  
  index<-sort(d,index.return=TRUE)
  cnt<-numeric(length(l))
  for (i in 1:k){
    cnt[Y[index$ix[i]]]<-cnt[Y[index$ix[i]]]+1
    
  }
  l[which.max(cnt)]
  
}
set.seed(0)


KNN.wrap<-function(X,Y,size,K=1){
## leave-one-out wrapper based on forward selection and KNN
  n<-ncol(X)
  N<-nrow(X)
  selected<-NULL
  while (length(selected)<size){
    miscl.tt<-numeric(n)+Inf
    for (j in 1:n){
      if (! is.element(j,selected)){ 
        select.temp<-c(selected,j)
        miscl<-numeric(N)
        for (i in 1:N) {
          X.tr<-array(X[-i,select.temp],c(N-1,length(select.temp)))
          Y.tr<-Y[-i]
          q<-array(X[i,select.temp],c(1,length(select.temp)))
          Y.ts<-Y[i]
          Y.hat.ts <- KNN(X.tr, Y.tr,K,q)
          miscl[i]<-Y.hat.ts!=Y.ts          
        }
        miscl.tt[j]<-mean(miscl)             
      }
    }
    selected<-c(selected,which.min(miscl.tt))
    cat(".")
    
  }
  cat("\n")
  selected
}



K=3 ## number of neighbours in KNN
load("golub.Rdata")  ## dataset upload


N<-nrow(X)



I<-sample(N)
X<-scale(X)


X<-X[I,]
Y<-Y[I]

## Training/test partition
N.tr<-40
X.tr<-X[1:N.tr,]
Y.tr<-Y[1:N.tr]
N.ts<-32
X.ts <- X[(N.tr+1):N,]
Y.ts<-Y[(N.tr+1):N]


## preliminary dimensionality reduction by ranking
ind.filter<-rankrho(X.tr,Y.tr,100)
X.tr=X.tr[,ind.filter]
X.ts=X.ts[,ind.filter]


## wrapper feature selection 
wrap.var<-KNN.wrap(X.tr,Y.tr,size=20,K)


###########################################
# Assessement of classification in the testset

for ( size in c(2:length(wrap.var))){
  
  miscl<-numeric(N.ts)
  Y.hat.ts<-numeric(N.ts)
  Conf.tt<-array(NA,c(2,2))
  for (i in 1:N.ts){
    q<-X.ts[i,]
    Y.hat.ts[i]<-KNN(X.tr[,wrap.var[1:size]],Y.tr,K,q[wrap.var[1:size]])
    miscl[i]<-Y.hat.ts[i]!=Y.ts[i]
  }
  
  miscl.tt<-mean(miscl)
  rownames(Conf.tt)=c("pred=0","pred=1")
  colnames(Conf.tt)=c("real=0","real=1")
  Conf.tt[1,1]<-length(which(Y.hat.ts=="0" & Y.ts =="0"))
  Conf.tt[1,2]<-length(which(Y.hat.ts=="0" & Y.ts =="1"))
  Conf.tt[2,1]<-length(which(Y.hat.ts=="1" & Y.ts =="0"))
  Conf.tt[2,2]<-length(which(Y.hat.ts=="1" & Y.ts =="1"))
  
  
  print(paste("K=",K, "size=",size,"; Misclass %=",miscl.tt))
  print(Conf.tt)
  
}

