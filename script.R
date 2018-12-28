set.seed(1111)

### 1

# Llegim el dataset com diu l'enunciat
Enquesta <- read.csv("Enquesta.csv", header=TRUE,sep=",",dec=",")
summary(Enquesta)

# Problema: s'ha inclòs el número de fila com a variable (Num), l'eliminem
Enquesta$Num <- NULL

# Recodifiquem la variable Salary, com diu l'enunciat
recodificarSalary <- function(x) {
  if (x == "35k")
    return("35k")
  if (x == "<18k" || x == "25k")
    return("<35k")
  return(">35k")
}

Enquesta$Salary.3 <- as.factor(mapply(recodificarSalary,Enquesta$Salary))
# Comprovem que s'ha fet bé:
sum(Enquesta$Salary.3 == "<35k") # 39
sum(Enquesta$Salary.3 == "35k") # 71
sum(Enquesta$Salary.3 == ">35k") # 37

# L'enunciat diu que utilitzem només les variables indicades, que entenem que són aquestes
vars <- c("Image","Exp.gene","Exp.spec","Qual.gen","Qual.spec","Value","Satisfaction","Startwork","Accgrade","Grade", "Salary.3")
Enquesta <- Enquesta[vars]

# Resampling

# precalculate the TR/TE partition and the cross-validation partitions on the TR part

N <- nrow(Enquesta)
all.indexes <- 1:N

learn.indexes <- sample(1:N, round(2*N/3))
test.indexes <- all.indexes[-learn.indexes]

learn.data <- Enquesta[learn.indexes,]
test.data <- Enquesta[test.indexes,]

nlearn <- length(learn.indexes)
ntest <- N - nlearn



### 2
library(randomForest)

#library(caret)

## specify 10x10 CV
#trc <- trainControl (method="repeatedcv", number=10, repeats=10)
#ntrees <- round(10^seq(1,3,by=0.2))
#mtrys <- (1:nrow(Enquesta)-1)

## WARNING: this takes some minutes
#model.10x10CV <- train (Salary.3 ~., data = learn.data, method='rf', trace = FALSE, metric="Accuracy", tuneGrid = expand.grid(.ntree=ntrees, trControl=trc))
#,.mtry=mtrys)


## Now we can try to optimize the number of trees, guided by OOB:

(ntrees <- round(10^seq(1,3,by=0.2)))

# prepare the structure to store the partial results

rf.results <- matrix (rep(0,2*length(ntrees)),nrow=length(ntrees))
colnames (rf.results) <- c("ntrees", "OOB")
rf.results[,"ntrees"] <- ntrees
rf.results[,"OOB"] <- 0

ii <- 1

# hauríem de fer el mateix amb el paràmetre mtry, de 1 a nrow(Enquesta)-1. Segurament hauria de ser un for anidat
# Sense for anidat, ja triga moltíssim
for (nt in ntrees)
{ 
  print(nt)
  
  model.rf <- randomForest(Salary.3 ~ ., data = learn.data, ntree=nt, proximity=FALSE)
                           #sampsize=c(yes=30, no=30), strata=learn.data$Salary.3)
  
  # get the OOB
  rf.results[ii,"OOB"] <- model.rf$err.rate[nt,1]
  
  ii <- ii+1
}

rf.results

# choose best value of 'ntrees'

lowest.OOB.error <- as.integer(which.min(rf.results[,"OOB"]))
(ntrees.best <- rf.results[lowest.OOB.error,"ntrees"]) # 25
# OOB =  0.5204082 -> 1-OOB = 0.4795918

### 3
library(e1071)

bestSVM <- tune(svm, Salary.3~., data = learn.data,
     ranges = list(gamma = 10^seq(-5,2), cost =10^seq(-2,3), kernel=c("linear","polynomial","radial"),degree=c(2,3)), #gamma = 2^(-1:1), cost = 2^(2:4), class.weights = "inverse"
     tunecontrol = tune.control(sampling = "cross",nrepeat=10), coef0=1, scale = TRUE,type="C-classification"#,class.weights=c('<35k' = 40, '35k' = 20, '>35k'  =40)
)

#Parameter tuning of ‘svm’:
  
#  - sampling method: 10-fold cross validation 

#- best parameters:
#  gamma cost     kernel degree
#0.1    1 polynomial      2

#- best performance: 0.4733333 

#Call:
#svm(formula = Salary.3 ~ ., data = learn.data, type = "C-classification", coef0 = 1, kernel = bestSVM$best.parameters$kernel, gamma = bestSVM$best.parameters$gamma, 
#    cost = bestSVM$best.parameters$cost, degree = bestSVM$best.parameters$degree, scale = TRUE)


#Parameters:
#  SVM-Type:  C-classification 
#SVM-Kernel:  polynomial 
#cost:  1 
#degree:  2 
#gamma:  0.1 
#coef.0:  1 

#Number of Support Vectors:  90

(modelSVM <- svm(Salary.3~., data = learn.data, type="C-classification", scale = TRUE,coef0 = 1, kernel=bestSVM$best.parameters$kernel,
                 gamma=bestSVM$best.parameters$gamma,cost=bestSVM$best.parameters$cost,degree=bestSVM$best.parameters$degree))





## Ara decidirem quin és el millor model d'entre el millor model de random forest i el millor d'SVM, que han tingut accés a exactament les mateixes dades de learn
# I a més cap dels dos ha utilitzat test.
# Per decidir quin és el millor *NOOO* utilitzarem test. Agafarem el que ha donat millors resultats en la validació, que és _____
# Ara sí, utilitzarem el millor model trobat en el test. Això servirà per donar una estimació realista del seu rendiment

# SVM té menys error
library(MASS)
library(caret)
p <- factor(predict (modelSVM, newdata=test.data, type="raw"),levels=Enquesta[,11])
c <- confusionMatrix(test.data$Salary.3,p)

overall <- c$overall
overall.accuracy <- overall['Accuracy']  #Accuracy 0.3265306 

# Cal aplicar tècniques per imbalanced (optimitzar per F1, stratitified, class weights...)