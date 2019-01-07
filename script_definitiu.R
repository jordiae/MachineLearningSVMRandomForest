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

#N <- nrow(Enquesta)

#all.indexes <- 1:N

#learn.indexes <- sample(1:N, round(2*N/3))
#test.indexes <- all.indexes[-learn.indexes]

#learn.data <- Enquesta[learn.indexes,]
#test.data <- Enquesta[test.indexes,]

#nlearn <- length(learn.indexes)
#ntest <- N - nlearn

# Hem provat de fer el resampling "estàndard" vist a laboratori (codi anterior, comentat), però el problema és que
# en aquest cas és molt fàcil que les particions quedin amb proporcions de cada classe excessivament diferents

# Farem el remostreig estratificat per mantenir les proporcions (el dataset està desbalancejat i hi ha molt poques dades)
# Hem cercat una llibreria que ens permeti fer-ho: splitstackshape amb la seva funció stratified()
N <- nrow(Enquesta)
library(splitstackshape)
indexs <- Enquesta["Salary.3"]
indexs$num <- seq.int(nrow(Enquesta))
elementsPerClasseALearn = c(round((2*N/3)*(sum(indexs$Salary.3 == "<35k")/N)),round((2*N/3)*(sum(indexs$Salary.3 == "35k")/N)),round((2*N/3)*sum(indexs$Salary.3 == ">35k")/N))
names(elementsPerClasseALearn) = c("<35k","35k",">35k")
particions <- stratified(indexs,c("Salary.3"),bothSets = TRUE,size=elementsPerClasseALearn)

learn.indexes <- particions$SAMP1$num
test.indexes <- particions$SAMP2$num
learn.data <- Enquesta[learn.indexes,]
test.data <- Enquesta[test.indexes,]



# Comentaris sobre el dataset:

# No escalarem les dades perquè ja estan escalades (el creador del dataset ja ho ha fet, observem mitjana = 0 i var = 1 en les contínues)
# Segons la documentació, les categòriques ja es codifiquen amb dummy variables al fer les crides per, exemple, a SVM. A l'estar ordenades,
# també hem provat de convertir-les a enters. Però els resultats no han millorat.
# Pel fet que el dataset estigui desbalancejat, podríem reportar mesures com ara F1, però no està tan desbalancejat com el cas del laboratori.

### 2
library(randomForest)
set.seed(1111)

# Tenint tan poques dades, ens podem permetre optimitzar tant ntrees com mtry (encara que l'heurística pel default de mtry és molt bona)
# Com que està basat en bagging, no cal CV.
# Agafarem els paràmetres que donin menys OOB (out of bag error)

(ntrees <- round(10^seq(1,3,by=0.2)))
mtrys <- (1:(ncol(Enquesta)-1))
# Matriu per desar OOBs per cada model provat
rf.results <- matrix (rep(0,3*(length(ntrees))*(length(mtrys))),nrow=(length(ntrees)*length(mtrys)))
colnames (rf.results) <- c("ntrees", "mtrys", "OOB")
rf.results[,"ntrees"] <- ntrees
rf.results[,"mtrys"] <- mtrys
rf.results[,"OOB"] <- 0
ii <- 1
for (nt in ntrees)
{ 
  for (mtr in mtrys) 
  {
    model.rf <- randomForest(Salary.3 ~ ., data = learn.data, ntree=nt, mtry = mtr, proximity=FALSE, 
                             strata=learn.data$Salary.3)
    
    # get the OOB
    rf.results[ii,"OOB"] <- model.rf$err.rate[nt,1]
    
    ii <- ii+1
  }
}
# choose best value of 'ntrees'
(lowest.OOB.error <- as.integer(which.min(rf.results[,"OOB"])))
(ntrees.best <- rf.results[lowest.OOB.error,"ntrees"])
(mtry.best <- rf.results[lowest.OOB.error,"mtrys"])
# OOB més baix: 0.4795918, obtingut amb 16 arbres i mtry = 5
# Ens guardem aquesta informació per si la necessitem més endavant.
# Ja veiem que els resultats no són gaire bons perquè la proporció de la classe gran és 0.4829932
# Hem provat tècniques per a datasets desbalancejats, com estratifiació, oversampling o undersampling, però no hem aconseguit millores.
# El dataset no està tan desbalancejat com el cas del laboratori 7, però. El principal problema en aquest cas sembla ser un altre.

### 3
library(e1071)

# Hem d'escollir el millor SVM d'entre els que hem vist (lineal, polinòmic amb graus 2 i 3 i RBF).
# Farem tota la cerca de cop amb una funció per cercar els millors paràmetres fent CV.
# Hi ha paràmetres que no tenen cap efecte en alguns mètodes (per exemple, el kernel radial no té coef0, i el polinòmic no té gamma)
# Però en aquests casos la funció svm simplement ignora el paràmetre, i a canvi ho podem tenir tot en una sola crida.

bestSVM <- tune(svm, Salary.3~., data = learn.data,
     ranges = list(gamma = 10^seq(-5,2), cost =10^seq(-2,3), kernel=c("linear","polynomial","radial"),degree=c(2,3)),
     tunecontrol = tune.control(sampling = "cross",nrepeat=10), coef0=1, scale = FALSE,type="C-classification"
)

save(file="svmmodel.mod",object=bestSVM)

#(modelSVM <- svm(Salary.3~., data = learn.data, type="C-classification", scale = FALSE,coef0 = 1, kernel=bestSVM$best.parameters$kernel,
#                 gamma=bestSVM$best.parameters$gamma,cost=bestSVM$best.parameters$cost,degree=bestSVM$best.parameters$degree))


# També hem provat tècniques per datasets desbalancejats, com ara utilitzar el paràmetre class.weights per donar més pes a les classes minoritàries.
# Però, de nou, no hem obtingut millores utilitzant aquestes tècniques.

# El millor SVM és el lineal, amb cost = 1 (la resta de paràmetres són ignorats).
# L'accuracy de CV és 0.4611111.
# Mirant el nombre de support vectors (per exemple, reentrenant els millors paràmetres però per tot learn) observem que n'hi ha un nombre molt elevat
# cosa que no parla bé del model (poca sparsity).
# És un resultat dolent (el millor model constant hauria donar més accuracy) i pitjor que el de random forest.
# L'OOB de random forest recordem que era 0.4795918. 1 - 0.4795918 = 0.5204082
# Per això, escollirem random forest (amb els millors paràmetres trobats).




# Així doncs, procedim a aplicar test al millor model trobat (random forest amb ntrees = 16 i mtry = 5).
# No cal dir que fins ara de cap manera cap mètode ha tingut accés a test i que no utilitzem test per escollir el millor model.

# Re-entrenem randomForest amb tot learn i els millors paràmetres trobats:
modelFinal <- randomForest(Salary.3 ~ ., data = learn.data, ntree=16, mtry = 5, proximity=FALSE, 
             strata=learn.data$Salary.3)
library(caret)
p <- factor(predict (modelFinal, newdata=test.data, type="class"),levels=Enquesta[,11])
c <- confusionMatrix(test.data$Salary.3,p)

overall <- c$overall
overall.accuracy <- overall['Accuracy'] 
# Accuracy 0.5918367 
# Aquesta és l'estimació honesta de l'error de generalització del nostre model final. El que passa és que a l'haver-hi
# tan poques files, s'ha d'anar en compte al fer aquesta estimació. És millor que el millor model constant, però només
# uns 11 punts. A part que n és molt petita, la relació n/d també és molt petita. També podria ser que encara que
# tinguéssim moltes dades hi hagués problemes inherents (per exemple, al tractar-se de dades "socials" i de valoracions,
# és conegut que en aquests camps acostuma a ser més difícil que hi hagi relacions clares entre les variables).

