# CS5310 - Course Project
# Team:6
# Team Members: 
# Member-1: Javier Berdejo 33.33% contribution
# Member-2: Ndubuisi Chibuogwu 33.33% contribution
# Member-3: Hassan Sarr 33.33% contribution

#Submission Date: 04/29/2022

#Problem description: predict outcome of football games applying regression and classification algorithms,
#and answer research questions. 
#Algorithms to use:
#kNN: we have to use different k's,and distances
#Boosting: AdaBoost
#Regression: Poisson Regression (for int counts)
#We can use several approaches to preprocess the data
#Approach-1: raw data
#Approach-2: standardize all the features (Min-max, or z-score)
#Approach-3: standard according to some criteria (Some features are more important than others)

#----------------------------------------------------------------------------------
#Clean/Prepare data
library(gmodels)
library(glmnet)
library(ggplot2) 
library(tidyverse)
library(caret)
library(adabag)
library(class)
#read premier league matches file 
matches <- read.csv('CS5310-Project/premiermatches-1819.csv')
str(matches) #read columns and types
ncol(matches) #number of columns
nrow(matches) #number of rows

matches[is.na(matches), ] #no missing values
for (i in 1:64) {
  print(table(matches[, i]))
} #no inconsistencies

#new nominal column with result of the games
matches$result <- ifelse(matches$home_team_goal_count > matches$away_team_goal_count, 'Home Team Win', ifelse(
  matches$home_team_goal_count == matches$away_team_goal_count, 'Draw', 'Away Team Win'))
matches[, c('home_team_goal_count', 'away_team_goal_count', 'result')]
round(prop.table(table(matches$result)) * 100, digits = 2) #proportion of resutls

#remove columns not needed for analysis
#leave only numerical data (stats from the matches)
#remove 1, 2, 3, 7, 8, 13-20 (goal columns, may save col 15), 27-30, 45-52, 58-61, 64 and 65 columns
removecols <- c(-c(1:3), -c(7:8), -c(13:20), -c(27:30), -c(45:52), -c(58:61), -c(64,65))
new_matches <- matches[,removecols]
#remove team names as well
new_matches <- new_matches[, c(-2,-3)]

#---------------------------------------------------------------------------------------------------------------------------
#EXPLORATORY ANALYSIS
#function to bootstrap variable and analyze normality
explAnalysis <- function(vect, s){
  par(mfrow=c(1,2))
  N <- 10^4
  n <- nrow(matches)
  boot <- numeric(N)
  set.seed(14)
  for (i in 1:N) {
    x <- sample(vect, n, replace = T)
    boot[i] <- mean(x)
  }
  hist(boot, prob = T, xlab = s, main = c('Histogram of ', s))
  lines(density(boot), col='red', lwd=4)
  qqnorm(boot)
  qqline(boot)
  print(summary(boot))
  print(summary(vect))
}
#few visualizations from distinct variables
explAnalysis(new_matches$away_team_possession, 'Away Team Possession')
explAnalysis(new_matches$home_team_shots, 'Home Team Shots')
explAnalysis(new_matches$away_team_fouls, 'Away Team Fouls')

#---------------------------------------------------------------------------
#Q1
#Classify teams by winners, losers, and average. 
teams <- unique(matches$home_team_name) #list of teams
#data frame for season records
premier_table <- data.frame(team = teams, W = numeric(20), L = numeric(20), D = numeric(20), Pts = numeric(20))
#get records for each team 
for (t in teams) {
  for (i in 1:nrow(matches)) { #teams check games one by one
    if(q1$away_team_name[i] == t) { #result when team plays away
      if(q1$result[i] == 'Away Team Win') {
        premier_table[premier_table$team == t, 'W'] <- premier_table[premier_table$team == t, 'W'] + 1
        premier_table[premier_table$team == t, 'Pts'] <- premier_table[premier_table$team == t, 'Pts'] + 3
      } else if (q1$result[i] == 'Home Team Win') {
        premier_table[premier_table$team == t, 'L'] <- premier_table[premier_table$team == t, 'L'] + 1
      } else {
        premier_table[premier_table$team == t, 'D'] <- premier_table[premier_table$team == t, 'D'] + 1
        premier_table[premier_table$team == t, 'Pts'] <- premier_table[premier_table$team == t, 'Pts'] + 1
      }
    }
    if(q1$home_team_name[i] == t) { #result when team plays home
      if(q1$result[i] == 'Home Team Win') {
        premier_table[premier_table$team == t, 'W'] <- premier_table[premier_table$team == t, 'W'] + 1
        premier_table[premier_table$team == t, 'Pts'] <- premier_table[premier_table$team == t, 'Pts'] + 3
      } else if (q1$result[i] == 'Away Team Win') {
        premier_table[premier_table$team == t, 'L'] <- premier_table[premier_table$team == t, 'L'] + 1
      } else {
        premier_table[premier_table$team == t, 'D'] <- premier_table[premier_table$team == t, 'D'] + 1
        premier_table[premier_table$team == t, 'Pts'] <- premier_table[premier_table$team == t, 'Pts'] + 1
      }
    }
  }
}
rownames(premier_table) <- premier_table$team
#scatterplot analysis of wins vs points for each team
ggplot(premier_table, aes(W, Pts, colour = team, label = team)) + geom_point() + geom_text() + 
  scale_y_continuous(breaks = seq(0,110,10))

#CLUSTER TEAMS
install.packages("factoextra")
library(factoextra)
set.seed(14)
prem_clust <- premier_table[2:5] #numeric variables only
z_prem <- as.data.frame(lapply(prem_clust, scale))
rownames(z_prem) <- rownames(prem_clust)
fviz_nbclust(z_prem, kmeans, method = "wss")+labs(subtitle = "Elbow method") #suggest k = 3,5,8
cluster_prem5 <- kmeans(z_prem, centers = 5)
cluster_prem3 <- kmeans(z_prem, centers = 3)
kmcluster5 <- cluster_prem5$cluster
kmcluster3 <- cluster_prem3$cluster
#plot for 3 clusters
fviz_cluster(list(data = z_prem, cluster = kmcluster3))

#----------------------------------------------------------------------------------------
#Q3
teamsta <- unique(matches[, c('home_team_name', 'stadium_name')]) #stadium with corresponding home team
city_result <- as.data.frame(table(matches$stadium_name, matches$result)) #game results per stadium
colnames(city_result) <- c('Stadium', 'Result', 'Freq') #change column name
city_result <- cbind(city_result, teamsta[order(teamsta$stadium_name), 1]) #assign corresponding team to each stadium
colnames(city_result)[4] <- 'Team'
table(city_result$Result)
city_result$Result <-  ifelse(city_result$Result=='Away Team Win', 'A', ifelse(city_result$Result=='Draw', 'D', 'H'))
ggplot(data = city_result, aes(Result, Freq, fill = Result)) + geom_col() + facet_grid(~Team)

#--------------------------------------------------------------------------------------------------------------
#Preparing data for PCA
#standardize variables
#z-score normalization(min-max would not be accurate for some features)
z_new_matches <- as.data.frame(lapply(new_matches, scale))

#apply PCA
z_pc <- princomp(z_new_matches)
summary(z_pc)
z_pc$loadings #16 PCs give 90% variance
z_pc$center
screeplot(z_pc, col = "red", pch = 16, type = "lines", cex = 2, lwd = 2, main = "") #visualization for PCs selection
matches_pca <- z_pc$scores[, 1:16]

#Split 80/20 
set.seed(14)
totrain <- sample(nrow(matches), 0.8*nrow(matches)) #row indexes that belong to train set
train_matches_pca <- as.data.frame(matches_pca[totrain, ]) #80% training
test_matches_pca <- as.data.frame(matches_pca[-totrain, ]) #20% evaluation
#home
train_home_goals <- matches$home_team_goal_count[totrain]
test_home_goals <- matches$home_team_goal_count[-totrain]
home_train_pca <- cbind(train_matches_pca, train_home_goals)
#away
train_away_goals <- matches$away_team_goal_count[totrain]
test_away_goals <- matches$away_team_goal_count[-totrain]
away_train_pca <- cbind(train_matches_pca, train_away_goals)

#apply Poisson regression
#for home goals
mhome_poisson <- glm(train_home_goals~., data = home_train_pca, family = poisson())
phome_goals <- predict.glm(mhome_poisson, test_matches_pca, type = 'response')
sum(ifelse((floor(phome_goals) - test_home_goals) == 0, 1, 0)) #33 guess right
#for away goals
maway_poisson <- glm(train_away_goals~., data = away_train_pca, family = poisson())
paway_goals <- predict.glm(maway_poisson, test_matches_pca, type = 'response')
sum(ifelse((floor(paway_goals) - test_away_goals) == 0, 1, 0)) #32 guess right
#get the cross table of the goal predictions vs actual scores
CrossTable(test_home_goals, floor(phome_goals), prop.chisq = F,
           prop.t = F, prop.r = F, dnn = c('actual', 'predicted'))

predicted_home <- floor(phome_goals) #int values for home goal predictions
predicted_away <- floor(paway_goals) #int values for away goal predictions
test_results <- ifelse(predicted_home > predicted_away, 'Home Team Win', ifelse(
  predicted_home == predicted_away, 'Draw', 'Away Team Win'))
all_results <- ifelse(matches$home_team_goal_count > matches$away_team_goal_count, 'Home Team Win', ifelse(
  matches$home_team_goal_count == matches$away_team_goal_count, 'Draw', 'Away Team Win'))
#crosstable predicted vs actual match results
CrossTable(all_results[-totrain], test_results, prop.chisq = F,
           prop.t = F, prop.r = F, dnn = c('actual', 'predicted'))
confusionMatrix(factor(test_results, c('Home Team Win', 'Draw', 'Away Team Win')), factor(all_results[-totrain]), c('Home Team Win', 'Draw', 'Away Team Win'))

#---------------------------------------------------------------------------------------------------------
#WITHOUT PCA
#prepare for regression (apply z-normalization as well)
#home goal data frame
home_goal <- z_new_matches
home_goal$home_goal_count <- matches$home_team_goal_count
#80% for training set, 20% evaluation
home_goal_training <- home_goal[totrain, ]
home_goal_test <- home_goal[-totrain, -33]
head(home_goal_test)
#away goal data frame
away_goal <- z_new_matches
away_goal$away_goal_count <- matches$away_team_goal_count
away_goal_training <- away_goal[totrain, ]
away_goal_test <- away_goal[-totrain, -33]
head(away_goal_test)

#Poisson Regression to calculate Home and Away goals
set.seed(14)
#for home goals
home_poisson <- glm(home_goal_count~., data = home_goal_training, family = poisson())
summary(home_poisson)
predict_home <- predict.glm(home_poisson, home_goal_test, type = 'response')
sum(ifelse((floor(predict_home) - home_goal[-totrain, 33])==0, 1, 0)) #39
#for away goals
away_poisson <- glm(away_goal_count~., data = away_goal_training, family = poisson())
predict_away <- predict.glm(away_poisson, away_goal_test, type = 'response')
sum(ifelse((floor(predict_away) - away_goal[-totrain, 33])==0, 1, 0)) #28
#get the cross table of the goal predictions vs actual scores
CrossTable(home_goal$home_goal_count[-totrain], floor(predict_home), prop.chisq = F,
           prop.t = F, prop.r = F, dnn = c('actual', 'predicted'))

predicted_home <- floor(predict_home)
predicted_away <- floor(predict_away)
test_results <- ifelse(predicted_home > predicted_away, 'Home Team Win', ifelse(
  predicted_home == predicted_away, 'Draw', 'Away Team Win'))
#predicted vs actual match results
CrossTable(all_results[-totrain], test_results, prop.chisq = F,
           prop.t = F, prop.r = F, dnn = c('actual', 'predicted'))
levels(all_results[-totrain]) <- c('Home Team Win', 'Draw', 'Away Team Win')
levels(test_results)
confusionMatrix(factor(test_results, c('Home Team Win', 'Draw', 'Away Team Win')), factor(all_results[-totrain]), c('Home Team Win', 'Draw', 'Away Team Win'))

#-------------------------------------------------------------------
#last regression approach: only game variables (ALTERNATIVE)
removecols <- c(-c(1:5), -c(22:32))
game <- new_matches[, removecols]
set.seed(14)
pca_game <- princomp(game)
pca_game$scores
summary(pca_game)
biplot(pca_game)
screeplot(pca_game, col = "red", pch = 16, type = "lines", cex = 2, lwd = 2, main = "")

games_pca <- as.data.frame(pca_game$scores[, 1:8])
goals_home_train <- matches$home_team_goal_count[totrain]
goals_home_test <- matches$home_team_goal_count[-totrain]
goals_away_train <- matches$away_team_goal_count[totrain]
goals_away_test <- matches$away_team_goal_count[-totrain]

home_train <- cbind(games_pca[totrain, ], goals_home_train)
away_train <- cbind(games_pca[totrain, ], goals_away_train)
model_home <- glm(goals_home_train~., data = home_train, family = 'poisson')
model_away <- glm(goals_away_train~., data = away_train, family = 'poisson')
predict_goal_home <- round(predict.glm(model_home, games_pca[-totrain, ], type = 'response'))-1
predict_goal_away <- round(predict.glm(model_away, games_pca[-totrain, ], type = 'response'))-1
test_results <- ifelse(predict_goal_home > predict_goal_away, 'Home Team Win', ifelse(
  predict_goal_home == predict_goal_away, 'Draw', 'Away Team Win'))
CrossTable(all_results[-totrain], test_results, prop.chisq = F,
           prop.t = F, prop.r = F, dnn = c('actual', 'predicted'))

#-------------------------------------------------------------------------
#kNN and Boosting approaches
matches <- read_csv("CS5310-Project/premiermatches-1819.csv")
# Remove useless column
matches <- matches[, c(-1,-2,-3,-7,-8,-(15:20),-(27:30),-(45:52),-(58:61))]
# Add the target feature
matches$results <- ifelse(matches$home_team_goal_count > matches$away_team_goal_count, 'Home Team Win', ifelse(
  matches$home_team_goal_count == matches$away_team_goal_count, 'Draw', 'Away Team Win'))
# factor the values of the target feature
matches$results<- factor(matches$results, levels =c('Away Team Win','Draw', 'Home Team Win'), labels =c('Away Team Win','Draw', 'Home Team Win'))
str(matches$results)

# remove the stadium column
matches<-matches[-37]

# Normalize the data using min-max approach
minmaxnorm <- function(x){
  return((x - min(x))/(max(x) - min(x)))
}

# remove team name from the data 
new_matches<-matches[-(2:3)]

matches_n<-as.data.frame(lapply(new_matches[1:34], minmaxnorm))
matches_n<-cbind(matches_n, new_matches[35])
#str(matches)

# getting our training set and test set
matches_train<-matches_n[1:305,-35]# 80% for the training set
matches_test<-matches_n[-(1:305),-35]#20% for the test set

# Getting labels for both set
matches_train_label<- new_matches$results[1:305]
matches_test_label<- new_matches$results[-(1:305)]

#----------------------------------------------------------------------------
#Apply kNN
# create the model and predict from model
k_value<-sqrt(nrow(matches)) # we will use 19 as k value
matches_test_pred <- knn(train = matches_train, test = matches_test, cl = matches_train_label, k = k_value )

# Evaluate the model performance using CrossTable function
CrossTable(x = matches_test_label, y = matches_test_pred, 
           prop.chisq = FALSE, prop.r = FALSE, prop.c = FALSE)

#------------------------------------------------------------------------
# Improve the model using 
matches_train<-matches_n[1:305,]# 80% for the training set
matches_test<-matches_n[-(1:305),]#20% for the test set

ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
modelLookup("knn")
grid_knn <- expand.grid(k = seq(3,23,2))
grid_knn

set.seed(300)
m_knn<-train(results~., data = matches_train, method = "knn",
             metric = "Kappa",
             trControl = ctrl, tuneGrid = grid_knn)
m_knn

# apply the best knn candidate model to make predictions
p_knn <- predict(m_knn, matches_test, type = "raw")
p_knn
CrossTable(x = matches_test_label, y = p_knn, 
           prop.chisq = FALSE, prop.r = FALSE, prop.c = FALSE)

#-----------------------------------------------------------------------------
#Apply Boosting
#Use boosting method to improve the model
set.seed(300)
m_adaboost<-boosting(results ~., data = matches_train)
m_adaboost1<-boosting(results ~., data = matches_train)

p_adaboost<-predict(m_adaboost,matches_test)

CrossTable(x = matches_test_label, y = p_adaboost$class, 
           prop.chisq = FALSE, prop.r = FALSE, prop.c = FALSE)
