# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import math
from sklearn.preprocessing import scale
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import combinations

results_data = pd.read_csv("champions league 20-21 results.csv")
players_data = pd.read_csv("PlayersAvg.csv")


#inizialize all results to lost.
results_data['Result'] = -1
for index, row in results_data.iterrows():
    home_goals = row['Score'][0]
    away_goals = row['Score'][2]
    results_data.loc[index, 'HGoals'] = home_goals
    results_data.loc[index, 'AGoals'] = away_goals
    #if the home team won, change the result to win.
    if home_goals > away_goals:
        results_data.loc[index, 'Result'] = 1
    #in case of a draw.
    elif home_goals == away_goals:
        results_data.loc[index, 'Result'] = 0
    

# number of games are 98 , number of features are 24, hence the matrix will be 98*24
X = np.ndarray(shape=(98, 24))
y = np.ndarray(shape=(96))

#inserts tha labels of the games (the result).
k = 0
for res in results_data['Result']:
    if k<96:
        y[k] = res
    else:
        break
    k+=1

#inserts all the features.
k = 0
for _, row in results_data.iterrows():
    if k<96:
        X[k] = [row['HxG'], row['HPoss'], row['HSoT'], row['Hsave'], row['HPass'], row['HPlayer1'], 
                row['HPlayer2'], row['HPlayer3'], row['HRank'], row['HKP'], row['HxA'], row['H1/3'],
                row['AxG'], row['APoss'], row['ASoT'], row['ASave'], row['APass'], row['APlayer1'], 
                row['APlayer2'], row['APlayer3'], row['ARank'], row['AKP'], row['AxA'], row['A1/3']]
    else: 
        break
    k+=1


avg=0
#inizialized the remaining teams, all the teams that are still in the compitition (sorted by the last 16 matches).
remaning_teams = ['Barcelona', 'Paris Saint Germain','RB Leipzig','Liverpool','Porto','Juventus',
                  'Sevilla','Dortmund','Lazio','Bayern Munich','Atletico Madrid',
                  'Chelsea','Gladbach','Manchester City','Atalanta','Real Madrid']

#will be used to calculate each team chances to win the league
teams_score = {'Paris Saint Germain':0,'Liverpool':0,'Juventus':0,'Sevilla':0,'Bayern Munich':0,
               'Chelsea':0,'Manchester City':0,'Atalanta':0, 'Barcelona':0, 'Lazio':0, 'RB Leipzig':0, 'Gladbach':0,
               'Real Madrid':0, 'Dortmund':0, 'Atletico Madrid':0, 'Porto':0}
#indicates the level of effect of each player (best players rating in their team)
players_1_score = {'Neymar':0,'Gomez':0,'Sandro':0,'Kimmich':0, 'Messi':0, 'Felix':0, 'Acuna':0,
                   'Halland':0, 'Hofman':0, 'Immobile':0, 'Ilicic':0, 'Oliveira':0, 'Nkunku':0, 'De Bruyne':0,
                   'Mendy':0, 'Benzema':0}
#indicates the level of effect of each player (2nd best players rating  in their team)
players_2_score={'Dembele':0,'Mbappe':0,'Jones':0,'Ronaldo':0, 'Coman':0, 'Torres':0, 'Havertz':0,
                   'Ramos':0, 'Llorente':0, 'Hummels':0, 'Bounou':0, 'Stindl':0, 'Angelino':0, 'Savic':0,
                   'Gosens':0, 'Otavio':0}
#indicates the level of effect of each player (3rd best players rating  in their team)
players_3_score={'Fati':0,'Haidara':0,'Marquinhos':0,'Akelleher':0, 'Morata':0, 'Sule':0, 'Gundogan':0,
                   'Silva':0, 'Eden Hazard':0, 'Gimenez':0, 'Torgen Hazard':0, 'De Jong':0, 'Elvedi':0, 'Alberto':0,
                   'Romero':0, 'Taremi':0}


#this method gets as input two teams and a condition (original, rank1, rank2, rank3) and returns the matchup of the 1st and 2nd legs between them.
def game_generator(teams, current_team_a, current_team_b, cond):
     X_a = np.ndarray(shape=(1, 12))
     X_b = np.ndarray(shape=(1, 12))
     
     team_a_name = teams[current_team_a]
     team_b_name = teams[current_team_b]
     
     for _, row in results_data.iterrows():
           #retrive the average data of the current team in their home and away matches.
           if row['Home'] == team_a_name and row['Away'] == team_a_name:
               X_a= [row['HxG'], row['HPoss'], row['HSoT'], row['Hsave'], row['HPass'], row['HPlayer1'],
               row['HPlayer2'], row['HPlayer3'], row['HRank'], row['HKP'], row['HxA'], row['H1/3']]
               
           #retrive the average data of the current team's opponent in their home and away matches.
           elif row['Home'] == team_b_name and row['Away'] == team_b_name:
               X_b = [row['AxG'], row['APoss'], row['ASoT'], row['ASave'], row['APass'], row['APlayer1'], 
               row['APlayer2'], row['APlayer3'], row['ARank'], row['AKP'], row['AxA'], row['A1/3']] 
               
     
                
     #normalize the posession feature (to be 100% for both teams)
     sum_of_posessions= X_a[1] + X_b[1]
     X_a[1] = (X_a[1] / sum_of_posessions) * 100
     X_b[1] = (X_b[1] / sum_of_posessions) * 100

          
     #rank x is switching the players on the x rank of both teams between them.          
     if cond == 'rank1':
         player_a = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[0]
         player_b = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[0]
         X_a[5]=players_data.loc[players_data['Player'] == player_b]['Home_Avg'].values[0]
         X_b[5]=players_data.loc[players_data['Player'] == player_a]['Away_Avg'].values[0]
     elif cond == 'rank2':
         player_a = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[1]
         player_b = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[1]
         X_a[6]=players_data.loc[players_data['Player'] == player_b]['Home_Avg'].values[0]
         X_b[6]=players_data.loc[players_data['Player'] == player_a]['Away_Avg'].values[0] 
     elif cond == 'rank3':
         player_a = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[2]
         player_b = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[2]
         X_a[7]=players_data.loc[players_data['Player'] == player_b]['Home_Avg'].values[0]
         X_b[7]=players_data.loc[players_data['Player'] == player_a]['Away_Avg'].values[0]  

     return np.concatenate((X_a, X_b))       
               
#this method predicts the last 8 teams of the competition according to a condition (original, rank1, rank2, rank3) and the methods (LogisticRegression, LinearRegression, svm or DecisionTreeClassifier)               
def last_8_preds(cond='original', method='LogisticRegression'):
    teams=remaning_teams[:]
    current_team_a = 0
    while len(teams) > 8:
        current_team_b = current_team_a+1
        first_leg=96
        second_leg=97
        
        team_a_name = teams[current_team_a]
        team_b_name = teams[current_team_b]
        
        if cond == 'rank1':
            player_a = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[0]
            player_b = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[0]
        elif cond == 'rank2':
            player_a = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[1]
            player_b = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[1]
        elif cond == 'rank3':
            player_a = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[2]
            player_b = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[2]
        elif cond == 'original':
            player_a_1 = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[0]
            player_b_1 = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[0]
            player_a_2 = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[1]
            player_b_2 = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[1]
            player_a_3 = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[2]
            player_b_3 = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[2]
            
        #merge the features of both teams such that we will have the first leg (current team is the host)
        #and the second leg (current team's opponent is the host)
        X[first_leg] = game_generator(teams, current_team_a, current_team_b,cond)
        X[second_leg] = game_generator(teams, current_team_b, current_team_a,cond)
        
        
        #scaling the features to make them be in the same range of numbers.
        X_scaled = scale(X)
        sum = 0
        #retrive the features and labels of the matches that already have been played.
        X_played = X_scaled[0:96]
        
        #inizialize variables for counting the probability for each result (team a wins, draw, teams b wins)
        win_pred_team_a = 0
        win_pred_team_b = 0
        draw_pred = 0
        
        #runs the training and the prediction 100 times and calculates the probability for each result.
        for i in range(100):
            X_train, X_test, y_train, y_test = train_test_split(X_played, y, test_size = 0.1)
            if method == 'LogisticRegression':
                reg = LogisticRegression().fit(X_train, y_train)
            elif  method == 'svm':     
                reg = svm.SVC().fit(X_train, y_train)
            elif  method == 'LinearRegression':     
                reg = LinearRegression().fit(X_train, y_train)
            elif  method == 'DecisionTreeClassifier':     
                reg = DecisionTreeClassifier().fit(X_train, y_train)
            preds = reg.predict(X_scaled[first_leg:second_leg+1])
            #if team a beats team b in the first or the second leg.
            if preds[0] > 0 or preds[1] < 0:
                win_pred_team_a += 1 
                teams_score[team_a_name] += 1
                if cond == 'original':
                    players_1_score[player_a_1] += 1
                    players_2_score[player_a_2] += 1
                    players_3_score[player_a_3] += 1
                elif cond=='rank1' :
                    players_1_score[player_b] += 1
                elif cond=='rank2' :
                     players_2_score[player_b] += 1
                elif cond=='rank3' :
                     players_3_score[player_b] += 1      
            #if team b beats team a in the first or the second.     
            elif preds[0] < 0 or preds[1] > 0:
                win_pred_team_b += 1 
                teams_score[team_b_name] += 1
                if cond == 'original':
                    players_1_score[player_b_1] += 1
                    players_2_score[player_b_2] += 1
                    players_3_score[player_b_3] += 1
                elif cond=='rank1' :
                    players_1_score[player_a] += 1
                elif cond=='rank2' :
                     players_2_score[player_a] += 1
                elif cond=='rank3' :
                     players_3_score[player_a] += 1  
            #if one of the matches ended with a draw.   
            elif preds[0] == 0 or preds[1] == 0:
                draw_pred+=1
            
            #calculates the accuracy of the training.
            sum += reg.score(X_test, y_test)
        
        #print(teams[current_team_a], " vs ", teams[current_team_b], " Draw: ", draw_pred, " A wins: ", win_pred_team_a, " B wins: ", win_pred_team_b)
        
        #if the probability for team a to win is the highest, emove team b from the remaining teams list (teams a stays for the next round).
        if win_pred_team_a > win_pred_team_b and win_pred_team_a > draw_pred:
            teams.remove(teams[current_team_b])
            
        #if the probability for team b to win is the highest, remove team a from the remaining teams list (teams b stays for the next round).
        elif win_pred_team_b > win_pred_team_a and win_pred_team_b > draw_pred:
            teams.remove(teams[current_team_a])
        
        #if the probability for a draw is the highest, check which probability is higher, team a wins or team b wins, and remove the worse team.
        elif draw_pred > win_pred_team_a and draw_pred > win_pred_team_b:
            if win_pred_team_a > win_pred_team_b:
                teams.remove(teams[current_team_b])
            else:
                teams.remove(teams[current_team_a])
        
            
        if current_team_a < len(teams)-2:
            current_team_a += 1
        else:
            #if all the matches were played in the last 16
            return teams
                
      
#pridicting the last 8 teams of the competitions
last_8_teams_original= last_8_preds()
last_8_teams_rank1= last_8_preds('rank1')
last_8_teams_rank2= last_8_preds('rank2')
last_8_teams_rank3= last_8_preds('rank3')

#this method gets the last 8 teams and predicts the chances for each team to win the competition according to the relevant method and condition.
#we took all the possible games because the quarter final phase hasn't been drawn.
def final_prediction(last_8_teams, cond='original', method='LogisticRegression'):
    last_8 = last_8_teams[:]
    #all possible games combinations.
    games_comb = combinations(last_8, 2)
    games_comb = list(games_comb)
    
    #the team we are about to predict their games (current_team_b is the opponent).
    current_team_a = 0 
    index = 0
    avg_per_matchup=0
    #runs as long as there are games to predict
    while index < len(games_comb):
        current_team_b = 1
        sum = 0
        first_leg=96
        second_leg=97
        
        team_a_name= games_comb[index][current_team_a]
        team_b_name= games_comb[index][current_team_b]
        
        #initializing players data according to the switching method.
        if cond == 'rank1':
            player_a = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[0]
            player_b = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[0]
        elif cond == 'rank2':
            player_a = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[1]
            player_b = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[1]
        elif cond == 'rank3':
            player_a = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[2]
            player_b = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[2]
        elif cond == 'original':
            player_a_1 = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[0]
            player_b_1 = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[0]
            player_a_2 = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[1]
            player_b_2 = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[1]
            player_a_3 = players_data.loc[players_data['Team'] == team_a_name]['Player'].values[2]
            player_b_3 = players_data.loc[players_data['Team'] == team_b_name]['Player'].values[2]
        
        
        #merge the features of both teams such that we will have the first leg (current team is the host)
        #and the second leg (current team's opponent is the host)
        X[first_leg] = game_generator(last_8, current_team_a, current_team_b,cond)
        X[second_leg] = game_generator(last_8, current_team_b, current_team_a,cond)
        
        #scaling the features to make them be in the same range of numbers.
        X_scaled = scale(X)
        

        #retrive the features and labels of the matches that already have been played.
        X_played = X_scaled[0:96]

        #inizialize variables for counting the probability for each result (team a wins, draw, teams b wins)
        win_pred_team_a = 0
        win_pred_team_b = 0
        draw_pred = 0
        
        #runs the training and the prediction 100 times and calculates the probability for each result.
        for i in range(100):
            X_train, X_test, y_train, y_test = train_test_split(X_played, y, test_size = 0.1)
            if method == 'LogisticRegression':
                reg = LogisticRegression().fit(X_train, y_train)
            elif  method == 'svm':     
                reg = svm.SVC().fit(X_train, y_train)
            elif  method == 'LinearRegression':     
                reg = LinearRegression().fit(X_train, y_train)
            elif  method == 'DecisionTreeClassifier':     
                reg = DecisionTreeClassifier().fit(X_train, y_train)
            preds = reg.predict(X_scaled[first_leg:second_leg+1])
            #if team a beats team b in the first or the second leg.
            if preds[0] > 0 or preds[1] < 0:
                if cond == 'original':
                    teams_score[games_comb[index][current_team_a]] += 1
                    players_1_score[player_a_1] += 1
                    players_2_score[player_a_2] += 1
                    players_3_score[player_a_3] += 1
                elif cond=='rank1' :
                    players_1_score[player_b] += 1
                elif cond=='rank2' :
                     players_2_score[player_b] += 1
                elif cond=='rank3' :
                     players_3_score[player_b] += 1      
            #if team b beats team a in the first or the second leg.                
            elif preds[0] < 0 or preds[1] > 0:
                if cond == 'original':
                    teams_score[games_comb[index][current_team_b]] += 1
                    players_1_score[player_b_1] += 1
                    players_2_score[player_b_2] += 1
                    players_3_score[player_b_3] += 1
                elif cond=='rank1' :
                    players_1_score[player_a] += 1
                elif cond=='rank2' :
                     players_2_score[player_a] += 1
                elif cond=='rank3' :
                     players_3_score[player_a] += 1  
                    
            #calculates the accuracy of the training.
            sum += reg.score(X_test, y_test)
        #accuracy 
        avg_per_matchup += sum / 100
        index += 1
        
    print('avg: ',avg_per_matchup/28)    
        
    if cond == 'original':
        #returns the winning chances for each team
        sum_of_score=0
        for team in teams_score.keys():
            sum_of_score += teams_score[team]
        for team in teams_score.keys():   
            teams_score[team] = round((teams_score[team]/sum_of_score *100), 3)
        return teams_score
    elif cond == 'rank1':
        return players_1_score
    elif cond == 'rank2':
        return players_2_score
    elif cond == 'rank3':
        return players_3_score        


teams_chances=final_prediction(last_8_teams_original)
print ('teams_chances: \n', teams_chances)

rank1_players=final_prediction(last_8_teams_rank1,'rank1')
print ('rank1_players: \n', rank1_players)

rank2_players=final_prediction(last_8_teams_rank2,'rank2')
print ('rank2_players: \n', rank2_players)

rank3_players=final_prediction(last_8_teams_rank3,'rank3')
print ('rank3_players: \n', rank3_players)



#making a graph
plt.rcParams.update({'font.size': 10})
f, ax = plt.subplots(figsize=(25,5))
plt.bar(teams_chances.keys(), teams_chances.values())

f1, ax1 = plt.subplots(figsize=(25,5))
plt.bar(rank1_players.keys(), rank1_players.values())

f2, ax2 = plt.subplots(figsize=(25,5))
plt.bar(rank2_players.keys(), rank2_players.values())

f3, ax3 = plt.subplots(figsize=(25,5))
plt.bar(rank3_players.keys(), rank3_players.values())

