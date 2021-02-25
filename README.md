# UEFA_champions_league_predictions

#### This program tries to predict the outcome of the 2020-2021 UEFA champions league with machine learning techniques. ####

***

The first step for us was to take the data we built on "champions league 20-21 results.csv" and try to figure out which data is telling the story of the game.
Eventually we took 27 fetures (devided to home and away statistics such as expected goals, shots on target, posession, saves, leading players rank etc.) and manipulate our program based on those fetures. 

After we finished with the data we devided the predection to two functions, one will predict the outcome of the last 16 round of the compatition.
The second function will predict the chance to win for every team of the last 8 teams left. 

Eventually we'll have the chances of every team of the competition to win the title.

After we predicted the chances for every team to win, we tried to find out which player, of every players quality groups (rank 1, rank 2 and rank 3).
Those groups are built based on the data of the 3 best players of each team such that the best player on the team is on rank 1, the second best player on the team is in rank 2 and likewise rank 3.


***

## The results ##

### Logistic regression ###
The best preformance algorithm for us was **Logistic regression** with over 80% accuracy (on the test). we belive that is because the statistics in such a game as soccer, doesn't tell the true story of the end result of the games many times.

That is the team chances to win (in precentage) based on the **Logistic regression** algorithm:  
![logisticRegreRegular](https://user-images.githubusercontent.com/44767003/109039832-e6f72400-76d5-11eb-8175-dea2e8b3aa0c.jpg)

We can see here that the most likely teams to win the competitions are Paris Saint Germain and Liverpool with over 17% chance to win.

In the player side, we can see that according to Logistic regression method, those are the most impactfull players of rank 1 (best ranking players on there teams):


![logisticRegrerank1](https://user-images.githubusercontent.com/44767003/109041281-88cb4080-76d7-11eb-8f45-9d471d4be833.jpg)
 We can see here that the rank 1 players such as Ilicic(Atalanta), gomez (Liverpool), Kimmich (Bayern Munich) and Neymar (Paris Saint Germain) are the most influencial players in the competition.


***

## svm ##

70e second best preformance algorithm for us was **svm** (Supporting Vector Machine) with over 75% accuracy (on the test).

That is the team chances to win (in precentage) (based on the **svm** algorithm):

![svmRegular png](https://user-images.githubusercontent.com/44767003/109120085-2c573800-774e-11eb-93fc-183b2b135ce9.jpg)

We can see here that the most likely teams to win the competitions is Barcelona with over 17.5% chance to win.
Those results are very different from the **Logistic regression** algorithm as we can see that once a team goes through the last 16 stage, then the team chances to win gets higher.

Those are the most impactfull players of rank 2 (second best ranking players on there teams):

![svmRank2 png](https://user-images.githubusercontent.com/44767003/109121073-742a8f00-774f-11eb-8b66-7f1f65756b7f.jpg)

***

## Decision Tree Classifier ##

The third best preformance algorithm for us was **Decision Tree Classifier** with over 65% accuracy (on the test).

That is the team chances to win (in precentage) (based on the **Decision Tree Classifier** algorithm):

![DesTreeRegular](https://user-images.githubusercontent.com/44767003/109121611-26625680-7750-11eb-8c28-eaf88354a39b.jpg)


We can see here that the most likely teams to win the competitions is Barcelona with over 17.5% chance to win.
Those results are very different from the **Logistic regression** algorithm but more similar to **svm**.

Those are the most impactfull players of rank 3 (third best ranking players on there teams):

![DesTreeRank3](https://user-images.githubusercontent.com/44767003/109121791-6590a780-7750-11eb-9a34-7a77cd2d71fd.jpg)

***

## Linear Regression ##

The last best preformance algorithm for us was **Linear Regression** with over 60% accuracy (on the test)..

That is the team chances to win (in precentage) (based on the **Linear Regression** algorithm):

![LinearRegRegular](https://user-images.githubusercontent.com/44767003/109122703-a63cf080-7751-11eb-902b-7fa8f5e8e302.jpg)

We can see here that the most likely teams to win the competitions is Barcelona with over 17.5% chance to win.
Those results are very different from the **Logistic regression** algorithm but more similar to **svm** and **Desicion Tree Classifier**.

Those are the most impactfull players of rank 2 (second best ranking players on there teams):

![LinearRegRank2](https://user-images.githubusercontent.com/44767003/109122717-accb6800-7751-11eb-9413-5340528622a6.jpg)


***

So which method is the best? 

we assume that the **Logistic regression** is the best preformance algorithm because the accuracy on the test was over 80% and the results seemed to us the most resonable results looking at the teams chances to win.

In conclusion, we can see that soccer is a sport that is different from baseball/basketball in terms that the statistics doesn't tell the story of the game in big part of the games. Also the chances for each team changed drastically between the different algorithms because once a team qualified from the last 16 phase, then it's chances to win the competition went higher because we devided the program into two functions (last 16 and last 8).









