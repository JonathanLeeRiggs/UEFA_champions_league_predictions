# UEFA_champions_league_predictions
**This program tries to predict the outcome of the 2020-2021 UEFA champions league with machine learning techniques.**

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

The best preformance algorithm for us was Logistic regression. we belive that is because the statistics in such a game as soccer, doesn't tell the true story of the end result of the games many times.

That is the team chances to win (in precentage) based on the **Logistic regression** algorithm:  
![logisticRegreRegular](https://user-images.githubusercontent.com/44767003/109039832-e6f72400-76d5-11eb-8175-dea2e8b3aa0c.jpg)

We can see here that the most likely teams to win the competitions are Paris Saint Germain and Liverpool with over 17% chance to win.

In the player side, we can see that according to Logistic regression method, those are the most impactfull players of rank 1 (best ranking players on there teams):


![logisticRegrerank1](https://user-images.githubusercontent.com/44767003/109041281-88cb4080-76d7-11eb-8f45-9d471d4be833.jpg)
 We can see here that the rank 1 players such as Ilicic(Atalanta), gomez (Liverpool), Kimmich (Bayern Munich) and Neymar (Paris Saint Germain) are the most influencial players in the competition.
