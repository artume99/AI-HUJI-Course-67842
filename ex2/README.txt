323379768
206974586
*****
Comments:
Our evaluation function consists of many ideas:
1. Keeping the lower left corner fixed in its place:
It is important to keep a fixed tile in one corner and merge the other tiles
in its direction, so we can keep our solution as clean as possible:
ALWAYS try to move down.
If can't - move left.
If can't - move right.
Only when there's ABSOLUTELY no other option - move up.

2. Maintaining the form of our board as a "snake":
We want our lowest row to have the biggest values, in descending order, and the
rows on top of it need to follow a "snake" pattern, where the tile above the
row can easily be merged with the row when needed.

3. Making sure that there are enough "empty tiles" on the board:
As the game goes on, the amount of empty tiles decreases (sometimes
drastically) which means that the option of maneuvering the board decreases
as well.
We don't want that to happen, so we prioritize boards with bigger chances to
have the "perfect amount" of empty tiles.
DISCLAIMER: PERFECT AMOUNT, IN OUR OPINION, IS 9, SINCE IT MANAGES TO KEEP A
SMALL PYRAMID OF EMPTY TILES IN THE UPPER-RIGHT SIDE OF THE BOARD, WHERE MOST
OF THE MOTION IS USUALLY HAPPENING :)

4. Prioritizing boards with bigger chances to merge with each move:
We calculated the "board's chance to merge" when merging columns (going left
or right) and when merging rows (going up or down) by summing up the
subtractions of each 2 adjacent tiles.
The lower the sum - the better our situation is, since low sums indicate
summing up lots of 0's AKA lots of merges.
We gave the cols a lower weight, so we won't contradict the "snake pattern".

We have tried to play with the weights, make them grant us with higher scores,
but couldn't find a formula that quite suited our expectations :(
The average case is a score of 20,000,
With winning (reaching 2048) in 3/5 cases