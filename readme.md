#### This problem is part of the assignment of course CS-551 (Elements of Artificial Intelligence) taught by Prof. David Crandall.
## Problem Statement
The 2021 Puzzle: Consider the 2021 puzzle, which is a lot like the 15-puzzle we talked about in class, but:  (1) it has 25 tiles, so there are no empty spots on the board (2) instead of moving a single tile into an open space,a move in this puzzle consists of either 
 (a) sliding an entire row of tiles left or right one space, with the left- or right-mosttile ‘wrapping around’ to the other side of the board, 
 (b) sliding an entire column of tiles up or down onespace, with the top- or bottom-most tile ‘wrapping around’ to the other side of the board, 
 (c) rotating theouter ‘ring’ of tiles either clockwise or counterclockwise, or 
 (d) rotating the inner ring either clockwise orcounterclockwise.

* There's a 5X5 matrix numberbed from 1-25 and certain moves can be perfomed on the board to move the elements of the matrix. A shuffled matrix is given and we have to find the goal state while performing operations on the matrix.
### Approach:
A star approach to find optimal solution to the solution. 
Heuristics:
Attempt 1: Grouped elements and their direction together and take max of their movements needed to go the ideal state.
This approach failed as it was not optimal, because as after 1 movement the grouping would change and the heuristic cost would "spill" to other groups
Attempt 2: Take a more simplistic view on the heuristic and group L/U/D/R movements mimicing the ring operations. This is the closest we got to the final answer.
Attempt 3: As the attempt 2 is quickly finding the answers, it was not the admissible heuristic. We combined the heuristics of 1 and 2 and we took the minimum of the cost of the costs from these two heuristics. This seemed intuitive but it was not optimal. 
Attempt 4: As the attempt 2 was the closest to the being an admissible heuristic, we tweaked the cost by of the heuristic by trying to estimate the "spill" of different groups. This approach seemed to be admissible but it was underestimating too much so we finalised the 2nd approach.

**Start state**: The given initial shuffled board .

**Succesor function**: Each state of the board after performing one of the 24 possible moves on the board. 

**Goal state**: The ideal position of numbers in the board. 1 to 25 in ascending order.

**Cost function**: The movements made till now + heuristic cost (max cost in the groups). 
