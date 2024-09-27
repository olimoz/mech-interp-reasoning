================================================================
NOTE: 

Example short prompt for ARC challenge 017c7c7b from the training set.
The text has been prepended to the matrices, which are taken directly from the challenge file.
We prepend the same text to all challenge matrices and submit the concatenation to the model as the prompt.

================================================================
PROMPT:

Below are pairs of matrices. 
There is a mapping which operates on each input to give the output, only one mapping applies to all matrices. 
Review the matrices to learn that mapping and then estimate the missing output for the final input matrix.

FIRST score your confidence that you understand the mapping pattern, 0-5 where 0 is no confidence and 5 is highly confident. 
This score must be the FIRST output you give, no preamble, no prefix, no punctuation, just a single digit score.
THEN Present your predicted output in np.array format

TRAIN Pair 0
INPUT. Shape=(6, 3)
array([[0, 1, 0],
       [1, 1, 0],
       [0, 1, 0],
       [0, 1, 1],
       [0, 1, 0],
       [1, 1, 0]])
OUTPUT. Shape=(9, 3)
array([[0, 2, 0],
       [2, 2, 0],
       [0, 2, 0],
       [0, 2, 2],
       [0, 2, 0],
       [2, 2, 0],
       [0, 2, 0],
       [0, 2, 2],
       [0, 2, 0]])
TRAIN Pair 1
INPUT. Shape=(6, 3)
array([[0, 1, 0],
       [1, 0, 1],
       [0, 1, 0],
       [1, 0, 1],
       [0, 1, 0],
       [1, 0, 1]])
OUTPUT. Shape=(9, 3)
array([[0, 2, 0],
       [2, 0, 2],
       [0, 2, 0],
       [2, 0, 2],
       [0, 2, 0],
       [2, 0, 2],
       [0, 2, 0],
       [2, 0, 2],
       [0, 2, 0]])
TRAIN Pair 2
INPUT. Shape=(6, 3)
array([[0, 1, 0],
       [1, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       [1, 1, 0],
       [0, 1, 0]])
OUTPUT. Shape=(9, 3)
array([[0, 2, 0],
       [2, 2, 0],
       [0, 2, 0],
       [0, 2, 0],
       [2, 2, 0],
       [0, 2, 0],
       [0, 2, 0],
       [2, 2, 0],
       [0, 2, 0]])
TEST Pair 0
INPUT. Shape=(6, 3)
array([[1, 1, 1],
       [0, 1, 0],
       [0, 1, 0],
       [1, 1, 1],
       [0, 1, 0],
       [0, 1, 0]])
OUTPUT. 