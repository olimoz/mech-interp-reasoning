================================================================
NOTE: 

Example long prompt for ARC challenge 017c7c7b from the training set.
The text has been prepended to the matrices, which are taken directly from the challenge file.
We prepend the same text to all challenge matrices and submit the concatenation to the model as the prompt.

================================================================
PROMPT:

# PROJECT INSTRUCTIONS

The training data consists of pairs of input and output grids, presented as numpy arrays of varying shapes.
Your task is to discover the single mapping which converts each input grid to its corresponding output grid and apply that to the test input, arriving at a test output.

## 1. OBSERVE AND HYPOTHESISE THE MAPPING LOGIC FOR ALL TRAINING PAIRS

When building your hypotheses on the mappings, be aware of the following common transformations:

    Grid Expansion and Repetition (Tiling):
    - Simply expand the grid and repeat (tile) the input grid into the output grid
    Symmetry and Mirroring (flipping):
    - Horizontally or vertically
    Propagation of patterns:
    - Identify non-zero clusters or shapes in the input grid and propagating them in the output. Proceeding horizontally, vertically or diagonally.
    Mathematical Operations:
    - Incrementing values, taking modulo, or performing addition.
    Color/Value Substitution:
    - Values in the input grid replaced with different values in the output grid, often changing all instances of one number to another
    Shape Detection and Transformation:
    - Identifying geometric shapes in the input grid and applying transformations such as rotation, scaling, flipping, translation and/or overlapping.
    Grid Segmentation:
    - Divide the input grid into sections and apply transformations to each section.
    Boundary Detection and Fill:
    - Identify the boundaries of shapes or patterns and fill them with specific values. This sometimes involved propagating values from the edges inward.
    Connectivity-based Transformations:
    - Using connected component analysis to identify and transform groups of connected cells.
    Rule-based Transformations:
    - Applying specific rules based on the arrangement of values in the input grid. These rules often considered the neighboring cells of each position.
    Coordinate-based Transformations:
    - Using the coordinates of cells to determine how they should be transformed or moved in the output grid.
    When the pattern is more complex than originally assumed:
    - Review all training pairs again and try to describe the transformation in plain language

Use these patterns to guide your own hypotheses on the training data.

## 2. PREDICT THE OUTPUT GRID FOR THE TEST INPUT GRID

Having considered the data...
FIRST score your confidence that you understand the mapping pattern, 0-5 where 0 is no confidence and 5 is highly confident. 
This score must be the FIRST output you give, no preamble, no prefix, no punctuation, just a single digit score.
THEN Present your predicted output in np.array format

## 3. THE DATA

================================================================
END OF PROMPT
EXAMPLE DATA:

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