# ICP Analysis Setup



## Setting up environment

. use the environment provided in excercise 5.

You can download it using this link

` https://github.com/seva100/3DSMC-exercise-5-core `

. Install the dependencies from requirements.txt

`pip install -r requirements.txt`





## Configuring the project 

### make configurations using config.h

Symmetric ICP optimizer enabled (USE_SYMMETRIC_ICP = 1)

Point-to-plane constraints enabled (USE_POINT_TO_PLANE = 1)

Sequence ICP testing enabled (RUN_SEQUENCE_ICP = 1)

Medium downsampling level (75% point reduction)

Colored ICP disabled (geometry-only processing)



#### Configuration Options
- ICP Optimizer Selection

Choose exactly one ICP implementation by setting it to 1 and all others to 0:

USE_LINEAR_ICP = 1 Fast, closed-form solution, best for small rotations 

USE_LM_ICP = 1 Levenberg-Marquardt, robust for large rotations 

USE_SYMMETRIC_ICP = 1 Bidirectional optimization, balanced approach 


- Constraint Type

USE_POINT_TO_PLANE = 1 Point-to-plane constraints (better surface alignment) USE_POINT_TO_PLANE = 0 Point-to-point constraints (faster computation)

- Color Information
USE_COLORED_ICP = 1 Uses RGB information for better correspondence 

USE_COLORED_ICP = 0 Geometry-only processing (faster)

- Test Scenarios

RUN_SHAPE_ICP = 1 Stanford Bunny alignment (development/testing) 

RUN_SEQUENCE_ICP = 1 Room reconstruction sequence (comprehensive)

- Performance Tuning

DOWNSAMPLING_LEVEL options:

"low" : No downsampling (0% reduction, highest quality, slowest)

"medium" : 4x downsampling (75% reduction, balanced performance)

"high" : 8x downsampling (87.5% reduction, fastest processing)

- Troubleshooting Common Issues

Multiple optimizers enabled: Ensure only one USE_*_ICP is set to 1

Invalid downsampling level: Use exact strings "low", "medium", or "high"

Build errors after config changes: Clean build directory and reconfigure

Performance issues: Start with "medium" downsampling for initial testing


## Running the project
1. move to build directory

`cd /config/workspace/Exercises/Project/ICP_Analysis/ICP-variants/src/build`

2. Build the project 

`make`

3. Run 

`./ICP_Analysis`



