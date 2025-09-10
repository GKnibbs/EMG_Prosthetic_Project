goal of project:
Want to create machine agent classifier for gestures using sEMG data.
The agent will be deployed on an embedded system thus will need to be runnable on hardware without a companion SBC or GPU.
The agent must run on the microcontroller itself.
The agent must be able to classify 10 different gestures.
The agent must be able to reliably classify getsures with an undetermined accuracy, and within as short a time as practically permissable.

Project development specifications:
1. The agent must be developed using python and TensorFlow/Keras.
2. Training is performed using google colab T4 GPU acceleration - the program to do this is editable in the Model_Training.ipynb notebook.
3. The gestures are the following; 
    1: 'REST',
    1: 'EXTENSION',
    2: 'FLEXION',
    3: 'ULNAR_DEVIATION',
    4: 'RADIAL_DEVIATION',
    5: 'GRIP',
    6: 'ABDUCTION',
    7: 'ADDUCTION',
    8: 'SUPINATION',
    9: 'PRONATION',
4. The project is being pursued by a single developer, who is also the project owner and is a novice to machine learning.
5. The project is separated out according to which iteration is being pursued. shown by the folder structure in Scripts where each iteration is in its own folder and there are copies of scripts just to make it intuitive what relates to what.
6. The project is being developed in an agile manner, with the developer learning as they go and iterating on the design of the agent as they learn more about machine learning and the constraints of the embedded system.

Data structure:
1. All 10 gestures are structured the same way.
2. Data is collected from 40 participants, each participant performs each of the ten gestures 5 times for 6 seconds each sampled at a rate of 2000Hz. The data collection procedure was as follows (for more information see image Data_Structure.png for a sample):
    1. A gesture is performed for 6 seconds
    2. Rest period
    3. Next gesture is performed for 6 seconds
    4. and so on until all 10 gestures are performed
    5. Therefore, there is transition data between gestures and at the start and end of each extracted gesture.
    6. Data was collected from 4 channels (4 sEMG sensors)
3. The data was then stored in 40 csv files (one of each participant) containing every data sample 
4. The csv files have already been read, and the 5 repetitions of each gesture have been extracted from each participant and concatenated into a single gesture csv file for each gesture called the following:
    1. 0_REST.csv
    2. 1_EXTENSION.csv
    3. 2_FLEXION.csv
    4. 3_ULNAR_DEVIATION.csv
    5. 4_RADIAL_DEVIATION.csv
    6. 5_GRIP.csv
    7. 6_ABDUCTION.csv
    8. 7_ADDUCTION.csv
    9. 8_SUPINATION.csv
    10. 9_PRONATION.csv
5. Each of these csv files contains 40*5*6*2000 = 2,400,000 rows and 5 columns (iD, one for each sEMG channel) titled the following:
    1. 'iD'
    2. 'ch1'
    3. 'ch2'
    4. 'ch3'
    5. 'ch4'
6. So the resulting gesture csv has the first 6 seconds of the gesture from participant 1, then the next 6 seconds of the gesture from participant 1, and so on until all 5 repetitions of the gesture from participant 1 are recorded, then the same for participant 2 and so on until all 40 participants have their data recorded.

Instructions for copilot:
0. Familiarise yourself with the project goal, development specifications, and data structure outlined above, and thoroughly read every line of code in the repository to understand how the project is structured and what has been done so far and read the Notes folder (see the images too) and read the research papers collected in Research_Papers folder to understand the context of the project.
1. When generating code, ensure that it adheres to the project development specifications outlined above.
2. Ensure that the code is well-documented with comments explaining the purpose of each function and class, as well as any complex logic.
3. When suggesting code, consider the constraints of running on an embedded system without a companion SBC or GPU.
4. When suggesting code, consider the need for the agent to classify 10 different gestures reliably and within a short time frame.
5. When suggesting code, consider the novice level of the developer and provide explanations for any advanced concepts or techniques used.
6. When suggesting code, consider the iterative nature of the project and provide suggestions for improvements or optimizations that can be made in future iterations.
7. When suggesting code, ensure modularity and separation of concerns to facilitate easy updates and modifications in future iterations.
8. When suggesting code, ensure that it is compatible with TensorFlow/Keras and can be run on the specified hardware.
9. When suggesting code, ensure that it is compatible with the existing codebase and follows the same coding style and conventions.
10. Always refresh understanding of the conversation and the project context before generating code and ensure that the suggestions align with the overall project goals and specifications.
11. Always above all else, ask me questions if you are unsure about any aspect of the project or the codebase before suggesting code and collaborate with me to ensure that the suggestions meet the project requirements and expectations rather than just generating code.