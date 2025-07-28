# EMG_Prosthetic_Project
Developing gesture classifying agent to drive robotic porsthetics

Data information:
1. Sampling Rate: 2000Hz
2. Gesture Durations: 6 seconds
3. Total Participants: 40
4. Total gesture repeats: 5
5. Between each gesture, the participants rested
6. Following gestures: 
    Rest
    Extension
    Flexion
    Ulnar Deviation
    Radial Deviation
    Grip
    Abduction
    Adduction
    Supination
    Pronation

Project Constraints:  
1. Classify 10 gestures
2. One agent
3. Small - must fit on an embedded system (STM32XX) without relying on SBC
4. The training set used is has a sample size of 40 people of diverse backgrounds
5. Each gesture is repeated five times and llasts for approximately six seconds
6. dataset was collected using four sensor positions
7. Low latency - 100ms training windows (model takes 100ms data in deployment)

Current Identifications:
1. The ten gestures are comprised of two distinct classes: Amplitude-biased, and phase-biased
2. Amplitude-biase (grip, rest...) are easily distinguishable by amplitude differences between the four sensor positions. Phase difference (complex gestures) are less easily distinguishable - require further training through creation of virtual sensor (READ: subtrcative difference between real sensors) brining total sensor count to 10 (4 real, 6 virtual [1-2, 1-3, 1-4, 2-3, 2-4, 3-4])
3. Difficulties differentiating between amplitude-bias and phase-bias do persist. Current models reach approx. 60% accuracy (optimisitic). Majority voting will be necessary post-processing in deployment - 3 votes results in 300ms.