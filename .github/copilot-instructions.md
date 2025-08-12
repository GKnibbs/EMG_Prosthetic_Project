# Copilot Instructions for EMG_Prosthetic_Project

## Project Overview
- This project develops a gesture classification agent for robotic prosthetics using EMG data.
- Data is sampled at 2000Hz, with 10 gestures performed by 40 participants, each gesture repeated 5 times for 6 seconds.
- The goal is low-latency classification (100ms windows) suitable for embedded deployment (STM32XX, no SBC).

## Architecture & Data Flow
- **Data Source:** EMG CSV files, each containing multi-channel sensor data for gestures.
- **Preprocessing:**
  - Windowing: Data is split into 100ms windows with overlap (see `Scripts/data_steam.py`).
  - Virtual Sensors: Six additional channels are created by subtracting real sensor pairs (e.g., 1-2, 1-3, ...).
  - Label Mapping: Each file/segment is mapped to a gesture label.
- **Model Training:**
  - Models are trained to distinguish amplitude-biased and phase-biased gestures.
  - Majority voting is used for post-processing (3 votes = 300ms window).

## Key Files & Directories
- `Scripts/data_steam.py`: Core streaming, windowing, and batching logic for EMG data. Handles large files efficiently.
- `Models/`: (Not shown) Presumed location for model definitions and training scripts.
- `Notes/`, `Reports/`, `Research_Papers/`: Documentation and reference material.
- `README.md`: Project constraints, data details, and classification challenges.

## Developer Workflows
- **Data Preparation:**
  - Use `Scripts/data_steam.py` to stream and preprocess EMG data for ML tasks.
- **Model Training:**
  - Integrate with TensorFlow via the provided generator/dataset wrappers.
- **Deployment:**
  - Models must be small and fast enough for STM32XX microcontrollers.
- **Post-processing:**
  - Implement majority voting for gesture prediction smoothing.

## Project-Specific Patterns
- **Windowing:** Always use 100ms windows for training and inference.
- **Sensor Channels:** Use 4 real + 6 virtual sensors for feature extraction.
- **Labeling:** Map each data segment to one of 10 gestures.
- **Low Latency:** All code should be optimized for real-time, embedded use.

## Integration Points
- **TensorFlow/Keras:** For model training and evaluation.
- **NumPy/Pandas:** For data manipulation and streaming.

## Example: Streaming EMG Data
```python
from Scripts.data_steam import EMGStream
stream = EMGStream([...], label_map, window_size=200, step_size=100)
for X, y in stream.stream():
    # X: [batch_size, window_size, num_channels]
    # y: [batch_size]
```

## Conventions
- Keep all preprocessing and streaming logic in `Scripts/data_steam.py`.
- Document any new model architectures or post-processing steps in `README.md`.
- Use majority voting for final gesture prediction.

---
_If any section is unclear or incomplete, please provide feedback to improve these instructions._
