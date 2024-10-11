# MEGA: Memory Episode Gaze Anticipation 👀

## Introduction
Welcome to the repository for the paper "Seeing the Future: Anticipatory Eye Gaze as a Marker of Memory." This study introduces the MEGA (Memory Episode Gaze Anticipation) paradigm, which utilizes eye tracking to quantify memory retrieval without relying on verbal reports. By monitoring anticipatory gaze patterns, we can index event memory traces, providing a novel "no-report" method for assessing episodic memory.

## Getting Started

### Prerequisites
To run the scripts in this repository, you need:
- Python 3.9
- Required Python packages (listed in `requirements.txt`)

### Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/dyamin/MEGA.git
cd MEGA
```
Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
Preprocessing starts with two main components: `Preprocess-subjects` and `decentralized`.

1. **Preprocess-subjects**:
   - Choose between `ParseEyeLinkAsc` or `Cili` to read the EyeLink `.asc` files (after conversion using the EyeLink tool `edf2asc`). This will return all the data from the EyeLink parser as a large pandas DataFrame.

2. **Decentralized**:
   - Since each subject watches the movies in a different order, you will have one long file with movie number markers. This code extracts data for each movie independently, making it easier to load and process.

```bash
To run this step, use the file `PreprocessingController.py` under the folder pre_processing
```

### Data Postprocessing
Run the postprocessing module to organize and classify the eye-tracking data, with the memory reports:
```bash
To run this step, use the file `PostprocessingController.py` under the folder post_processing
```

### RoI
Run the RoI module to add the RoI definitions
```bash
To run this step, use the file `src/roi/RoiController.py`
```

### Feature Extraction
To extract eye-tracking features, use the module features_extraction
```bash
To run this step, use the file `src/features_extraction/FeatureExtractionController.py'
```

### Statistical Analysis
To carry out statistical analysis, you can use the Statistical Analysis module to create visualizations and calculate the statistical power for the effect. Here's how to do it:
```bash
1. Start by defining the configuration in the `src/statistical_analysis/config.py` file.
2. Next, execute the pre-plotter file located at 'src/statistical_analysis/Preplotter.py'
3. Finally, run either 'src/statistical_analysis/GazeAnalysisPlotter.py' or 'src/statistical_analysis/BetweenPopulationsAnalysisPlotter.py' based on the specific test you need to perform.
```

### Machine Learning
Use the provided Jupyter notebooks to perform machine learning classification on the eye-tracking features (under src/features_extraction).
```bash
The version we utilized in the paper is: `src/classification/paired_session_classification_loso_ALL_ROI_v2.ipynby`
```
These notebooks contain detailed instructions and scripts for preparing data, training, and evaluating machine learning models. You can also perform single-trial classification using XGBoost and other models we experimented with.

### Visualization
To generate videos with eye-tracking data like the ones we used [here](https://drive.google.com/file/d/1-W7XKSghak0-JkF8k19or83d4DMbNXbJ/view?usp=drive_link)

```bash
1. Start by defining the configuration in the `src/visualize/config.py` file.
2. Next, execute the 'src/visualize/VisualizeController.py' file
```

## Stimuli
The movie clips used in the experiments are available at [Yuval Nir Lab](https://yuvalnirlab.com/)

## Experiment
The experiment code is available [here](https://github.com/dyamin/MEGA-Experiment)

## Data
The data used in this study is available upon request due to privacy and ethical considerations. Please contact the corresponding author for access.

## Results
- **Figures and Tables**: Generated results can be found in the `results` directory.
- **Supplementary Materials**: Additional analysis and figures are available in the `supplementary` directory.

## Citing This Work
If you use any part of this code or data in your research, please cite our paper:

Yamin D., Schmidig J.F., Sharon O., Nadu Y., Nir J., Ranganath C., Nir Y. (2024). Seeing the future: anticipatory eye gaze as a marker of memory.

```
TODO
```

## Contributing
We welcome contributions from the community. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add Your Feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## Authors
- Daniel Yamin
- Flavio Schmidig
- Omer Sharon
- Jonathan Nir
- Yuval Shapira
- Yuval Nir

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
For more detailed information, please refer to our published paper:
Yamin D., Schmidig J.F., Sharon O., Nadu Y., Nir J., Ranganath C., Nir Y. (2024). Seeing the future: anticipatory eye gaze as a marker of memory.
