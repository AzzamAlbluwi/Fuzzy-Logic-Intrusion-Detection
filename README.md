# Fuzzy Logic-based Intrusion Detection System

## Introduction

This project implements a Fuzzy Logic-based Intrusion Detection System (IDS) to detect network intrusions in cybersecurity data. The IDS uses the k-means clustering algorithm for dimensionality reduction through Principal Component Analysis (PCA), and then applies Fuzzy Logic to classify network traffic as either "Normal" or "Attack." The system provides an accuracy evaluation and visualizes the results through heatmaps and evaluation metrics.

## Features

- Utilizes PCA and Fuzzy Logic to classify network traffic.
- Provides an accuracy evaluation, True Positive Percentage, False Positive Percentage, True Negative Percentage, False Negative Percentage, Precision, Recall, and F-score.
- Visualizes the results using heatmaps and evaluation metric plots.

## Requirements

- Python 3.6 or higher
- NumPy
- pandas
- scikit-learn
- skfuzzy
- matplotlib
- seaborn

## Installation

1. Clone the repository: `git clone https://github.com/AzzamAlbluwi/fuzzy.git`
2. Navigate to the project folder: `cd fuzzy`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Download the dataset "UNSW_NB15.csv" and place it in the project root directory.
2. Run the script "Final_code.py": `python Final_code.py`
3. The script will perform PCA, Fuzzy Logic classification, and output the evaluation metrics and heatmaps in the terminal.

## Results

The system was tested using PCA with 3 components, and it achieved the following results:

- Overall Accuracy: 93.96%
- True Positive Percentage (TPP): 88.46%
- False Positive Percentage (FPP): 0.53%
- True Negative Percentage (TNP): 99.47%
- False Negative Percentage (FNP): 11.54%
- Precision: 0.9940
- Recall (True Positive Rate): 0.8846
- F-score: 0.9361

## Contributing

Contributions are welcome! If you have any ideas or suggestions, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to Dr. Liyth and Yousef Al-Suiwdan for their support and guidance during this project.
 
## Contact

For any questions or feedback, please contact Me at AlbluwiAzzam@gmail.com
