Loan Approval Prediction using Machine Learning

This project is an end-to-end Machine Learning pipeline designed to predict whether a loan application should be approved based on applicant data. It includes data preprocessing, model training, evaluation, visualization, and a simple application interface for real-time predictions.

🚀 Features

Data cleaning and preprocessing

Training ML models for loan approval classification

Visualization of decision paths and important features

Ready-to-use prediction script

Lightweight app (app.py) for user input and instant prediction

📂 Project Structure
├── app.py
├── data/
│   ├── RawLoan.csv
│   └── ProcessedLoan.csv
├── outputs/
│   └── decision_tree_top3.png
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── prediction.py
│   └── visualization.py
├── requirements.txt
└── tree.png

🧠 How It Works

Preprocessing
Raw data is cleaned, missing values handled, and features encoded using preprocess.py.

Model Training
ML algorithms such as Decision Trees are trained and evaluated using train.py.

Visualization
Important features and decision paths are visualized with visualization.py.

Prediction
New loan applications can be evaluated through prediction.py or the app interface.

▶️ Running the Project
1. Install dependencies
pip install -r requirements.txt

2. Preprocess the data
python scripts/preprocess.py

3. Train the model
python scripts/train.py

4. Run predictions
python scripts/prediction.py

5. Launch the app
python app.py

📊 Sample Visualization

A decision-tree visualization is included in the outputs/ folder to help understand how predictions are made.

👨‍💻 Author

Vivek Rao GitHub: https://github.com/vivekkrao07
Machine Learning & Data Science Enthusiast

