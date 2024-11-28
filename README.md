# Dynamic Music Data Visualization and Prediction System

This project offers an interactive platform for dynamic data visualization and personalized predictions. Designed with **HTML** for visualizations and enhanced by **JavaScript** functionalities, the system leverages a **Flask server** in Python to handle requests. A dedicated service layer integrates machine learning models for streamlined training and prediction processes.

Users can upload their own `.csv` files for analysis or rely on an available **Kaggle dataset** [link](https://www.kaggle.com/datasets/abdulszz/spotify-most-streamed-songs). containing Spotify music data. The system allows data exploration, model training, and stream predictions for uploaded datasets.

---

## Features

### **Dynamic Data Visualization**

- Import `.csv` files locally to analyze your custom data.
- Use the preloaded Spotify music dataset for immediate insights.
- Interactive charts and graphs for exploring trends, artists, platforms, and more.

### **Custom Model Training**

Train your data with one of four available machine learning models:  

- **RandomForestRegressor**  
- **GradientBoostingRegressor**  
- **LightGBM**  
- **AdaBoostRegressor**  

Users can fine-tune model parameters to improve prediction accuracy.

### **Stream Prediction**

In the **Predictions** tab, the system provides:

- Row-by-row predictions of expected music streams for your uploaded dataset.  
- Easy-to-understand results formatted directly into the table.

---

## Setup and Usage

This project is managed by **UV**, a powerful Python management tool that simplifies the environment setup and application execution.

### **Prerequisites**

Ensure that UV is installed on your system. You can find more about UV [here](https://docs.astral.sh/uv/getting-started/installation/).

### **Steps to Run the Project**

1. **Install Dependencies**  
   Run the following command to install all necessary dependencies:

   ```bash
   uv sync
   ```

2. **Set Up a Virtual Environment**
   Create an isolated virtual environment:

   ```bash
   uv venv
   ```

3. **Start the Application**
   Launch the project on your local machine with:

   ```bash
   uv run flask --app app/app.py --debug run
   ```

4. **Access the Application**
   Open your browser and navigate to <http://127.0.0.1:5000> to interact with the system.
