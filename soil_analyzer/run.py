import os
import subprocess
import sys

def main():
    print("🌱 Starting Soil Health Analyzer...")
    
    # Check if .env file exists and has Groq API key
    if not os.path.exists(".env"):
        print("Warning: .env file not found. Creating a template .env file.")
        with open(".env", "w") as f:
            f.write("# Groq API Key - Replace with your actual key\n")
            f.write("GROQ_API_KEY=your_groq_api_key_here\n")
        print("Please edit the .env file to add your Groq API key before continuing.")
        sys.exit(1)
    
    # Check if soil_report_dataset_500.csv exists
    if not os.path.exists("soil_report_dataset_500.csv"):
        print("Error: soil_report_dataset_500.csv not found.")
        print("Please make sure the dataset file is in the same directory as this script.")
        sys.exit(1)
    
    # Launch Streamlit app
    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install required packages:")
        print("pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main() 