# Soil Health Analyzer - Android App

This is the Android mobile application version of the Soil Health Analyzer, built using Kivy and KivyMD.

## Features

- **Mobile-optimized interface** for soil health analysis
- **Real-time visualization** of soil parameters using gauge charts
- **Comprehensive soil reports** based on pH, Nitrogen, Phosphorus, and Potassium levels
- **Report history** to track soil changes over time
- **Offline mode** with pre-generated reports based on dataset
- **Share and export** functionality for reports
- **Mobile-responsive design** for various screen sizes

## Installation

### Prerequisites

- Python 3.8 or higher
- Buildozer for Android packaging
- Required Python packages (see requirements.txt)

### Development Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/soil-health-analyzer.git
   cd soil-health-analyzer/soil_analyzer/android_app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Buildozer:
   ```
   pip install buildozer
   ```

4. For Windows users, ensure you have the necessary tools:
   ```
   pip install docutils pygments pypiwin32 kivy_deps.sdl2 kivy_deps.glew
   ```

### Building the Android APK

1. Initialize buildozer (if not already done):
   ```
   buildozer init
   ```

2. Build the APK (this will take some time):
   ```
   buildozer android debug
   ```

3. The APK will be generated in the `bin` directory.

## Usage

### Running in Development Mode

You can run the app on your computer for development:

```
python main.py
```

### Using the App

1. Launch the Soil Health Analyzer app on your Android device
2. Enter your soil parameters:
   - pH level (0-14)
   - Nitrogen (N) content (0-150 kg/ha)
   - Phosphorus (P) content (0-100 kg/ha)
   - Potassium (K) content (0-100 kg/ha)
3. Tap "Analyze Soil" to generate a report
4. View the detailed soil health report with recommendations
5. Save or share your results

## API Setup

For full functionality, a Groq API key is required:

1. Go to the Settings tab in the app
2. Enter your Groq API key
3. Save the settings

Without an API key, the app will use pre-generated reports from the included dataset.

## Data Source

The app uses the `soil_report_dataset_500.csv` dataset which contains:
- Soil parameter values (pH, N, P, K)
- Suitable crops
- Pre-generated reports

## Customization

You can customize the app by modifying:
- The soil parameter ranges in main.py
- The UI layout in soilanalyzer.kv
- The buildozer.spec file for packaging options

## Troubleshooting

### Common Issues

- **App crashes during analysis**: Check your internet connection and API key
- **Blank screen on startup**: Ensure all dependencies are installed correctly
- **Build errors**: Make sure you have all Android SDK components installed

### Getting Help

If you encounter issues:
1. Check the log files in the .buildozer directory
2. Refer to the Kivy documentation at kivy.org
3. Post issues on our GitHub repository

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kivy framework for cross-platform mobile development
- KivyMD for Material Design components
- Groq LLM API for soil report generation
- All contributors to the project 