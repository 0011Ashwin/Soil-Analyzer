import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import base64
from dotenv import load_dotenv

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar
from kivy_garden.graph import Graph, MeshLinePlot
from kivy.properties import StringProperty, NumericProperty, ListProperty, ObjectProperty
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.metrics import dp

from kivymd.app import MDApp
from kivymd.uix.card import MDCard
from kivymd.uix.tab import MDTabsBase
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.list import OneLineListItem, MDList
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import Snackbar

# Load environment variables
load_dotenv()

# Configure Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY and GROQ_API_KEY.startswith('"') and GROQ_API_KEY.endswith('"'):
    GROQ_API_KEY = GROQ_API_KEY[1:-1]  # Remove quotes if present

# Define soil parameter ranges
soil_ranges = {
    "pH": {
        "very_acidic": (0, 5.5),
        "acidic": (5.5, 6.5),
        "neutral": (6.5, 7.5),
        "alkaline": (7.5, 8.5),
        "very_alkaline": (8.5, 14)
    },
    "N": {
        "very_low": (0, 50),
        "low": (50, 70),
        "medium": (70, 90),
        "high": (90, 110),
        "very_high": (110, 200)
    },
    "P": {
        "very_low": (0, 20),
        "low": (20, 30),
        "medium": (30, 40),
        "high": (40, 50),
        "very_high": (50, 100)
    },
    "K": {
        "very_low": (0, 30),
        "low": (30, 40),
        "medium": (40, 50),
        "high": (50, 60),
        "very_high": (60, 100)
    }
}

# Function to classify soil parameters
def classify_soil_parameter(value, parameter):
    for category, (min_val, max_val) in soil_ranges[parameter].items():
        if min_val <= value < max_val:
            return category
    return "unknown"

# Function to get color based on category
def get_color_for_category(category):
    colors = {
        "very_acidic": "#d32f2f",
        "acidic": "#f57c00",
        "neutral": "#4caf50",
        "alkaline": "#f57c00",
        "very_alkaline": "#d32f2f",
        "very_low": "#d32f2f",
        "low": "#f57c00",
        "medium": "#4caf50",
        "high": "#2196f3",
        "very_high": "#673ab7"
    }
    return colors.get(category, "#9e9e9e")

# Load dataset
def load_data():
    try:
        # First try to load from the app directory
        return pd.read_csv("soil_report_dataset_500.csv")
    except:
        try:
            # Then try to load from the parent directory
            return pd.read_csv("../soil_report_dataset_500.csv")
        except:
            # Finally try to load from absolute path (for development only)
            app_path = os.path.dirname(os.path.abspath(__file__))
            parent_path = os.path.dirname(app_path)
            return pd.read_csv(os.path.join(parent_path, "soil_report_dataset_500.csv"))

# Analyze Screen
class AnalyzeScreen(Screen):
    ph_value = NumericProperty(7.0)
    n_value = NumericProperty(75)
    p_value = NumericProperty(40)
    k_value = NumericProperty(50)
    
    def __init__(self, **kwargs):
        super(AnalyzeScreen, self).__init__(**kwargs)
        self.data = load_data()
    
    def on_enter(self):
        # Update gauge displays when entering the screen
        self.update_gauges()
    
    def update_gauges(self):
        # Update the gauge displays based on current values
        self.ids.ph_gauge.update_gauge(self.ph_value, "pH")
        self.ids.n_gauge.update_gauge(self.n_value, "N")
        self.ids.p_gauge.update_gauge(self.p_value, "P")
        self.ids.k_gauge.update_gauge(self.k_value, "K")
    
    def analyze_soil(self):
        # Show progress during analysis
        self.ids.analyze_progress.opacity = 1
        
        # Schedule the actual analysis to allow UI to update
        Clock.schedule_once(self.perform_analysis, 0.5)
    
    def perform_analysis(self, dt):
        # Generate soil report
        report, suitable_crop, closest_matches = self.generate_soil_report(
            self.ph_value, self.n_value, self.p_value, self.k_value
        )
        
        # Save to history
        app = App.get_running_app()
        app.save_report_to_history(
            self.ph_value, self.n_value, self.p_value, self.k_value, 
            report, suitable_crop
        )
        
        # Hide progress indicator
        self.ids.analyze_progress.opacity = 0
        
        # Navigate to results screen with the data
        results_screen = app.root.get_screen('results')
        results_screen.display_results(
            self.ph_value, self.n_value, self.p_value, self.k_value,
            report, suitable_crop, closest_matches
        )
        app.root.current = 'results'
    
    def generate_soil_report(self, ph, n, p, k):
        # Find the most suitable crop based on soil parameters
        def weighted_distance(row):
            ph_diff = abs(row['pH'] - ph) / 14  # Normalize pH (0-14 range)
            n_diff = abs(row['N'] - n) / 150    # Normalize N (assuming 0-150 range)
            p_diff = abs(row['P'] - p) / 100    # Normalize P
            k_diff = abs(row['K'] - k) / 100    # Normalize K
            return ph_diff * 0.4 + n_diff * 0.2 + p_diff * 0.2 + k_diff * 0.2

        self.data['distance'] = self.data.apply(weighted_distance, axis=1)
        closest_matches = self.data.sort_values('distance').head(3)
        
        # Get the most suitable crop and its ideal values
        best_match = closest_matches.iloc[0]
        suitable_crop = best_match['Crop']
        
        # If no API key available, use the existing report from the dataset
        app = App.get_running_app()
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            app.show_snackbar("No Groq API key found. Using pre-generated reports.")
            report = best_match['Report']
            return report, suitable_crop, closest_matches
        
        # Create prompt for Groq LLM
        prompt = f"""
        Generate a comprehensive soil health report based on the following soil test results:
        
        pH: {ph} (Classification: {classify_soil_parameter(ph, "pH")})
        Nitrogen (N): {n} kg/ha (Classification: {classify_soil_parameter(n, "N")})
        Phosphorus (P): {p} kg/ha (Classification: {classify_soil_parameter(p, "P")})
        Potassium (K): {k} kg/ha (Classification: {classify_soil_parameter(k, "K")})
        
        Most suitable crop based on similar soil profiles: {suitable_crop}
        
        The report should include the following sections:
        1. Soil Deficiency Analysis: Analyze each parameter (pH, N, P, K) stating whether it's optimal, too low, or too high for general plant growth and specifically for {suitable_crop}.
        2. Detailed Recommendations: Suggest specific fertilizers or amendments to address any deficiencies or excesses.
        3. Soil Type Context: Indicate what soil type might have these characteristics and how suitable it is for {suitable_crop}.
        4. Irrigation Recommendation: Suggest appropriate irrigation methods.
        5. Additional Suggestions: Provide 2-3 practical tips for improving soil health for {suitable_crop} cultivation.
        
        Keep the report professional but easy to understand. Use specific measurements and product recommendations where appropriate.
        Format the report with clear section headings and proper paragraph breaks.
        """
        
        try:
            # Call Groq API for report generation
            messages = [
                {
                    "role": "system",
                    "content": "You are a soil science expert specializing in agricultural soil health analysis. Provide informative, accurate and actionable soil reports."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            try:
                chat_completion = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama3-70b-8192",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 1024,
                        "top_p": 0.9
                    }
                )
                
                if chat_completion.status_code != 200:
                    app.show_snackbar(f"API Error: {chat_completion.status_code}")
                    # Fallback to the report from the dataset
                    report = best_match['Report']
                    return report, suitable_crop, closest_matches
                    
                report = chat_completion.json()["choices"][0]["message"]["content"]
            except Exception as e:
                app.show_snackbar(f"Error calling Groq API: {str(e)}")
                # Fallback to the report from the dataset
                report = best_match['Report']
            
            return report, suitable_crop, closest_matches
        except Exception as e:
            app.show_snackbar(f"Error generating report: {str(e)}")
            
            # Fallback to the report from the dataset
            report = best_match['Report']
            return report, suitable_crop, closest_matches

# Results Screen
class ResultsScreen(Screen):
    report_text = StringProperty("")
    suitable_crop = StringProperty("")
    
    def __init__(self, **kwargs):
        super(ResultsScreen, self).__init__(**kwargs)
        self.ph = 0
        self.n = 0
        self.p = 0
        self.k = 0
        self.closest_matches = None
    
    def display_results(self, ph, n, p, k, report, suitable_crop, closest_matches):
        self.ph = ph
        self.n = n
        self.p = p
        self.k = k
        self.report_text = report
        self.suitable_crop = suitable_crop
        self.closest_matches = closest_matches
        
        # Update the report tab
        self.ids.report_content.text = report
        
        # Update comparison tab
        self.update_comparison()
        
        # Update similar profiles tab
        self.update_similar_profiles()
    
    def update_comparison(self):
        # Clear previous widgets
        self.ids.comparison_content.clear_widgets()
        
        # Add the radar chart (this would be a custom widget in a real app)
        # Here we're just adding a placeholder
        comparison_label = Label(
            text=f"Your Soil vs. Ideal for {self.suitable_crop}\n\n"
                 f"pH: {self.ph} (Ideal: {self.closest_matches.iloc[0]['pH']})\n"
                 f"N: {self.n} (Ideal: {self.closest_matches.iloc[0]['N']})\n"
                 f"P: {self.p} (Ideal: {self.closest_matches.iloc[0]['P']})\n"
                 f"K: {self.k} (Ideal: {self.closest_matches.iloc[0]['K']})",
            size_hint_y=None,
            height=dp(200),
            halign='center'
        )
        self.ids.comparison_content.add_widget(comparison_label)
    
    def update_similar_profiles(self):
        # Clear previous widgets
        self.ids.profiles_content.clear_widgets()
        
        # Add similar profiles
        for i in range(min(3, len(self.closest_matches))):
            profile = self.closest_matches.iloc[i]
            profile_card = MDCard(
                orientation='vertical',
                size_hint=(1, None),
                height=dp(120),
                padding=dp(10)
            )
            
            title = Label(
                text=f"Profile {i+1}: {profile['Crop']}",
                size_hint_y=None,
                height=dp(30),
                bold=True
            )
            
            details = Label(
                text=f"pH: {profile['pH']}, N: {profile['N']}, P: {profile['P']}, K: {profile['K']}",
                size_hint_y=None,
                height=dp(30)
            )
            
            similarity = Label(
                text=f"Similarity Score: {profile['distance']:.4f}",
                size_hint_y=None,
                height=dp(30)
            )
            
            profile_card.add_widget(title)
            profile_card.add_widget(details)
            profile_card.add_widget(similarity)
            
            self.ids.profiles_content.add_widget(profile_card)
    
    def share_report(self):
        App.get_running_app().show_snackbar("Sharing report...")
        # In a real app, this would open the Android share dialog
    
    def download_report(self):
        App.get_running_app().show_snackbar("Report saved to downloads")
        # In a real app, this would save the report as a file

# History Screen
class HistoryScreen(Screen):
    def on_enter(self):
        self.load_history()
    
    def load_history(self):
        app = App.get_running_app()
        history_list = self.ids.history_list
        history_list.clear_widgets()
        
        if not app.report_history:
            history_list.add_widget(
                Label(text="No report history yet", halign='center')
            )
            return
        
        # Display history in reverse order (newest first)
        for i, history_item in enumerate(reversed(app.report_history)):
            item_card = MDCard(
                orientation='vertical',
                size_hint=(1, None),
                height=dp(150),
                padding=dp(10)
            )
            
            title = Label(
                text=f"Report {i+1}: {history_item['timestamp']}",
                size_hint_y=None,
                height=dp(30),
                bold=True
            )
            
            crop = Label(
                text=f"Crop: {history_item['suitable_crop']}",
                size_hint_y=None,
                height=dp(20)
            )
            
            params = Label(
                text=f"pH: {history_item['ph']}, N: {history_item['n']}, P: {history_item['p']}, K: {history_item['k']}",
                size_hint_y=None,
                height=dp(30)
            )
            
            button_layout = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=dp(40)
            )
            
            view_button = Button(
                text="View Report",
                on_press=lambda btn, index=i: self.load_report(index)
            )
            
            button_layout.add_widget(view_button)
            
            item_card.add_widget(title)
            item_card.add_widget(crop)
            item_card.add_widget(params)
            item_card.add_widget(button_layout)
            
            history_list.add_widget(item_card)
    
    def load_report(self, index):
        app = App.get_running_app()
        if index < len(app.report_history):
            # Use the reversed index since we're displaying newest first
            report = app.report_history[len(app.report_history) - 1 - index]
            
            # Update analyze screen values
            analyze_screen = app.root.get_screen('analyze')
            analyze_screen.ph_value = report['ph']
            analyze_screen.n_value = report['n']
            analyze_screen.p_value = report['p']
            analyze_screen.k_value = report['k']
            
            # Manually trigger analysis
            analyze_screen.analyze_soil()

# Settings Screen
class SettingsScreen(Screen):
    def __init__(self, **kwargs):
        super(SettingsScreen, self).__init__(**kwargs)
        self.api_key = GROQ_API_KEY if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here" else ""
    
    def on_enter(self):
        self.ids.api_key_input.text = self.api_key
    
    def save_api_key(self):
        new_api_key = self.ids.api_key_input.text
        self.api_key = new_api_key
        
        # Save to .env file
        try:
            with open(".env", "w") as f:
                f.write(f"GROQ_API_KEY={new_api_key}")
            App.get_running_app().show_snackbar("API key saved successfully!")
        except Exception as e:
            App.get_running_app().show_snackbar(f"Error saving API key: {str(e)}")
    
    def save_settings(self):
        App.get_running_app().show_snackbar("Settings saved!")
    
    def clear_history(self):
        app = App.get_running_app()
        app.report_history = []
        app.save_history_to_file()
        App.get_running_app().show_snackbar("History cleared")

# Custom Gauge Widget
class GaugeWidget(BoxLayout):
    parameter = StringProperty("")
    value = NumericProperty(0)
    
    def update_gauge(self, value, parameter):
        self.value = value
        self.parameter = parameter
        
        # Update the value display
        category = classify_soil_parameter(value, parameter)
        self.ids.gauge_value.text = f"{value:.1f}"
        self.ids.gauge_category.text = category.replace('_', ' ').title()
        
        # Update gauge color
        color = get_color_for_category(category)
        self.ids.gauge_value.color = get_color_from_hex(color)
        
        # Update gauge progress
        max_val = 14 if parameter == "pH" else 150
        self.ids.gauge_bar.value = (value / max_val) * 100

# Main App
class SoilAnalyzerApp(MDApp):
    def __init__(self, **kwargs):
        super(SoilAnalyzerApp, self).__init__(**kwargs)
        self.report_history = []
        self.load_history_from_file()
    
    def build(self):
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.accent_palette = "LightGreen"
        self.theme_cls.theme_style = "Light"
        
        # Create the screen manager
        sm = ScreenManager()
        sm.add_widget(AnalyzeScreen(name='analyze'))
        sm.add_widget(ResultsScreen(name='results'))
        sm.add_widget(HistoryScreen(name='history'))
        sm.add_widget(SettingsScreen(name='settings'))
        
        return sm
    
    def show_snackbar(self, text):
        Snackbar(text=text).open()
    
    def save_report_to_history(self, ph, n, p, k, report, suitable_crop):
        """Saves the current report to history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_history.append({
            "timestamp": timestamp,
            "ph": ph,
            "n": n,
            "p": p,
            "k": k,
            "suitable_crop": suitable_crop,
            "report": report
        })
        
        # Keep only the last 10 reports
        if len(self.report_history) > 10:
            self.report_history = self.report_history[-10:]
        
        # Save to file
        self.save_history_to_file()
    
    def save_history_to_file(self):
        """Saves history to a file"""
        try:
            with open("report_history.json", "w") as f:
                json.dump(self.report_history, f)
        except Exception as e:
            print(f"Error saving history: {str(e)}")
    
    def load_history_from_file(self):
        """Loads history from a file"""
        try:
            with open("report_history.json", "r") as f:
                self.report_history = json.load(f)
        except:
            # If file doesn't exist, use empty history
            self.report_history = []

if __name__ == '__main__':
    SoilAnalyzerApp().run() 