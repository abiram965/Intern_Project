import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import pickle
import csv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class MLModelGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Machine Learning Model GUI")
        self.model = None
        self.data = None
        
        # Create frame for feature entry fields
        self.feature_frame = tk.Frame(master)
        self.feature_frame.pack(padx=10, pady=5, anchor='w')
        
        # Features to be entered by the user
        self.features = ['Sex', 'Age', 'AI1', 'AI2', 'AI3', 'AI4', 'PCGPA', 'Failures', 'Reason',
                         'Test Preparation Course', 'Traveltime', 'Studytime', 'Medu', 'Fedu',
                         'Mjob', 'Fjob', 'Activities', 'Internet', 'Freetme', 'Absence', 'LC1',
                         'LC2', 'LC3']
        
        # Entry fields for each feature
        self.feature_entries = {}
        self.create_feature_entries()
        
        # Button to save feature values to CSV
        self.save_button = tk.Button(master, text="Save to CSV", command=self.save_to_csv)
        self.save_button.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Button to load model and data
        self.load_button = tk.Button(master, text="Load Model and Data", command=self.load_model_and_data)
        self.load_button.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Button to analyze input and display suggestions
        self.analyze_button = tk.Button(master, text="Analyze and Display Suggestions", command=self.display_suggestions)
        self.analyze_button.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Button to predict
        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.output_label = tk.Label(master, text="")
        self.output_label.pack(pady=10)
        
        # Frame to display model performances
        self.performance_frame = tk.Frame(master)
        self.performance_frame.pack(anchor='w', padx=10, pady=10)
        
        # Model performance labels
        self.performance_labels = {}
        self.display_model_performances()
        
        # Conclusion
        self.conclusion_label = tk.Label(master, text="Conclusion:", font=("Arial", 12, "bold"))
        self.conclusion_label.pack(anchor='w', padx=10, pady=5)
        self.conclusion_text = tk.Label(master, text="As the R-squared value for SVR and Multiple Linear Regression models is higher, we are using the SVR regression model to predict the CGPA.", wraplength=500)
        self.conclusion_text.pack(anchor='w', padx=10, pady=5)

    def create_feature_entries(self):
        num_features = len(self.features)
        num_columns = 3  # Number of columns for feature entry fields
        num_rows = (num_features + num_columns - 1) // num_columns
        
        for i in range(num_rows):
            frame = tk.Frame(self.feature_frame)
            frame.pack(side=tk.LEFT, padx=10)
            
            for j in range(num_columns):
                index = i * num_columns + j
                if index < num_features:
                    feature = self.features[index]
                    tk.Label(frame, text=feature + ": ").pack(side=tk.TOP, anchor='w')
                    entry = tk.Entry(frame, width=10)
                    entry.pack(side=tk.TOP)
                    self.feature_entries[feature] = entry
    
    def save_to_csv(self):
        # Get feature values from entry widgets
        feature_values = {}
        for feature, entry in self.feature_entries.items():
            value = entry.get()
            if not value:
                messagebox.showerror("Error", f"Please enter a value for {feature}.")
                return
            feature_values[feature] = value
        
        # Prompt user to choose CSV file to save
        csv_filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if csv_filename:
            try:
                # Write feature values to CSV file
                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=feature_values.keys())
                    writer.writeheader()
                    writer.writerow(feature_values)
                messagebox.showinfo("Success", f"Feature values saved to {csv_filename}!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving feature values to CSV: {e}")

    def load_model_and_data(self):
        model_filename = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if model_filename:
            try:
                with open(model_filename, 'rb') as file:
                    self.model = pickle.load(file)
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {e}")
        
        data_filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if data_filename:
            try:
                self.data = pd.read_csv(data_filename)
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {e}")
    
    def preprocess_data(self):
        try:
            self.data['Sex'] = self.data['Sex'].map({'M': 0, 'F': 1})
            self.data['Mjob'] = self.data['Mjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
            self.data['Fjob'] = self.data['Fjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
            self.data['Reason'] = self.data['Reason'].map({'home': 0, 'reputation': 1, 'course': 2, 'other': 3})
            self.data['Activities'] = self.data['Activities'].map({'no': 0, 'yes': 1})
            self.data['Internet'] = self.data['Internet'].map({'no': 0, 'yes': 1})
            self.data['Test Preparation Course'] = self.data['Test Preparation Course'].map({'none': 0, 'completed': 1})

            categorical_columns = self.data.select_dtypes(include=['object']).columns
            numeric_cols = self.data.select_dtypes(include=['number']).columns
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(drop='first')
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_columns)
                ])
            preprocessor.fit_transform(self.data)
        except Exception as e:
            messagebox.showerror("Error", f"Error preprocessing data: {e}")
        
    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        if self.data is None:
            messagebox.showerror("Error", "Data not loaded!")
            return
        
        try:
            self.preprocess_data()
            predictions = self.model.predict(self.data)
            messagebox.showinfo("Predictions", f"CGPA: {predictions[0]}")
        except Exception as e:
            messagebox.showerror("Error", f"Error making predictions: {e}")

    def analyze_values(self):
        suggestions = []
        for feature, entry in self.feature_entries.items():
            value = entry.get()
            if feature == 'AI1' or feature == 'AI2' or feature == 'AI3' or feature == 'AI4':
                value = float(value)  # Convert value to float for comparison
                if value < 75:
                    suggestions.append(f"Low score in {feature}. Concentrate more on {feature} subject.")
            elif feature == 'Studytime':
                value = float(value)  # Convert value to float for comparison
                if value < 3:
                    suggestions.append("Low study time. Increase study hours.")
            elif feature == 'Freetme':
                value = float(value)  # Convert value to float for comparison
                if value > 3:
                    suggestions.append("Utilize free time efficiently for academic purposes.")
            elif feature == 'Absence':
                value = float(value)  # Convert value to float for comparison
                if value > 2:
                    suggestions.append("High number of absences. Regular attendance is important for learning.")
            elif feature.startswith('LC'):
                value = float(value)  # Convert value to float for comparison
                if value < 0.8:
                    suggestions.append(f"Low score in {feature}. Concentrate on basic subjects to improve.")

        return suggestions

    def display_suggestions(self):
        suggestions = self.analyze_values()
        if suggestions:
            messagebox.showinfo("Suggestions", "\n".join(suggestions))
        else:
            messagebox.showinfo("Suggestions", "No suggestions for improvement.")
    
    def display_model_performances(self):
        model_performances = {
            "SVR Model": 0.8886503261814326,
            "Random Forest Model": 0.8454821013508875,
            "Decision Tree Model": 0.7787007726420497,
            "Multiple Linear Regression Model": 0.8893790426951274,
            "Artificial Neural Network (ANN) Model": 0.8450600505963838,
            "XGBoost Model": 0.7804996830245308
        }
        tk.Label(self.performance_frame, text="Model Performances:", font=("Arial", 12, "bold")).pack(anchor='w', padx=10, pady=5)
        for model, r_squared in model_performances.items():
            label_text = f"{model}: R-squared: {r_squared}"
            label = tk.Label(self.performance_frame, text=label_text, font=("Arial", 10, "bold"))
            label.pack(anchor='w', padx=10)

def main():
    root = tk.Tk()
    app = MLModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
