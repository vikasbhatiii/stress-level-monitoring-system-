import numpy as np
import pandas as pd
import webbrowser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle

# Load dataset
df = pd.read_csv('Data/heart.csv')

# Models dictionary
models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(random_state=100),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(max_depth=10, random_state=100)
}

# Global variables
scaler = None
normalizer = None

# Function to train models and show graphs
def train_models():
    global scaler, normalizer  # Make scalers global
    selected_model = model_var.get()
    if not selected_model:
        messagebox.showerror("Error", "Please select a model to train.")
        return

    X = df.drop(['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    scaler = StandardScaler().fit(X_train)
    normalizer = Normalizer().fit(X_train)

    model = models[selected_model]

    # Train with scaled data
    model.fit(scaler.transform(X_train), y_train)
    y_pred_scaled = model.predict(scaler.transform(X_test))
    accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

    # Train with normalized data
    model.fit(normalizer.transform(X_train), y_train)
    y_pred_normalized = model.predict(normalizer.transform(X_test))
    accuracy_normalized = accuracy_score(y_test, y_pred_normalized)

    # Save the trained model
    with open(f"{selected_model}_model.pkl", "wb") as file:
        pickle.dump(model, file)

    # Show training graphs
    show_training_graph(accuracy_scaled, accuracy_normalized, selected_model)

    messagebox.showinfo("Training Complete", f"{selected_model} trained!\nScaled Accuracy: {accuracy_scaled:.2f}\nNormalized Accuracy: {accuracy_normalized:.2f}")

# Function to show training graphs
def show_training_graph(accuracy_scaled, accuracy_normalized, selected_model):
    graph_window = tk.Toplevel(root)
    graph_window.title(f"{selected_model} Training Results")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Scaled Data", "Normalized Data"], [accuracy_scaled, accuracy_normalized], color=['skyblue', 'orange'])
    ax.set_title(f"{selected_model} Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to load models
def load_models():
    selected_model = model_var.get()
    if not selected_model:
        messagebox.showerror("Error", "Please select a model to load.")
        return None

    try:
        with open(f"{selected_model}_model.pkl", "rb") as file:
            return {selected_model: pickle.load(file)}
    except FileNotFoundError:
        messagebox.showerror("Error", f"Model {selected_model} not found! Train the model first.")
        return None

# Function to predict from user input
def predict():
    loaded_models = load_models()
    if loaded_models is None:
        return

    selected_model = model_var.get()
    model = loaded_models[selected_model]

    input_data = [float(entry.get()) for entry in input_entries]
    input_data = scaler.transform([input_data])  # Apply the same scaler

    probability = model.predict_proba(input_data)[0][1] * 100
    prediction = "Disease Predicted" if probability > 50 else "No Disease"

    result_text = f"{selected_model}: {prediction} ({probability:.2f}%)"
    messagebox.showinfo("Prediction", result_text)

    # Show prediction graph for the selected model
    show_prediction_graph(selected_model, probability)

    # Check if the model predicts disease
    if prediction == "Disease Predicted":
        show_recommendations()
        find_doctors_near_me()


# Function to display prediction probabilities
def show_prediction_graph(model_name, probability):
    graph_window = tk.Toplevel(root)
    graph_window.title("Prediction Probability")
    fig, ax = plt.subplots(figsize=(7, 5))

    # Create a bar graph for the selected model's probability
    ax.bar([model_name], [probability], color='skyblue')
    ax.set_title(f"Prediction Probability for {model_name}")
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)  # Set y-axis limit to 0-100%

    # Display the graph in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to show recommendations in a beautiful window
def show_recommendations():
    recommendations_window = tk.Toplevel(root)
    recommendations_window.title("Health Recommendations")
    recommendations_window.geometry("600x400")
    recommendations_window.configure(bg="#f0f0f0")

    # Title
    title_label = tk.Label(recommendations_window, text="Health Recommendations", font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333333")
    title_label.pack(pady=10)

    # Recommendations text
    recommendations_text = """
    If a heart disease is predicted, consider the following recommendations:
    1. Consult a cardiologist as soon as possible.
    2. Maintain a healthy diet low in salt, fat, and cholesterol.
    3. Exercise regularly, but consult your doctor before starting any new exercise program.
    4. Avoid smoking and limit alcohol consumption.
    5. Monitor your blood pressure and cholesterol levels regularly.
    6. Reduce stress through relaxation techniques such as meditation or yoga.
    """
    text_frame = tk.Frame(recommendations_window, bg="#f0f0f0")
    text_frame.pack(pady=10, padx=20, fill="both", expand=True)

    text_box = tk.Text(text_frame, wrap="word", font=("Arial", 12), bg="#ffffff", fg="#333333", padx=10, pady=10)
    text_box.insert("end", recommendations_text)
    text_box.config(state="disabled")
    text_box.pack(fill="both", expand=True)

    # Button to find doctors
    find_doctors_button = tk.Button(recommendations_window, text="Find Cardiologists Near Me", command=find_doctors_near_me, font=("Arial", 12, "bold"), bg="#3498db", fg="white", padx=10, pady=5)
    find_doctors_button.pack(pady=10)

# Function to open Google Maps to find heart doctors near the user
def find_doctors_near_me():
    url = "https://www.google.com/maps/search/cardiologist+near+me"
    webbrowser.open(url)

# Function to import CSV and make predictions
def import_csv_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    data = pd.read_csv(file_path)

    # Check if CSV columns match the expected format
    if list(data.columns) != list(df.columns[:-1]):
        messagebox.showerror("Error", "CSV file columns do not match the required format.")
        return

    # Ensure the scaler is available
    if 'scaler' not in globals():
        messagebox.showerror("Error", "Please train the models first before importing CSV.")
        return

    loaded_models = load_models()
    if loaded_models is None:
        return

    selected_model = model_var.get()
    model = loaded_models[selected_model]

    probabilities_list = []
    disease_status_list = []

    for _, row in data.iterrows():
        input_data = np.array(row).reshape(1, -1)
        input_data = scaler.transform(input_data)  # Apply scaler

        probability = model.predict_proba(input_data)[0][1] * 100
        disease_status = "Disease" if probability > 50 else "No Disease"

        probabilities_list.append(probability)
        disease_status_list.append(disease_status)

    # Convert predictions to DataFrames
    results_df = pd.DataFrame({selected_model: probabilities_list})
    disease_status_df = pd.DataFrame({selected_model: disease_status_list})

    if disease_status == "Disease":
        show_recommendations()
        find_doctors_near_me()

    # Display results in a tabular format
    show_results_table(results_df, disease_status_df)

    # Plot bar graph
    plot_prediction_graph(results_df, selected_model)

# Function to display results in a table
def show_results_table(results_df, disease_status_df):
    """Display results in a tabular format using Tkinter Treeview."""
    result_window = tk.Toplevel(root)
    result_window.title("Prediction Results")

    # Create a Treeview widget
    tree = ttk.Treeview(result_window)
    tree["columns"] = ["Sample", "Probability (%)", "Diagnosis"]
    tree["show"] = "headings"

    # Add column headings
    for col in tree["columns"]:
        tree.heading(col, text=col)
        tree.column(col, width=150)  # Adjust column width

    # Insert rows
    for i, (prob, status) in enumerate(zip(results_df.iloc[:, 0], disease_status_df.iloc[:, 0])):
        tree.insert("", "end", values=[f"Sample {i + 1}", f"{prob:.2f}%", status])

    # Pack the Treeview widget
    tree.pack(expand=True, fill="both")

# Function to plot prediction graph
def plot_prediction_graph(results_df, selected_model):
    """Plots a bar graph for the predicted probabilities."""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(results_df)), results_df.iloc[:, 0], color='skyblue')
    plt.xlabel("Samples")
    plt.ylabel("Predicted Probability (%)")
    plt.title(f"Predicted Probabilities by {selected_model}")
    plt.xticks(range(len(results_df)), [f"Sample {i + 1}" for i in range(len(results_df))])
    plt.show()

# Function to clear input fields
def clear_inputs():
    for entry in input_entries:
        entry.delete(0, tk.END)

# GUI setup
root = tk.Tk()
root.title("Stress Level Prediction")
root.geometry("1000x400")
root.configure(bg="#34495e")

title_label = tk.Label(root, text="Stress Level Prediction", font=("Arial", 20, "bold"), bg="#34495e", fg="white")
title_label.pack(pady=10)

desc_label = tk.Label(root, text="Enter patient details below and predict Stress disease likelihood.", font=("Arial", 12), bg="#34495e", fg="white")
desc_label.pack(pady=5)

# Model selection dropdown
model_var = tk.StringVar()
model_label = ttk.Label(root, text="Select Model:", font=("Arial", 12), background="#34495e", foreground="white")
model_label.pack(pady=5)
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=list(models.keys()), font=("Arial", 12))
model_dropdown.pack(pady=5)

input_frame = ttk.LabelFrame(root, text="Input Parameters", padding=10)
input_frame.pack(pady=10)

input_labels = df.columns[:-1]
input_entries = []
for i, label in enumerate(input_labels):
    row, col = divmod(i, 6)
    ttk.Label(input_frame, text=label, font=("Arial", 10)).grid(row=row, column=col * 2, padx=5, pady=5)
    entry = ttk.Entry(input_frame, font=("Arial", 10), width=12)
    entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
    input_entries.append(entry)

button_frame = tk.Frame(root, bg="#34495e")
button_frame.pack(pady=10)

def button_style(widget, bg_normal, bg_hover):
    widget.bind("<Enter>", lambda e: e.widget.config(bg=bg_hover))
    widget.bind("<Leave>", lambda e: e.widget.config(bg=bg_normal))

train_button = tk.Button(button_frame, text="Train Models", command=train_models, font=("Arial", 12, "bold"), bg="#27ae60", fg="white", padx=10, pady=5)
train_button.grid(row=0, column=0, padx=10, pady=5)
button_style(train_button, "#27ae60", "#2ecc71")

predict_button = tk.Button(button_frame, text="Predict", command=predict, font=("Arial", 12, "bold"), bg="#3498db", fg="white", padx=10, pady=5)
predict_button.grid(row=0, column=1, padx=10, pady=5)
button_style(predict_button, "#3498db", "#2980b9")

clear_button = tk.Button(button_frame, text="Clear", command=clear_inputs, font=("Arial", 12, "bold"), bg="#e74c3c", fg="white", padx=10, pady=5)
clear_button.grid(row=0, column=2, padx=10, pady=5)
button_style(clear_button, "#e74c3c", "#c0392b")

import_button = tk.Button(button_frame, text="Import CSV & Predict", command=import_csv_and_predict, font=("Arial", 12, "bold"), bg="#f39c12", fg="white", padx=10, pady=5)
import_button.grid(row=0, column=3, padx=10, pady=5)
button_style(import_button, "#f39c12", "#e67e22")

exit_button = tk.Button(button_frame, text="Exit", command=root.quit, font=("Arial", 12, "bold"), bg="#e74c3c", fg="white", padx=10, pady=5)
exit_button.grid(row=0, column=4, padx=10, pady=5)
button_style(exit_button, "#e74c3c", "#c0392b")

root.mainloop()