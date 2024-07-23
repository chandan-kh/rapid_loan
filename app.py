import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import innvestigate
from IPython.core.display import display, HTML
import plotly.graph_objects as go
import sys


# Increase recursion limit
sys.setrecursionlimit(1500)


# Function to preprocess data
def preprocess_data(train_df, test_df):
    loan_ids_test = test_df['loan_id']
    train_df.drop(columns=['loan_id'], inplace=True)
    test_df.drop(columns=['loan_id'], inplace=True)

    train_df.rename(columns=lambda X: X.strip(), inplace=True)
    test_df.rename(columns=lambda X: X.strip(), inplace=True)

    combined_df = pd.concat([train_df, test_df])
    
    label_encoder = LabelEncoder()
    combined_df['education'] = label_encoder.fit_transform(combined_df['education'])
    combined_df['self_employed'] = label_encoder.fit_transform(combined_df['self_employed'])
    combined_df['loan_status'] = label_encoder.fit_transform(combined_df['loan_status'])

    train_df = combined_df[:len(train_df)]
    test_df = combined_df[len(train_df):]

    X_train = train_df.drop(columns=['loan_status'])
    y_train = train_df['loan_status']
    X_test = test_df.drop(columns=['loan_status'])
    y_test = test_df['loan_status']

    feature_names = X_train.columns

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, feature_names

def preprocess_input(input_data):
    # Label encoding for categorical variables
    label_encoders = {}
    categorical_cols = ['education', 'self_employed']

    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        input_data[col] = label_encoders[col].fit_transform(input_data[col])

    # Standardization
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    return input_data


# Function to train and evaluate model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, classification_rep, conf_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    st.pyplot(plt)

# Function to display model metrics
def display_metrics(model_name, accuracy, precision, recall, f1, classification_rep, conf_matrix):
    st.subheader(f"{model_name} Metrics")
    st.write(f"**Accuracy**: {accuracy}")
    st.write(f"**Precision**: {precision}")
    st.write(f"**Recall**: {recall}")
    st.write(f"**F1 Score**: {f1}")
    st.text("**Classification Report**:\n")
    st.text(classification_rep)
    plot_confusion_matrix(conf_matrix, f"{model_name} Confusion Matrix")

tf.compat.v1.disable_eager_execution()



def perform_crp_analysis(model, instance_to_explain, feature_names):
    analyzer = innvestigate.create_analyzer("lrp.epsilon", model)
    analysis = analyzer.analyze(instance_to_explain)
    feature_importances = np.sum(analysis, axis=0)

    # Plot feature relevance using CRP
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances, align='center')
    plt.yticks(range(len(feature_importances)), feature_names)
    plt.xlabel('Relevance Score')
    plt.title('Feature Relevance using CRP')
    st.pyplot(plt)
    return analysis

# Helper functions (defined outside the main function)
def calculate_feature_contributions(activations, analysis, instance_to_explain, feature_names):
    feature_contributions = []
    for i, activation in enumerate(activations):
        layer_contributions = []
        if len(activation.shape) == 2:  # Multiple neurons in this layer
            for j in range(activation.shape[1]):
                neuron_contributions = analysis[0, :] * instance_to_explain[0]
                layer_contributions.append(neuron_contributions)
        else:  # Single neuron (output layer)
            neuron_contributions = analysis[0, :] * instance_to_explain[0]
            layer_contributions.append(neuron_contributions)
        feature_contributions.append(np.array(layer_contributions))
    return feature_contributions

def plot_feature_contributions(feature_contributions, activations, feature_names):
    for i, layer_contributions in enumerate(feature_contributions):
        st.write(f"Layer {i+1} - Neuron Information:")
        neuron_info = pd.DataFrame(layer_contributions, columns=feature_names)
        neuron_info['Neuron Number'] = neuron_info.index
        if len(activations[i].shape) == 2:
            neuron_info['Output Value'] = activations[i].flatten()
        else:
            neuron_info['Output Value'] = activations[i]
        neuron_info = neuron_info.melt(id_vars=['Neuron Number', 'Output Value'], var_name='Feature', value_name='Contribution')
        st.write(neuron_info)

        # Plot heatmap of feature contributions
        plt.figure(figsize=(10, 6))
        heatmap = sns.heatmap(layer_contributions, cmap='viridis', annot=True, fmt='.2f', xticklabels=feature_names)
        plt.title(f'Layer {i+1} - Feature Contributions to Neuron Outputs')
        plt.xlabel('Feature')
        plt.ylabel('Neuron Index')
        st.pyplot(plt)
        plt.clf()  # Clear the figure after displaying it in Streamlit

def construct_decision_tree(layer_contributions, feature_names, activations, layer_idx=0, neuron_idx=0, path=[]):
    if layer_idx >= len(layer_contributions):
        return path

    neuron_contributions = layer_contributions[layer_idx][neuron_idx]
    top_features_idx = np.argsort(np.abs(neuron_contributions))[-4:][::-1]
    top_features = [(feature_names[idx], neuron_contributions[idx]) for idx in top_features_idx]
    path.append((layer_idx + 1, neuron_idx + 1, top_features))

    next_layer_idx = layer_idx + 1
    if next_layer_idx < len(layer_contributions):
        next_neuron_idx = np.argmax(activations[next_layer_idx][0])
        path = construct_decision_tree(layer_contributions, feature_names, activations, next_layer_idx, next_neuron_idx, path)
    return path

def visualize_ann_with_path(model, decision_tree_path, feature_names, activations):
    fig = go.Figure()

    # Spacing factors
    x_spacing = 2  # Increase this to spread out layers horizontally
    y_spacing = 2  # Increase this to spread out neurons vertically

    # Add nodes for input features
    for i, feature in enumerate(feature_names):
        fig.add_trace(go.Scatter(
            x=[0],
            y=[i * y_spacing],
            text=feature,
            mode='markers+text',
            textposition='middle left',
            marker=dict(size=20, color='lightblue')
        ))

    # Add nodes and edges for each layer
    for layer_idx, layer in enumerate(model.layers):
        for neuron_idx in range(layer.output_shape[1]):
            fig.add_trace(go.Scatter(
                x=[(layer_idx + 1) * x_spacing],
                y=[neuron_idx * y_spacing],
                text=f'Neuron {neuron_idx + 1}',
                mode='markers+text',
                textposition='middle right',
                marker=dict(size=20, color='lightgray')
            ))
            if layer_idx == 0:
                for input_idx in range(len(feature_names)):
                    fig.add_trace(go.Scatter(
                        x=[0, x_spacing],
                        y=[input_idx * y_spacing, neuron_idx * y_spacing],
                        mode='lines',
                        line=dict(color='black')
                    ))
            else:
                prev_layer_neuron_count = model.layers[layer_idx - 1].output_shape[1]
                for prev_neuron_idx in range(prev_layer_neuron_count):
                    fig.add_trace(go.Scatter(
                        x=[layer_idx * x_spacing, (layer_idx + 1) * x_spacing],
                        y=[prev_neuron_idx * y_spacing, neuron_idx * y_spacing],
                        mode='lines',
                        line=dict(color='black')
                    ))

    # Add decision path highlights
    for layer_idx, neuron_idx, top_features in decision_tree_path:
        for feature, contribution in top_features:
            if layer_idx == 1:
                feature_idx = list(feature_names).index(feature)
                fig.add_trace(go.Scatter(
                    x=[0, x_spacing],
                    y=[feature_idx * y_spacing, (neuron_idx - 1) * y_spacing],
                    mode='lines',
                    line=dict(color='red', width=2 + abs(contribution))
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[(layer_idx - 1) * x_spacing, layer_idx * x_spacing],
                    y=[np.argmax(activations[layer_idx - 1][0]) * y_spacing, (neuron_idx - 1) * y_spacing],
                    mode='lines',
                    line=dict(color='red', width=2 + abs(contribution))
                ))

    fig.update_layout(title='ANN Decision Path', showlegend=False)
    return fig

def save_shap_plot(shap_values, feature_names, filename):
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_shap_bar_plot(shap_values, instance_idx, filename):
    plt.figure()
    shap.plots.bar(shap_values[instance_idx], show=False)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def main():
    st.title("Loan Approval Prediction and Explanation")

    page = st.sidebar.selectbox("Choose a Page", ["Home", "Traditional Methods", "Neural Networks", "Developer"])

    if page == "Home":
        st.write("Welcome to the Loan Approval Prediction and Explanation App")
        st.write("Choose an option from the sidebar")

    elif page == "Traditional Methods":
        traditional_methods_page()

    elif page == "Neural Networks":
        neural_networks_page()

    elif page == "Developer":
        developer_page()

# Traditional methods page function
def traditional_methods_page():
    st.header("Traditional Methods")
    model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"])

    if st.button("Proceed"):
        st.session_state.show_form = True

    if 'show_form' in st.session_state and st.session_state.show_form:
        traditional_method_form(model_choice)

# Traditional method form function
def traditional_method_form(model_choice):
    # Load pre-defined datasets
    train_df = pd.read_csv('train_loan_approval_dataset.csv')
    test_df = pd.read_csv('test_loan_approval_dataset.csv')
    X_train, y_train, X_test, y_test, feature_names = preprocess_data(train_df, test_df)

    # Initialize the selected model
    if model_choice == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_choice == "XGBoost":
        model = XGBClassifier(random_state=42)


    if st.sidebar.checkbox('Manually Input Instance Details', value=True):
        st.subheader('Enter Instance Details')

        # Form to manually input instance details
        form = st.form(key='input_form')

        # Input fields for each feature
        no_of_dependents = form.number_input('Number of Dependents', min_value=0, step=1)
        education = form.selectbox('Education', ['Graduate', 'Not Graduate'])
        self_employed = form.selectbox('Self Employed', ['Yes', 'No'])
        income_annum = form.number_input('Income Per Annum', min_value=0)
        loan_amount = form.number_input('Loan Amount', min_value=0)
        loan_term = form.number_input('Loan Term', min_value=0, step=1)
        cibil_score = form.number_input('CIBIL Score', min_value=0)
        residential_assets_value = form.number_input('Residential Assets Value', min_value=0)
        commercial_assets_value = form.number_input('Commercial Assets Value', min_value=0)
        luxury_assets_value = form.number_input('Luxury Assets Value', min_value=0)
        bank_asset_value = form.number_input('Bank Asset Value', min_value=0)

        # Submit button to predict
        submit_button = form.form_submit_button(label='Predict')

        if submit_button:
            # Create a DataFrame from user input
            input_data = {
                'no_of_dependents': [no_of_dependents],
                'education': [education],
                'self_employed': [self_employed],
                'income_annum': [income_annum],
                'loan_amount': [loan_amount],
                'loan_term': [loan_term],
                'cibil_score': [cibil_score],
                'residential_assets_value': [residential_assets_value],
                'commercial_assets_value': [commercial_assets_value],
                'luxury_assets_value': [luxury_assets_value],
                'bank_asset_value': [bank_asset_value]
            }
            input_df = pd.DataFrame(input_data)

            # Preprocess the input data
            input_df = preprocess_input(input_df)

            # Convert DataFrame to numpy array (Lime expects numpy array)
            input_np = input_df.values.reshape(1, -1)  # Reshape to fit Lime's expected input format

            explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
            exp = explainer.explain_instance(input_np[0], model.predict_proba, num_features=len(feature_names))
            st.write("LIME Explanation Plot")
            exp.show_in_notebook(show_table=True, show_all=False)
            html = exp.as_html()
            styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
            st.components.v1.html(styled_html, height=800, scrolling=True)

# Neural networks page function
def neural_networks_page():
    st.header("Neural Networks")
    model_choice = st.selectbox("Choose a Model", ["ANN"])

    if st.button("Proceed"):
        st.session_state.show_form = True

    if 'show_form' in st.session_state and st.session_state.show_form:
        neural_networks_form(model_choice)

def neural_networks_form(model):
            
            # Load pre-defined datasets
            train_df = pd.read_csv('train_loan_approval_dataset.csv')
            test_df = pd.read_csv('test_loan_approval_dataset.csv')
            X_train, y_train, X_test, y_test, feature_names = preprocess_data(train_df, test_df)
            
            model = Sequential()
            model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dense(15, activation='relu'))
            model.add(Dense(5, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
            
            
            st.sidebar.header("LIME, CRP/LRP, and Integrated Gradients Explanations")

            if st.sidebar.checkbox('Manually Input Instance Details', value=True):
                st.subheader('Enter Instance Details')

                # Form to manually input instance details
                form = st.form(key='input_form')
                
                # Input fields for each feature
                no_of_dependents = form.number_input('Number of Dependents', min_value=0, step=1)
                education = form.selectbox('Education', ['Graduate', 'Not Graduate'])
                self_employed = form.selectbox('Self Employed', ['Yes', 'No'])
                income_annum = form.number_input('Income Per Annum', min_value=0)
                loan_amount = form.number_input('Loan Amount', min_value=0)
                loan_term = form.number_input('Loan Term', min_value=0, step=1)
                cibil_score = form.number_input('CIBIL Score', min_value=0)
                residential_assets_value = form.number_input('Residential Assets Value', min_value=0)
                commercial_assets_value = form.number_input('Commercial Assets Value', min_value=0)
                luxury_assets_value = form.number_input('Luxury Assets Value', min_value=0)
                bank_asset_value = form.number_input('Bank Asset Value', min_value=0)

                # Submit button to predict
                submit_button = form.form_submit_button(label='Predict')

                if submit_button:
                    # Create a DataFrame from user input
                    input_data = {
                        'no_of_dependents': [no_of_dependents],
                        'education': [education],
                        'self_employed': [self_employed],
                        'income_annum': [income_annum],
                        'loan_amount': [loan_amount],
                        'loan_term': [loan_term],
                        'cibil_score': [cibil_score],
                        'residential_assets_value': [residential_assets_value],
                        'commercial_assets_value': [commercial_assets_value],
                        'luxury_assets_value': [luxury_assets_value],
                        'bank_asset_value': [bank_asset_value]
                    }
                    input_df = pd.DataFrame(input_data)

                    # Preprocess the input data
                    input_df = preprocess_input(input_df)

                    # Convert DataFrame to numpy array (Lime expects numpy array)
                    input_np = input_df.reshape(1, -1)  # Reshape to fit Lime's expected input format

                    instance = input_np

                    # Define a simplified prediction function for debugging
                    def predict_fn(instance):
                        prediction = model.predict(instance).flatten()
                        return np.vstack([1 - prediction, prediction]).T
                    
                    st.write("LIME Explanation Plot")
                    # LIME Explanation
                    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
                    # Explain the prediction of the chosen instance
                    exp = explainer.explain_instance(instance[0], predict_fn, num_features=10)

                    # Display LIME explanation as HTML in Streamlit
                    html = exp.as_html()
                    styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
                    st.components.v1.html(styled_html, height=800, scrolling=True)
                    
                    # CRP/LRP Explanation - Replace with your specific CRP/LRP method and code
                    
                    instance_to_explain=instance
                    st.write("CRP Explanation Plot")
                    analysis = perform_crp_analysis(model, instance_to_explain, feature_names)

        
                    # Capture neuron outputs at each layer
                    layer_outputs = [layer.output for layer in model.layers]
                    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
                    activations = activation_model.predict(instance_to_explain)

                    # Calculate feature contributions
                    feature_contributions = calculate_feature_contributions(activations, analysis, instance_to_explain, feature_names)

                    # Plot feature contributions
                    plot_feature_contributions(feature_contributions, activations, feature_names)

                    

                    # Construct the decision tree
                    decision_tree_path = construct_decision_tree(feature_contributions, feature_names, activations)
                    # st.write("Decision Tree Path:\n", decision_tree_path)

                    fig = visualize_ann_with_path(model, decision_tree_path, feature_names,activations)
                    st.plotly_chart(fig)

                    # Predict the loan status for the instance
                    predicted_loan_status = (model.predict(instance_to_explain) > 0.5).astype("int32")

                    # Display the prediction result
                    if predicted_loan_status == 1:
                        st.write("\nThe loan is rejected.")
                    else:
                        st.write("\nThe loan is approved.")

                # Example using Integrated Gradients
                # with tf.GradientTape() as tape:
                #     tape.watch(instance)
                #     prediction = model(instance)
                # grads = tape.gradient(prediction, instance)
                # integrated_grads = np.sum(grads, axis=0)
                # plt.figure(figsize=(10, 6))
                # plt.barh(range(len(integrated_grads)), integrated_grads, align='center')
                # plt.yticks(range(len(integrated_grads)), feature_names)
                # plt.xlabel('Integrated Gradients')
                # plt.title('Integrated Gradients Explanation')
                # st.pyplot(plt)


# Developer page function
def developer_page():

    st.header("Test For Your Custom Datasets")
    
    st.sidebar.header("Upload Dataset")
    train_file = st.sidebar.file_uploader("Upload the training dataset", type=["csv"])
    test_file = st.sidebar.file_uploader("Upload the test dataset", type=["csv"])
    
    if train_file and test_file:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        X_train, y_train, X_test, y_test, feature_names = preprocess_data(train_df, test_df)
        
        st.sidebar.header("Choose Model")
        model_option = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "ANN"])
        
        if model_option == "Logistic Regression":
            model = LogisticRegression(random_state=42)
            accuracy, precision, recall, f1, classification_rep, conf_matrix = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
            display_metrics("Logistic Regression", accuracy, precision, recall, f1, classification_rep, conf_matrix)
        
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
            accuracy, precision, recall, f1, classification_rep, conf_matrix = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
            display_metrics("Decision Tree", accuracy, precision, recall, f1, classification_rep, conf_matrix)
            
            st.sidebar.header("LIME and SHAP Explanations")
            explain_instance = st.sidebar.checkbox("Explain a specific instance")
            if explain_instance:
                instance_idx = st.sidebar.number_input("Choose an instance index", min_value=0, max_value=len(X_test)-1, value=0)
                instance = X_test[instance_idx]

                st.write(instance)

                explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
                exp = explainer.explain_instance(instance, model.predict_proba, num_features=len(feature_names))
                st.write("LIME Explanation Plot")
                exp.show_in_notebook(show_table=True, show_all=False)
                html = exp.as_html()
                styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
                st.components.v1.html(styled_html, height=800, scrolling=True)

            if st.sidebar.checkbox('Manually Input Instance Details'):
                st.subheader('Enter Instance Details')

                # Form to manually input instance details
                form = st.form(key='input_form')
                
                # Input fields for each feature
                no_of_dependents = form.number_input('Number of Dependents', min_value=0, step=1)
                education = form.selectbox('Education', ['Graduate', 'Not Graduate'])
                self_employed = form.selectbox('Self Employed', ['Yes', 'No'])
                income_annum = form.number_input('Income Per Annum', min_value=0)
                loan_amount = form.number_input('Loan Amount', min_value=0)
                loan_term = form.number_input('Loan Term', min_value=0, step=1)
                cibil_score = form.number_input('CIBIL Score', min_value=0)
                residential_assets_value = form.number_input('Residential Assets Value', min_value=0)
                commercial_assets_value = form.number_input('Commercial Assets Value', min_value=0)
                luxury_assets_value = form.number_input('Luxury Assets Value', min_value=0)
                bank_asset_value = form.number_input('Bank Asset Value', min_value=0)

                # Submit button to predict
                submit_button = form.form_submit_button(label='Predict')

                if submit_button:
                    # Create a DataFrame from user input
                    input_data = {
                        'no_of_dependents': [no_of_dependents],
                        'education': [education],
                        'self_employed': [self_employed],
                        'income_annum': [income_annum],
                        'loan_amount': [loan_amount],
                        'loan_term': [loan_term],
                        'cibil_score': [cibil_score],
                        'residential_assets_value': [residential_assets_value],
                        'commercial_assets_value': [commercial_assets_value],
                        'luxury_assets_value': [luxury_assets_value],
                        'bank_asset_value': [bank_asset_value]
                    }
                    input_df = pd.DataFrame(input_data)

                    # Preprocess the input data
                    input_df = preprocess_input(input_df)

                    # Convert DataFrame to numpy array (Lime expects numpy array)
                    input_np = input_df.reshape(1, -1)  # Reshape to fit Lime's expected input format

                    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
                    exp = explainer.explain_instance(input_np[0], model.predict_proba, num_features=len(feature_names))
                    st.write("LIME Explanation Plot")
                    exp.show_in_notebook(show_table=True, show_all=False)
                    html = exp.as_html()
                    styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
                    st.components.v1.html(styled_html, height=800, scrolling=True)




                # SHAP explanation
                # shap_explainer = shap.Explainer(model.predict, X_test, feature_names=feature_names)
                # shap_values = shap_explainer(X_test)
                
                # # Save SHAP summary plot as image and display it
                # shap_summary_filename = "shap_summary_plot.png"
                # save_shap_plot(shap_values, feature_names, shap_summary_filename)
                # st.write("SHAP Summary Plot")
                # st.image(shap_summary_filename)

                # # Save SHAP explanation for specific instance as image and display it
                # shap_instance_filename = f"shap_instance_plot_{instance_idx}.png"
                # save_shap_bar_plot(shap_values, instance_idx, shap_instance_filename)
                # st.write(f"SHAP Explanation for Instance {instance_idx}")
                # st.image(shap_instance_filename)

        
        elif model_option == "Random Forest":
            model = RandomForestClassifier(random_state=42)
            accuracy, precision, recall, f1, classification_rep, conf_matrix = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
            display_metrics("Random Forest", accuracy, precision, recall, f1, classification_rep, conf_matrix)
            
            st.sidebar.header("LIME and SHAP Explanations")
            explain_instance = st.sidebar.checkbox("Explain a specific instance")
            if explain_instance:


                instance_idx = st.sidebar.number_input("Choose an instance index", min_value=0, max_value=len(X_test)-1, value=0)
                instance = X_test[instance_idx]

                explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
                exp = explainer.explain_instance(instance, model.predict_proba, num_features=len(feature_names))
                st.write("LIME Explanation Plot")
                exp.show_in_notebook(show_table=True, show_all=False)
                html = exp.as_html()
                styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
                st.components.v1.html(styled_html, height=800, scrolling=True)
            
            if st.sidebar.checkbox('Manually Input Instance Details'):
                st.subheader('Enter Instance Details')

                # Form to manually input instance details
                form = st.form(key='input_form')
                
                # Input fields for each feature
                no_of_dependents = form.number_input('Number of Dependents', min_value=0, step=1)
                education = form.selectbox('Education', ['Graduate', 'Not Graduate'])
                self_employed = form.selectbox('Self Employed', ['Yes', 'No'])
                income_annum = form.number_input('Income Per Annum', min_value=0)
                loan_amount = form.number_input('Loan Amount', min_value=0)
                loan_term = form.number_input('Loan Term', min_value=0, step=1)
                cibil_score = form.number_input('CIBIL Score', min_value=0)
                residential_assets_value = form.number_input('Residential Assets Value', min_value=0)
                commercial_assets_value = form.number_input('Commercial Assets Value', min_value=0)
                luxury_assets_value = form.number_input('Luxury Assets Value', min_value=0)
                bank_asset_value = form.number_input('Bank Asset Value', min_value=0)

                # Submit button to predict
                submit_button = form.form_submit_button(label='Predict')

                if submit_button:
                    # Create a DataFrame from user input
                    input_data = {
                        'no_of_dependents': [no_of_dependents],
                        'education': [education],
                        'self_employed': [self_employed],
                        'income_annum': [income_annum],
                        'loan_amount': [loan_amount],
                        'loan_term': [loan_term],
                        'cibil_score': [cibil_score],
                        'residential_assets_value': [residential_assets_value],
                        'commercial_assets_value': [commercial_assets_value],
                        'luxury_assets_value': [luxury_assets_value],
                        'bank_asset_value': [bank_asset_value]
                    }
                    input_df = pd.DataFrame(input_data)

                    # Preprocess the input data
                    input_df = preprocess_input(input_df)

                    # Convert DataFrame to numpy array (Lime expects numpy array)
                    input_np = input_df.reshape(1, -1)  # Reshape to fit Lime's expected input format

                    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
                    exp = explainer.explain_instance(input_np[0], model.predict_proba, num_features=len(feature_names))
                    st.write("LIME Explanation Plot")
                    exp.show_in_notebook(show_table=True, show_all=False)
                    html = exp.as_html()
                    styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
                    st.components.v1.html(styled_html, height=800, scrolling=True)

                # SHAP explanation
                # shap_explainer = shap.Explainer(model.predict, X_test, feature_names=feature_names)
                # shap_values = shap_explainer(X_test)

                # # Save SHAP summary plot as image and display it
                # shap_summary_filename = "shap_summary_plot.png"
                # save_shap_plot(shap_values, feature_names, shap_summary_filename)
                # st.write("SHAP Summary Plot")
                # st.image(shap_summary_filename)

                # # Save SHAP explanation for specific instance as image and display it
                # shap_instance_filename = f"shap_instance_plot_{instance_idx}.png"
                # save_shap_bar_plot(shap_values, instance_idx, shap_instance_filename)
                # st.write(f"SHAP Explanation for Instance {instance_idx}")
                # st.image(shap_instance_filename)

        
        elif model_option == "XGBoost":
            model = XGBClassifier(random_state=42)
            accuracy, precision, recall, f1, classification_rep, conf_matrix = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
            display_metrics("XG Boost", accuracy, precision, recall, f1, classification_rep, conf_matrix)
            
            st.sidebar.header("LIME and SHAP Explanations")
            explain_instance = st.sidebar.checkbox("Explain a specific instance")
            if explain_instance:
                instance_idx = st.sidebar.number_input("Choose an instance index", min_value=0, max_value=len(X_test)-1, value=0)
                instance = X_test[instance_idx]

                explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
                exp = explainer.explain_instance(instance, model.predict_proba, num_features=len(feature_names))
                st.write("LIME Explanation Plot")
                exp.show_in_notebook(show_table=True, show_all=False)
                html = exp.as_html()
                styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
                st.components.v1.html(styled_html, height=800, scrolling=True)
            
            if st.sidebar.checkbox('Manually Input Instance Details'):
                st.subheader('Enter Instance Details')

                # Form to manually input instance details
                form = st.form(key='input_form')
                
                # Input fields for each feature
                no_of_dependents = form.number_input('Number of Dependents', min_value=0, step=1)
                education = form.selectbox('Education', ['Graduate', 'Not Graduate'])
                self_employed = form.selectbox('Self Employed', ['Yes', 'No'])
                income_annum = form.number_input('Income Per Annum', min_value=0)
                loan_amount = form.number_input('Loan Amount', min_value=0)
                loan_term = form.number_input('Loan Term', min_value=0, step=1)
                cibil_score = form.number_input('CIBIL Score', min_value=0)
                residential_assets_value = form.number_input('Residential Assets Value', min_value=0)
                commercial_assets_value = form.number_input('Commercial Assets Value', min_value=0)
                luxury_assets_value = form.number_input('Luxury Assets Value', min_value=0)
                bank_asset_value = form.number_input('Bank Asset Value', min_value=0)

                # Submit button to predict
                submit_button = form.form_submit_button(label='Predict')

                if submit_button:
                    # Create a DataFrame from user input
                    input_data = {
                        'no_of_dependents': [no_of_dependents],
                        'education': [education],
                        'self_employed': [self_employed],
                        'income_annum': [income_annum],
                        'loan_amount': [loan_amount],
                        'loan_term': [loan_term],
                        'cibil_score': [cibil_score],
                        'residential_assets_value': [residential_assets_value],
                        'commercial_assets_value': [commercial_assets_value],
                        'luxury_assets_value': [luxury_assets_value],
                        'bank_asset_value': [bank_asset_value]
                    }
                    input_df = pd.DataFrame(input_data)

                    # Preprocess the input data
                    input_df = preprocess_input(input_df)

                    # Convert DataFrame to numpy array (Lime expects numpy array)
                    input_np = input_df.reshape(1, -1)  # Reshape to fit Lime's expected input format

                    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
                    exp = explainer.explain_instance(input_np[0], model.predict_proba, num_features=len(feature_names))
                    st.write("LIME Explanation Plot")
                    exp.show_in_notebook(show_table=True, show_all=False)
                    html = exp.as_html()
                    styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
                    st.components.v1.html(styled_html, height=800, scrolling=True)

                # SHAP explanation
                # shap_explainer = shap.Explainer(model, X_test, feature_names=feature_names)
                # shap_values = shap_explainer(X_test)
                
                # # Save SHAP summary plot as image and display it
                # shap_summary_filename = "shap_summary_plot.png"
                # save_shap_plot(shap_values, feature_names, shap_summary_filename)
                # st.write("SHAP Summary Plot")
                # st.image(shap_summary_filename)

                # # Save SHAP explanation for specific instance as image and display it
                # shap_instance_filename = f"shap_instance_plot_{instance_idx}.png"
                # save_shap_bar_plot(shap_values, instance_idx, shap_instance_filename)
                # st.write(f"SHAP Explanation for Instance {instance_idx}")
                # st.image(shap_instance_filename)

        
        elif model_option == "ANN":
            # Build and train the ANN model
            model = Sequential()
            model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dense(15, activation='relu'))
            model.add(Dense(5, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
            
            # Evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test)
            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            display_metrics("ANN", accuracy, precision, recall, f1, classification_rep, conf_matrix)
            
            st.sidebar.header("LIME, CRP/LRP, and Integrated Gradients Explanations")
            explain_instance = st.sidebar.checkbox("Explain a specific instance")
            if explain_instance:
                instance_idx = st.sidebar.number_input("Choose an instance index", min_value=0, max_value=len(X_test)-1, value=0)
                instance = X_test[instance_idx].reshape(1, -1)

                # Define a simplified prediction function for debugging
                def predict_fn(instance):
                    prediction = model.predict(instance).flatten()
                    return np.vstack([1 - prediction, prediction]).T
                
                st.write("LIME Explanation Plot")
                # LIME Explanation
                explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
                # Explain the prediction of the chosen instance
                exp = explainer.explain_instance(instance[0], predict_fn, num_features=10)

                # Display LIME explanation as HTML in Streamlit
                html = exp.as_html()
                styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
                st.components.v1.html(styled_html, height=800, scrolling=True)
                
                # CRP/LRP Explanation - Replace with your specific CRP/LRP method and code
                
                instance_to_explain=instance
                st.write("CRP Explanation Plot")
                analysis = perform_crp_analysis(model, instance_to_explain, feature_names)

    
                # Capture neuron outputs at each layer
                layer_outputs = [layer.output for layer in model.layers]
                activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
                activations = activation_model.predict(instance_to_explain)

                # Calculate feature contributions
                feature_contributions = calculate_feature_contributions(activations, analysis, instance_to_explain, feature_names)

                # Plot feature contributions
                plot_feature_contributions(feature_contributions, activations, feature_names)

                

                # Construct the decision tree
                decision_tree_path = construct_decision_tree(feature_contributions, feature_names, activations)
                # st.write("Decision Tree Path:\n", decision_tree_path)

                fig = visualize_ann_with_path(model, decision_tree_path, feature_names,activations)
                st.plotly_chart(fig)

                # Predict the loan status for the instance
                predicted_loan_status = (model.predict(instance_to_explain) > 0.5).astype("int32")

                # Display the prediction result
                if predicted_loan_status == 1:
                    st.write("\nThe loan is rejected.")
                else:
                    st.write("\nThe loan is approved.")

            if st.sidebar.checkbox('Manually Input Instance Details'):
                st.subheader('Enter Instance Details')

                # Form to manually input instance details
                form = st.form(key='input_form')
                
                # Input fields for each feature
                no_of_dependents = form.number_input('Number of Dependents', min_value=0, step=1)
                education = form.selectbox('Education', ['Graduate', 'Not Graduate'])
                self_employed = form.selectbox('Self Employed', ['Yes', 'No'])
                income_annum = form.number_input('Income Per Annum', min_value=0)
                loan_amount = form.number_input('Loan Amount', min_value=0)
                loan_term = form.number_input('Loan Term', min_value=0, step=1)
                cibil_score = form.number_input('CIBIL Score', min_value=0)
                residential_assets_value = form.number_input('Residential Assets Value', min_value=0)
                commercial_assets_value = form.number_input('Commercial Assets Value', min_value=0)
                luxury_assets_value = form.number_input('Luxury Assets Value', min_value=0)
                bank_asset_value = form.number_input('Bank Asset Value', min_value=0)

                # Submit button to predict
                submit_button = form.form_submit_button(label='Predict')

                if submit_button:
                    # Create a DataFrame from user input
                    input_data = {
                        'no_of_dependents': [no_of_dependents],
                        'education': [education],
                        'self_employed': [self_employed],
                        'income_annum': [income_annum],
                        'loan_amount': [loan_amount],
                        'loan_term': [loan_term],
                        'cibil_score': [cibil_score],
                        'residential_assets_value': [residential_assets_value],
                        'commercial_assets_value': [commercial_assets_value],
                        'luxury_assets_value': [luxury_assets_value],
                        'bank_asset_value': [bank_asset_value]
                    }
                    input_df = pd.DataFrame(input_data)

                    # Preprocess the input data
                    input_df = preprocess_input(input_df)

                    # Convert DataFrame to numpy array (Lime expects numpy array)
                    input_np = input_df.reshape(1, -1)  # Reshape to fit Lime's expected input format

                    instance = input_np

                    # Define a simplified prediction function for debugging
                    def predict_fn(instance):
                        prediction = model.predict(instance).flatten()
                        return np.vstack([1 - prediction, prediction]).T
                    
                    st.write("LIME Explanation Plot")
                    # LIME Explanation
                    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=feature_names, class_names=['Approved', 'Rejected'])
                    # Explain the prediction of the chosen instance
                    exp = explainer.explain_instance(instance[0], predict_fn, num_features=10)

                    # Display LIME explanation as HTML in Streamlit
                    html = exp.as_html()
                    styled_html = f'<div style="background-color: white; padding: 10px">{html}</div>'
                    st.components.v1.html(styled_html, height=800, scrolling=True)
                    
                    # CRP/LRP Explanation - Replace with your specific CRP/LRP method and code
                    
                    instance_to_explain=instance
                    st.write("CRP Explanation Plot")
                    analysis = perform_crp_analysis(model, instance_to_explain, feature_names)

        
                    # Capture neuron outputs at each layer
                    layer_outputs = [layer.output for layer in model.layers]
                    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
                    activations = activation_model.predict(instance_to_explain)

                    # Calculate feature contributions
                    feature_contributions = calculate_feature_contributions(activations, analysis, instance_to_explain, feature_names)

                    # Plot feature contributions
                    plot_feature_contributions(feature_contributions, activations, feature_names)

                    

                    # Construct the decision tree
                    decision_tree_path = construct_decision_tree(feature_contributions, feature_names, activations)
                    # st.write("Decision Tree Path:\n", decision_tree_path)

                    fig = visualize_ann_with_path(model, decision_tree_path, feature_names,activations)
                    st.plotly_chart(fig)

                    # Predict the loan status for the instance
                    predicted_loan_status = (model.predict(instance_to_explain) > 0.5).astype("int32")

                    # Display the prediction result
                    if predicted_loan_status == 1:
                        st.write("\nThe loan is rejected.")
                    else:
                        st.write("\nThe loan is approved.")

                # Example using Integrated Gradients
                # with tf.GradientTape() as tape:
                #     tape.watch(instance)
                #     prediction = model(instance)
                # grads = tape.gradient(prediction, instance)
                # integrated_grads = np.sum(grads, axis=0)
                # plt.figure(figsize=(10, 6))
                # plt.barh(range(len(integrated_grads)), integrated_grads, align='center')
                # plt.yticks(range(len(integrated_grads)), feature_names)
                # plt.xlabel('Integrated Gradients')
                # plt.title('Integrated Gradients Explanation')
                # st.pyplot(plt)


if __name__ == "__main__":
    main()