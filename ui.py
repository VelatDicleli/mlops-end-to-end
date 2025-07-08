import gradio as gr
import requests
import pandas as pd
from typing import Literal
from src.exception import exception_handler_decorator


API_URL = "http://localhost:8000/predict"  


@exception_handler_decorator
def predict_interface(
    hours_studied: float,
    attendance: float,
    sleep_hours: float,
    previous_scores: float,
    tutoring_sessions: float,
    family_income: float,
    teacher_quality: float,
    peer_influence: float,
    physical_activity: float,
    parental_involvement: Literal["Low", "Medium", "High"],
    access_to_resources: Literal["Low", "Medium", "High"],
    extracurricular_activities: Literal["Yes", "No"],
    motivation_level: Literal["Low", "Medium", "High"],
    internet_access: Literal["Yes", "No"],
    school_type: Literal["Public", "Private"],
    learning_disabilities: Literal["Yes", "No"],
    parental_education_level: Literal["High School", "College", "Postgraduate"],
    distance_from_home: Literal["Near", "Moderate", "Far"],
    gender: Literal["Male", "Female", "Other"]
):
    
    input_data = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular_activities,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "School_Type": school_type,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education_level,
        "Distance_from_Home": distance_from_home,
        "Gender": gender
    }
    
    
       
    response = requests.post(API_URL, json=input_data)
    response.raise_for_status()
    
    
    prediction = response.json()["prediction"]
    
    
    if isinstance(prediction, list):
        if len(prediction) == 1:
            return f"Predicted Score: {prediction[0]:.2f}"
        return f"Predictions: {prediction}"
    return f"Prediction: {prediction}"



with gr.Blocks(title="Student Performance Predictor") as demo:
    gr.Markdown("# 🎓 Student Performance Prediction")
    gr.Markdown("Enter student details to predict academic performance")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Numeric Features")
            hours_studied = gr.Slider(0, 24, value=2, label="Hours Studied per day")
            attendance = gr.Slider(0, 100, value=90, label="Attendance Percentage")
            sleep_hours = gr.Slider(0, 12, value=8, label="Sleep Hours per night")
            previous_scores = gr.Slider(0, 100, value=75, label="Previous Test Scores")
            tutoring_sessions = gr.Slider(0, 10, value=2, label="Tutoring Sessions per month")
            family_income = gr.Slider(0, 200000, step=1000, value=50000, label="Family Income ($)")
            teacher_quality = gr.Slider(1, 10, value=7, label="Teacher Quality Rating (1-10)")
            peer_influence = gr.Slider(1, 10, value=5, label="Peer Influence Rating (1-10)")
            physical_activity = gr.Slider(0, 20, value=5, label="Physical Activity Hours per week")
        
        with gr.Column():
            gr.Markdown("### 📝 Categorical Features")
            parental_involvement = gr.Radio(["Low", "Medium", "High"], value="Medium", label="Parental Involvement")
            access_to_resources = gr.Radio(["Low", "Medium", "High"], value="Medium", label="Access to Resources")
            extracurricular_activities = gr.Radio(["Yes", "No"], value="Yes", label="Extracurricular Activities")
            motivation_level = gr.Radio(["Low", "Medium", "High"], value="Medium", label="Motivation Level")
            internet_access = gr.Radio(["Yes", "No"], value="Yes", label="Internet Access")
            school_type = gr.Radio(["Public", "Private"], value="Public", label="School Type")
            learning_disabilities = gr.Radio(["Yes", "No"], value="No", label="Learning Disabilities")
            parental_education_level = gr.Radio(["High School", "College", "Postgraduate"], value="College", label="Parental Education Level")
            distance_from_home = gr.Radio(["Near", "Moderate", "Far"], value="Moderate", label="Distance from Home")
            gender = gr.Radio(["Male", "Female", "Other"], value="Male", label="Gender")
    
    submit_btn = gr.Button("Predict Performance", variant="primary")
    
    output = gr.Textbox(label="Prediction Result")
    
    submit_btn.click(
        fn=predict_interface,
        inputs=[
            hours_studied,
            attendance,
            sleep_hours,
            previous_scores,
            tutoring_sessions,
            family_income,
            teacher_quality,
            peer_influence,
            physical_activity,
            parental_involvement,
            access_to_resources,
            extracurricular_activities,
            motivation_level,
            internet_access,
            school_type,
            learning_disabilities,
            parental_education_level,
            distance_from_home,
            gender
        ],
        outputs=output
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0")