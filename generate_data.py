"""
Natural Data Generator for Student Performance Predictor
=========================================================
Generates 10,000+ realistic student records with correlated features.
Data is designed to produce natural pass/fail outcomes based on real-world patterns.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_STUDENTS = 55000  # Large dataset for better model training
PASS_THRESHOLD = 0.40  # 40% threshold as specified by user

def generate_student_data():
    """
    Generate natural-looking student performance data.
    Features are correlated to create realistic patterns:
    - Higher study hours → better performance
    - Higher attendance → better performance
    - Higher previous scores → better performance
    - Moderate internet usage can help (research) but excessive hurts
    - Social activity has mixed effects
    - Sleep is crucial - too little or too much affects performance
    """

    print("Generating natural student data...")

    # Initialize arrays
    study_hours = np.zeros(NUM_STUDENTS)
    sleep_hours = np.zeros(NUM_STUDENTS)
    attendance_percentage = np.zeros(NUM_STUDENTS)
    previous_score = np.zeros(NUM_STUDENTS)
    internet_usage = np.zeros(NUM_STUDENTS)
    social_activity_level = np.zeros(NUM_STUDENTS)

    # Generate correlated data with realistic distributions

    # Study hours: Most students study 1-6 hours, some study more, few study very little
    # Distribution: Skewed towards lower values with long tail
    study_hours = np.random.lognormal(mean=1.2, sigma=0.6, size=NUM_STUDENTS)
    study_hours = np.clip(study_hours, 0.5, 16)

    # Sleep hours: Most students sleep 5-9 hours, follows normal distribution
    sleep_hours = np.random.normal(loc=7, scale=1.5, size=NUM_STUDENTS)
    sleep_hours = np.clip(sleep_hours, 3, 12)

    # Attendance: Follows a beta distribution - most attend regularly, some skip often
    attendance_percentage = np.random.beta(a=5, b=1.5, size=NUM_STUDENTS) * 100
    attendance_percentage = np.clip(attendance_percentage, 30, 100)

    # Previous score: Correlated with study habits, with realistic distribution
    # Students who study more tend to have better previous scores
    base_previous = 50 + (study_hours / 10) * 30 + np.random.normal(0, 15, NUM_STUDENTS)
    previous_score = np.clip(base_previous, 15, 100)

    # Internet usage: Students use internet 0-10 hours
    # Some use for study (beneficial), some for entertainment (neutral to harmful)
    internet_usage = np.random.exponential(scale=2.5, size=NUM_STUDENTS)
    internet_usage = np.clip(internet_usage, 0, 12)

    # Social activity: 1-5 scale
    social_activity_level = np.random.randint(1, 6, size=NUM_STUDENTS)

    # Now calculate the performance score based on weighted features
    # This creates natural correlations between features and outcomes

    performance_scores = np.zeros(NUM_STUDENTS)
    pass_fail = np.zeros(NUM_STUDENTS, dtype=int)

    for i in range(NUM_STUDENTS):
        # Calculate score based on multiple factors with weights
        # Adjusted to create a more balanced pass/fail distribution (~60-65% pass rate)

        # Study hours contribution (25%) - important but not everything
        study_score = (study_hours[i] / 10) * 100

        # Attendance contribution (20%) - important
        attendance_score = attendance_percentage[i] * 0.7

        # Previous score contribution (30%) - strong predictor
        previous_contribution = previous_score[i] * 0.75

        # Sleep contribution (15%) - crucial for performance
        sleep_diff = abs(sleep_hours[i] - 7.5)
        if sleep_diff <= 0.5:
            sleep_score = 100
        elif sleep_diff <= 1:
            sleep_score = 90
        elif sleep_diff <= 2:
            sleep_score = 75
        else:
            sleep_score = max(40, 100 - (sleep_diff * 20))

        # Internet usage (5%) - can help or hurt
        if internet_usage[i] <= 2:
            internet_score = 70
        elif internet_usage[i] <= 5:
            internet_score = 80
        else:
            internet_score = max(50, 90 - (internet_usage[i] - 5) * 8)

        # Social activity (5%) - balanced impact
        if social_activity_level[i] == 2 or social_activity_level[i] == 3:
            social_score = 85
        elif social_activity_level[i] == 1:
            social_score = 75
        else:
            social_score = max(60, 90 - (social_activity_level[i] - 3) * 12)

        # Add random noise
        noise = np.random.normal(0, 8)

        # Weighted final score (adjusted for more balanced pass/fail)
        final_score = (
            study_score * 0.25 +
            attendance_score * 0.20 +
            previous_contribution * 0.30 +
            sleep_score * 0.15 +
            internet_score * 0.05 +
            social_score * 0.05 +
            noise
        )

        # Apply penalties for poor habits
        if study_hours[i] < 2:
            final_score -= random.uniform(8, 15)
        if attendance_percentage[i] < 50:
            final_score -= random.uniform(10, 20)
        if sleep_hours[i] < 5:
            final_score -= random.uniform(10, 18)
        if previous_score[i] < 30:
            final_score -= random.uniform(5, 12)

        # Apply bonuses for good habits
        if study_hours[i] > 6 and attendance_percentage[i] > 80:
            final_score += random.uniform(3, 8)
        if previous_score[i] > 75:
            final_score += random.uniform(2, 6)

        performance_scores[i] = np.clip(final_score, 0, 100)

        # Calculate pass/fail based on threshold (40%)
        pass_fail[i] = 1 if performance_scores[i] >= 40 else 0

    # Create DataFrame
    data = {
        'Study_Hours': np.round(study_hours, 2),
        'Sleep_Hours': np.round(sleep_hours, 2),
        'Attendance_Percentage': np.round(attendance_percentage, 2),
        'Previous_Score': np.round(previous_score, 2),
        'Internet_Usage': np.round(internet_usage, 2),
        'Social_Activity_Level': social_activity_level.astype(int),
        'Performance_Score': np.round(performance_scores, 2),
        'Pass_Fail': pass_fail
    }

    df = pd.DataFrame(data)

    # Add more edge cases to make data more realistic (scaled for larger dataset)
    edge_cases = pd.DataFrame({
        'Study_Hours': [0.5, 0.8, 1.0, 10.0, 12.0, 14.0, 15.0, 0.5, 1.5, 8.0, 9.0, 10.0],
        'Sleep_Hours': [3.0, 4.0, 10.0, 11.0, 12.0, 3.5, 4.0, 2.5, 3.0, 7.0, 8.0, 6.5],
        'Attendance_Percentage': [30, 35, 95, 98, 100, 40, 45, 25, 32, 85, 90, 95],
        'Previous_Score': [20, 25, 90, 95, 98, 30, 35, 15, 22, 70, 80, 90],
        'Internet_Usage': [0, 0.5, 10.0, 11.0, 12.0, 8.0, 9.0, 0, 1, 5, 4, 3],
        'Social_Activity_Level': [1, 5, 1, 2, 3, 5, 4, 1, 5, 2, 3, 2],
        'Performance_Score': [15, 20, 88, 92, 95, 25, 30, 10, 18, 75, 82, 90],
        'Pass_Fail': [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    })

    df = pd.concat([df, edge_cases], ignore_index=True)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Add ID column
    df.insert(0, 'id', range(1, len(df) + 1))

    return df


def save_data(df, output_dir='data'):
    """Save generated data to CSV files"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save full dataset
    full_path = os.path.join(output_dir, 'student_data_full.csv')
    df.to_csv(full_path, index=False)
    print(f"Full dataset saved to: {full_path}")
    print(f"Total records: {len(df)}")

    # Split into train (80%) and test (20%)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Training data saved to: {train_path} ({len(train_df)} records)")
    print(f"Testing data saved to: {test_path} ({len(test_df)} records)")

    # Print statistics
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print(f"\nPass Rate: {df['Pass_Fail'].mean()*100:.1f}%")
    print(f"\nFeature Statistics:")
    print(df.describe().round(2))

    return train_df, test_df


def export_for_mongodb(df, output_dir='data'):
    """Export data in format suitable for MongoDB"""

    # Remove ID for MongoDB (MongoDB adds _id automatically)
    df_mongodb = df.drop('id', axis=1)

    # Convert to records (JSON-like format)
    records = df_mongodb.to_dict(orient='records')

    # Save as JSON
    json_path = os.path.join(output_dir, 'student_data.json')
    import json
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"MongoDB format saved to: {json_path}")


if __name__ == "__main__":
    print("="*60)
    print("STUDENT PERFORMANCE DATA GENERATOR")
    print("="*60)
    print(f"\nGenerating {NUM_STUDENTS} student records...")
    print(f"Pass threshold: {PASS_THRESHOLD*100}%\n")

    # Generate data
    df = generate_student_data()

    # Save data
    train_df, test_df = save_data(df)

    # Export for MongoDB
    export_for_mongodb(df)

    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE!")
    print("="*60)