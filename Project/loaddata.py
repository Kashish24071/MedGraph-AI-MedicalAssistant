import pandas as pd

# Load all datasets
symptoms_df = pd.read_csv("C:/Users/Kashish U Singh/Downloads/archive (2)/Diseases_and_Symptoms_dataset.csv")
description_df = pd.read_csv("C:/Users/Kashish U Singh/Downloads/archive (2)/description.csv")
medications_df = pd.read_csv("C:/Users/Kashish U Singh/Downloads/archive (2)/medications.csv")
precautions_df = pd.read_csv("C:/Users/Kashish U Singh/Downloads/archive (2)/precautions.csv")
diets_df = pd.read_csv("C:/Users/Kashish U Singh/Downloads/archive (2)/diets.csv")
workout_df = pd.read_csv("C:/Users/Kashish U Singh/Downloads/archive (2)/workout.csv")

print("Loaded all datasets successfully!")

def clean_text(text):
    if isinstance(text, str):
        return text.strip().lower()
    return text

# Apply cleaning
description_df['Disease'] = description_df['Disease'].apply(clean_text)
medications_df['Disease'] = medications_df['Disease'].apply(clean_text)
precautions_df['Disease'] = precautions_df['Disease'].apply(clean_text)
diets_df['Disease'] = diets_df['Disease'].apply(clean_text)
workout_df['Disease'] = workout_df['Disease'].apply(clean_text)

# Also clean main dataset
symptoms_df.iloc[:, 0] = symptoms_df.iloc[:, 0].apply(clean_text)

print("Disease names cleaned!")

# First column = disease
disease_col = symptoms_df.columns[0]

# Get symptom column names
symptom_columns = symptoms_df.columns[1:]

# Create new dataframe
disease_symptom_list = []

for _, row in symptoms_df.iterrows():
    disease = row[disease_col]
    
    symptoms = []
    for symptom in symptom_columns:
        if row[symptom] == 1:
            symptoms.append(symptom.lower())
    
    disease_symptom_list.append({
        "Disease": disease,
        "Symptoms": symptoms
    })

symptom_clean_df = pd.DataFrame(disease_symptom_list)

print(symptom_clean_df.head())

master_df = symptom_clean_df.merge(description_df, on="Disease", how="left")

import ast

def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

medications_df['Medication'] = medications_df['Medication'].apply(safe_eval)

master_df = master_df.merge(medications_df, on="Disease", how="left")

precautions_df['Precautions'] = precautions_df.iloc[:, 1:].values.tolist()
precautions_df = precautions_df[['Disease', 'Precautions']]

master_df = master_df.merge(precautions_df, on="Disease", how="left")

diets_df['Diet'] = diets_df.iloc[:, 1:].values.tolist()
diets_df = diets_df[['Disease', 'Diet']]

master_df = master_df.merge(diets_df, on="Disease", how="left")

workout_df['Workout'] = workout_df.iloc[:, 1:].values.tolist()
workout_df = workout_df[['Disease', 'Workout']]

master_df = master_df.merge(workout_df, on="Disease", how="left")

# Fill missing values
master_df = master_df.fillna("")

# Save master dataset
master_df.to_csv("master_disease_data.csv", index=False)

print("MASTER DATASET CREATED ✅")
print(master_df.head())

edges = []

for _, row in master_df.iterrows():
    disease = row['Disease']
    for symptom in row['Symptoms']:
        edges.append({
            "Symptom": symptom,
            "Disease": disease
        })

edges_df = pd.DataFrame(edges)
edges_df.to_csv("symptom_disease_edges.csv", index=False)

print("Symptom-Disease edges created!")

import pandas as pd
import ast

master_df = pd.read_csv("master_disease_data.csv")

def to_list(val):
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
        return [str(parsed).strip().lower()]
    except:
        return [str(val).strip().lower()]

# Convert string columns back to lists
master_df["Symptoms"] = master_df["Symptoms"].apply(to_list)
master_df["Medication"] = master_df["Medication"].apply(to_list)
master_df["Precautions"] = master_df["Precautions"].apply(to_list)
master_df["Diet"] = master_df["Diet"].apply(to_list)
master_df["Workout"] = master_df["Workout"].apply(to_list)

def build_edges(df, source_col, list_col, target_name):
    rows = []
    for _, row in df.iterrows():
        source = str(row[source_col]).strip().lower()
        for item in row[list_col]:
            item = str(item).strip().lower()
            if item:
                rows.append({source_col: source, target_name: item})
    return pd.DataFrame(rows)

med_edges = build_edges(master_df, "Disease", "Medication", "Medication")
prec_edges = build_edges(master_df, "Disease", "Precautions", "Precaution")
diet_edges = build_edges(master_df, "Disease", "Diet", "Diet")
work_edges = build_edges(master_df, "Disease", "Workout", "Workout")

med_edges.to_csv("disease_medication_edges.csv", index=False)
prec_edges.to_csv("disease_precaution_edges.csv", index=False)
diet_edges.to_csv("disease_diet_edges.csv", index=False)
work_edges.to_csv("disease_workout_edges.csv", index=False)

print("All graph edge files created successfully.")

import pandas as pd
import ast

master_df = pd.read_csv("C:/Users/Kashish U Singh/master_disease_data.csv")

def to_list(val):
    if pd.isna(val) or val == "":
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
        return [str(parsed).strip().lower()]
    except:
        return [str(val).strip().lower()]

master_df["Symptoms"] = master_df["Symptoms"].apply(to_list)

# Disease nodes
disease_nodes = master_df[["Disease", "Description"]].copy()
disease_nodes["Disease"] = disease_nodes["Disease"].astype(str).str.strip().str.lower()
disease_nodes["Description"] = disease_nodes["Description"].astype(str).str.strip()
disease_nodes.drop_duplicates().to_csv("disease_nodes.csv", index=False)

# Symptom nodes
symptom_rows = []
for _, row in master_df.iterrows():
    for symptom in row["Symptoms"]:
        symptom_rows.append({"Symptom": symptom})

symptom_nodes = pd.DataFrame(symptom_rows).drop_duplicates()
symptom_nodes.to_csv("symptom_nodes.csv", index=False)

print("Node files created.")

from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "Kashish@2001"

driver = GraphDatabase.driver(uri, auth=(username, password))

def get_disease_details(disease_name):
    query = """
    MATCH (d:Disease {name: $disease})
    OPTIONAL MATCH (s:Symptom)-[:INDICATES]->(d)
    OPTIONAL MATCH (d)-[:TREATED_BY]->(m:Medication)
    OPTIONAL MATCH (d)-[:NEEDS_PRECAUTION]->(p:Precaution)
    OPTIONAL MATCH (d)-[:RECOMMENDS_DIET]->(di:Diet)
    OPTIONAL MATCH (d)-[:SUGGESTS_WORKOUT]->(w:Workout)
    RETURN d.name AS disease,
           d.description AS description,
           collect(DISTINCT s.name) AS symptoms,
           collect(DISTINCT m.name) AS medications,
           collect(DISTINCT p.name) AS precautions,
           collect(DISTINCT di.name) AS diets,
           collect(DISTINCT w.name) AS workouts
    """
    with driver.session() as session:
        result = session.run(query, disease=disease_name.lower())
        return [record.data() for record in result]

print(get_disease_details("panic disorder"))