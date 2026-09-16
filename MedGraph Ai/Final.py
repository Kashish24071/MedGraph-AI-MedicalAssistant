import streamlit as st
import pandas as pd
import ast
import chromadb
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import io
import datetime

st.set_page_config(page_title="MedGraph AI", layout="wide", page_icon="🩺")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        opacity: 0.75;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
        font-size: 1rem;
    }

    .severity-emergency {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        border-left: 6px solid #c0392b;
    }

    .severity-high {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        border-left: 6px solid #e67e22;
    }

    .severity-medium {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        border-left: 6px solid #2980b9;
    }

    .severity-low {
        background: linear-gradient(135deg, #43e97b, #38f9d7);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        border-left: 6px solid #27ae60;
    }

    .disease-card {
        background: #ffffff;
        border: 1px solid #e8ecf0;
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1.2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    }

    .disease-card-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a2332;
        margin-bottom: 0.4rem;
    }

    .tag-pill {
        display: inline-block;
        background: #f0f4f8;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-family: 'IBM Plex Mono', monospace;
        color: #4a5568;
        margin: 2px;
        border: 1px solid #e2e8f0;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #718096;
        margin: 1.2rem 0 0.4rem 0;
    }

    .info-box {
        background: #f7fafc;
        border-left: 4px solid #4299e1;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #2d3748;
    }

    .warning-box {
        background: #fffaf0;
        border-left: 4px solid #ed8936;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #2d3748;
    }

    .symptom-slider-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .stSlider > div > div > div > div {
        background-color: #4299e1 !important;
    }

    .download-btn {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🩺 MedGraph AI</h1>
    <p>Symptom-Based Disease & Treatment Assistant — powered by ChromaDB · RAG · Neo4j</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    master_df = pd.read_csv("C:/Users/Kashish U Singh/master_disease_data.csv")
    return master_df

master_df = load_data()

# -----------------------------
# HELPERS
# -----------------------------
def to_list(val):
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [str(parsed).strip()]
    except:
        return [str(val).strip()]


master_df["Symptoms"] = master_df["Symptoms"].apply(to_list)
master_df["Medication"] = master_df["Medication"].apply(to_list)
master_df["Precautions"] = master_df["Precautions"].apply(to_list)
master_df["Diet"] = master_df["Diet"].apply(to_list)
master_df["Workout"] = master_df["Workout"].apply(to_list)


def make_retrieval_text(row):
    disease = row["Disease"]
    symptoms = ", ".join(row["Symptoms"]) if isinstance(row["Symptoms"], list) else ""
    return f"Disease: {disease}\nSymptoms: {symptoms}"


master_df["retrieval_text"] = master_df.apply(make_retrieval_text, axis=1)


def flatten_field(val):
    if isinstance(val, list):
        flat = []
        for item in val:
            if isinstance(item, str):
                try:
                    parsed = ast.literal_eval(item)
                    if isinstance(parsed, list):
                        flat.extend([str(x).strip() for x in parsed])
                    else:
                        flat.append(item.strip())
                except:
                    flat.append(item.strip())
            else:
                flat.append(str(item).strip())
        return flat
    return []


def clean_result_list(values):
    if not values:
        return []
    return flatten_field(values)


def clean_text_input(value):
    if not value:
        return ""
    value = value.strip().lower()
    if value in ["none", "no", "nil", "na", "n/a"]:
        return ""
    return value


master_df = master_df.drop_duplicates(subset=["Disease"])

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()

# -----------------------------
# CHROMA SETUP
# -----------------------------
@st.cache_resource
def setup_chroma(_master_df):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "disease_collection_ui"
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name)
    documents = _master_df["retrieval_text"].tolist()
    metadatas = [{"disease": d} for d in _master_df["Disease"].tolist()]
    ids = [str(i) for i in range(len(_master_df))]
    embeddings = model.encode(documents, convert_to_numpy=True)
    batch_size = 50
    for i in range(0, len(embeddings), batch_size):
        collection.add(
            embeddings=embeddings[i:i + batch_size].tolist(),
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )
    return collection


collection = setup_chroma(master_df)

# -----------------------------
# NEO4J CONNECTION
# -----------------------------
@st.cache_resource
def get_neo4j_driver():
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "Kashish@2001"
    return GraphDatabase.driver(uri, auth=(username, password))


driver = get_neo4j_driver()


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


# -----------------------------
# SEARCH + RANKING
# -----------------------------
def build_user_query(symptom_list):
    return "Symptoms: " + ", ".join([s.strip().lower() for s in symptom_list])


def search_disease_chroma(user_query, top_k=10):
    query_embedding = model.encode([user_query], convert_to_numpy=True)
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
    output = []
    for i in range(len(results["metadatas"][0])):
        output.append({
            "rank": i + 1,
            "disease": results["metadatas"][0][i]["disease"],
            "distance": results["distances"][0][i]
        })
    return output


def exact_symptom_match(user_symptoms):
    user_symptoms = set([s.strip().lower() for s in user_symptoms])
    results = []
    for _, row in master_df.iterrows():
        disease = row["Disease"]
        disease_symptoms = set([s.strip().lower() for s in row["Symptoms"]])
        overlap = len(user_symptoms.intersection(disease_symptoms))
        total_user = len(user_symptoms)
        match_ratio = overlap / total_user if total_user > 0 else 0
        results.append({"disease": disease, "overlap": overlap, "match_ratio": match_ratio})
    results = sorted(results, key=lambda x: (-x["overlap"], -x["match_ratio"]))
    return results


def hybrid_rank(user_symptoms, top_k_chroma=20):
    query = build_user_query(user_symptoms)
    chroma_results = search_disease_chroma(query, top_k=top_k_chroma)
    chroma_map = {r["disease"].lower(): r["distance"] for r in chroma_results}
    exact_results = exact_symptom_match(user_symptoms)
    combined = []
    for r in exact_results:
        disease = r["disease"]
        disease_key = disease.lower()
        chroma_distance = chroma_map.get(disease_key, 999.0)
        combined.append({
            "disease": disease,
            "overlap": r["overlap"],
            "match_ratio": r["match_ratio"],
            "chroma_distance": chroma_distance
        })
    combined = sorted(combined, key=lambda x: (-x["overlap"], -x["match_ratio"], x["chroma_distance"]))
    return combined


def generate_response(symptoms):
    ranked_results = hybrid_rank(symptoms)
    final_output = []
    for r in ranked_results[:3]:
        disease_name = r["disease"]
        details = get_disease_details(disease_name)
        if details:
            d = details[0]
            medications = clean_result_list(d["medications"])
            precautions = clean_result_list(d["precautions"])
            diets = clean_result_list(d["diets"])
            workouts = clean_result_list(d["workouts"])
            disease_symptoms = clean_result_list(d["symptoms"])
            final_output.append({
                "disease": d["disease"],
                "overlap": r["overlap"],
                "match_ratio": r["match_ratio"],
                "description": d["description"],
                "symptoms": disease_symptoms,
                "medications": medications,
                "precautions": precautions,
                "diets": diets,
                "workouts": workouts
            })
    return final_output


# -----------------------------
# SEVERITY ENGINE
# -----------------------------
emergency_symptoms = {
    "chest pain", "chest tightness", "chest pressure", "shortness of breath",
    "difficulty breathing", "severe headache", "loss of consciousness", "fainting",
    "seizure", "stroke", "sudden weakness", "blood in vomit", "coughing blood",
    "severe abdominal pain", "high fever", "confusion", "blue lips"
}

medium_risk_symptoms = {
    "fever", "persistent cough", "vomiting", "diarrhea", "dizziness",
    "fatigue", "abdominal pain", "body pain", "dehydration", "rash"
}

high_risk_conditions = {
    "Diabetes", "High Blood Pressure", "Asthma", "Heart Disease", "Kidney Disease", "Liver Disease"
}


def compute_weighted_severity_score(symptom_severity_map):
    """
    Takes a dict of {symptom: severity_score (1-10)}
    Emergency symptoms get multiplied; moderate ones less so.
    Returns a composite score.
    """
    score = 0
    for symptom, severity in symptom_severity_map.items():
        s = symptom.strip().lower()
        if s in emergency_symptoms:
            score += severity * 3.0
        elif s in medium_risk_symptoms:
            score += severity * 1.5
        else:
            score += severity * 0.8
    return score


def detect_severity(symptoms, patient_profile, symptom_severity_map=None):
    symptoms_lower = set([s.strip().lower() for s in symptoms])
    emergency_matches = symptoms_lower.intersection(emergency_symptoms)
    medium_matches = symptoms_lower.intersection(medium_risk_symptoms)
    existing_conditions = patient_profile.get("existing_conditions", [])
    age = patient_profile.get("age", 0)
    pregnancy_status = patient_profile.get("pregnancy_status", "Not Applicable")
    risk_reasons = []

    # Factor in symptom severity scores if provided
    weighted_score = 0
    if symptom_severity_map:
        weighted_score = compute_weighted_severity_score(symptom_severity_map)
        avg_severity = sum(symptom_severity_map.values()) / len(symptom_severity_map) if symptom_severity_map else 0
        if avg_severity >= 8:
            risk_reasons.append(f"High average symptom severity score: {avg_severity:.1f}/10")
        elif avg_severity >= 5:
            risk_reasons.append(f"Moderate average symptom severity score: {avg_severity:.1f}/10")

    if emergency_matches:
        risk_reasons.append("Emergency symptoms detected: " + ", ".join(emergency_matches))
        return "Emergency", risk_reasons, weighted_score

    if age >= 60:
        risk_reasons.append("Patient age is 60 or above, which may increase health risk.")
    if age <= 5:
        risk_reasons.append("Patient is very young, so symptoms may need closer attention.")

    condition_risk = set(existing_conditions).intersection(high_risk_conditions)
    if condition_risk:
        risk_reasons.append("Existing condition may increase risk: " + ", ".join(condition_risk))

    if pregnancy_status == "Pregnant":
        risk_reasons.append("Pregnancy status may require extra medical caution.")

    if medium_matches:
        risk_reasons.append("Moderate-risk symptoms found: " + ", ".join(medium_matches))

    # Boost severity level if weighted score is high
    if weighted_score >= 40:
        if condition_risk or age >= 60 or pregnancy_status == "Pregnant":
            return "Emergency", risk_reasons, weighted_score
        return "High", risk_reasons, weighted_score

    if condition_risk or age >= 60 or pregnancy_status == "Pregnant":
        if medium_matches:
            return "High", risk_reasons, weighted_score
        else:
            return "Medium", risk_reasons, weighted_score

    if medium_matches:
        return "Medium", risk_reasons, weighted_score

    return "Low", ["No major emergency or high-risk symptoms detected."], weighted_score


def personalize_recommendations(result, patient_profile):
    personalized_notes = []
    age = patient_profile.get("age")
    existing_conditions = patient_profile.get("existing_conditions", [])
    allergies = patient_profile.get("allergies", "")
    current_medications = patient_profile.get("current_medications", "")
    pregnancy_status = patient_profile.get("pregnancy_status", "Not Applicable")
    lifestyle = patient_profile.get("lifestyle", "")

    if age >= 60:
        personalized_notes.append("Because the patient is 60 or above, medical consultation is recommended before taking any medication.")
    if age <= 12:
        personalized_notes.append("For children, medicine dosage and treatment should be confirmed by a doctor.")
    if current_medications:
        personalized_notes.append("Patient is already taking medication, so possible drug interactions should be checked by a healthcare professional.")
    if "Diabetes" in existing_conditions:
        personalized_notes.append("Since the patient has diabetes, diet suggestions should be followed carefully, especially sugar and carbohydrate intake.")
    if "High Blood Pressure" in existing_conditions:
        personalized_notes.append("Since the patient has high blood pressure, avoid high-salt diet and consult a doctor before taking new medication.")
    if "Asthma" in existing_conditions:
        personalized_notes.append("Since the patient has asthma, breathing-related symptoms should be monitored carefully.")
    if "Heart Disease" in existing_conditions:
        personalized_notes.append("Since the patient has a heart condition, symptoms like chest pain, breathlessness, or fatigue should not be ignored.")
    if "Kidney Disease" in existing_conditions:
        personalized_notes.append("Since the patient has kidney disease, medication and diet recommendations should be reviewed by a doctor.")
    if "Liver Disease" in existing_conditions:
        personalized_notes.append("Since the patient has liver disease, medication should be reviewed carefully by a healthcare professional.")
    if "Thyroid" in existing_conditions:
        personalized_notes.append("Since the patient has thyroid history, symptoms and medication guidance should be reviewed with a doctor if symptoms persist.")
    if pregnancy_status == "Pregnant":
        personalized_notes.append("Pregnancy requires extra caution. Please consult a doctor before taking any medication or treatment.")
    if lifestyle == "Low Activity":
        personalized_notes.append("A low activity lifestyle may increase risk for long-term health issues. Light physical activity may help if medically safe.")
    if not personalized_notes:
        personalized_notes.append("No major personalization warnings found based on the entered profile.")
    return personalized_notes


def medicine_safety_check(result, patient_profile):
    safety_notes = []
    warning_found = False
    medications = [str(m).lower() for m in result.get("medications", [])]
    allergies = patient_profile.get("allergies", "")
    current_medications = patient_profile.get("current_medications", "")
    existing_conditions = patient_profile.get("existing_conditions", [])
    pregnancy_status = patient_profile.get("pregnancy_status", "Not Applicable")
    age = patient_profile.get("age", 0)

    if not medications:
        return ["No medicine data available for safety checking."]

    if allergies:
        allergy_terms = [a.strip() for a in allergies.split(",") if a.strip()]
        for allergy in allergy_terms:
            for med in medications:
                if allergy in med or med in allergy:
                    safety_notes.append(f"⚠️ Allergy Alert: '{med}' may conflict with the entered allergy '{allergy}'.")
                    warning_found = True

    if current_medications:
        safety_notes.append("⚠️ Interaction Caution: The patient is already taking medication. Please verify possible drug interactions before using any suggested medicine.")
        warning_found = True

    if age <= 12:
        safety_notes.append("⚠️ Pediatric Safety: Medicine dosage for children must be confirmed by a doctor.")
        warning_found = True

    if age >= 60:
        safety_notes.append("⚠️ Elderly Patient Safety: Older adults may need adjusted dosage or extra monitoring.")
        warning_found = True

    condition_warning_map = {
        "Diabetes": "Patients with diabetes should avoid medicines that may affect blood sugar unless approved by a doctor.",
        "High Blood Pressure": "Patients with high blood pressure should avoid medicines that may increase blood pressure unless approved by a doctor.",
        "Asthma": "Patients with asthma should be careful with medicines that may worsen breathing symptoms.",
        "Heart Disease": "Patients with heart disease should check medicine safety with a doctor before use.",
        "Kidney Disease": "Patients with kidney disease may require dosage adjustment for many medicines.",
        "Liver Disease": "Patients with liver disease should avoid medicines that may stress the liver unless approved by a doctor.",
        "Thyroid": "Patients with thyroid conditions should check medicine compatibility with current thyroid medication."
    }

    for condition in existing_conditions:
        if condition in condition_warning_map:
            safety_notes.append("⚠️ " + condition_warning_map[condition])
            warning_found = True

    if pregnancy_status == "Pregnant":
        safety_notes.append("⚠️ Pregnancy Safety: Medication use during pregnancy should be confirmed by a healthcare professional.")
        warning_found = True

    if not warning_found:
        safety_notes.append("No major medicine safety warnings detected from the entered profile. Still, medication should be verified by a healthcare professional.")

    return safety_notes


def recommend_lab_tests(result, selected_symptoms):
    disease = result.get("disease", "").lower()
    symptoms = [s.lower() for s in selected_symptoms]
    lab_tests = []

    disease_lab_map = {
        "diabetes": ["Fasting Blood Sugar", "HbA1c", "Random Blood Sugar", "Urine Sugar Test"],
        "hypertension": ["Blood Pressure Monitoring", "Lipid Profile", "Kidney Function Test", "ECG"],
        "heart": ["ECG", "Troponin Test", "Lipid Profile", "Echocardiogram"],
        "asthma": ["Spirometry", "Peak Flow Test", "Chest X-ray if symptoms are severe"],
        "pneumonia": ["Chest X-ray", "CBC", "Sputum Test", "Oxygen Saturation Test"],
        "bronchitis": ["Chest X-ray", "CBC", "Pulmonary Function Test"],
        "covid": ["COVID-19 Antigen Test", "RT-PCR Test", "Oxygen Saturation Test"],
        "influenza": ["Flu Test", "CBC"],
        "dengue": ["CBC", "Platelet Count", "NS1 Antigen Test", "Dengue IgM/IgG Test"],
        "malaria": ["Malaria Parasite Test", "Rapid Malaria Antigen Test", "CBC"],
        "typhoid": ["Widal Test", "Blood Culture", "CBC"],
        "thyroid": ["TSH", "T3", "T4"],
        "anemia": ["CBC", "Hemoglobin Test", "Iron Studies", "Vitamin B12 Test"],
        "migraine": ["Neurological Evaluation", "MRI/CT scan only if recommended by doctor"],
        "jaundice": ["Liver Function Test", "Bilirubin Test", "Hepatitis Panel"],
        "hepatitis": ["Liver Function Test", "Hepatitis Panel", "Bilirubin Test"],
        "kidney": ["Kidney Function Test", "Creatinine", "Urine Routine Test", "Electrolyte Test"],
        "urinary": ["Urine Routine Test", "Urine Culture", "Kidney Function Test"],
        "allergy": ["Allergy Panel Test", "IgE Test"],
        "skin": ["Dermatology Examination", "Allergy Test if needed"],
        "infection": ["CBC", "CRP", "Blood Culture if severe"],
        "gastro": ["Stool Test", "Liver Function Test", "Abdominal Ultrasound if needed"]
    }

    for keyword, tests in disease_lab_map.items():
        if keyword in disease:
            lab_tests.extend(tests)

    if "fever" in symptoms or "high fever" in symptoms:
        lab_tests.extend(["CBC", "CRP", "Blood Culture if fever is persistent"])
    if "chest pain" in symptoms or "chest tightness" in symptoms:
        lab_tests.extend(["ECG", "Troponin Test", "Blood Pressure Check"])
    if "shortness of breath" in symptoms or "difficulty breathing" in symptoms:
        lab_tests.extend(["Oxygen Saturation Test", "Chest X-ray", "Spirometry"])
    if "vomiting" in symptoms or "diarrhea" in symptoms:
        lab_tests.extend(["Electrolyte Test", "Stool Test", "CBC"])
    if "fatigue" in symptoms:
        lab_tests.extend(["CBC", "Thyroid Profile", "Vitamin B12 Test"])

    lab_tests = list(dict.fromkeys(lab_tests))

    if not lab_tests:
        lab_tests = ["General Physician Evaluation", "CBC", "Basic Metabolic Panel if recommended by doctor"]

    return lab_tests


def recommend_specialist(result, selected_symptoms):
    disease = result.get("disease", "").lower()
    symptoms = [s.lower() for s in selected_symptoms]
    specialist = "General Physician"
    reason = "A general physician can evaluate the symptoms and refer to a specialist if required."

    specialist_rules = [
        {"keywords": ["heart", "cardiac", "chest pain", "chest tightness", "chest pressure"], "specialist": "Cardiologist", "reason": "Heart-related or chest symptoms may require cardiac evaluation."},
        {"keywords": ["asthma", "pneumonia", "bronchitis", "breathing", "shortness of breath", "difficulty breathing"], "specialist": "Pulmonologist", "reason": "Breathing or lung-related symptoms may require respiratory evaluation."},
        {"keywords": ["diabetes", "thyroid", "hormone"], "specialist": "Endocrinologist", "reason": "Diabetes, thyroid, and hormonal conditions are usually managed by an endocrinologist."},
        {"keywords": ["skin", "rash", "allergy", "itching"], "specialist": "Dermatologist", "reason": "Skin-related symptoms may require dermatology evaluation."},
        {"keywords": ["kidney", "urinary", "urine"], "specialist": "Nephrologist / Urologist", "reason": "Kidney or urinary symptoms may require nephrology or urology evaluation."},
        {"keywords": ["liver", "jaundice", "hepatitis", "gastro", "stomach", "abdominal", "diarrhea", "vomiting"], "specialist": "Gastroenterologist", "reason": "Digestive, liver, or abdominal symptoms may require gastroenterology evaluation."},
        {"keywords": ["migraine", "seizure", "stroke", "headache", "confusion", "weakness"], "specialist": "Neurologist", "reason": "Neurological symptoms may require a neurologist."},
        {"keywords": ["joint", "bone", "back pain", "muscle"], "specialist": "Orthopedic Specialist", "reason": "Bone, joint, or muscle-related symptoms may require orthopedic evaluation."},
        {"keywords": ["fever", "infection", "dengue", "malaria", "typhoid", "covid", "influenza"], "specialist": "General Physician / Infectious Disease Specialist", "reason": "Fever or infection-related symptoms should first be evaluated by a physician."}
    ]

    combined_text = disease + " " + " ".join(symptoms)
    for rule in specialist_rules:
        for keyword in rule["keywords"]:
            if keyword in combined_text:
                return rule["specialist"], rule["reason"]

    return specialist, reason


# -----------------------------
#  PLOTLY CONFIDENCE CHART
# -----------------------------
def render_confidence_chart(results):
    if not results:
        return

    diseases = [r["disease"].title() for r in results]
    match_ratios = [round(r["match_ratio"] * 100, 1) for r in results]
    overlaps = [r["overlap"] for r in results]

    colors = ["#4299e1", "#63b3ed", "#90cdf4"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=match_ratios,
        y=diseases,
        orientation='h',
        marker=dict(
            color=colors[:len(diseases)],
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        text=[f"{v}%  ({overlaps[i]} symptom matches)" for i, v in enumerate(match_ratios)],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Match: %{x}%<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Condition Match Confidence", font=dict(size=15, family="IBM Plex Sans"), x=0),
        xaxis=dict(
            title="Match Confidence (%)",
            range=[0, max(match_ratios) * 1.35 if match_ratios else 100],
            showgrid=True,
            gridcolor="#f0f4f8",
            tickfont=dict(family="IBM Plex Mono", size=11)
        ),
        yaxis=dict(
            tickfont=dict(family="IBM Plex Sans", size=13, color="#2d3748"),
            autorange="reversed"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=20, t=50, b=40),
        height=220,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_severity_radar(symptom_severity_map):
    if not symptom_severity_map or len(symptom_severity_map) < 3:
        return

    symptoms = list(symptom_severity_map.keys())
    values = list(symptom_severity_map.values())
    symptoms_closed = symptoms + [symptoms[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=symptoms_closed,
        fill='toself',
        fillcolor='rgba(66, 153, 225, 0.2)',
        line=dict(color='#4299e1', width=2),
        name='Severity'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10, family="IBM Plex Sans"))
        ),
        showlegend=False,
        title=dict(text="Symptom Severity Radar", font=dict(size=14, family="IBM Plex Sans"), x=0),
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# PDF EXPORT
# -----------------------------
def safe_text(text):
    """Strip any non-latin1 characters so Helvetica never crashes."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(patient_profile, final_symptoms, symptom_severity_map,
                        severity_level, severity_reasons, results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---- HEADER BLOCK ----
    pdf.set_fill_color(15, 32, 39)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 12)
    pdf.cell(0, 10, "MedGraph AI - Symptom Analysis Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(200, 220, 235)
    pdf.set_xy(10, 26)
    generated_time = datetime.datetime.now().strftime("%d %b %Y  %I:%M %p")
    pdf.cell(0, 8, safe_text(f"Generated: {generated_time}   |   Educational Use Only"), ln=True)

    pdf.set_text_color(30, 30, 30)
    pdf.set_xy(10, 48)

    # ---- HELPER: section header ----
    def section_header(title):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(230, 236, 245)
        pdf.set_text_color(30, 50, 80)
        pdf.cell(0, 9, safe_text(f"  {title}"), ln=True, fill=True)
        pdf.ln(2)

    # ---- HELPER: body text ----
    def body(text):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 6, safe_text(str(text)))

    # ---- HELPER: bullet row (uses dash, never special char) ----
    def bullet(text):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 7, safe_text(f"  - {text}"), ln=True)

    # ---- PATIENT PROFILE ----
    section_header("Patient Profile")
    conditions_str = ", ".join(patient_profile["existing_conditions"]) if patient_profile["existing_conditions"] else "None"
    allergies_str  = patient_profile["allergies"] or "None"
    meds_str       = patient_profile["current_medications"] or "None"
    profile_rows = [
        f"Age: {patient_profile['age']}   |   Gender: {patient_profile['gender']}",
        f"Existing Conditions: {conditions_str}",
        f"Allergies: {allergies_str}",
        f"Current Medications: {meds_str}",
        f"Pregnancy Status: {patient_profile['pregnancy_status']}",
        f"Lifestyle: {patient_profile['lifestyle']}",
    ]
    for row in profile_rows:
        bullet(row)
    pdf.ln(4)

    # ---- REPORTED SYMPTOMS ----
    section_header("Reported Symptoms")
    for sym in final_symptoms:
        sev      = symptom_severity_map.get(sym, 5)
        duration = st.session_state.get(f"dur_{sym}", 1)
        sev_lbl  = "Mild" if sev <= 3 else "Moderate" if sev <= 6 else "Severe"
        bullet(f"{sym.title()}   Severity: {sev}/10 ({sev_lbl})   Duration: {duration} day(s)")
    pdf.ln(4)

    # ---- RISK LEVEL ----
    sev_colors = {
        "Emergency": (192, 57,  43),
        "High":      (211, 84,   0),
        "Medium":    ( 41,128, 185),
        "Low":       ( 39,174,  96),
    }
    r, g, b = sev_colors.get(severity_level, (100, 100, 100))
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, safe_text(f"  Risk Level: {severity_level.upper()}"), ln=True, fill=True)
    pdf.ln(2)
    pdf.set_text_color(60, 60, 60)
    for reason in severity_reasons:
        # strip any emoji / non-latin chars from reason text
        bullet(reason)
    pdf.ln(4)

    # ---- MATCHED CONDITIONS ----
    for idx, result in enumerate(results, 1):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(66, 153, 225)
        pdf.set_text_color(255, 255, 255)
        cond_title = f"  Condition {idx}: {result['disease'].title()}   (Match: {result['match_ratio']*100:.0f}%)"
        pdf.cell(0, 10, safe_text(cond_title), ln=True, fill=True)
        pdf.ln(2)

        if result.get("description"):
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(80, 80, 80)
            desc = str(result["description"])[:300]
            pdf.multi_cell(0, 6, safe_text(desc))
            pdf.ln(2)

        def subsection(label, items):
            if not items:
                return
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 7, safe_text(label + ":"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            line = "  " + ", ".join(safe_text(str(x)) for x in items[:6])
            pdf.multi_cell(0, 6, line)
            pdf.ln(1)

        subsection("Medications",  result.get("medications", []))
        subsection("Precautions",  result.get("precautions", []))
        subsection("Diet",         result.get("diets", []))
        subsection("Workouts",     result.get("workouts", []))
        pdf.ln(5)

    # ---- FOOTER ----
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    footer = "This report is for educational purposes only. It does not replace professional medical advice."
    pdf.cell(0, 6, safe_text(footer), ln=True, align="C")

    return bytes(pdf.output())


# -----------------------------
# PATIENT PROFILE SIDEBAR
# -----------------------------
st.sidebar.markdown("## 👤 Patient Profile")

age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
existing_conditions = st.sidebar.multiselect(
    "Existing Medical Conditions",
    ["None", "Diabetes", "High Blood Pressure", "Asthma", "Heart Disease", "Kidney Disease", "Liver Disease", "Thyroid"]
)

if "None" in existing_conditions and len(existing_conditions) > 1:
    existing_conditions = [c for c in existing_conditions if c != "None"]

allergies = st.sidebar.text_input("Known Allergies", placeholder="Example: penicillin, aspirin, peanuts")
current_medications = st.sidebar.text_input("Current Medications", placeholder="Example: insulin, metformin")

pregnancy_status = "Not Applicable"
if gender == "Female":
    pregnancy_status = st.sidebar.selectbox("Pregnancy Status", ["Not Pregnant", "Pregnant", "Prefer not to say"])

lifestyle = st.sidebar.selectbox("Lifestyle Activity Level", ["Low Activity", "Moderate Activity", "Highly Active"])

patient_profile = {
    "age": age,
    "gender": gender,
    "existing_conditions": existing_conditions,
    "allergies": clean_text_input(allergies),
    "current_medications": clean_text_input(current_medications),
    "pregnancy_status": pregnancy_status,
    "lifestyle": lifestyle
}

# -----------------------------
# SYMPTOM INPUT
# -----------------------------
all_symptoms = sorted(set(sym for row in master_df["Symptoms"] for sym in row if sym))

col_input1, col_input2 = st.columns([3, 2])
with col_input1:
    selected_symptoms = st.multiselect("Select Symptoms", options=all_symptoms)
    manual_symptoms = st.text_input("Or type symptoms (comma-separated)", placeholder="fever, headache, fatigue")

# -----------------------------
# ★ NEW FEATURE 3: SYMPTOM SEVERITY + DURATION SLIDERS
# -----------------------------
symptom_severity_map = {}

if selected_symptoms or (manual_symptoms and manual_symptoms.strip()):
    combined_preview = list(set(selected_symptoms))
    if manual_symptoms.strip():
        typed = [x.strip().lower() for x in manual_symptoms.split(",") if x.strip()]
        combined_preview = list(set(combined_preview + typed))

    if combined_preview:
        st.markdown("---")
        st.markdown("### 🎚️ Rate Each Symptom")
        st.caption("For each symptom, rate the severity and how many days you've had it. This improves risk assessment accuracy.")

        num_cols = min(3, len(combined_preview))
        cols = st.columns(num_cols)

        for i, symptom in enumerate(combined_preview):
            col = cols[i % num_cols]
            with col:
                st.markdown(f"""
                <div class="symptom-slider-card">
                    <b style="font-size:0.9rem; color:#2d3748;">{symptom.title()}</b>
                </div>
                """, unsafe_allow_html=True)
                severity = st.slider(
                    f"Severity",
                    min_value=1, max_value=10, value=5,
                    key=f"sev_{symptom}",
                    help="1 = Very mild, 10 = Unbearable"
                )
                duration = st.number_input(
                    f"Duration (days)",
                    min_value=1, max_value=365, value=1,
                    key=f"dur_{symptom}"
                )
                symptom_severity_map[symptom] = severity

st.markdown("---")
analyze_btn = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)

# -----------------------------
# ANALYSIS OUTPUT
# -----------------------------
if analyze_btn:
    final_symptoms = list(set(selected_symptoms))
    if manual_symptoms.strip():
        typed = [x.strip().lower() for x in manual_symptoms.split(",") if x.strip()]
        final_symptoms = list(set(final_symptoms + typed))

    if not final_symptoms:
        st.warning("Please select or type at least one symptom.")
    else:
        # Selected symptoms display
        st.markdown("### Selected Symptoms")
        pills_html = " ".join([f'<span class="tag-pill">{s}</span>' for s in final_symptoms])
        st.markdown(pills_html, unsafe_allow_html=True)
        st.markdown("")

        results = generate_response(final_symptoms)
        severity_level, severity_reasons, weighted_score = detect_severity(
            final_symptoms, patient_profile, symptom_severity_map
        )

        # ---- SEVERITY DISPLAY ----
        st.markdown("### 🚦 Risk Assessment")
        sev_class = {
            "Emergency": "severity-emergency",
            "High": "severity-high",
            "Medium": "severity-medium",
            "Low": "severity-low"
        }.get(severity_level, "severity-low")

        sev_icon = {"Emergency": "🚨", "High": "⚠️", "Medium": "🟡", "Low": "🟢"}.get(severity_level, "🟢")
        sev_label = f"{sev_icon} {severity_level} Risk"
        if weighted_score > 0:
            sev_label += f"  ·  Weighted Score: {weighted_score:.1f}"

        st.markdown(f'<div class="{sev_class}">{sev_label}</div>', unsafe_allow_html=True)

        for reason in severity_reasons:
            st.markdown(f'<div class="info-box">• {reason}</div>', unsafe_allow_html=True)

        if severity_level == "Emergency":
            st.error("🚨 Emergency symptoms detected. Please seek immediate medical attention or contact emergency services.")

        # ---- CHARTS ROW ----
        st.markdown("### 📊 Visual Analysis")
        chart_col1, chart_col2 = st.columns([3, 2])

        with chart_col1:
            render_confidence_chart(results)

        with chart_col2:
            if len(symptom_severity_map) >= 3:
                render_severity_radar(symptom_severity_map)
            else:
                if symptom_severity_map:
                    st.markdown("**Symptom Severity Summary**")
                    for sym, sev in symptom_severity_map.items():
                        label = "🔴 Severe" if sev >= 8 else "🟡 Moderate" if sev >= 5 else "🟢 Mild"
                        st.markdown(f"**{sym.title()}** — {sev}/10 {label}")

        # ---- DISEASE RESULTS ----
        if not results:
            st.error("No matching conditions found.")
        else:
            st.markdown("### 🏥 Possible Conditions")

            for idx, result in enumerate(results, start=1):
                confidence_pct = result['match_ratio'] * 100
                conf_color = "#4299e1" if confidence_pct >= 60 else "#ed8936" if confidence_pct >= 30 else "#a0aec0"

                st.markdown(f"""
                <div class="disease-card">
                    <div class="disease-card-title">#{idx} — {result['disease'].title()}</div>
                    <div style="margin-top:6px;">
                        <span style="font-size:0.85rem; color:#718096;">Confidence: </span>
                        <span style="font-weight:600; color:{conf_color}; font-family:'IBM Plex Mono';">{confidence_pct:.0f}%</span>
                        <span style="font-size:0.8rem; color:#a0aec0; margin-left:8px;">({result['overlap']} symptom matches)</span>
                    </div>
                """, unsafe_allow_html=True)

                # Progress bar for confidence
                st.progress(min(result['match_ratio'], 1.0))
                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander(f"View full details for {result['disease'].title()}", expanded=(idx == 1)):
                    tab1, tab2, tab3, tab4 = st.tabs(["🩺 Clinical", "💊 Medications & Safety", "🧪 Tests & Specialist", "👤 Personal Notes"])

                    with tab1:
                        st.markdown(f"**Description:** {result.get('description', 'N/A')}")
                        st.markdown('<div class="section-label">Common Symptoms</div>', unsafe_allow_html=True)
                        if result["symptoms"]:
                            pills = " ".join([f'<span class="tag-pill">{s}</span>' for s in result["symptoms"][:8]])
                            st.markdown(pills, unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Diet Suggestions</div>', unsafe_allow_html=True)
                        if result["diets"]:
                            st.markdown(" | ".join(result["diets"][:6]))
                        st.markdown('<div class="section-label">Recommended Workouts</div>', unsafe_allow_html=True)
                        if result["workouts"]:
                            st.markdown(" | ".join(result["workouts"][:6]))

                    with tab2:
                        st.markdown('<div class="section-label">Medications</div>', unsafe_allow_html=True)
                        if result["medications"]:
                            pills = " ".join([f'<span class="tag-pill">{m}</span>' for m in result["medications"][:8]])
                            st.markdown(pills, unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Precautions</div>', unsafe_allow_html=True)
                        if result["precautions"]:
                            for p in result["precautions"][:5]:
                                st.markdown(f'<div class="info-box">• {p}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="section-label">Medicine Safety Check</div>', unsafe_allow_html=True)
                        safety_notes = medicine_safety_check(result, patient_profile)
                        for note in safety_notes:
                            box_class = "warning-box" if "⚠️" in note else "info-box"
                            st.markdown(f'<div class="{box_class}">{note}</div>', unsafe_allow_html=True)

                    with tab3:
                        lab_tests = recommend_lab_tests(result, final_symptoms)
                        specialist, specialist_reason = recommend_specialist(result, final_symptoms)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown('<div class="section-label">Suggested Lab Tests</div>', unsafe_allow_html=True)
                            for test in lab_tests:
                                st.markdown(f'<div class="info-box">🧪 {test}</div>', unsafe_allow_html=True)
                        with c2:
                            st.markdown('<div class="section-label">Recommended Specialist</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="info-box">🧑‍⚕️ <b>{specialist}</b><br>{specialist_reason}</div>', unsafe_allow_html=True)

                    with tab4:
                        personalized_notes = personalize_recommendations(result, patient_profile)
                        for note in personalized_notes:
                            st.markdown(f'<div class="info-box">💡 {note}</div>', unsafe_allow_html=True)

            # ---- PDF EXPORT ----
            st.markdown("---")
            st.markdown("### 📄 Export Report")
            st.caption("Download a full summary PDF you can bring to your doctor.")

            try:
                pdf_bytes = generate_pdf_report(
                    patient_profile, final_symptoms, symptom_severity_map,
                    severity_level, severity_reasons, results
                )
                report_filename = f"MedGraphAI_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=report_filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}. Make sure fpdf2 is installed: pip install fpdf2")

st.markdown("---")
st.caption("⚠️ Educational use only. This system does not replace professional medical advice, diagnosis, or treatment.")
