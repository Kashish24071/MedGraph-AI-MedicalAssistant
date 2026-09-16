
import streamlit as st
import pandas as pd
import ast
import chromadb
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="MedGraph AI", layout="wide")

st.title("🩺 MedGraph AI: Symptom-Based Disease & Treatment Assistant")
st.markdown(
    "Enter symptoms to get possible conditions with medications, precautions, "
    "diet suggestions, workouts, personalized health notes, severity risk level, "
    "medicine safety checks, lab test suggestions, and specialist recommendations."
)

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
print(len(master_df))

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

    # Replace this with your Neo4j password.
    # For public submission or GitHub, use environment variables or st.secrets.
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

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

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

        results.append({
            "disease": disease,
            "overlap": overlap,
            "match_ratio": match_ratio
        })

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

    combined = sorted(
        combined,
        key=lambda x: (-x["overlap"], -x["match_ratio"], x["chroma_distance"])
    )

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
# PATIENT PERSONALIZATION + SEVERITY ENGINE
# -----------------------------
emergency_symptoms = {
    "chest pain",
    "chest tightness",
    "chest pressure",
    "shortness of breath",
    "difficulty breathing",
    "severe headache",
    "loss of consciousness",
    "fainting",
    "seizure",
    "stroke",
    "sudden weakness",
    "blood in vomit",
    "coughing blood",
    "severe abdominal pain",
    "high fever",
    "confusion",
    "blue lips"
}

medium_risk_symptoms = {
    "fever",
    "persistent cough",
    "vomiting",
    "diarrhea",
    "dizziness",
    "fatigue",
    "abdominal pain",
    "body pain",
    "dehydration",
    "rash"
}

high_risk_conditions = {
    "Diabetes",
    "High Blood Pressure",
    "Asthma",
    "Heart Disease",
    "Kidney Disease",
    "Liver Disease"
}


def detect_severity(symptoms, patient_profile):
    symptoms_lower = set([s.strip().lower() for s in symptoms])

    emergency_matches = symptoms_lower.intersection(emergency_symptoms)
    medium_matches = symptoms_lower.intersection(medium_risk_symptoms)

    existing_conditions = patient_profile.get("existing_conditions", [])
    age = patient_profile.get("age", 0)
    pregnancy_status = patient_profile.get("pregnancy_status", "Not Applicable")

    risk_reasons = []

    if emergency_matches:
        risk_reasons.append(
            "Emergency symptoms detected: " + ", ".join(emergency_matches)
        )
        return "Emergency", risk_reasons

    if age >= 60:
        risk_reasons.append(
            "Patient age is 60 or above, which may increase health risk."
        )

    if age <= 5:
        risk_reasons.append(
            "Patient is very young, so symptoms may need closer attention."
        )

    condition_risk = set(existing_conditions).intersection(high_risk_conditions)

    if condition_risk:
        risk_reasons.append(
            "Existing condition may increase risk: " + ", ".join(condition_risk)
        )

    if pregnancy_status == "Pregnant":
        risk_reasons.append(
            "Pregnancy status may require extra medical caution."
        )

    if medium_matches:
        risk_reasons.append(
            "Moderate-risk symptoms found: " + ", ".join(medium_matches)
        )

    if condition_risk or age >= 60 or pregnancy_status == "Pregnant":
        if medium_matches:
            return "High", risk_reasons
        else:
            return "Medium", risk_reasons

    if medium_matches:
        return "Medium", risk_reasons

    return "Low", ["No major emergency or high-risk symptoms detected."]


def personalize_recommendations(result, patient_profile):
    personalized_notes = []

    age = patient_profile.get("age")
    existing_conditions = patient_profile.get("existing_conditions", [])
    allergies = patient_profile.get("allergies", "")
    current_medications = patient_profile.get("current_medications", "")
    pregnancy_status = patient_profile.get("pregnancy_status", "Not Applicable")
    lifestyle = patient_profile.get("lifestyle", "")

    if age >= 60:
        personalized_notes.append(
            "Because the patient is 60 or above, medical consultation is recommended before taking any medication."
        )

    if age <= 12:
        personalized_notes.append(
            "For children, medicine dosage and treatment should be confirmed by a doctor."
        )

    if current_medications:
        personalized_notes.append(
            "Patient is already taking medication, so possible drug interactions should be checked by a healthcare professional."
        )

    if "Diabetes" in existing_conditions:
        personalized_notes.append(
            "Since the patient has diabetes, diet suggestions should be followed carefully, especially sugar and carbohydrate intake."
        )

    if "High Blood Pressure" in existing_conditions:
        personalized_notes.append(
            "Since the patient has high blood pressure, avoid high-salt diet and consult a doctor before taking new medication."
        )

    if "Asthma" in existing_conditions:
        personalized_notes.append(
            "Since the patient has asthma, breathing-related symptoms should be monitored carefully."
        )

    if "Heart Disease" in existing_conditions:
        personalized_notes.append(
            "Since the patient has a heart condition, symptoms like chest pain, breathlessness, or fatigue should not be ignored."
        )

    if "Kidney Disease" in existing_conditions:
        personalized_notes.append(
            "Since the patient has kidney disease, medication and diet recommendations should be reviewed by a doctor."
        )

    if "Liver Disease" in existing_conditions:
        personalized_notes.append(
            "Since the patient has liver disease, medication should be reviewed carefully by a healthcare professional."
        )

    if "Thyroid" in existing_conditions:
        personalized_notes.append(
            "Since the patient has thyroid history, symptoms and medication guidance should be reviewed with a doctor if symptoms persist."
        )

    if pregnancy_status == "Pregnant":
        personalized_notes.append(
            "Pregnancy requires extra caution. Please consult a doctor before taking any medication or treatment."
        )

    if lifestyle == "Low Activity":
        personalized_notes.append(
            "A low activity lifestyle may increase risk for long-term health issues. Light physical activity may help if medically safe."
        )

    if not personalized_notes:
        personalized_notes.append(
            "No major personalization warnings found based on the entered profile."
        )

    return personalized_notes


# -----------------------------
# MEDICINE SAFETY CHECK
# -----------------------------
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

    # Allergy check
    if allergies:
        allergy_terms = [a.strip() for a in allergies.split(",") if a.strip()]

        for allergy in allergy_terms:
            for med in medications:
                if allergy in med or med in allergy:
                    safety_notes.append(
                        f"⚠️ Allergy Alert: '{med}' may conflict with the entered allergy '{allergy}'."
                    )
                    warning_found = True

    # Current medication interaction reminder
    if current_medications:
        safety_notes.append(
            "⚠️ Interaction Caution: The patient is already taking medication. Please verify possible drug interactions before using any suggested medicine."
        )
        warning_found = True

    # Age-based safety
    if age <= 12:
        safety_notes.append(
            "⚠️ Pediatric Safety: Medicine dosage for children must be confirmed by a doctor."
        )
        warning_found = True

    if age >= 60:
        safety_notes.append(
            "⚠️ Elderly Patient Safety: Older adults may need adjusted dosage or extra monitoring."
        )
        warning_found = True

    # Condition-based medicine safety
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

    # Pregnancy safety
    if pregnancy_status == "Pregnant":
        safety_notes.append(
            "⚠️ Pregnancy Safety: Medication use during pregnancy should be confirmed by a healthcare professional."
        )
        warning_found = True

    if not warning_found:
        safety_notes.append(
            "No major medicine safety warnings detected from the entered profile. Still, medication should be verified by a healthcare professional."
        )

    return safety_notes


# -----------------------------
# LAB TEST RECOMMENDATION
# -----------------------------
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

    # Symptom-based test suggestions
    if "fever" in symptoms or "high fever" in symptoms:
        lab_tests.extend(["CBC", "CRP", "Blood Culture if fever is persistent"])

    if "chest pain" in symptoms or "chest tightness" in symptoms or "chest pressure" in symptoms:
        lab_tests.extend(["ECG", "Troponin Test", "Blood Pressure Check"])

    if "shortness of breath" in symptoms or "difficulty breathing" in symptoms:
        lab_tests.extend(["Oxygen Saturation Test", "Chest X-ray", "Spirometry"])

    if "vomiting" in symptoms or "diarrhea" in symptoms:
        lab_tests.extend(["Electrolyte Test", "Stool Test", "CBC"])

    if "fatigue" in symptoms:
        lab_tests.extend(["CBC", "Thyroid Profile", "Vitamin B12 Test"])

    lab_tests = list(dict.fromkeys(lab_tests))

    if not lab_tests:
        lab_tests = [
            "General Physician Evaluation",
            "CBC",
            "Basic Metabolic Panel if recommended by doctor"
        ]

    return lab_tests


# -----------------------------
# DOCTOR / SPECIALIST RECOMMENDATION
# -----------------------------
def recommend_specialist(result, selected_symptoms):
    disease = result.get("disease", "").lower()
    symptoms = [s.lower() for s in selected_symptoms]

    specialist = "General Physician"
    reason = "A general physician can evaluate the symptoms and refer to a specialist if required."

    specialist_rules = [
        {
            "keywords": ["heart", "cardiac", "chest pain", "chest tightness", "chest pressure"],
            "specialist": "Cardiologist",
            "reason": "Heart-related or chest symptoms may require cardiac evaluation."
        },
        {
            "keywords": ["asthma", "pneumonia", "bronchitis", "breathing", "shortness of breath", "difficulty breathing"],
            "specialist": "Pulmonologist",
            "reason": "Breathing or lung-related symptoms may require respiratory evaluation."
        },
        {
            "keywords": ["diabetes", "thyroid", "hormone"],
            "specialist": "Endocrinologist",
            "reason": "Diabetes, thyroid, and hormonal conditions are usually managed by an endocrinologist."
        },
        {
            "keywords": ["skin", "rash", "allergy", "itching"],
            "specialist": "Dermatologist",
            "reason": "Skin-related symptoms may require dermatology evaluation."
        },
        {
            "keywords": ["kidney", "urinary", "urine"],
            "specialist": "Nephrologist / Urologist",
            "reason": "Kidney or urinary symptoms may require nephrology or urology evaluation."
        },
        {
            "keywords": ["liver", "jaundice", "hepatitis", "gastro", "stomach", "abdominal", "diarrhea", "vomiting"],
            "specialist": "Gastroenterologist",
            "reason": "Digestive, liver, or abdominal symptoms may require gastroenterology evaluation."
        },
        {
            "keywords": ["migraine", "seizure", "stroke", "headache", "confusion", "weakness"],
            "specialist": "Neurologist",
            "reason": "Neurological symptoms may require a neurologist."
        },
        {
            "keywords": ["joint", "bone", "back pain", "muscle"],
            "specialist": "Orthopedic Specialist",
            "reason": "Bone, joint, or muscle-related symptoms may require orthopedic evaluation."
        },
        {
            "keywords": ["fever", "infection", "dengue", "malaria", "typhoid", "covid", "influenza"],
            "specialist": "General Physician / Infectious Disease Specialist",
            "reason": "Fever or infection-related symptoms should first be evaluated by a physician."
        }
    ]

    combined_text = disease + " " + " ".join(symptoms)

    for rule in specialist_rules:
        for keyword in rule["keywords"]:
            if keyword in combined_text:
                specialist = rule["specialist"]
                reason = rule["reason"]
                return specialist, reason

    return specialist, reason


# -----------------------------
# PATIENT PROFILE INPUT
# -----------------------------
st.sidebar.header("👤 Patient Profile")

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Prefer not to say", "Male", "Female", "Other"]
)

existing_conditions = st.sidebar.multiselect(
    "Existing Medical Conditions",
    [
        "None",
        "Diabetes",
        "High Blood Pressure",
        "Asthma",
        "Heart Disease",
        "Kidney Disease",
        "Liver Disease",
        "Thyroid"
    ]
)

if "None" in existing_conditions and len(existing_conditions) > 1:
    existing_conditions = [
        condition for condition in existing_conditions if condition != "None"
    ]

allergies = st.sidebar.text_input(
    "Known Allergies",
    placeholder="Example: penicillin, aspirin, peanuts"
)

current_medications = st.sidebar.text_input(
    "Current Medications",
    placeholder="Example: insulin, metformin, blood pressure medicine"
)

pregnancy_status = "Not Applicable"

if gender == "Female":
    pregnancy_status = st.sidebar.selectbox(
        "Pregnancy Status",
        ["Not Pregnant", "Pregnant", "Prefer not to say"]
    )

lifestyle = st.sidebar.selectbox(
    "Lifestyle Activity Level",
    ["Low Activity", "Moderate Activity", "Highly Active"]
)

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
# UI INPUT
# -----------------------------
all_symptoms = sorted(set(sym for row in master_df["Symptoms"] for sym in row if sym))

selected_symptoms = st.multiselect(
    "Select symptoms",
    options=all_symptoms
)

manual_symptoms = st.text_input(
    "Or type symptoms separated by commas",
    placeholder="fever, headache, fatigue"
)

if st.button("Analyze Symptoms"):
    final_symptoms = selected_symptoms.copy()

    if manual_symptoms.strip():
        typed = [x.strip().lower() for x in manual_symptoms.split(",") if x.strip()]
        final_symptoms.extend(typed)

    final_symptoms = list(set(final_symptoms))

    if not final_symptoms:
        st.warning("Please select or type at least one symptom.")

    else:
        st.subheader("Selected Symptoms")
        st.write(", ".join(final_symptoms))

        results = generate_response(final_symptoms)

        severity_level, severity_reasons = detect_severity(
            final_symptoms,
            patient_profile
        )

        # -----------------------------
        # SEVERITY AND RISK DISPLAY
        # -----------------------------
        st.subheader("🚦 Severity and Risk Level")

        if severity_level == "Emergency":
            st.error("🚨 Emergency Risk Detected")
        elif severity_level == "High":
            st.warning("⚠️ High Risk")
        elif severity_level == "Medium":
            st.info("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")

        st.markdown("### Risk Explanation")

        for reason in severity_reasons:
            st.write("- " + reason)

        if severity_level == "Emergency":
            st.error(
                "This system has detected possible emergency symptoms. "
                "Please seek immediate medical attention or contact emergency services."
            )

        # -----------------------------
        # DISEASE RESULT DISPLAY
        # -----------------------------
        if not results:
            st.error("No matching conditions found.")

        else:
            for idx, result in enumerate(results, start=1):
                with st.expander(
                    f"Possible Condition {idx}: {result['disease']}",
                    expanded=True
                ):
                    st.markdown(f"**Symptom Match Count:** {result['overlap']}")
                    st.markdown(f"**Match Ratio:** {result['match_ratio']:.2f}")

                    # -----------------------------
                    # PERSONALIZED NOTES
                    # -----------------------------
                    personalized_notes = personalize_recommendations(
                        result,
                        patient_profile
                    )

                    st.markdown("### 👤 Personalized Health Notes")
                    for note in personalized_notes:
                        st.write("- " + note)

                    # -----------------------------
                    # MEDICINE SAFETY CHECK
                    # -----------------------------
                    safety_notes = medicine_safety_check(
                        result,
                        patient_profile
                    )

                    st.markdown("### 💊 Medicine Safety Check")
                    for note in safety_notes:
                        st.write("- " + note)

                    # -----------------------------
                    # LAB TEST RECOMMENDATION
                    # -----------------------------
                    lab_tests = recommend_lab_tests(
                        result,
                        final_symptoms
                    )

                    st.markdown("### 🧪 Suggested Lab Tests")
                    for test in lab_tests:
                        st.write("- " + test)

                    # -----------------------------
                    # DOCTOR / SPECIALIST RECOMMENDATION
                    # -----------------------------
                    specialist, specialist_reason = recommend_specialist(
                        result,
                        final_symptoms
                    )

                    st.markdown("### 🧑‍⚕️ Recommended Doctor / Specialist")
                    st.write(f"**Specialist:** {specialist}")
                    st.write(f"**Reason:** {specialist_reason}")

                    # -----------------------------
                    # EXISTING OUTPUT
                    # -----------------------------
                    st.markdown("### Description")
                    st.write(result["description"])

                    st.markdown("### Common Symptoms")
                    st.write(", ".join(result["symptoms"][:8]))

                    st.markdown("### Medications")
                    st.write(", ".join(result["medications"][:6]))

                    st.markdown("### Precautions")
                    st.write(", ".join(result["precautions"][:6]))

                    st.markdown("### Diet Suggestions")
                    st.write(", ".join(result["diets"][:6]))

                    st.markdown("### Workouts")
                    st.write(", ".join(result["workouts"][:6]))

st.markdown("---")
st.caption(
    "Educational use only. This system does not replace professional medical advice, diagnosis, or treatment."
)