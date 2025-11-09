import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import Counter

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring GenerativeAI: {e}")

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static')
CORS(app) 

# --- AI Model Setup ---

# Model 1: JSON/Chemistry Predictions
CHEMISTRY_PROMPT = """
You are a chemistry expert. A user will provide a list of atoms.
Each atom will have an 'element' and its grid 'row' and 'col'.
Your task is to predict ALL chemical bonds for the most likely stable structure.
Use the spatial information (row, col) to inform your predictions: atoms that are closer together are much more likely to be bonded.
Respond *only* with a JSON object that adheres to the provided schema.
Do not include any other text or markdown formatting.
The 'from' and 'to' fields in the bonds MUST be 0-based indices
corresponding to the user's input atom list.

Example user query: "Predict all bonds for the molecule CH4, based on this 0-indexed atom list: ['C', 'H', 'H', 'H', 'H']"
Example JSON response:
{
  "bonds": [
    {"from": 0, "to": 1, "type": "SINGLE"},
    {"from": 0, "to": 2, "type": "SINGLE"},
    {"from": 0, "to": 3, "type": "SINGLE"},
    {"from": 0, "to": 4, "type": "SINGLE"}
  ]
}
"""
json_model = genai.GenerativeModel(
    'gemini-2.5-flash-preview-09-2025',
    system_instruction=CHEMISTRY_PROMPT
)
CHEMISTRY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "bonds": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "from": {"type": "INTEGER"},
                    "to": {"type": "INTEGER"},
                    "type": {"type": "STRING"}
                },
                "required": ["from", "to", "type"]
            }
        }
    },
    "required": ["bonds"]
}
json_generation_config = genai.GenerationConfig(
    response_mime_type="application/json",
    response_schema=CHEMISTRY_SCHEMA
)

# Model 2: Text/Fun Facts
text_model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

# Model 3: Molecule Info
INFO_PROMPT = """
You are a brilliant chemist and data analyst. A user will provide a list of atoms.
Your task is to analyze this list and return structured information about the most
likely stable molecule they form.
If the atoms do not form a viable, well-known molecule, set 'common_name' to 'N/A' 
but still calculate the formula and molar mass.
Respond *only* with a JSON object.
"""
INFO_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "chemical_formula": { "type": "STRING" },
        "common_name": { "type": "STRING" },
        "molar_mass_g_mol": { "type": "NUMBER" }
    },
    "required": ["chemical_formula", "common_name", "molar_mass_g_mol"]
}
info_model = genai.GenerativeModel(
    'gemini-2.5-flash-preview-09-2025',
    system_instruction=INFO_PROMPT
)
info_generation_config = genai.GenerationConfig(
    response_mime_type="application/json",
    response_schema=INFO_SCHEMA
)

# Model 4: Structure Analysis (NEW)
ANALYZE_PROMPT = """
You are a meticulous chemistry expert. A user will provide a JSON object describing a molecule
with atoms (and their IDs) and bonds (connecting atom IDs).
Your task is to analyze this structure and return a JSON object.

**YOUR ANALYSIS MUST FOLLOW THESE RULES:**

**1. Bond Type Analysis (For each bond):**
* **Rule:** A bond is 'IONIC' if it is between a metal and a non-metal.
* **Rule:** A bond is 'COVALENT' if it is between two non-metals.
* (Metals include: Al, Na, K, Mg, Ca, Fe. Non-metals include: C, H, O, N, Cl, F, S, P, I, Br).

**2. Atom Analysis (For each atom):**

* **If the atom is part of an IONIC bond (Metal + Non-metal):**
    * `formal_charge`: This is the atom's typical ionic charge.
        * Examples: Na is +1, Mg is +2, Al is +3.
        * Examples: Cl is -1, O is -2, N is -3.
    * `electrons_shared_or_given`: This is the number of electrons the atom *gives* (for metals) or *takes* (for non-metals).
        * Example: For AlCl3, Al 'gives' 3, so its value is 3. Each Cl 'takes' 1, so its value is 1.
        * Example: For NaCl, Na 'gives' 1, so its value is 1. Cl 'takes' 1, so its value is 1.

* **If the atom is part of a COVALENT bond (Non-metal + Non-metal):**
    * `formal_charge`: Calculate the standard formal charge based on the number of bonds (e.g., in O3, the central O with 3 bonds is +1, the single-bonded O is -1, the double-bonded O is 0).
    * `electrons_shared_or_given`: This is the total number of electrons that atom is *sharing* in its bonds (e.g., a single bond = 2, a double bond = 4. An atom with one single and one double bond is sharing 6).

* **Priority:** The IONIC rules take priority. If you see Al bonded to Cl, treat all bonds as 'IONIC' even if the user drew them as covalent.

Respond *only* with a JSON object.
"""
ANALYZE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "atoms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "atom_id": {"type": "INTEGER"},
                    "formal_charge": {"type": "INTEGER"},
                    "electrons_shared_or_given": {"type": "INTEGER"}
                },
                "required": ["atom_id", "formal_charge", "electrons_shared_or_given"]
            }
        },
        "bonds": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "bond_id": {"type": "INTEGER"},
                    "type": {"type": "STRING"}
                },
                "required": ["bond_id", "type"]
            }
        }
    },
    "required": ["atoms", "bonds"]
}
analyze_model = genai.GenerativeModel(
    'gemini-2.5-flash-preview-09-2025',
    system_instruction=ANALYZE_PROMPT
)
analyze_generation_config = genai.GenerationConfig(
    response_mime_type="application/json",
    response_schema=ANALYZE_SCHEMA
)

# --- Frontend Route ---
@app.route('/')
def serve_frontend():
    if app.static_folder is None:
        return "Server configuration error: Static folder not found.", 500
    return send_from_directory(app.static_folder, 'User_Interface.html')

# --- API Endpoint 1: Predict Bonds ---
@app.route('/api/predict_bonds', methods=['POST'])
def handle_predict_bonds():
    if not API_KEY:
        return jsonify({"error": "Server is missing GEMINI_API_KEY"}), 500
    data = request.get_json()
    if not data or 'atoms' not in data:
        return jsonify({"error": "Invalid request: 'atoms' list missing"}), 400
    
    atom_list = data['atoms'] # This is now a list of objects
    
    if len(atom_list) < 2:
        return jsonify({"error": "At least 2 atoms are required"}), 400
    
    # Get just the element symbols from the list of objects
    atom_symbols = [atom.get('element', 'X') for atom in atom_list]
    
    # Build chemical formula to help AI
    atom_counts = Counter(atom_symbols) # Count the symbols, not the objects
    
    formula = ""
    if 'C' in atom_counts:
        count = atom_counts.pop('C')
        formula += f"C{count if count > 1 else ''}"
    if 'H' in atom_counts:
        count = atom_counts.pop('H')
        formula += f"H{count if count > 1 else ''}"
    for element in sorted(atom_counts.keys()):
        count = atom_counts[element]
        formula += f"{element}{count if count > 1 else ''}"

    # Send the FULL atom_list (with coordinates) to the AI
    user_query = f"Predict all bonds for the molecule {formula}, based on this 0-indexed atom list with grid coordinates: {str(atom_list)}"
    try:
        response = json_model.generate_content(
            user_query,
            generation_config=json_generation_config
        )
        predicted_json_text = response.candidates[0].content.parts[0].text
        return jsonify(json.loads(predicted_json_text))
    except Exception as e:
        print(f"An error occurred calling the Gemini API for bonds: {e}")
        return jsonify({"error": f"AI prediction failed: {e}"}), 500
    
# --- API Endpoint 2: Get Fun Fact ---
@app.route('/api/get_fun_fact', methods=['POST'])
def get_fun_fact():
    if not API_KEY:
        return jsonify({"error": "Server is missing GEMINI_API_KEY"}), 500
    data = request.get_json()
    if not data or 'element' not in data:
        return jsonify({"error": "Invalid request: 'element' missing"}), 400
    element_name = data['element']
    prompt = f"Tell me a single, one-sentence fun fact about the element {element_name}. Respond with only the fact, nothing else."
    try:
        response = text_model.generate_content(prompt)
        fact = response.candidates[0].content.parts[0].text
        return jsonify({"fact": fact.strip()})
    except Exception as e:
        print(f"An error occurred calling the Gemini API for fact: {e}")
        return jsonify({"error": f"AI fact generation failed: {e}"}), 500

# --- API Endpoint 3: Get Molecule Info ---
@app.route('/api/get_molecule_info', methods=['POST'])
def get_molecule_info():
    if not API_KEY:
        return jsonify({"error": "Server is missing GEMINI_API_KEY"}), 500
    data = request.get_json()
    if not data or 'atoms' not in data:
        return jsonify({"error": "Invalid request: 'atoms' list missing"}), 400
    
    atom_list = data['atoms'] # This is a list of objects
    
    # === THIS IS THE FIX ===
    # We must extract the symbols from the objects before counting
    atom_symbols = [atom.get('element', 'X') for atom in atom_list]
    # =======================

    # Calculate chemical formula to help the AI
    atom_counts = Counter(atom_symbols) # Now this counts symbols (e.g., 'C', 'H')
    formula = ""
    if 'C' in atom_counts:
        count = atom_counts.pop('C')
        formula += f"C{count if count > 1 else ''}"
    if 'H' in atom_counts:
        count = atom_counts.pop('H')
        formula += f"H{count if count > 1 else ''}"
    for element in sorted(atom_counts.keys()):
        count = atom_counts[element]
        formula += f"{element}{count if count > 1 else ''}"

    # Send the original list of atom *objects* to the AI
    user_query = f"Analyze the molecule {formula}, based on this 0-indexed atom list: {str(atom_list)}"

    try:
        response = info_model.generate_content(user_query, generation_config=info_generation_config )
        
        info_text = response.candidates[0].content.parts[0].text
        return jsonify(json.loads(info_text))
        
    except Exception as e:
        print(f"An error occurred calling the Gemini API for info: {e}")
        return jsonify({"error": f"AI info generation failed: {e}"}), 500

# --- API Endpoint 4: Analyze Structure (NEW) ---
@app.route('/api/analyze_structure', methods=['POST'])
def analyze_structure():
    if not API_KEY:
        return jsonify({"error": "Server is missing GEMINI_API_KEY"}), 500
    data = request.get_json()
    if not data or 'atoms' not in data or 'bonds' not in data:
        return jsonify({"error": "Invalid request: 'atoms' and 'bonds' lists missing"}), 400
    
    atom_list = data['atoms']

    if not atom_list:
         return jsonify({"error": "No atoms to analyze"}), 400

    user_query = f"Analyze this molecule structure: {json.dumps(data)}"

    try:
        response = analyze_model.generate_content(
            user_query, 
            generation_config=analyze_generation_config 
        )
        analysis_text = response.candidates[0].content.parts[0].text
        return jsonify(json.loads(analysis_text))
        
    except Exception as e:
        print(f"An error occurred calling the Gemini API for analysis: {e}")
        return jsonify({"error": f"AI analysis failed: {e}"}), 500

# --- Run the Server ---
if __name__ == '__main__':
    print("Starting Python Flask server for molecule prediction...")
    app.run(port=5000, debug=True)