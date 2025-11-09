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

# --- Frontend Route ---
@app.route('/')
def serve_frontend():
    if app.static_folder is None:
        return "Server configuration error: Static folder not found.", 500
    # This assumes your HTML file is named User_Interface.html and is in a 'static' folder
    # If not, adjust the 'static_folder' in app = Flask(__name__, static_folder='static')
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
    
    # --- THIS IS THE FIX ---
    # Get just the element symbols from the list of objects
    atom_symbols = [atom.get('element', 'X') for atom in atom_list]
    
    # Build chemical formula to help AI
    atom_counts = Counter(atom_symbols) # Count the symbols, not the objects
    # --- END OF FIX ---
    
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
    
    # Build chemical formula to help AI
    atom_counts = Counter(atom_list)
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
    
    atom_list = data['atoms']
    atom_symbols = [atom.get('element', 'X') for atom in atom_list]
    # Calculate chemical formula to help the AI
    atom_counts = Counter(atom_symbols)
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

    user_query = f"Analyze the molecule {formula}, based on this 0-indexed atom list: {str(atom_list)}"

    try:
        response = info_model.generate_content(user_query, generation_config=info_generation_config )
        
        # **This is the fix from our previous conversation**
        info_text = response.candidates[0].content.parts[0].text
        return jsonify(json.loads(info_text))
        
    except Exception as e:
        print(f"An error occurred calling the Gemini API for info: {e}")
        return jsonify({"error": f"AI info generation failed: {e}"}), 500


# --- Run the Server ---
if __name__ == '__main__':
    print("Starting Python Flask server for molecule prediction...")
    # Make sure the 'static' folder exists and User_Interface.html is inside it.
    app.run(port=5000, debug=True)