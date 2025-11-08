import os
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

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

# Model for JSON/Chemistry Predictions
CHEMISTRY_PROMPT = """
You are a chemistry expert. A user will provide a list of atoms. 
Your task is to predict the most likely stable bonding structure for these atoms.
Respond *only* with a JSON object that adheres to the provided schema. 
Do not include any other text or markdown formatting.
The 'from' and 'to' fields in the bonds should be 0-based indices 
corresponding to the user's input atom list.
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

# Model for Text/Fun Facts
text_model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

# --- Frontend Route ---
@app.route('/')
def serve_frontend():
    """
    Serves the User_Interface.html file from the 'static' folder.
    """
    # === FIX: Added check for None to satisfy Pylance/make code safer ===
    if app.static_folder is None:
        return "Server configuration error: Static folder not found.", 500
        
    return send_from_directory(app.static_folder, 'User_Interface.html')


# --- API Endpoint 1: Predict Bonds ---
@app.route('/api/predict_bonds', methods=['POST'])
def handle_predict_bonds():
    """
    Receives a list of atoms and returns predicted bonds from the AI.
    """
    if not API_KEY:
        return jsonify({"error": "Server is missing GEMINI_API_KEY"}), 500

    data = request.get_json()
    if not data or 'atoms' not in data:
        return jsonify({"error": "Invalid request: 'atoms' list missing"}), 400

    atom_list = data['atoms']
    if len(atom_list) < 2:
        return jsonify({"error": "At least 2 atoms are required"}), 400

    user_query = f"Atoms: {str(atom_list)}"
    
    try:
        response = json_model.generate_content(
            user_query,
            generation_config=json_generation_config
        )
        predicted_json = response.candidates[0].content.parts[0].text
        import json
        return jsonify(json.loads(predicted_json))

    except Exception as e:
        print(f"An error occurred calling the Gemini API: {e}")
        return jsonify({"error": f"AI prediction failed: {e}"}), 500

# --- API Endpoint 2: Get Fun Fact (NEW) ---
@app.route('/api/get_fun_fact', methods=['POST'])
def get_fun_fact():
    """
    Receives an element symbol and returns a fun fact.
    """
    if not API_KEY:
        return jsonify({"error": "Server is missing GEMINI_API_KEY"}), 500

    data = request.get_json()
    if not data or 'element' not in data:
        return jsonify({"error": "Invalid request: 'element' missing"}), 400

    element_name = data['element']
    
    # Simple prompt for a text-only response
    prompt = f"Tell me a single, one-sentence fun fact about the element {element_name}. Respond with only the fact, nothing else."

    try:
        response = text_model.generate_content(prompt)
        fact = response.candidates[0].content.parts[0].text
        return jsonify({"fact": fact.strip()})

    except Exception as e:
        print(f"An error occurred calling the Gemini API for a fact: {e}")
        return jsonify({"error": f"AI fact generation failed: {e}"}), 500


# --- Run the Server ---
if __name__ == '__main__':
    print("Starting Python Flask server for molecule prediction...")
    app.run(port=5000, debug=True)