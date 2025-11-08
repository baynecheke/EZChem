import os
import google.generativeai as genai
# === ADD THIS IMPORT ===
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
# === TELL FLASK ABOUT THE 'static' FOLDER ===
app = Flask(__name__, static_folder='static')
CORS(app) 

# --- AI Model Setup (Your code, all good) ---

# Define the System Prompt *before* the model
SYSTEM_PROMPT = """
You are a chemistry expert. A user will provide a list of atoms. 
Your task is to predict the most likely stable bonding structure for these atoms.
Respond *only* with a JSON object that adheres to the provided schema. 
Do not include any other text or markdown formatting.
The 'from' and 'to' fields in the bonds should be 0-based indices 
corresponding to the user's input atom list.
"""

# === FIX 1: Pass the system prompt string directly to the model constructor ===
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    system_instruction=SYSTEM_PROMPT
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

generation_config = genai.GenerationConfig(
    response_mime_type="application/json",
    response_schema=CHEMISTRY_SCHEMA
)

# === FIX 2: This dictionary is no longer needed ===
# system_instruction = {"parts": [{"text": SYSTEM_PROMPT}]}


# --- === ADD THIS NEW ROUTE TO SERVE THE HTML FILE === ---
@app.route('/')
def serve_frontend():
    """
    Serves the User_Interface.html file from the 'static' folder.
    """
    # === FIX FOR PYLANCE ERROR ===
    # Pylance correctly notes that app.static_folder *could* be None.
    # We add a check to handle this, even though our app's setup
    # ensures it will always be 'static'.
    if app.static_folder is None:
        # This should never be reached in our app
        return "Server configuration error: Static folder not found.", 500
        
    return send_from_directory(app.static_folder, 'User_Interface.html')


# --- API Endpoint (Your existing code, all good) ---
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
        # === FIX 3: Remove the invalid system_instruction argument ===
        response = model.generate_content(
            user_query,
            generation_config=generation_config
        )
        predicted_json = response.candidates[0].content.parts[0].text
        import json
        return jsonify(json.loads(predicted_json))

    except Exception as e:
        print(f"An error occurred calling the Gemini API: {e}")
        return jsonify({"error": f"AI prediction failed: {e}"}), 500

# --- Run the Server ---
if __name__ == '__main__':
    # Render will use the gunicorn command instead of this
    print("Starting Python Flask server for molecule prediction...")
    print("Access at http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
