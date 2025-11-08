import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration ---

# We'll get the API key from an environment variable for security.
# You would set this in your terminal: export GEMINI_API_KEY='Your_API_Key_Here'
# IMPORTANT: For this to work, you must set the API key in your environment.
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    # In a real app, you might exit or raise an exception
    # For this example, we'll continue so the server can run,
    # but API calls will fail until the key is set.
    
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring GenerativeAI: {e}")

# --- Flask App Setup ---
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing) to allow
# your HTML file to call the server from a different origin.
CORS(app)

# --- AI Model Setup ---
model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

# Define the JSON schema the AI must follow
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

# Define the System Prompt
SYSTEM_PROMPT = """
You are a chemistry expert. A user will provide a list of atoms. 
Your task is to predict the most likely stable bonding structure for these atoms.
Respond *only* with a JSON object that adheres to the provided schema. 
Do not include any other text or markdown formatting.
The 'from' and 'to' fields in the bonds should be 0-based indices 
corresponding to the user's input atom list.
"""

# Configure the generation config with the schema
generation_config = genai.GenerationConfig(
    response_mime_type="application/json",
    response_schema=CHEMISTRY_SCHEMA
)

system_instruction = {"parts": [{"text": SYSTEM_PROMPT}]}

# --- API Endpoint ---

@app.route('/api/predict_bonds', methods=['POST'])
def handle_predict_bonds():
    """
    Receives a list of atoms and returns predicted bonds from the AI.
    """
    if not API_KEY:
        return jsonify({"error": "Server is missing GEMINI_API_KEY"}), 500

    # 1. Get the list of atoms from the frontend's request
    data = request.get_json()
    if not data or 'atoms' not in data:
        return jsonify({"error": "Invalid request: 'atoms' list missing"}), 400

    atom_list = data['atoms']
    if len(atom_list) < 2:
        return jsonify({"error": "At least 2 atoms are required"}), 400

    # 2. Create the user query for the AI
    user_query = f"Atoms: {str(atom_list)}"
    
    # 3. Call the AI
    try:
        response = model.generate_content(
            user_query,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        # 4. Extract the JSON and send it back to the frontend
        # The AI is already forced to return JSON text, so we parse it.
        # Note: The 'text' will be a string of JSON.
        predicted_json = response.candidates[0].content.parts[0].text
        
        # We send the JSON *string* back, and the browser will parse it.
        # Or, we can parse it here and send it as a Flask JSON response.
        # Let's let Flask handle it.
        import json
        return jsonify(json.loads(predicted_json))

    except Exception as e:
        print(f"An error occurred calling the Gemini API: {e}")
        return jsonify({"error": f"AI prediction failed: {e}"}), 500

# --- Run the Server ---

if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    print("Starting Python Flask server for molecule prediction...")
    print("Access at http://127.0.0.1:5000")
    app.run(port=5000, debug=True)