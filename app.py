from flask import Flask, request, jsonify
from flask_cors import CORS
import os, uuid
import openai

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_API_KEY = "OPEN_AI_Key"  # Replace with your key
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

openai.api_key = OPENAI_API_KEY

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Transcribe audio with Whisper
# -----------------------------
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    # transcript.text has the text
    return {
        "text": transcript.text,
        "language": getattr(transcript, "language", "unknown")
    }

# -----------------------------
# Refine text using GPT
# -----------------------------
def refine_text(text, language):
    prompt = (
        f"Refine the following text in {language}. "
        f"Fix any misheard words or grammar, but keep the SAME language:\n\n{text}"
    )
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# Translate text using GPT
# -----------------------------
def translate_text(text, target_language):
    prompt = (
        f"Translate the following text to {target_language} language. "
        f"Keep the meaning accurate and proper grammar:\n\n{text}"
    )
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio file provided"}), 400

    filename = f"{uuid.uuid4()}_{audio.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    audio.save(path)

    try:
        result = transcribe_audio(path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.route("/refine", methods=["POST"])
def refine():
    data = request.json
    text = data.get("text")
    language = data.get("language", "same language")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        refined = refine_text(text, language)
        return jsonify({"refined_text": refined})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text")
    target = data.get("target")
    if not text or not target:
        return jsonify({"error": "Text or target language missing"}), 400

    # Make target language human-readable for GPT
    lang_map = {
        "en": "English",
        "es": "Spanish",
        "te": "Telugu",
        "hi": "Hindi",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "zh": "Chinese",
        "pt": "Portuguese"
    }
    target_lang = lang_map.get(target, target)

    try:
        translated = translate_text(text, target_lang)
        return jsonify({"translated_text": translated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Audio-to-Text API running!"

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
