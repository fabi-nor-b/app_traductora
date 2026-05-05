"""
app.py - Servidor Flask para traducción Inglés → Español
Modelo: Qwen2-0.5B (local, sin GPU, sin APIs externas)
"""

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

app = Flask(__name__)

# ─── Carga del modelo (se hace UNA vez al iniciar el servidor) ───────────────
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

print(f"[INFO] Cargando modelo '{MODEL_NAME}'... (puede tardar unos minutos la primera vez)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # float32 para CPU
    device_map="cpu"
)
model.eval()

print("[INFO] Modelo cargado correctamente ✓")


# ─── Función de traducción ────────────────────────────────────────────────────
def translate_to_spanish(text: str) -> str:
    """
    Traduce texto del inglés al español usando Qwen2-0.5B-Instruct.
    Retorna la traducción como string.
    """
    # Prompt en formato chat para modelos Instruct
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translator. "
                "Translate the given English text to Spanish. "
                "Reply with ONLY the Spanish translation, nothing else. "
                "Do not add explanations, notes, or extra text."
            )
        },
        {
            "role": "user",
            "content": f"Translate to Spanish: {text}"
        }
    ]

    # Aplicar el template de chat del tokenizador
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenizar el prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # Generar la respuesta
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,       # baja temperatura = respuestas más deterministas
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    # Decodificar solo los tokens generados (no el prompt)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Limpieza básica: quitar líneas vacías iniciales
    result = result.strip().splitlines()[0] if result.strip() else result

    return result


# ─── Rutas Flask ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Página principal con la interfaz de traducción."""
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    """
    Endpoint POST que recibe JSON con campo 'text'
    y devuelve JSON con campo 'translation'.
    """
    data = request.get_json()

    # Validación de entrada
    if not data or "text" not in data:
        return jsonify({"error": "No se recibió texto para traducir."}), 400

    text = data["text"].strip()

    if not text:
        return jsonify({"error": "El campo de texto está vacío."}), 400

    if len(text) > 500:
        return jsonify({"error": "El texto es demasiado largo (máximo 500 caracteres)."}), 400

    try:
        translation = translate_to_spanish(text)
        return jsonify({"translation": translation})
    except Exception as e:
        print(f"[ERROR] Fallo en la traducción: {e}")
        return jsonify({"error": "Ocurrió un error interno al traducir. Intenta de nuevo."}), 500


# ─── Punto de entrada ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
