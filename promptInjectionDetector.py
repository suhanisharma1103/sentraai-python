from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = Flask(__name__)
CORS(app)

# === Thresholds ===
HIGH_CONFIDENCE_THRESHOLD = 0.75   # block outright
LOW_CONFIDENCE_THRESHOLD = 0.45    # warn but allow through

# === Load training data ===
print("Loading training data...")
df = pd.read_csv("train.csv")
df = df.sample(n=1000, random_state=42)
documents = [
    Document(page_content=text, metadata={"label": int(label)})
    for text, label in zip(df["text"].astype(str), df["label"].astype(int))
]

# === Create embeddings and in-memory Chroma ===
print("Creating embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Initializing Chroma in-memory...")

batch_size = 500
vectordb = None
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    if vectordb is None:
        vectordb = Chroma.from_documents(batch, embedding=embedding_model)
    else:
        vectordb.add_documents(batch)
    print(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents...")

print("Chroma ready!")

@app.route("/check", methods=["POST"])
def check_prompt():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request"}), 400

    message = data["message"]

    # Perform similarity search
    try:
        results = vectordb.similarity_search_with_score(message, k=5)
    except Exception as e:
        return jsonify({"error": f"Vector search failed: {str(e)}"}), 500

    if not results:
        return jsonify({"status": "unknown", "reason": "No similar examples found"}), 200

    # Count malicious votes
    malicious_votes = sum(int(doc.metadata.get("label", 0)) for doc, _ in results)
    total = len(results)

    # Confidence = ratio of malicious votes out of total
    confidence = malicious_votes / total

    # Determine status based on confidence thresholds
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        status = "malicious"
    elif confidence >= LOW_CONFIDENCE_THRESHOLD:
        status = "warning"
    else:
        status = "safe"

    return jsonify({
        "message": message,
        "status": status,
        "confidence": round(confidence, 2),
        "malicious_votes": malicious_votes,
        "total_considered": total,
        "similar_examples": [
            {
                "text": doc.page_content,
                "label": doc.metadata.get("label", 0),
                "score": score
            }
            for doc, score in results
        ]
    })

if __name__ == "__main__":
    print("Starting Flask server on port 9000...")
    app.run(host="0.0.0.0", port=9000)