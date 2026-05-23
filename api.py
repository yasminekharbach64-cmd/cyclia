from flask import Flask, request, jsonify, render_template
from agents.orchestrator import route_question
from logger import HealthChatLogger
from conversation_memory import ConversationMemory
import time
import traceback

app = Flask(__name__)
app.json.ensure_ascii = False


logger = HealthChatLogger()
memory = ConversationMemory()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    start_time = time.time()
    question = None
    
    try:
        data = request.json
        question = data.get("question", "")
        session_id = data.get("session_id", "default_user")
        
        
        memory.add_message(session_id, "user", question)
        
        
        answer = route_question(question)
        
        
        memory.add_message(session_id, "assistant", answer)
        
        
        response_time = time.time() - start_time
        
        
        logger.log_interaction(
            question=question,
            answer=answer,
            language="auto-detect",
            category="health",
            response_time=response_time,
            metadata={"model": "mistral", "session_id": session_id}
        )
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        # DETAILED ERROR LOGGING
        print("=" * 60)
        print("❌ ERROR OCCURRED:")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print("=" * 60)
        
        # Log error
        logger.log_error(
            error_type="api_error",
            error_message=str(e),
            question=question if question else None
        )
        return jsonify({"error": "حدث خطأ"}), 500


@app.route("/stats")
def stats():
    return jsonify(logger.get_stats())


@app.route("/conversation/<session_id>", methods=["GET"])
def get_conversation(session_id):
    """Get conversation history for a session"""
    history = memory.get_history(session_id)
    stats = memory.get_stats(session_id)
    
    return jsonify({
        "session_id": session_id,
        "history": history,
        "stats": stats
    })

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/conversation/<session_id>", methods=["DELETE"])
def clear_conversation(session_id):
    """Clear conversation history"""
    memory.clear_conversation(session_id)
    return jsonify({"message": "Conversation cleared", "session_id": session_id})

if __name__ == "__main__":
    app.run(debug=True)