import os
import sys
import re
import requests 
import urllib.parse
from flask import Flask, render_template, request, jsonify, send_from_directory

# LangChain and Gemini Engine Configuration Maps
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import config
import utils

# Setup system structural parent locations linked to active assets inside static folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Initialize the Premium Flask Application Gateway Subsystem
app = Flask(__name__)  

@app.route('/static/<path:subpath>')
def serve_shop_images(subpath):
    """Safely serves physical craft images from the localized static directory."""
    return send_from_directory(STATIC_DIR, subpath)

# =====================================================================
# CHROMA VECTOR DATABASE INITIALIZATION AND PIPELINE COMPILATION
# =====================================================================
embeddings = None
db = None
llm = None

try:
    if os.path.exists(config.DB_DIR):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=config.GEMINI_API_KEY
        )

        db = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=config.DB_DIR
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash", 
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.2
        )
    else:
        print(f"⚠️ Subsystem Error: ChromaDB files missing. Execute 'python ingest.py' first.")
except Exception:
    pass

# =====================================================================
# FLASK WEB INTERFACE ROUTING ENGINE AND ENDPOINT MANAGEMENT
# =====================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # Defining fallback variables at the root level to prevent UnboundLocalError
    clean_phone = "918220079860"
    try:
        user_query = request.form.get('message', '').strip()
        if not user_query:
            return jsonify({"status": "empty", "answer": "Query payload cannot be empty."})

        final_answer = ""
        image_urls = []  
        query_lower = user_query.lower()

        # Safely pull and format target business configuration values
        raw_phone = str(getattr(config, 'WHATSAPP_NUMBER', '918220079860')).strip()
        clean_phone = re.sub(r'[^0-9]', '', raw_phone)

        # CENTRALIZED CATEGORY RESOLUTION (Eliminates repeated evaluation expressions)
        if any(kw in query_lower for kw in ['crochet', 'toy', 'class', 'frock', 'woolen', 'baby', 'sweater', 'bag', 'purse', 'amigurumi', 'training', 'academy']):
            target_category = "crochet" 
        elif any(kw in query_lower for kw in ['painting', 'paint', 'canva', 'canvas', 'sketch', 'drawing', 'portrait', 'realistic', 'watercolor', 'commission', 'oil painting']):
            target_category = "painting"
        else:
            target_category = "crochet" if "crochet" in query_lower or "bag" in query_lower or "toy" in query_lower else "painting"

        # LOCAL HELPER: Dynamic image metadata parser to avoid repeating loops
        def extract_images_from_docs(documents, active_cat):
            for doc in documents:
                metadata = getattr(doc, 'metadata', {})
                img_name = metadata.get("image_name") or metadata.get("image_path")
                if img_name:
                    resolved_url = f"/static/{img_name}" if "/" in str(img_name) else f"/static/{metadata.get('category', active_cat)}/{img_name}"
                    if resolved_url not in image_urls:
                        image_urls.append(resolved_url)

        # STEP 1: EXECUTE HANDWRITTEN FAQ LOOKUP LOGIC
        faq_match = None
        try:
            faq_match = utils.resolve_handwritten_faq(user_query)
        except Exception as faq_internal_err:
            print(f"⚠️ FAQ Execution Warning: {faq_internal_err}")

        if faq_match and isinstance(faq_match, dict) and faq_match.get("status") == "success":
            final_answer = faq_match.get("answer", "")
            raw_image_path = faq_match.get("image_path") 
            
            if raw_image_path and str(raw_image_path) != "None":
                image_urls.append(f"/static/{raw_image_path}")
            
            if len(image_urls) == 0 and db:
                try:
                    search_filter = {"category": target_category} if target_category else None
                    img_results = db.similarity_search(user_query, k=3, filter=search_filter)
                    extract_images_from_docs(img_results, target_category)
                except Exception as rag_img_err:
                    print(f"⚠️ FAQ Image Extraction from RAG failed: {rag_img_err}")

            wa_text = "Hello! I am reading your Crochet training specs and want to register." if target_category == "crochet" else "Hello! I am interested in booking a Custom Painting Commission."
            whatsapp_url = f"https://wa.me/{clean_phone}?text={urllib.parse.quote(wa_text)}"

            return jsonify({
                "status": "success",
                "answer": final_answer,
                "image_urls": image_urls,
                "whatsapp_url": whatsapp_url
            })

        # STEP 2: FALLBACK TO CATEGORY-FILTERED RAG VIA CHROMADB
        search_filter = {"category": target_category} if target_category != "None" else None
        search_results = []
        
        try:
            if db:
                search_results = db.similarity_search(user_query, k=3, filter=search_filter)
                if not search_results and search_filter:
                    search_results = db.similarity_search(user_query, k=3)
        except Exception as db_err:
            print(f"⚠️ ChromaDB Error: {db_err}")

        if search_results:
            context = "\n".join([doc.page_content for doc in search_results])
            extract_images_from_docs(search_results, target_category)
            
            try:
                if llm is not None:
                    system_prompt = f"You are an expert shop assistant. Answer the user's question using ONLY the provided context.\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"
                    ai_response = llm.invoke(system_prompt)
                    final_answer = ai_response.content
                else:
                    raise Exception("LLM offline")
            except Exception:
                final_answer = "We offer custom-made crochet items including beautiful baby clothing sets, boutique purses, and custom anime Amigurumi toys!" if target_category == "crochet" else "We specialize in gorgeous custom fine art canvas commissions, landscape paintings, and highly realistic pencil sketches."
            
            live_image_url = "https://githubusercontent.com"
            wa_text = f"Hello! I am reading your Crochet training specs and want to register.\n\nSample Design: {live_image_url}" if target_category == "crochet" else f"Hello! I am interested in booking a Custom Painting Commission.\n\nSample Design: {live_image_url}"
            # ✅ FIXED: Added missing forward slash after wa.me
            whatsapp_url = f"https://wa.me/{clean_phone}?text={urllib.parse.quote(wa_text)}&preview_url=1"
        else:
            final_answer = "Yes! We specialize in gorgeous custom fine art canvas commissions, landscape paintings, acrylic artworks, and highly realistic pencil sketches." if target_category == "painting" else "Yes! We sell custom crochet woven items like sweaters, boutique bags, accessories, and cute amigurumi toys. Let me know what you would like us to make!"
            # ✅ FIXED: Added missing forward slash after wa.me
            whatsapp_url = f"https://wa.me/{clean_phone}"

        return jsonify({
            "status": "success",
            "answer": final_answer,
            "image_urls": image_urls,
            "whatsapp_url": whatsapp_url
        })

    except Exception as general_crash:
        print(f"❌ Catch-All Intercept: {general_crash}")
        return jsonify({
            "status": "success",
            "answer": "We sell high-quality custom handmade crochet items along with bespoke fine art paintings!",
            "image_urls": [],
            # ✅ FIXED: Added missing forward slash after wa.me to prevent link corruption
            "whatsapp_url": f"https://wa.me/{clean_phone}"
        })

# =====================================================================
# REAL-TIME WHATSAPP WEBHOOK GATEWAY FOR LIVE CUSTOMER REPLIES
# =====================================================================
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", config.WHATSAPP_TOKEN)
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", config.PHONE_NUMBER_ID)
VERIFY_TOKEN = os.environ.get("WHATSAPP_WEBHOOK_VERIFY_TOKEN", "my_secure_token_123")

@app.route('/webhook', methods=['GET'])
def verify_whatsapp_webhook_gateway():
    """Handles Meta Developer Hub initial verification validation."""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        print("✅ Webhook verified successfully by Meta Developer Platform!")
        return challenge, 200
    return "Forbidden", 403

# =====================================================================
# REAL-TIME WHATSAPP WEBHOOK GATEWAY FOR LIVE CUSTOMER REPLIES
# =====================================================================
@app.route('/webhook', methods=['POST'])
def handle_incoming_whatsapp():
    """Listens to real-time client traffic, parses messages, extracts parameters, and submits replies."""
    body = request.get_json()
    
    # Trace dynamic structure mapping inside console log
    print(f"📡 Webhook Hit Received! Payload: {body}")
    
    if not body or "object" not in body or body["object"] != "whatsapp_business_account":
        return jsonify({"status": "ignored"}), 200

    customer_phone = None
    default_fail_reply = "Hello! We offer premium custom handmade crochet items and fine art paintings. We are looking into your request right now!"
    
    try:
        entry_list = body.get("entry", [])
        if entry_list and isinstance(entry_list, list) and len(entry_list) > 0:
            # Safely capture entry root component
            entry = entry_list[0] if isinstance(entry_list[0], dict) else {}
            changes_list = entry.get("changes", [])
            
            if changes_list and isinstance(changes_list, list) and len(changes_list) > 0:
                # Capture change details structure object
                change_item = changes_list[0] if isinstance(changes_list[0], dict) else {}
                change_val = change_item.get("value", {}) if isinstance(change_item, dict) else {}
                messages_list = change_val.get("messages", [])
                
                if messages_list and isinstance(messages_list, list) and len(messages_list) > 0:
                    # Safely map client message attributes 
                    message_payload = messages_list[0] if isinstance(messages_list[0], dict) else {}
                    customer_phone = message_payload.get("from")
                    
                    if message_payload.get("type") == "text":
                        text_data = message_payload.get("text", {}) if isinstance(message_payload.get("text"), dict) else {}
                        customer_query = text_data.get("body", "").strip()
                        
                        print(f"📩 Live WhatsApp message from +{customer_phone}: '{customer_query}'")
                        
                        try:
                            # Step 1: Structural verification using handwritten FAQ layer
                            faq_match = utils.resolve_handwritten_faq(customer_query)
                            if faq_match and faq_match.get("status") == "success":
                                rag_reply = faq_match.get("answer")
                            else:
                                if llm is not None:
                                    portfolio_info = utils.generate_output_string(None)
                                    rag_reply = utils.answer_user_msg(customer_query, portfolio_info)
                                else:
                                    rag_reply = "Fallback Triggered"
                                    
                            # Step 2: Fallback text mapping if response strings execute blank or yield fallbacks
                            if not rag_reply or str(rag_reply).strip() == "" or "fallback" in str(rag_reply).lower():
                                if any(kw in customer_query.lower() for kw in ["crochet", "class", "academy", "bag", "toy"]):
                                    rag_reply = "Hello! We specialize in premium custom-made crochet items including beautiful baby clothes, custom boutique bags, and specialized weekly masterclasses."
                                else:
                                    rag_reply = "Hello! We specialize in gorgeous custom fine art canvas commissions, landscape paintings, acrylic artworks, and highly realistic pencil sketches."
                                    
                            # Step 3: Dispatch final response string straight to target client via Meta Cloud Gateway
                            send_live_whatsapp_reply(customer_phone, rag_reply)
                            
                        except Exception as inner_pipeline_err:
                            print(f"⚠️ Internal RAG Pipeline Blocked: {inner_pipeline_err}")
                            send_live_whatsapp_reply(customer_phone, default_fail_reply)
                            
    except Exception as general_err:
        print(f"❌ Real-time Webhook Structural Parsing Error: {str(general_err)}")
        if customer_phone:
            send_live_whatsapp_reply(customer_phone, default_fail_reply)
            
    return jsonify({"status": "success"}), 200


# =====================================================================
# CORE OUTBOUND LIVE WEBHOOK TRANSMISSION PIPELINE TO META GRAPH API
# =====================================================================
def send_live_whatsapp_reply(to_number, text_content):
    """Submits standardized JSON structural outbound payloads straight to the verified Cloud Graph URL."""
    # ✅ FIXED: Corrected domain routing to graph.facebook.com and added missing forward slashes along with API versioning
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"

    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": False, "body": text_content}
    }
    try:
        res = requests.post(url, headers=headers, json=payload)
        print(f"📡 Meta Dashboard API Response Code: {res.status_code}")
        # 🌟 CHANGE 1: Always print raw response text to bypass json decoding failures
        print(f"📡 Meta Raw Server Response Body: {res.text}")

        if res.status_code != 200:
            try:
                print(f"⚠ Cloud API Error Details: {res.json()}")
            except Exception:
                pass
    except Exception as network_err:
        print(f"❌ Failed to reach Meta Graph servers: {network_err}")



# =====================================================================
# AUTOMATED NGROK INTEGRATION ENGINE AND FLASK BOOT ENGINE PROCESS
# =====================================================================
if __name__ == "__main__":
    try:
        from pyngrok import ngrok
        # Open a standard clean HTTP tunnel endpoint tracking our local port
        public_tunnel = ngrok.connect(5000)
        print("\n========================================================================")
        print("🚀 AUTOMATED NGROK HTTPS WEBHOOK TUNNEL ACTIVATED")
        print(f"🔗 Public URL Target: {public_tunnel.public_url}")
        print(f"📝 Copy this endpoint for Meta Webhook URL: {public_tunnel.public_url}/webhook")
        print("========================================================================\n")
    except Exception as ngrok_err:
        print(f"⚠️ Ngrok tunnel failed to start automatically: {ngrok_err}")
        
    # Standard server execution boot pattern with safety flags preventing duplicate loop tunnels
    app.run(debug=True, port=5000, use_reloader=False)
