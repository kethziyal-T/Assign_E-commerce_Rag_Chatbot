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
PUBLIC_APP_URL = os.environ.get("PUBLIC_APP_URL", "").rstrip("/")
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
            temperature=0.0
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
    clean_phone = "918220079860"
    try:
        user_query = request.form.get('message', '').strip()
        if not user_query:
            return jsonify({"status": "empty", "answer": "Query payload cannot be empty."})

        final_answer = ""
        image_urls = []  
        query_lower = user_query.lower()

        raw_phone = str(getattr(config, 'WHATSAPP_NUMBER', '918220079860')).strip()
        clean_phone = re.sub(r'[^0-9]', '', raw_phone)

        if any(kw in query_lower for kw in ['crochet', 'toy', 'class', 'frock', 'woolen', 'baby', 'sweater', 'bag', 'purse', 'amigurumi', 'training', 'academy']):
            target_category = "crochet" 
        elif any(kw in query_lower for kw in ['painting', 'paint', 'canva', 'canvas', 'sketch', 'drawing', 'portrait', 'realistic', 'watercolor', 'commission', 'oil painting']):
            target_category = "painting"
        else:
            target_category = "crochet" if "crochet" in query_lower or "bag" in query_lower or "toy" in query_lower else "painting"

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
                    system_prompt = (
    f"You are a strict shop assistant. Answer ONLY using the Context below. "
    f"If the exact answer, price, or product is not in the Context, reply ONLY: "
    f"'Sorry, I don't have details on that right now.' Do NOT guess.\n\n"
    f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
)

                    ai_response = llm.invoke(system_prompt)
                    final_answer = ai_response.content
                else:
                    raise Exception("LLM offline")
            except Exception:
                final_answer = "We offer custom-made crochet items including beautiful baby clothing sets, boutique purses, and custom anime Amigurumi toys!" if target_category == "crochet" else "We specialize in gorgeous custom fine art canvas commissions, landscape paintings, and highly realistic pencil sketches."
            
            wa_text = f"Hello! I am reading your Crochet training specs and want to register." if target_category == "crochet" else f"Hello! I am interested in booking a Custom Painting Commission."
            whatsapp_url = f"https://wa.me/{clean_phone}?text={urllib.parse.quote(wa_text)}&preview_url=1"
        else:
            final_answer = "Yes! We specialize in gorgeous custom fine art canvas commissions, landscape paintings, acrylic artworks, and highly realistic pencil sketches." if target_category == "painting" else "Yes! We sell custom crochet woven items like sweaters, boutique bags, accessories, and cute amigurumi toys. Let me know what you would like us to make!"
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
            "whatsapp_url": f"https://wa.me/{clean_phone}"
        })

# =====================================================================
# REAL-TIME WHATSAPP WEBHOOK GATEWAY FOR LIVE CUSTOMER REPLIES
# =====================================================================
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", config.WHATSAPP_TOKEN)
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", config.PHONE_NUMBER_ID)
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "my_secure_token_123")

@app.route('/webhook', methods=['GET'])
def verify_whatsapp_webhook_gateway():
    """Handles webhook verification handshake with Meta Developer Platform."""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    print(f"🔍 Webhook Verification Attempt -> Mode: {mode}, Token: {token}")
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        print("✅ Webhook verified successfully by Meta Developer Platform!")
        return str(challenge), 200, {'Content-Type': 'text/plain'}
    
    print("❌ Webhook verification failed! Token mismatch.")
    return "Forbidden", 403

@app.route('/webhook', methods=['POST'])
def handle_incoming_whatsapp():
    """Processes incoming real-time WhatsApp messages via Meta Cloud API v27.0."""
    try:
        body = request.get_json(silent=True)
        print(f"📡 Webhook Hit Received! Payload: {body}")
        
        if not body or "object" not in body or body["object"] != "whatsapp_business_account":
            return jsonify({"status": "ignored"}), 200

        customer_phone = None
        default_fail_reply = "Hello! We offer premium custom handmade crochet items and fine art paintings. We are looking into your request right now!"
        
        entry_list = body.get("entry", [])
        for entry in entry_list:
            changes_list = entry.get("changes", [])
            for change_item in changes_list:
                change_val = change_item.get("value", {})
                messages_list = change_val.get("messages", [])
                
                for message_payload in messages_list:
                    customer_phone = message_payload.get("from")
                    msg_type = message_payload.get("type")
                    
                    if msg_type == "text":
                        text_data = message_payload.get("text", {})
                        customer_query = text_data.get("body", "").strip()
                        
                        print(f"📩 Live WhatsApp message from +{customer_phone}: '{customer_query}'")
                        
                        try:
                            # STEP 1: Execute Handwritten FAQ lookup logic
                                             
                            faq_match = utils.resolve_handwritten_faq(customer_query)
                            rag_reply = ""
                            image_attachment = None
                            
                            if faq_match and isinstance(faq_match, dict) and faq_match.get("status") == "success":
                                rag_reply = faq_match.get("answer", "")
                                raw_image_path = faq_match.get("image_path")
                                if raw_image_path and str(raw_image_path) != "None":
                                    clean_img = str(raw_image_path).lstrip("/")
                                    image_attachment = f"{PUBLIC_APP_URL}/static/{clean_img}" if PUBLIC_APP_URL else None
                            else:
                                # STEP 2: Fallback to category-filtered RAG via ChromaDB
                                query_lower = customer_query.lower()
                                target_category = "crochet" if any(kw in query_lower for kw in ['crochet', 'toy', 'class', 'frock', 'sweater', 'bag']) else "painting"
                                
                                search_filter = {"category": target_category}
                                search_results = []
                                if db:
                                    search_results = db.similarity_search(customer_query, k=2, filter=search_filter)
                                    if not search_results:
                                        search_results = db.similarity_search(customer_query, k=2)
                                
                                if search_results:
                                    context = "\n".join([doc.page_content for doc in search_results])
                                    meta = getattr(search_results[0], 'metadata', {})
                                    img_name = meta.get("image_name")
                                    img_cat = meta.get("category", target_category)
                                    if img_name:
                                        clean_img = str(img_name).lstrip("/")
                                        if "/" not in clean_img:
                                            clean_img = f"{img_cat}/{clean_img}"
                                        image_attachment = f"{PUBLIC_APP_URL}/static/{clean_img}" if PUBLIC_APP_URL else None 
                                    
                                    if llm is not None:
                                        system_prompt = (
                                            f"You are a strict shop assistant. Answer ONLY using the Context below. "
                                            f"If the exact answer, price, or product is not in the Context, reply ONLY: "
                                            f"'Sorry, I don't have details on that right now.' Do NOT guess.\n\n"
                                            f"Context:\n{context}\n\nQuestion: {customer_query}\nAnswer:"
                                        )
                                        ai_response = llm.invoke(system_prompt)
                                        rag_reply = ai_response.content
                                    else:
                                        rag_reply = "We offer bespoke custom handmade crochet items and fine art canvas paintings!"
                                else:
                                    rag_reply = "Yes! We sell custom handmade crochet items, sweaters, boutique bags, and fine art commissions. Let us know how we can help!"

                            if not rag_reply or not str(rag_reply).strip():
                                rag_reply = default_fail_reply
                                
                            # STEP 3: Dispatch Reply via Meta Cloud API v27.0
                            send_live_whatsapp_reply(customer_phone, rag_reply, image_url=image_attachment)
                            
                        except Exception as inner_pipeline_err:
                            print(f"⚠️ Internal RAG Pipeline Blocked: {inner_pipeline_err}")
                            send_live_whatsapp_reply(customer_phone, default_fail_reply)
                            
        return jsonify({"status": "success"}), 200

    except Exception as general_err:
        print(f"❌ Real-time Webhook Structural Parsing Error: {str(general_err)}")
        return jsonify({"status": "success"}), 200

# ==============================================================================
# 1. LIVE WHATSAPP REPLY ENGINE (DIRECT META CLOUD API DELIVERY)
# ==============================================================================
def send_live_whatsapp_reply(to_number, text_content, image_url=None):
    """Sends the text reply first, followed by the related image directly via Meta Cloud API."""

    # --- 1.1 Recipient Phone Number Validation ---
    clean_to_number = re.sub(r'[^0-9]', '', str(to_number))
    if not clean_to_number:
        print(f"❌ Outbound Error: Invalid recipient phone number provided: {to_number}")
        return

    # --- 1.2 Fallback Text Protection ---
    if not text_content or not str(text_content).strip():
        text_content = "Sorry, I do not have that specific information right now."
    else:
        text_content = str(text_content).strip()

    # --- 1.3 Meta Graph API Endpoint & Authorization Headers ---
    url = f"https://graph.facebook.com/v27.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    # --- 1.4 Step 1: Send Text Reply First ---
    text_payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": clean_to_number,
        "type": "text",
        "text": {"preview_url": False, "body": text_content}
    }

    try:
        res = requests.post(url, headers=headers, json=text_payload, timeout=10)
        print(f"📩 Step 1 - Meta Text API Response Code: {res.status_code}")
        if res.status_code != 200:
            print(f"⚠️ Text Delivery Error: {res.text}")
    except Exception as network_err:
        print(f"❌ Failed to send text to Meta: {network_err}")

    # --- 1.5 Step 2: Send Related Image After Text ---
    if image_url and str(image_url).startswith("https://"):
        img_payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_to_number,
            "type": "image",
            "image": {
                "link": image_url
            }
        }
        try:
            img_res = requests.post(url, headers=headers, json=img_payload, timeout=15)
            print(f"🖼️ Step 2 - Meta Image API Response Code: {img_res.status_code}")
            if img_res.status_code != 200:
                print(f"⚠️ Image Delivery Warning: {img_res.text}")
        except Exception as img_err:
            print(f"❌ Failed to send image to Meta: {img_err}")

# ==============================================================================
# 2. PRODUCTION & LOCAL SERVER BOOT ENGINE & Cloud Run uses Port 8080 by default
# ==============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Chatbot Server Starting on Port {port} with Public URL: {PUBLIC_APP_URL}")
    app.run(host="0.0.0.0", port=port, debug=False)