import os
import sys
import re
import urllib.parse
from flask import Flask, render_template, request, jsonify, send_from_directory

# ChromaDB and Gemini Config Imports
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import config
import utils

# Setup system structural parent locations linked to your active assets inside 'static'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FIX: Ensuring static directory points to the exact absolute folder location path
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Initialize the Premium Flask Application Gateway
app = Flask(__name__)  

@app.route('/static/<path:subpath>')
def serve_shop_images(subpath):
    """
    Safely serves images from the static/ folder. 
    Verifies existence before serving to avoid pipeline disruptions.
    """
    return send_from_directory(STATIC_DIR, subpath)

# =====================================================================
# CHROMA VECTOR DATABASE INITIALIZATION
# =====================================================================
if not os.path.exists(config.DB_DIR):
    print(f"⚠️ Subsystem Error: ChromaDB files missing. Execute 'python ingest.py' first.")
    sys.exit(1)

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

# =====================================================================
# FLASK WEB INTERFACE ROUTING ENGINE
# =====================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_query = request.form.get('message', '').strip()
        if not user_query:
            return jsonify({"status": "empty", "answer": "Query payload cannot be empty."})

        final_answer = ""
        image_urls = []  
        whatsapp_url = None
        query_lower = user_query.lower()

        raw_phone = str(getattr(config, 'WHATSAPP_NUMBER', '1234567890')).strip()
        clean_phone = re.sub(r'[^0-9]', '', raw_phone)

        # -----------------------------------------------------------------
        # STEP 1: EXECUTE HANDWRITTEN FAQ LOOKUP LOGIC
        # -----------------------------------------------------------------
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
            
            # 🌟 EXPERT RAG IMAGE EXTRACTION FOR FAQ TURN
            if len(image_urls) == 0 and db:
                try:
                    target_category = "crochet" if "crochet" in query_lower or "bag" in query_lower or "toy" in query_lower else "painting"
                    search_filter = {"category": target_category} if target_category else None
                    img_results = db.similarity_search(user_query, k=3, filter=search_filter)
                    
                    for doc in img_results:
                        metadata = getattr(doc, 'metadata', {})
                        image_name = metadata.get("image_name") or metadata.get("image_path")
                        if image_name:
                            if "/" in str(image_name):
                                full_url = f"/static/{image_name}"
                            else:
                                img_cat = metadata.get("category", target_category)
                                full_url = f"/static/{img_cat}/{image_name}"
                            if full_url not in image_urls:
                                image_urls.append(full_url)
                except Exception as rag_img_err:
                    print(f"⚠️ FAQ Image Extraction from RAG failed: {rag_img_err}")

            current_cat = "crochet" if "crochet" in query_lower else "painting"
            if current_cat == "crochet":
                wa_text = "Hello! I am reading your Crochet training specs and want to register."
            else:
                wa_text = "Hello! I am interested in booking a Custom Painting Commission."
                
            whatsapp_url = f"https://wa.me{clean_phone}?text={urllib.parse.quote(wa_text)}"
            
            return jsonify({
                "status": "success",
                "answer": final_answer,
                "image_urls": image_urls,
                "whatsapp_url": whatsapp_url
            })

        # -----------------------------------------------------------------
        # STEP 2: FALLBACK TO CATEGORY-FILTERED RAG VIA CHROMADB
        # -----------------------------------------------------------------
        target_category = None
        if any(kw in query_lower for kw in [
            'crochet', 'toy', 'class', 'frock', 'woolen', 'baby', 'clothes set',
            'customise baby clothes', 'hair accessories', 'sweater', 'skirt',
            'baby clothe set', 'bag', 'purse', 'shoulder bag', 'bubble bag', 'bunny',
            'rabbit', 'cat', 'dog', 'sheep', 'amigurumi', 'shark', 'turtle',
            'sea turtle', 'flower', 'sunflower', 'mat', 'hair band', 'christmas',
            'ornaments', 'earrings', 'rings', 'training', 'academy'
        ]):

            target_category = "crochet"
        elif any(kw in query_lower for kw in [
            'painting', 'paint', 'canva', 'canvas', 'sketch', 'drawing', 'draw', 
            'pencil', 'portrait', 'realistic', 'watercolor', 'couple', 'sitting', 
            'coffee', 'cup', 'wine', 'pouring', 'cow', 'rooster', 'animal', 
            'nature', 'landscape', 'city', 'prague', 'hamburger', 'food', 
            'bubble', 'illusion', 'racing', 'paralympics', 'commission', 
            'oil painting', 'colour pencil', 'live painting', 'customised painting', 
            'wildlife', 'farm', 'scenery', 'perspective', 'pop art', 'event', 
            'beverage', 'human', 'figure'
        ]):

            target_category = "painting"

        search_filter = {"category": target_category} if target_category else None
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
            
            for doc in search_results:
                metadata = getattr(doc, 'metadata', {})
                image_name = metadata.get("image_name") or metadata.get("image_path")
                if image_name:
                    if "/" in str(image_name):
                        full_url = f"/static/{image_name}"
                    else:
                        img_cat = metadata.get("category", target_category or "crochet")
                        full_url = f"/static/{img_cat}/{image_name}"
                    if full_url not in image_urls:
                        image_urls.append(full_url)
            
            system_prompt = f"""You are an expert shop assistant. Answer the user's question using ONLY the provided context.
Context:
{context}

Question: {user_query}
Answer:"""
            
            try:
                ai_response = llm.invoke(system_prompt)
                final_answer = ai_response.content
            except Exception as llm_err:
                print(f"⚠️ Gemini API Overloaded (503 Catch): {llm_err}")
                if target_category == "crochet":
                    final_answer = "We offer custom-made crochet items including beautiful baby clothing sets, sweaters, boutique purses with protective lining, and custom anime Amigurumi toys!"
                else:
                    final_answer = "We specialize in gorgeous custom fine art canvas commissions, landscape paintings, acrylic artworks, and highly realistic pencil sketches."
            
            active_mode = target_category if target_category else "shop"
            wa_text = f"Hello, I have an inquiry regarding your {active_mode} portfolio!"
            whatsapp_url = f"https://wa.me{clean_phone}?text={urllib.parse.quote(wa_text)}"
        else:
            final_answer = "Yes! We sell custom crochet woven items like sweaters, boutique bags, accessories, and cute amigurumi toys. Let me know what you would like us to make!"
            whatsapp_url = f"https://wa.me{clean_phone}"

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
            "whatsapp_url": f"https://wa.me{clean_phone}"
        })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
