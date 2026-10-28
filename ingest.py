import os
import sys
import shutil
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import config
import utils

def clean_string_normalization(text: str) -> str:
    """Standardizes string variations by converting to lowercase and stripping spacers."""
    return text.lower().replace('_', '').replace('-', '').replace(' ', '').strip()

def build_vector_database():
    """
    Scans local static asset directories, maps metadata flexibly 
    with normalized keys, and builds a clean Chroma Vector DB index.
    """
    print("⏳ Initializing Standardized Vector Ingestion Subsystem...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=config.GEMINI_API_KEY
    )

    products = utils.get_products()
    
    # Store items inside a flexible lookup map using normalized matching keys
    prod_mapping = {clean_string_normalization(name): val for name, val in products.items()}

    titles = []
    metadatas = []
    image_paths = []

    target_folders = {
        "crochet": "static/crochet",
        "painting": "static/painting"
    }

    for category, folder_path in target_folders.items():
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Directory missing, skipping: {folder_path}")
            continue

        print(f"📁 Scanning items inside {category} folder...")
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, file)
                
                # Strip extension and clean up punctuation variations
                base_name = os.path.splitext(file)[0]
                normalized_file_key = clean_string_normalization(base_name)
                
                matched_info = None
                
                # High tolerance partial string validation loop
                for prod_key, prod_val in prod_mapping.items():
                    if prod_key in normalized_file_key or normalized_file_key in prod_key:
                        matched_info = {
                            "name": prod_val.get("name", file.capitalize()),
                            "price": prod_val.get("price", "Price on Request"),
                            "description": prod_val.get("description", "Premium custom handmade craft work item.")
                        }
                        break
                
                if not matched_info:
                    matched_info = {
                        "name": base_name.replace('_', ' ').replace('-', ' ').title(),
                        "price": "Price on Request",
                        "description": f"Beautiful handmade {category} artwork asset item."
                    }

                text_content = f"Product: {matched_info['name']}. Category: {category}. Details: {matched_info['description']}. Price: Rs.{matched_info['price']}."
                
                titles.append(text_content)
                image_paths.append(img_path)
                
                metadatas.append({
                    "category": category,
                    "image_name": file,  # Saves exact physical file name (e.g., Crochet_bag1.jpeg)
                    "title": matched_info['name'],
                    "price": str(matched_info['price'])
                })

    if not image_paths:
        print("❌ Action Cancelled: No image assets found inside static directories.")
        return

    if os.path.exists(config.DB_DIR):
        print("🧹 Clearing old Chroma index database storage records...")
        shutil.rmtree(config.DB_DIR)

    print(f"🚀 Injecting {len(titles)} structured documents into ChromaDB...")
    
    db = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.DB_DIR
    )
    
    db.add_texts(texts=titles, metadatas=metadatas)
    print(f"🎉 RAG Data Ingestion Completed Successfully! Index built inside '{config.DB_DIR}'.")

if __name__ == "__main__":
    build_vector_database()
