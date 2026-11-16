import os
import json
from collections import defaultdict
from google import genai
import config

# ✅ File system storage paths
products_file = 'craft_products.json'
categories_file = 'craft_categories.json'
delimiter = "####"

# ✅ CRITICAL FIX: Client initialization linked to active config parameters
client = genai.Client(api_key=config.GEMINI_API_KEY)
MODEL_NAME = "gemini-3.8-flash"

def get_completion_from_messages(messages, model=MODEL_NAME, temperature=0.7, max_tokens=500):
    formatted_messages = []
    extracted_system_text = ""

    # 1. Isolate system instructions and format user payload turns cleanly
    for message in messages:
        if message.get("role") == "system":
            extracted_system_text = message.get("content", "")
        elif "content" in message:  
            role_map = "user" if message["role"] == "user" else "model"
            formatted_messages.append({
                "role": role_map,  
                "parts": [{"text": message["content"]}]  
            })

    try:
        # 2. DEFENSIVE PROTECTION LAYER: Build a single text prompt string safely
        prompt_content = ""
        if formatted_messages:
            # Safely grab the text out of the structured list format
            prompt_content = formatted_messages[0]["parts"][0]["text"]
        
        # Merge the system prompt text cleanly if it exists
        if extracted_system_text:
            prompt_content = f"{extracted_system_text}\n\n{prompt_content}"

        if not prompt_content:
            return "No input text query provided."

        # 3. Execute the single content generation call safely using modern syntax wrapping configurations
        response = client.models.generate_content(
            model=model,
            contents=prompt_content,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        return response.text.strip()
    except Exception as e:
        # Print the exact inner issue to the terminal screen so you can read the error live
        print(f"❌ Internal System Breakdown Details: {str(e)}")
        return "Sorry, I am unable to connect to the studio model workspace right now."



def create_categories():
    categories_dict = {
        "Painting Art Collection": [
            "3D_Bubble_illusion_Painting",
            "3point_perspective_city_sketch",
            "Acrylic_couple_sitting_painting",
            "Acrylic_cow_panting ",
            "Acrylic_painting_cup_coffee",
            "Acrylic_painting_cup_of_coffee1",
            "Acrylic_painting_hamburger",
            "Acrylic_painting_Prague_city",
            "Acyrlic_painting",
            "canva_paint",
            "Canva_wine_pouring_painting",
            "canvas_nature_landscape_painting",
            "classic_oil_painting",
            "Coffee_painting",
            "color_pencil_portrait_painting",
            "Custom_oil_portrait_painting",
            "oil_work_rooster_painting",
            "Paralympics_racing_painting ",
            "pencil_drawing_of_Morgan",
            "pencil_drawing",
            "Pencil_sketch_Padel",
            "Pencil_sketch_portrait_of_abraham",
            "Pencil_sketch_portrait_of_young_g",
            "Pencil_sketch_portrait",
            "Realistic_pencil_portrait",
            "Watercolor_portrait_Anna"
        ],
        "Crochet Woven Items": [
            "Baby_crochet_set",
            "Baby_crochet_set1",
            "Baby_crochet_set2",
            "baby_sweater",
            "Canva_shoulder_bag",
            "Crochet_bag1",
            "crochet_bag2",
            "crochet_bag3",
            "crochet_bag4",
            "crochet_bag6",
            "crochet_bags",
            "crochet_baby_accessories_set",
            "crochet_baby_boots",
            "crochet_bubble_bag ",
            "crochet_bunny_toy",
            "crochet_bunny",
            "Crochet_bunny1",
            "crochet_cat",
            "crochet_christmas_ornaments_set",
            "crochet_christmas_ornaments",
            "crochet_class",
            "crochet_dog",
            "crochet_dress",
            "crochet_ear_rings",
            "crochet_earrings",
            "crochet_earrings1",
            "crochet_flower_bag",
            "crochet_hooked_vest_backside",
            "crochet_hooked_vest",
            "crochet_play_food_item",
            "crochet_play_food_item1",
            "crochet_play_food_items",
            "crochet_play_food_items1",
            "crochet_Rabbit",
            "crochet_rainbow_skirt",
            "Crochet_sandles_babies",
            "crochet_sea_turtle",
            "crochet_shark",
            "crochet_sheep",
            "crochet_sunflower_mat",
            "crochet_toodler_frock",
            "Crochet_turtle",
            "crochet_winter_accessory_set",
            "customise_crochet_dress",
            "customised_baby_sweater",
            "Hair_band",
            "handmade_crochet_baby_clothset",
            "Handmade_crochet bag"
        ],
        "Crochet Training Academy": [
            "Crochet Training Class"
        ]
    }
    with open(categories_file, 'w') as file:
        json.dump(categories_dict, file, indent=4)
    return categories_dict

def get_categories():
    try:
        with open(categories_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return create_categories()


def create_products():
    """
    Initializes product details natively using the exact file names from your directory list
    to maximize vector similarity hits.
    """
    products = {}
    categories = get_categories()

    # --- 1. POPULATE PAINTING ART COLLECTION ---
    painting_extensions = {
        "3point_perspective_city_sketch": "jpeg",
        "Acrylic_painting_hamburger": "jpeg",
        "Acrylic_painting_Prague_city": "jpeg",
        "Canva_wine_pouring_painting": "jpeg",
        "color_pencil_portrait_painting": "jpeg",
        "Custom_oil_portrait_painting": "jpeg",
        "Pencil_sketch_Padel": "jpeg",
        "Pencil_sketch_portrait": "jpeg",
        "Realistic_pencil_portrait": "jpeg",
        "Watercolor_portrait_Anna": "jpeg"
    }

    for item in categories["Painting Art Collection"]:
        ext = painting_extensions.get(item, "jpg")
        
        clean_name = item.replace('_', ' ').strip().title()
        if "Panting" in clean_name:
            clean_name = clean_name.replace("Panting", "Painting")
            
        products[item] = {
            "name": clean_name,
            "category": "Painting Art Collection",
            "brand": "Custom Fine Arts",
            "type": "Physical Painting / Sketch",
            "features": ["Handcrafted", "High fidelity execution", "Varnish protected"],
            "description": f"Custom physical fine artwork created in alignment with the style of {clean_name}.",
            "price": 2000.00,
            "image_path": f"painting/{item}.{ext}"
        }

    # --- 2. POPULATE CROCHET WOVEN ITEMS ---
    crochet_extensions = {
        "crochet_cat": "jpg",
        "crochet_class": "jpg",
        "crochet_dog": "jpg",
        "crochet_ear_rings": "jpg",
        "crochet_rainbow_skirt": "jpg",
        "Crochet_sandles_babies": "jpg",
        "crochet_sheep": "jpg"
    }

    for item in categories["Crochet Woven Items"]:
        ext = crochet_extensions.get(item, "jpeg")
        clean_name = item.replace('_', ' ').strip().title()
        
        products[item] = {
            "name": clean_name,
            "category": "Crochet Woven Items",
            "brand": "Bespoke Yarn Works",
            "type": "Handmade Crochet",
            "features": ["100% Cotton Yarn", "Washable", "Hypoallergenic material"],
            "description": f"Premium handmade physical crochet product matching your portfolio asset {clean_name}.",
            "price": 850.00,
            "image_path": f"crochet/{item}.{ext}"
        }

    # --- 3. EXPLICIT TRAINING CLASS ENTRY WITH SEPARATE ASSETS AND PRICING ---
    products["Crochet Training Class"] = {
        "name": "Crochet Training Class",
        "category": "Crochet Training Academy",
        "brand": "Crochet Academy",
        "type": "Educational Course",
        "features": ["Step-by-step masterclass", "Reference designs manual included", "Includes practice yarn material kits"],
        "description": "Complete educational program for learning crochet techniques from beginner to expert level, featuring full access keys to our video guide files.",
        "price": 1500.00,
        "image_path": "crochet/crochet_class.jpg"
    }

    with open(products_file, 'w') as file:
        json.dump(products, file, indent=4)
    return products


def get_products():
    try:
        with open(products_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return create_products()


def get_product_list():
    return list(get_products().keys())


def get_products_and_category():
    products = get_products()
    products_by_category = defaultdict(list)
    for product_name, product_info in products.items():
        category = product_info.get('category', 'uncategorized')
        if category:
            products_by_category[category].append(product_name)
    return dict(products_by_category)

 
def find_category_and_product_only(user_input, products_and_category_data=None):
    if products_and_category_data is None:
        products_and_category_data = get_products_and_category()
        
    allowed_listings_str = ""
    for category, products_list in products_and_category_data.items():
        allowed_listings_str += f"- {category}:\n"
        for product in products_list:
            allowed_listings_str += f"  - {product}\n"

        system_message_content = f"""
    You will be provided with customer service queries.
    The query will be enclosed within {delimiter} characters.

    Extract both matching category configurations and specific products found in the Allowed Schema list below.

    ### Output Format:
    Output a Python list of objects, where each object follows this format exactly:
    [
        {{"category": "category Name", "products": ["Product 1", "Product 2"]}}
    ]

    ### Rules for Extraction:
    1. Categories & Products must be found or inferred from the user query.
    2. Products must be matched to the correct category.
    3. If a category is mentioned but no specific products, include ALL products from that category.
    4. If no relevant products or categories are found, return an empty list [].
    5. Ignore any irrelevant words or phrases not related to the predefined allowed listings.

    Allowed Products & Categories:

    {allowed_listings_str}
    """
    messages = [
    {"role": "system", "content": system_message_content},
    {"role": "user", "content": f"{delimiter}{user_input}{delimiter}"}
    ]
    return get_completion_from_messages(messages)

def get_product_by_name(name):
    return get_products().get(name, None)

def get_products_by_category(category):
    return [prod for prod in get_products().values() if prod.get("category") == category]

def get_mentioned_product_info(data_list):
    product_info_l = []
    if not data_list:
        return product_info_l
    for data in data_list:
        if isinstance(data, dict):
            if "products" in data:
                for product_name in data["products"]:
                    product = get_product_by_name(product_name)
                    if product: product_info_l.append(product)
                    elif "category" in data:
                        product_info_l.extend(get_products_by_category(data["category"]))
    return product_info_l

def read_string_to_list(input_data):
    if isinstance(input_data, list):
        return input_data
    if not input_data:
        return []
    try:
        cleaned_input = str(input_data).strip().strip("```").replace("json", "").strip()
        return json.loads(cleaned_input)
    except json.JSONDecodeError:
        return []
    
def generate_output_string(data_list):
    output_string = ""
    if not data_list:
        all_products = get_products()
        for product_name, product_info in all_products.items():
            output_string += json.dumps(product_info, indent=2) + "\n"
            return output_string
        for data in data_list:
            if isinstance(data, dict):
                if "products" in data:
                    for product_name in data["products"]:
                        product = get_product_by_name(product_name)
                        if product: output_string += json.dumps(product, indent=2) + "\n"
                        elif "category" in data:
                            for product in get_products_by_category(data["category"]):
                                output_string += json.dumps(product, indent=2) + "\n"
    return output_string    

def answer_user_msg(user_msg, product_info):
    system_message = (
    "You are an elegant AI Fashion Stylist and Craft Tutor for 'Ecommerce RAG Chatbot' boutique. "
    "Help clients register for classes, recommend custom styles, or explain commission details. "
    "Logistics: Crochet classes happen Saturdays 10-11am at Crossroad Library, Balbinova, Prague-2 for 250 CZK. "
    "Note: Painting courses do not exist. We only sell physical canvases and take art commission requests."
    )
    messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user','content': f"Bespoke Available Catalog/Workshop Details:\n{product_info}\n\nCustomer Inquiry:\n{delimiter}{user_msg}{delimiter}"}
    ]
    return get_completion_from_messages(messages, temperature=0.3)

def classify_user_query(user_msg):
    local_delimiter = "####"
    system_message_content = f"""
    You will be provided with a customer service conversation.
    The most recent user query will be delimited with {local_delimiter} characters.
    Output Format:
    Output a Python list containing a single JSON object, following this format exactly:
    [
    {{"primary": "Main Category", "secondary": "Sub-Category"}}
    ]
    Allowed Classifications:
    Fine Art (Sales & Commissions):
    • Finished painting purchase
    • Custom canvas commission
    • Art portfolio inquiry
    Custom Crochet Orders:
    • Custom toys & anime Amigurumi
    • Bags, purses, & accessories
    • Baby clothing & newborn apparel
    • Pet sweaters & custom items
    Crochet Academy (Classes):
    • Class schedule & location
    • Pricing & session costs
    • Material requirements & registration
    • Skill level or age groups
    General Inquiry & Support:
    • Contact details (Email/WhatsApp)
    • Feedback or complaints
    • Speak to a human
    Only output the list of objects without any extra text, spaces, explanations, or markdown blocks.
    """
    messages = [
    {'role': 'system', 'content': system_message_content},
    {'role': 'user', 'content': f"{local_delimiter}{user_msg}{local_delimiter}"}
    ]
    raw_response = get_completion_from_messages(messages, temperature=0.0)
    try:
        clean_response = raw_response.strip()
        if clean_response.startswith("```"):
            clean_response = clean_response.split("\n", 1)[1]
        if clean_response.endswith("```"):
            clean_response = clean_response.rsplit("\n", 1)[0]
            
        clean_response = clean_response.strip().replace("json", "", 1).strip()
        parsed_classification = json.loads(clean_response)
        
        if isinstance(parsed_classification, list) and len(parsed_classification) > 0:
            return parsed_classification[0]
            return parsed_classification

    except (json.JSONDecodeError, Exception):

        return {"primary": "General Inquiry & Support", "secondary": "Contact details (Email/WhatsApp)"}
    
    
def resolve_handwritten_faq(user_query):
    try:
        faq_matrix = get_handwritten_faq_data()
        query_lower = user_query.lower()
        
        # EXPERT DETERMINISTIC KEYWORD MAPPING (Bypasses 503 Overload Errors entirely)
        matched_key = None
        
        if "online" in query_lower or "virtual" in query_lower or "zoom" in query_lower:
            matched_key = "online_classes_policy"
        elif "offline" in query_lower or "library" in query_lower or "in-person" in query_lower or "physical" in query_lower:
            matched_key = "offline_vs_online"
        elif "who" in query_lower and ("join" in query_lower or "age" in query_lower or "kid" in query_lower or "children" in query_lower):
            matched_key = "who_can_join"
        elif "experience" in query_lower or "beginner" in query_lower or "know" in query_lower or "learn" in query_lower:
            matched_key = "experience_requirements"
        elif "stitch" in query_lower or "teach" in query_lower or "curriculum" in query_lower or "learn" in query_lower:
            matched_key = "class_curriculum"
        elif "time" in query_lower or "schedule" in query_lower or "when" in query_lower or "saturday" in query_lower:
            matched_key = "logistics_schedule"
        elif "where" in query_lower or "location" in query_lower or "address" in query_lower or "place" in query_lower:
            matched_key = "location_details"
        elif "price" in query_lower or "cost" in query_lower or "fee" in query_lower or "czk" in query_lower or "much" in query_lower:
            matched_key = "pricing_fees"
        elif "material" in query_lower or "hook" in query_lower or "yarn" in query_lower or "bring" in query_lower:
            matched_key = "material_requirements"
        elif "register" in query_lower or "book" in query_lower or "seat" in query_lower:
            matched_key = "registration_policy"
        elif "flexibility" in query_lower or "single" in query_lower or "one class" in query_lower:
            matched_key = "attendance_flexibility"
        elif "custom" in query_lower or "order" in query_lower or "commission" in query_lower or "email" in query_lower or "contact" in query_lower:
            matched_key = "custom_orders_and_commissions"
        elif "what" in query_lower and ("sell" in query_lower or "product" in query_lower or "item" in query_lower or "have" in query_lower or "bag" in query_lower or "toy" in query_lower or "sweater" in query_lower):
            matched_key = "specific_product_items"
        elif "painting class" in query_lower or "drawing class" in query_lower or "art workshop" in query_lower:
            matched_key = "painting_classes_restriction"

        # If a local keyword pattern matches, return it instantly!
        if matched_key in faq_matrix:
            faq_node = faq_matrix[matched_key]
            print(f"⚡ [FAST-TRACK MATCH] Local Keyword Router matched: {matched_key}")
            return {
                "status": "success",
                "key": matched_key,
                "answer": faq_node.get("answer", ""),
                "image_path": faq_node.get("image_path", None)
            }
            
        print("ℹ️ Local FAQ didn't match keywords. Passing query down to database context.")
        return {"status": "no_match", "answer": None, "image_path": None}
        
    except Exception as e:
        print(f"⚠️ Local FAQ Router Exception: {str(e)}")
        return {"status": "error", "answer": None, "image_path": None}


def get_handwritten_faq_data():
    return {
    "offline_vs_online": {
    "intent_description": "Inquiries about  physical/offline vs virtual/online classes.",
    "answer": "Yes, we absolutely provide offline, in-person classes! We meet every Saturday morning from 10:00 AM to 11:00 AM at the Crossroad Library (Balbinova, Prague-2). Please note that we do not offer online classes at this time, as we believe learning crochet is best experienced with hands-on, face-to-face guidance. We would love to have you join us at our physical workshop table!",
    "image_path": "crochet/crochet_class.jpg"
    },
    "who_can_join": {
    "intent_description": "Who can join the classes and age groups.",
    "answer": "Thank you so much for asking! Our crochet classes are completely open to everyone—from absolute beginners to intermediate crafters looking to refine their skills. Any and all age groups are warmly welcome, including both children and adults!",
    "image_path": "crochet/crochet_class.jpg"
    },
    "experience_requirements": {
    "intent_description": "Whether previous crochet experience or knowledge is required.",
    "answer": "Please don't worry at all! Previous crochet experience is not necessary. Our sessions are specifically designed to be perfectly suitable for complete beginners, and we will guide you step-by-step from the very start.",
    "image_path": "crochet/crochet_class.jpg"
    },
    "class_curriculum": {
    "intent_description": "What tools, stitches, or projects are taught in the lessons.",
    "answer": "In these classes, you will enjoy a full, supportive progression starting from basic foundational stitches all the way to executing advanced, custom projects of your choosing!",
    "image_path": "crochet/crochet_class.jpg"
    },
    "logistics_schedule": {
    "intent_description": "When and what time the classes start, finish, or are held.",
    "answer": "We would love to have you join us! Our crochet classes are held every single Saturday morning from 10:00 AM to 11:00 AM.",
    "image_path": "crochet/crochet_class.jpg"
    },
    "location_details": {
    "intent_description": "Where the classes are physically located.",
    "answer": "Our wonderful sessions are held in a cozy setting at the Crossroad Library, located at Balbinova, Prague-2. It is a fantastic environment for crafting together!",
    "image_path": "crochet/crochet_class.jpg"
    },
    "pricing_fees": {
    "intent_description": "Cost per class session or currency prices.",
    "answer": "The investment for each structural class session is exactly 250 CZK, which includes personalized, hands-on guidance throughout the hour.",
    "image_path": "crochet/crochet_class.jpg"
    },
    "material_requirements": {
    "intent_description": "Whether materials are provided or if students bring their own.",
    "answer": "To make your creation uniquely yours, we kindly ask that you bring along your own personal crochet materials—including your favorite choice of yarn and matching hooks to work with during the session!",
    "image_path": "crochet/crochet_class.jpg"
    },
    "registration_policy": {
    "intent_description": "Whether pre-booking or pre-registration is needed before showing up.",
    "answer": "Yes, please! Pre-registration is required before attending a class session. This helps us ensure that we save a dedicated, comfortable workspace just for you.",
    "image_path": "crochet/crochet_class.jpg"
    },
    "attendance_flexibility": {
    "intent_description": "Joining just a single class versus regular sequential attendance.",
    "answer": "We offer total flexibility to match your schedule! You can absolutely join us for just a single standalone class to master a specific technique, or choose to attend regularly to build your skills.",
    "image_path": "crochet/crochet_class.jpg"
    },
    "custom_orders_and_commissions": {
    "intent_description": "General custom orders, ordering process queries, contact information, how to reach out, email address, or phone number inquiries.",
    "answer": "I would be absolutely delighted to bring your ideas to life! I accept customized crochet orders tailored exactly to your personal style and preferences. You can easily place an order or join a class by writing to me at christi.gladish@gmail.com or via WhatsApp at +420 776756594.",
    "image_path": None
    },
    "specific_product_items": {
    "intent_description": "Queries about pet sweaters, customized bags, purses, or custom anime amigurumi characters.",
    "answer": "Yes, I handcraft a wide variety of custom pieces with love and care! I make cozy, tailored sweaters for pets, durable customized bags and purses with protective internal linings, and highly detailed custom anime characters. Feel free to share your design ideas or reference photos with me!",
    "image_path": None
    },
    "painting_classes_restriction": {
    "intent_description": "Inquiries asking about taking painting classes, canvas workshops, or drawing lessons.",
    "answer": "Thank you so much for reaching out to us! Please note that we specialize exclusively in fiber arts and only offer training classes for crochet making. We do not host any painting classes or drawing workshops at our academy. However, if you are looking to learn crochet, you are very welcome to join our weekly hands-on training workshop at the Crossroad Library (Balbinova, Prague-2) for 250 CZK every Saturday morning from 10:00 AM to 11:00 AM. Alternatively, if you are looking for paintings, I work strictly on a retail purchase or custom commission basis. As a professional figurative and fine artist, I create custom portraits, illustrative art, perspective art, landscapes, animal portraits, and abstract paintings. I work with high-quality mediums including oil paints, watercolors, graphite pencils, and colored pencils. Would you like me to help you reserve a workspace for our next crochet class, or would you like to discuss commissioning a custom fine art piece?",
    "image_path": None
    },
    "online_classes_policy": {
    "intent_description": "Inquiries about virtual lessons, digital courses, Zoom calls, internet sessions, or online classes.",
    "answer": "Thank you so much for your interest! Currently, we do not offer any online classes, as we prioritize a hands-on, face-to-face workshop environment to help refine your stitches in real time. We would absolutely love to have you join us at our in-person sessions held every Saturday morning at the Crossroad Library instead!",
    "image_path": "crochet/crochet_class.jpg"
    }
    }

