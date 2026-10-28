# services/gen_service.py
import os
import io
from PIL import Image
from google import genai
import config

def generate_ai_image(prompt: str):
    """
    Triggers Google Gemini Imagen 3 to generate a brand new image
    and immediately saves it inside the static folder for broken-free rendering.
    """
    try:
        if not config.GEMINI_API_KEY or "your-gemini" in config.GEMINI_API_KEY:
            return None, "Error: Valid GEMINI_API_KEY configuration is missing in config.py."
            
        # Instantiate official Google GenAI Client
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        
        print(f"\n🎨 Gemini Imagen 3 is processing your prompt layout. Please wait...")
        response = client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=dict(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="1:1"
            )
        )
        
        generated_image_bytes = response.generated_images[0].image.image_bytes
        img = Image.open(io.BytesIO(generated_image_bytes))
        
        # -----------------------------------------------------------------
        # FIX: Save inside the static folder so Flask can serve it clean!
        # -----------------------------------------------------------------
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # இமேஜை லோக்கல் ரூட்டில் சேமிக்காமல் static/crochet/ உள்ளே சேமிக்கிறோம்
        output_dir = os.path.join(base_dir, 'static', 'crochet')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_name = f"ai_output_{abs(hash(prompt))}.jpg"
        full_save_path = os.path.join(output_dir, file_name)
        
        img.save(full_save_path)
        print(f"✅ AI Image successfully saved locally at: {full_save_path}")
        
        # பிரண்ட்-எண்டிற்கு சரியான வெப் பாதையை ரிட்டர்ன் செய்கிறோம்
        web_image_url = f"/static/crochet/{file_name}"
        
        return web_image_url, "Success! Custom concept artwork generated."
        
    except Exception as e:
        return None, f"Gemini Image Generation subsystem failure: {str(e)}"