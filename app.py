import os
import base64
import aiohttp
import openai
from fastapi import FastAPI, Path, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from dotenv import load_dotenv
from PIL import Image
import io
import logging
import asyncio
import fal_client
import httpx
import mimetypes
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union, AsyncIterator
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.responses import Response, RedirectResponse
from datetime import datetime, timedelta
from fastapi import Depends, Cookie

# Pydantic model to validate input (you can add this to your existing models)
class ImageGenerationRequest(BaseModel):
    prompt: str
    width: Optional[int] = Field(default=1280, ge=64, le=4096)
    height: Optional[int] = Field(default=720, ge=64, le=4096)
    style: Optional[str] = Field(default="realistic_image")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Basic Auth for POC
security = HTTPBasic()

# Hardcoded credentials for POC
users_db = {
    "prabhudatta.tripathy@gmail.com": { 
        "username": "prabhudatta.tripathy@gmail.com",
        "password": "admin@123" 
    }
}


BASE_REMOTE_URL = "https://fal.media/files"

# User model
class User(BaseModel):
    username: str

# Dependency to get the current user from the session cookie
def get_current_user(session_key: Optional[str] = Cookie(None)):
    if session_key in session_keys:
        user_data = session_keys[session_key]
        if datetime.utcnow() < user_data["expiry"]:
            return User(username=user_data["username"])
    raise HTTPException(status_code=401, detail="Invalid or expired session key")

# Store session keys in memory
session_keys = {}

@app.post("/login")
def login(credentials: HTTPBasicCredentials = Depends(security), response: Response = Response()):
    # Retrieve user data from the database
    user_data = users_db.get(credentials.username)
    
    # Check if the username exists and the password matches
    if not user_data or credentials.password != user_data["password"]:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    # Generate a session key (or any other logic post-login)
    session_key = base64.b64encode(os.urandom(24)).decode('utf-8')
    session_keys[session_key] = {
        "username": credentials.username,
        "expiry": datetime.utcnow() + timedelta(minutes=30)
    }
    
    # Set the session key in a cookie
    response.set_cookie(key="session_key", value=session_key, httponly=True)
    return {"message": "Login successful"}

# Add CORS middleware
app.mount("/app", StaticFiles(directory="app"), name="app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # **Important:** Restrict this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys from environment
FAL_KEY = os.getenv("FAL_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables on startup
@app.on_event("startup")
def startup_event():
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set in environment variables.")
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")
    if not FAL_KEY:
        logger.error("FAL_KEY is not set in environment variables.")
        raise RuntimeError("FAL_KEY is not set in environment variables.")
    # Set OpenAI API key
    openai.api_key = OPENAI_API_KEY
    
    logger.info("Startup: OpenAI API key set.")


@app.get("/", dependencies=[Depends(get_current_user)])
def serve_frontend():
    """Serve the main HTML page"""
    logger.info("Serving frontend.")
    return FileResponse("app/index.html")


def get_image_type(image_content):
    """
    Detect image type using file signatures (magic numbers)

    Args:
        image_content (bytes): Image file content

    Returns:
        str: Image type (e.g., 'jpeg', 'png') or None
    """
    # Define file signatures for common image types
    signatures = {
        'jpeg': b'\xff\xd8\xff',
        'png': b'\x89PNG\r\n\x1a\n',
        'gif': b'GIF87a',
        'webp': b'RIFF',
    }

    # Check if content starts with any of the known signatures
    for img_type, signature in signatures.items():
        if image_content.startswith(signature):
            return img_type
    return None


def resize_image(image_content, size=(512, 512)):
    """
    Resize the image to the specified size while preserving the original format.

    Args:
        image_content (bytes): Original image file content.
        size (tuple): Desired image size.

    Returns:
        tuple: Resized image file content and its format.
    """
    try:
        with Image.open(io.BytesIO(image_content)) as img:
            original_format = img.format  # e.g., 'JPEG', 'PNG'
            img = img.convert("RGB")  # Ensure image is in RGB format
            img = img.resize(size, Image.ANTIALIAS)
            buffer = io.BytesIO()
            # Save in the original format to preserve MIME type
            img.save(buffer, format=original_format)
            resized_image = buffer.getvalue()
            return resized_image, original_format.lower()  # e.g., 'jpeg', 'png'
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to process the image.")


@app.post("/generate-prompt", dependencies=[Depends(get_current_user)])
def generate_prompt(image: UploadFile = File(...)):
    """
    Generate a descriptive prompt from the uploaded image using OpenAI GPT-4o-mini

    Args:
        image (UploadFile): Uploaded image file

    Returns:
        dict: Generated prompt
    """
    try:
        logger.info("Received image for prompt generation.")
        # Read image file
        image_content = image.file.read()

        # Set maximum image size (e.g., 5MB)
        MAX_SIZE = 5 * 1024 * 1024  # 5MB
        if len(image_content) > MAX_SIZE:
            logger.error("Image size exceeds 5MB limit.")
            raise HTTPException(status_code=400, detail="Image size exceeds 5MB limit.")

        # Determine image MIME type
        image_type = get_image_type(image_content)

        if not image_type:
            logger.error("Invalid image file type.")
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Resize the image to 512x512 while preserving format
        resized_image, image_format = resize_image(image_content, size=(512, 512))
        logger.info(f"Image resized to 512x512 in format {image_format.upper()}.")

        # Encode resized image in base64
        base64_image = base64.b64encode(resized_image).decode('utf-8')
        logger.info("Image encoded to base64.")

        # Prepare the message as per OpenAI's documentation
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert in analyzing images and generating highly detailed, structured textual descriptions suitable for recreating or manipulating visuals using generative tools like DALL·E or similar systems. Your task is to carefully examine the provided image and generate a comprehensive prompt that captures all visible details with precision. Ensure the following:"

                            "Accuracy and Objectivity:"

                            "Describe only what is present in the image without making assumptions or adding imaginative elements."
                            "Avoid biases or subjective commentary. Focus solely on observable details."
                            "Comprehensive Detail Extraction:"
                            "Provide a detailed account of the following:"

                            "Subjects: Describe people, animals, or objects, including their pose, orientation, facial expressions, clothing, accessories, and actions."
                            "Environment: Explain the setting, background details, furniture, decor, and visible objects in context."
                            "Lighting: Identify light sources, their direction, intensity, shadows, and reflections."
                            "Colors: Highlight the dominant and subtle colors in the image, including clothing, background, and props."
                            "Framing and Orientation: Specify the camera angle (e.g., front-facing, side view), shot type (e.g., close-up, medium shot), and subject placement within the frame."
                            "Security and Privacy Considerations:"

                            "Exclude identifiable details that could compromise privacy, such as text on personal documents, license plates, or other sensitive information."
                            "Refer to individuals in general terms (e.g., 'a young woman,' 'a man in casual attire') without implying specific identities or personal recognition."
                            "Prompt Structure:"

                            "Produce a single-paragraph description that is concise, grammatically correct, and formatted for direct use in generative tools."
                            "Avoid including any special characters, markdown, or additional formatting elements."
                            "Output Quality and Usability:"

                            "Ensure the prompt is actionable, meaning it contains sufficient detail to accurately recreate the scene in a generative AI tool."
                            "Maintain a balance between brevity and descriptiveness, focusing on key elements essential for visual reconstruction."
                            "Note: Description length should not exceed 900 characters."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{base64_image}"
                        }
                    },
                ],
            }
        ]

        logger.info("Sending request to OpenAI ChatCompletion API.")
        # Make the synchronous API call to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Ensure this is the correct model name as per OpenAI's documentation
            messages=messages,
            max_tokens=300
        )

        # Extract prompt from response
        prompt = response.choices[0].message['content'].strip()
        logger.info("Prompt generated successfully.")
        return {"prompt": prompt}

    except openai.error.OpenAIError as oe:
        logger.error(f"OpenAI API error: {str(oe)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(oe)}")
    except HTTPException as he:
        # Re-raise HTTPExceptions to be handled by FastAPI
        raise he
    except Exception as e:
        logger.error(f"Error generating prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {str(e)}")


# Update the generate_variants function
@app.post("/generate-variants", dependencies=[Depends(get_current_user)])
async def generate_variants(data: dict):
    """
    Generate image variants using FAL.AI's Recraft V3 model

    Args:
        data (dict): Request payload containing prompt

    Returns:
        JSONResponse: Generated image URLs
    """
    try:
        # Extract prompt from the incoming data
        prompt = data.get('prompt')
        height = data.get('height')
        width = data.get('width')
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        logger.info(f"Received prompt for variant generation: {prompt}")
        
        # Prepare the payload template
        # payload_template = {
        #     "prompt": prompt,
        #     "image_size": {
        #         "width": width,
        #         "height": height
        #         },
        #     "style": "realistic_image",
        #     "colors": []
        # }
        payload_template = {
        "prompt": prompt,
        "num_images": 4,
        "enable_safety_checker": "true",
        "safety_tolerance": "2",
        "output_format": "jpeg",
        "aspect_ratio": "16:9"
        }

        # Create tasks for parallel API calls
        tasks = []
        for _ in range(1):
            task = generate_single_variant(payload_template)
            tasks.append(task)
        
        # Wait for all tasks to complete
        logger.info("Sending parallel requests to FAL.AI Recraft V3 API.")
        results = await asyncio.gather(*tasks)
        
        # Collect all generated image URLs
        all_images = []
        for result in results:
            if result and 'images' in result:
                all_images.extend(result['images'])
        
        logger.info(f"Generated {len(all_images)} image variants successfully.")
        return JSONResponse(content={"images": all_images})

    except Exception as e:
        logger.error(f"Error generating variants: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
async def generate_single_variant(payload):
    """
    Generate a single image variant using fal-client
    
    Args:
        payload (dict): Request payload
    
    Returns:
        dict: API response or None
    """
    try:
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                # Ensure that update.logs is not None before iterating
                if update.logs is not None:
                    for log in update.logs:
                        print(log["message"])
            if isinstance(update, fal_client.Completed):
                # Ensure that update.logs is not None before iterating
                if update.logs is not None:
                    for log in update.logs:
                        print(log["message"])


        result = await fal_client.subscribe_async(
            "fal-ai/flux-pro/v1.1-ultra",
            arguments=payload,
            on_queue_update=on_queue_update,
        )

        for item in result.get("images", []):
            if "url" in item and item["url"].startswith("https://fal.media/files"):
                item["url"] = item["url"].replace("https://fal.media/files", "/getfiles")

        return result
    

    except Exception as e:
        logger.error(f"Error in single variant generation: {str(e)}")
        return None

async def fetch_file_from_remote(remote_url: str) -> AsyncIterator[bytes]:
    async with httpx.AsyncClient() as client:
        response = await client.get(remote_url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="File not found")
        for chunk in response.iter_bytes():
            yield chunk

@app.get("/getfiles/{file_path:path}")
async def proxy_file(file_path: str = Path(..., description="Path to the file")):
    remote_url = f"{BASE_REMOTE_URL}/{file_path}"
    mime_type, _ = mimetypes.guess_type(file_path)
    return StreamingResponse(
        fetch_file_from_remote(remote_url), 
        media_type=mime_type or "application/octet-stream"
    )


if __name__ == "__main__":
    logger.info("Starting FastAPI application.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

