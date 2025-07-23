import asyncio
from random import randint
from PIL import Image
import requests
from dotenv import get_key
import os
from time import sleep

def open_images(prompt):
    folder_path = r"Data"
    prompt = prompt.replace(" ", "_")
    Files = [f"{prompt}{i}.jpg" for i in range(1, 4)]

    for jpg_file in Files:
        image_path = os.path.join(folder_path, jpg_file)
        try:
            img = Image.open(image_path)
            print(f"Opening Image: {image_path}")
            img.show()
            sleep(1)
        except IOError:
            print(f"Error: {image_path} not found or cannot be opened.")

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
api_key = get_key('.env', 'HuggingFaceAPIKey')

if not api_key:
    print("ERROR: HuggingFaceAPIKey not found in .env file!")
    exit(1)

headers = {"Authorization": f"Bearer {api_key}"}

async def query(payload):
    try:
        response = await asyncio.to_thread(requests.post, API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            return None
        return response.content
    except Exception as e:
        print(f"Exception during API request: {e}")
        return None

async def generate_images(prompt: str):
    tasks = []
    for _ in range(3):
        payload = {
            "inputs": f"{prompt}, quality=4k, sharpness=maximum, Ultra High details, high resolution, seed={randint(0, 1000000)}",
        }
        tasks.append(asyncio.create_task(query(payload)))

    image_bytes_list = await asyncio.gather(*tasks)
    saved = False
    for i, image_bytes in enumerate(image_bytes_list):
        if image_bytes:
            os.makedirs("Data", exist_ok=True)
            filename = os.path.join("Data", f"{prompt.replace(' ', '_')}{i + 1}.jpg")
            with open(filename, "wb") as f:
                f.write(image_bytes)
            print(f"Saved image: {filename}")
            saved = True
        else:
            print(f"Image {i+1} generation failed.")
    return saved

def GenerateImages(prompt: str):
    saved = asyncio.run(generate_images(prompt))
    if saved:
        open_images(prompt)
    else:
        print("No images were generated.")

def main_loop():
    while True:
        try:
            with open(r"Frontend/Files/ImageGeneration.data", "r") as f:
                Data = f.read().strip()
            if not Data:
                sleep(1)
                continue

            parts = Data.split(",")
            if len(parts) != 2:
                print(f"Invalid data format: {Data}")
                sleep(1)
                continue

            Prompt, Status = parts[0].strip(), parts[1].strip()
            print(f"Read from file: Prompt='{Prompt}', Status='{Status}'")

            if Status == "True":
                print(f"Generating Images for prompt: '{Prompt}'")
                GenerateImages(prompt=Prompt)

                with open(r"Frontend/Files/ImageGeneration.data", "w") as f:
                    f.write("False,False")
                print("Image generation done, status reset.")
                break
            else:
                sleep(1)

        except FileNotFoundError:
            print("Data file not found, waiting...")
            sleep(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sleep(1)

if __name__ == "__main__":
    main_loop()
