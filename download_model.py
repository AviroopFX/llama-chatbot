import os
import requests
import argparse
from tqdm import tqdm

def download_file(url, destination):
    """Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file to
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def main():
    parser = argparse.ArgumentParser(description="Download a Llama model")
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama-2-7b-chat.Q4_K_M.gguf",
        help="Model filename to download"
    )
    parser.add_argument(
        "--url", 
        type=str, 
        default="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        help="URL to download the model from"
    )
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download the model
    destination = os.path.join("models", args.model)
    
    print(f"Downloading {args.model} from {args.url}")
    print(f"This may take a while depending on your internet connection...")
    
    try:
        download_file(args.url, destination)
        print(f"Model downloaded successfully to {destination}")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    main()
