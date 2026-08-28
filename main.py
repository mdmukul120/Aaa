import os
import json
import re
from datetime import datetime
from pathlib import Path
import config
from scraper import MHDTVScraper
from image_generator import FallbackImageGenerator

def sanitize_filename(name: str) -> str:
    """ ফাইল সেভ করার নিরাপদ নাম জেনারেট করে """
    clean = re.sub(r'[^\w\s-]', '', name).strip().lower()
    return re.sub(r'[-\s]+', '_', clean)[:30]

def cleanup_ended_matches(active_image_paths: set):
    """
    ম্যাচ শেষ হয়ে গেলে ফোল্ডার থেকে পুরাতন ডিলিট হওয়া ইমেজের ফাইল মুছে ফেলে।
    """
    if not config.IMAGE_OUTPUT_DIR.exists():
        return
    
    for img_file in config.IMAGE_OUTPUT_DIR.glob("*.png"):
        if str(img_file.resolve()) not in active_image_paths and str(img_file) not in active_image_paths:
            try:
                os.remove(img_file)
                print(f"[−] পুরাতন/শেষ হওয়া ম্যাচের থাম্বনেইল মুছে ফেলা হয়েছে: {img_file.name}")
            except Exception as e:
                print(f"[!] ফাইল মুছতে ব্যর্থ: {e}")

def main():
    print("==================================================")
    print("   MHDTV Multi-Server Live Match Pipeline         ")
    print("==================================================")

    scraper = MHDTVScraper()
    img_generator = FallbackImageGenerator()

    processed_data = []
    active_image_paths = set()
    matches = scraper.get_all_matches()

    for match in matches:
        file_slug = sanitize_filename(match['title'])
        final_image_path = match['original_image']

        # সাইটে ইমেজ না থাকলে Pillow/Willow দিয়ে অটো ব্যানার তৈরি
        if not final_image_path:
            img_filename = f"{file_slug}.png"
            local_img_path = config.IMAGE_OUTPUT_DIR / img_filename

            if not local_img_path.exists():
                print(f"     └─ [Pillow/Willow] নতুন ব্যানার তৈরি হচ্ছে: {img_filename}")
                generated_path = img_generator.create_custom_thumbnail(
                    match_title=match['title'],
                    output_path=str(local_img_path)
                )
                final_image_path = str(generated_path)
            else:
                final_image_path = str(local_img_path)

            active_image_paths.add(str(Path(final_image_path).resolve()))
            active_image_paths.add(final_image_path)

        match_entry = {
            "id": file_slug,
            "title": match['title'],
            "slug_url": match['slug_url'],
            "image_url": final_image_path,
            "total_servers": len(match['servers']),
            "servers": match['servers'],
            "status": "LIVE",
            "last_updated": datetime.now().isoformat()
        }

        processed_data.append(match_entry)

    # ১. সমাপ্ত ম্যাচের ইমেজ ক্লিনআপ
    cleanup_ended_matches(active_image_paths)

    # ২. সক্রিয় ডাটা দিয়ে JSON ফাইল সেভ
    print("\n[+] সক্রিয় লাইভ ডাটা দিয়ে JSON ফাইল সেভ করা হচ্ছে...")
    with open(config.JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"[✓] পাইপলাইন সফলভাবে সম্পন্ন হয়েছে! মোট লাইভ ম্যাচ: {len(processed_data)}")
    print(f"[✓] আউটপুট ফাইল: {config.JSON_OUTPUT_FILE}")
    print("==================================================")

if __name__ == "__main__":
    main()
