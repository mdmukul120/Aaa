import json
import re
from datetime import datetime
import config
from scraper import MHDTVScraper
from image_generator import FallbackImageGenerator

def sanitize_filename(name: str) -> str:
    """ নিরাপদ ফাইল নাম তৈরি করে """
    clean = re.sub(r'[^\w\s-]', '', name).strip().lower()
    return re.sub(r'[-\s]+', '_', clean)[:30]

def main():
    print("==================================================")
    print("      MHDTV Live Match Automated Pipeline         ")
    print("==================================================")

    scraper = MHDTVScraper()
    img_generator = FallbackImageGenerator()

    processed_data = []
    matches = scraper.get_all_matches()

    for match in matches:
        file_slug = sanitize_filename(match['title'])
        final_image_path = match['original_image']

        # যদি সাইটে অরিজিনাল ইমেজ না থাকে, Pillow/Willow দিয়ে তৈরি করা হবে
        if not final_image_path:
            img_filename = f"{file_slug}_{datetime.now().strftime('%H%M%S')}.png"
            local_img_path = config.IMAGE_OUTPUT_DIR / img_filename

            print(f"     └─ [Pillow/Willow] কাস্টম ইমেজ জেনারেট হচ্ছে: {img_filename}")
            generated_path = img_generator.create_custom_thumbnail(
                match_title=match['title'],
                output_path=str(local_img_path)
            )
            final_image_path = str(generated_path)

        match_entry = {
            "id": file_slug,
            "title": match['title'],
            "slug_url": match['slug_url'],
            "image_url": final_image_path,
            "iframe_url": match['iframe_url'],
            "m3u8_url": match['m3u8_url'],
            "embed_code": match['embed_code'],
            "timestamp": datetime.now().isoformat()
        }

        processed_data.append(match_entry)

    # JSON ফাইলে সেভ করা
    print("\n[+] JSON ফাইল আপডেট করা হচ্ছে...")
    with open(config.JSON_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"[✓] পাইপলাইন সফলভাবে সম্পন্ন হয়েছে! মোট ম্যাচ: {len(processed_data)}")
    print(f"[✓] ডাটা সেভ ফাইল: {config.JSON_OUTPUT_FILE}")
    print("==================================================")

if __name__ == "__main__":
    main()
