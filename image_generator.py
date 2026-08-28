import os
from io import BytesIO
from PIL import Image as PILImage, ImageDraw, ImageFont
from willow import Image
import config

class FallbackImageGenerator:
    """
    যদি সাইটে কোনো ইমেজ পাওয়া না যায়, তবে Pillow ও Willow ব্যবহার করে 
    স্বয়ংক্রিয়ভাবে প্রিমিয়াম স্পোর্টস থাম্বনেইল তৈরি করার ক্লাস।
    """

    def __init__(self):
        self.width = config.BANNER_WIDTH
        self.height = config.BANNER_HEIGHT

    def _create_base_canvas(self) -> PILImage.Image:
        """ কাস্টম কালার প্যালেট সহ গ্র্যাডিয়েন্ট ব্যাকগ্রাউন্ড তৈরি করে """
        img = PILImage.new('RGB', (self.width, self.height), config.BG_PRIMARY_COLOR)
        draw = ImageDraw.Draw(img)

        # বর্ডার এবং প্যানেল শেড আঁকা
        padding = 25
        draw.rectangle(
            [padding, padding, self.width - padding, self.height - padding],
            outline=config.ACCENT_COLOR,
            width=3
        )

        # ডিজাইন এলিমেন্ট (ডানপাশের এঙ্গেল ব্লকে থিম শেড)
        draw.polygon(
            [(self.width - 350, 0), (self.width, 0), (self.width, self.height), (self.width - 500, self.height)],
            fill=config.BG_SECONDARY_COLOR
        )

        return img

    def create_custom_thumbnail(self, match_title: str, output_path: str) -> str:
        """
        ম্যাচের নাম দিয়ে আকর্ষণীয় কাস্টম থাম্বনেইল জেনারেট করে
        """
        try:
            pil_img = self._create_base_canvas()
            draw = ImageDraw.Draw(pil_img)

            # ফন্ট লোড (Fallback সহ)
            try:
                title_font = ImageFont.truetype("arial.ttf", 44)
                badge_font = ImageFont.truetype("arial.ttf", 26)
                sub_font = ImageFont.truetype("arial.ttf", 22)
            except IOError:
                title_font = badge_font = sub_font = ImageFont.load_default()

            # ১. লাইভ ম্যাচ ব্যাজ
            draw.rectangle([60, 60, 360, 110], fill=config.ACCENT_COLOR)
            draw.text((75, 72), "  LIVE MATCH STREAM  ", fill=(0, 0, 0), font=badge_font)

            # ২. ম্যাচের শিরোনাম
            words = match_title.split()
            line1, line2 = "", ""
            for word in words:
                if len(line1 + " " + word) < 26:
                    line1 += " " + word
                else:
                    line2 += " " + word

            draw.text((60, 210), line1.strip(), fill=config.TEXT_WHITE, font=title_font)
            if line2:
                draw.text((60, 275), line2.strip(), fill=config.TEXT_WHITE, font=title_font)

            # ৩. ব্র্যান্ডিং ও ডট
            draw.ellipse([60, 490, 80, 510], fill=(239, 68, 68)) # লাল লাইভ ডট
            draw.text((95, 487), "MHDTV LIVE STREAMING", fill=config.TEXT_MUTED, font=sub_font)

            # Willow দিয়ে ইমেজ প্রসেসিং ও সেভ
            buffer = BytesIO()
            pil_img.save(buffer, format='PNG')
            buffer.seek(0)

            willow_image = Image.open(buffer)
            willow_image.save_as_png(output_path)
            return output_path

        except Exception as e:
            print(f"[-] ইমেজ জেনারেট করতে ত্রুটি: {e}")
            return ""
