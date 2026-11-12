"""রেজিস্ট্রি: বিশ্বের সবচেয়ে বড় সাইটগুলো + তাদের ফিড/API।

এই লিংকগুলোই Supabase-এর `sites` টেবিলে থাকে; ইন্টারনেট-লার্নিং এই
তালিকা ধরে ডেটা নামায়।
"""

BIG_SITES = [
    # ---------- encyclopedia / reference ----------
    {"domain": "en.wikipedia.org", "name": "Wikipedia (English)", "homepage": "https://en.wikipedia.org",
     "category": "encyclopedia", "lang": "en", "feed": None, "rank": 5},
    {"domain": "bn.wikipedia.org", "name": "উইকিপিডিয়া (বাংলা)", "homepage": "https://bn.wikipedia.org",
     "category": "encyclopedia", "lang": "bn", "feed": None, "rank": 6},
    {"domain": "stackoverflow.com", "name": "Stack Overflow", "homepage": "https://stackoverflow.com",
     "category": "tech-qa", "lang": "en", "feed": "https://stackoverflow.com/feeds", "rank": 60},

    # ---------- global news ----------
    {"domain": "bbc.com", "name": "BBC News", "homepage": "https://www.bbc.com/news",
     "category": "news", "lang": "en", "feed": "https://feeds.bbci.co.uk/news/rss.xml", "rank": 1},
    {"domain": "bbc.com", "name": "BBC বাংলা", "homepage": "https://www.bbc.com/bengali",
     "category": "news", "lang": "bn", "feed": "https://feeds.bbci.co.uk/bengali/rss.xml", "rank": 1},
    {"domain": "theguardian.com", "name": "The Guardian", "homepage": "https://www.theguardian.com",
     "category": "news", "lang": "en", "feed": "https://www.theguardian.com/world/rss", "rank": 9},
    {"domain": "aljazeera.com", "name": "Al Jazeera", "homepage": "https://www.aljazeera.com",
     "category": "news", "lang": "en", "feed": "https://www.aljazeera.com/xml/rss/all.xml", "rank": 11},
    {"domain": "cnn.com", "name": "CNN", "homepage": "https://edition.cnn.com",
     "category": "news", "lang": "en", "feed": "http://rss.cnn.com/rss/edition.rss", "rank": 10},
    {"domain": "nytimes.com", "name": "The New York Times", "homepage": "https://www.nytimes.com",
     "category": "news", "lang": "en",
     "feed": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "rank": 7},
    {"domain": "dw.com", "name": "DW বাংলা", "homepage": "https://www.dw.com/bn",
     "category": "news", "lang": "bn", "feed": "https://rss.dw.com/rdf/rss-bn-all", "rank": 40},
    {"domain": "voanews.com", "name": "VOA News", "homepage": "https://www.voanews.com",
     "category": "news", "lang": "en", "feed": "https://www.voanews.com/api/zq$omekvi_", "rank": 55},

    # ---------- Bangladesh ----------
    {"domain": "prothomalo.com", "name": "প্রথম আলো", "homepage": "https://www.prothomalo.com",
     "category": "news", "lang": "bn", "feed": "https://www.prothomalo.com/feed", "rank": 2},
    {"domain": "bdnews24.com", "name": "বিডিনিউজ২৪", "homepage": "https://bdnews24.com",
     "category": "news", "lang": "bn", "feed": "https://bdnews24.com/rss", "rank": 3},
    {"domain": "thedailystar.net", "name": "The Daily Star", "homepage": "https://www.thedailystar.net",
     "category": "news", "lang": "en", "feed": "https://www.thedailystar.net/frontpage/rss.xml", "rank": 4},
    {"domain": "banglanews24.com", "name": "বাংলানিউজ২৪", "homepage": "https://www.banglanews24.com",
     "category": "news", "lang": "bn", "feed": "https://www.banglanews24.com/rss/rss.xml", "rank": 15},

    # ---------- tech / knowledge ----------
    {"domain": "news.ycombinator.com", "name": "Hacker News", "homepage": "https://news.ycombinator.com",
     "category": "tech", "lang": "en", "feed": "https://hnrss.org/frontpage", "rank": 30},
    {"domain": "techcrunch.com", "name": "TechCrunch", "homepage": "https://techcrunch.com",
     "category": "tech", "lang": "en", "feed": "https://techcrunch.com/feed/", "rank": 25},
    {"domain": "theverge.com", "name": "The Verge", "homepage": "https://www.theverge.com",
     "category": "tech", "lang": "en", "feed": "https://www.theverge.com/rss/index.xml", "rank": 26},
    {"domain": "wired.com", "name": "WIRED", "homepage": "https://www.wired.com",
     "category": "tech", "lang": "en", "feed": "https://www.wired.com/feed/rss", "rank": 35},
    {"domain": "nasa.gov", "name": "NASA", "homepage": "https://www.nasa.gov",
     "category": "science", "lang": "en", "feed": "https://www.nasa.gov/rss/dyn/breaking_news.rss", "rank": 45},
    {"domain": "sciencedaily.com", "name": "ScienceDaily", "homepage": "https://www.sciencedaily.com",
     "category": "science", "lang": "en", "feed": "https://www.sciencedaily.com/rss/top/science.xml", "rank": 50},
    {"domain": "medium.com", "name": "Medium", "homepage": "https://medium.com",
     "category": "blog", "lang": "en", "feed": "https://medium.com/feed/tag/technology", "rank": 20},
    {"domain": "espncricinfo.com", "name": "ESPN Cricinfo", "homepage": "https://www.espncricinfo.com",
     "category": "sports", "lang": "en", "feed": "https://www.espncricinfo.com/rss/content/story/feeds/0.xml", "rank": 22},
]

# দ্রুত ডেমোতে কোন ফিড/উৎস কত পেজ দেবে
SOURCE_PLAN = {
    "wiki_en": {"kind": "wiki", "lang": "en", "default": 40},
    "wiki_bn": {"kind": "wiki", "lang": "bn", "default": 40},
    "bbc_en": {"kind": "feed", "feed": "https://feeds.bbci.co.uk/news/rss.xml", "lang": "en", "domain": "bbc.com", "default": 20},
    "bbc_bn": {"kind": "feed", "feed": "https://feeds.bbci.co.uk/bengali/rss.xml", "lang": "bn", "domain": "bbc.com", "default": 20},
    "prothomalo": {"kind": "feed", "feed": "https://www.prothomalo.com/feed", "lang": "bn", "domain": "prothomalo.com", "default": 20},
    "guardian": {"kind": "feed", "feed": "https://www.theguardian.com/world/rss", "lang": "en", "domain": "theguardian.com", "default": 20},
    "aljazeera": {"kind": "feed", "feed": "https://www.aljazeera.com/xml/rss/all.xml", "lang": "en", "domain": "aljazeera.com", "default": 20},
    "thedailystar": {"kind": "feed", "feed": "https://www.thedailystar.net/frontpage/rss.xml", "lang": "en", "domain": "thedailystar.net", "default": 15},
    "dw_bn": {"kind": "feed", "feed": "https://rss.dw.com/rdf/rss-bn-all", "lang": "bn", "domain": "dw.com", "default": 12},
    "bdnews24": {"kind": "feed", "feed": "https://bdnews24.com/rss", "lang": "bn", "domain": "bdnews24.com", "default": 12},
    "jagonews24": {"kind": "feed", "feed": "https://www.jagonews24.com/rss/rss.xml", "lang": "bn", "domain": "jagonews24.com", "default": 12},
    "techcrunch": {"kind": "feed", "feed": "https://techcrunch.com/feed/", "lang": "en", "domain": "techcrunch.com", "default": 12},
    "theverge": {"kind": "feed", "feed": "https://www.theverge.com/rss/index.xml", "lang": "en", "domain": "theverge.com", "default": 12},
    "nasa": {"kind": "feed", "feed": "https://www.nasa.gov/rss/dyn/breaking_news.rss", "lang": "en", "domain": "nasa.gov", "default": 10},
    "sciencedaily": {"kind": "feed", "feed": "https://www.sciencedaily.com/rss/top/science.xml", "lang": "en", "domain": "sciencedaily.com", "default": 10},
    "cricinfo": {"kind": "feed", "feed": "https://www.espncricinfo.com/rss/content/story/feeds/0.xml", "lang": "en", "domain": "espncricinfo.com", "default": 10},
    "test": {"kind": "test", "default": 60},
}


# ডোমেইন → তার অফিশিয়াল ফিড(গুলো) — সাইটে ক্লিক করলেই এখান থেকে ডেটা নামে
DOMAIN_FEEDS: dict[str, list[tuple[str, str]]] = {}
for _key, _cfg in SOURCE_PLAN.items():
    if _cfg.get("kind") == "feed":
        DOMAIN_FEEDS.setdefault(_cfg["domain"], []).append((_cfg["feed"], _cfg["lang"]))
