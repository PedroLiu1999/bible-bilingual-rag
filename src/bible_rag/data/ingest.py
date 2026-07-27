import json

import httpx
from pydantic import BaseModel
from tqdm import tqdm

from bible_rag.config import PROCESSED_DIR, PROCESSED_FILE, RAW_DIR

KJV_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"
CUV_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/zh_cuv.json"

BIBLE_BOOKS = [
    ("Genesis", "創世記", "Gen"), ("Exodus", "出埃及記", "Exo"), ("Leviticus", "利未記", "Lev"),
    ("Numbers", "民數記", "Num"), ("Deuteronomy", "申命記", "Deu"), ("Joshua", "約書亞記", "Jos"),
    ("Judges", "士師記", "Jdg"), ("Ruth", "路得記", "Rth"), ("1 Samuel", "撒母耳記上", "1Sa"),
    ("2 Samuel", "撒母耳記下", "2Sa"), ("1 Kings", "列王紀上", "1Ki"), ("2 Kings", "列王紀下", "2Ki"),
    ("1 Chronicles", "歷代志上", "1Ch"), ("2 Chronicles", "歷代志下", "2Ch"), ("Ezra", "以斯拉記", "Ezr"),
    ("Nehemiah", "尼希米記", "Neh"), ("Esther", "以斯帖記", "Est"), ("Job", "約伯記", "Job"),
    ("Psalms", "詩篇", "Psa"), ("Proverbs", "箴言", "Pro"), ("Ecclesiastes", "傳道書", "Ecc"),
    ("Song of Solomon", "雅歌", "Sng"), ("Isaiah", "以賽亞書", "Isa"), ("Jeremiah", "耶利米書", "Jer"),
    ("Lamentations", "耶利米哀歌", "Lam"), ("Ezekiel", "以西結書", "Ezk"), ("Daniel", "但以理書", "Dan"),
    ("Hosea", "何西阿書", "Hos"), ("Joel", "約珥書", "Jol"), ("Amos", "阿摩司書", "Amo"),
    ("Obadiah", "俄巴底亞書", "Obad"), ("Jonah", "約拿書", "Jon"), ("Micah", "彌迦書", "Mic"),
    ("Nahum", "那鴻書", "Nam"), ("Habakkuk", "哈巴谷書", "Hab"), ("Zephaniah", "西番雅書", "Zep"),
    ("Haggai", "哈該書", "Hag"), ("Zechariah", "撒迦利亞書", "Zec"), ("Malachi", "瑪拉基書", "Mal"),
    ("Matthew", "馬太福音", "Mat"), ("Mark", "馬可福音", "Mrk"), ("Luke", "路加福音", "Luk"),
    ("John", "約翰福音", "Jhn"), ("Acts", "使徒行傳", "Act"), ("Romans", "羅馬書", "Rom"),
    ("1 Corinthians", "哥林多前書", "1Co"), ("2 Corinthians", "哥林多後書", "2Co"), ("Galatians", "加拉太書", "Gal"),
    ("Ephesians", "以弗所書", "Eph"), ("Philippians", "腓立比書", "Php"), ("Colossians", "歌羅西書", "Col"),
    ("1 Thessalonians", "帖撒羅尼迦前書", "1Th"), ("2 Thessalonians", "帖撒羅尼迦後書", "2Th"),
    ("1 Timothy", "提摩太前書", "1Tm"), ("2 Timothy", "提摩太後書", "2Tm"), ("Titus", "提多書", "Tit"),
    ("Philemon", "腓利門書", "Phm"), ("Hebrews", "希伯來書", "Heb"), ("James", "雅各書", "Jas"),
    ("1 Peter", "彼得前書", "1Pe"), ("2 Peter", "彼得後書", "2Pe"), ("1 John", "約翰一書", "1Jn"),
    ("2 John", "約翰二書", "2Jn"), ("3 John", "約翰三書", "3Jn"), ("Jude", "猶大書", "Jud"),
    ("Revelation", "啟示錄", "Rev")
]

class BibleVerse(BaseModel):
    id: str
    book_name_en: str
    book_name_zh: str
    book_abbrev: str
    chapter: int
    verse: int
    text_kjv: str
    text_cuv: str

def fetch_raw_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {"kjv.json": KJV_URL, "cuv.json": CUV_URL}
    with httpx.Client(follow_redirects=True) as client:
        for file_name, url in files.items():
            dest = RAW_DIR / file_name
            if not dest.exists():
                print(f"Downloading {file_name}...")
                res = client.get(url)
                res.raise_for_status()
                dest.write_bytes(res.content)

def process_and_align():
    fetch_raw_data()
    with open(RAW_DIR / "kjv.json", "r", encoding="utf-8-sig") as f:
        kjv_raw = json.load(f)
    with open(RAW_DIR / "cuv.json", "r", encoding="utf-8-sig") as f:
        cuv_raw = json.load(f)

    aligned_verses = []
    for book_idx, (en_name, zh_name, abbrev) in enumerate(tqdm(BIBLE_BOOKS, desc="Processing Books")):
        if book_idx >= len(kjv_raw): break
        kjv_chapters = kjv_raw[book_idx].get("chapters", [])
        cuv_chapters = cuv_raw[book_idx].get("chapters", []) if book_idx < len(cuv_raw) else []

        for ch_idx, kjv_verses in enumerate(kjv_chapters):
            chapter_num = ch_idx + 1
            cuv_verses = cuv_chapters[ch_idx] if ch_idx < len(cuv_chapters) else []
            for v_idx, kjv_text in enumerate(kjv_verses):
                verse_num = v_idx + 1
                cuv_text = cuv_verses[v_idx] if v_idx < len(cuv_verses) else ""
                verse_id = f"{abbrev.upper()}_{chapter_num:03d}_{verse_num:03d}"

                verse_obj = BibleVerse(
                    id=verse_id, book_name_en=en_name, book_name_zh=zh_name, book_abbrev=abbrev,
                    chapter=chapter_num, verse=verse_num, text_kjv=kjv_text.strip(), text_cuv=cuv_text.strip()
                )
                aligned_verses.append(verse_obj.model_dump())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(aligned_verses, f, ensure_ascii=False, indent=2)
    print(f"\n Processing complete! Total verses aligned: {len(aligned_verses)}")

if __name__ == "__main__":
    process_and_align()

