import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

prsk_charactor_name_dict ={"初音ミク": "Hatsune_Miku",
                            "鏡音リン": "Kagamine_Rin",
                            "鏡音レン": "Kagamine_Ren",
                            "巡音ルカ": "Megurine_Luka",
                            "MEIKO": "MEIKO",
                            "KAITO": "KAITO",

                            "宵崎奏": "Yoisaki_Kanade",
                            "東雲絵名": "Shinonome_Ena",
                            "朝比奈まふゆ": "Asahina_Mafuyu",
                            "暁山瑞希": "Akiyama_Mizuki",

                            "天馬司": "Tennma_Tsukasa",
                            "鳳えむ": "Ootori_Emu",
                            "草薙寧々": "Kusanagi_Nene",
                            "神代類": "Kamishiro_Rui",

                            "小豆沢こはね": "Azukisawa_Kohane",
                            "白石杏": "Shiraishi_Ann",
                            "東雲彰人": "Shinonome_Akito",
                            "青柳冬弥": "Aoyagi_Touya",

                            "花里みのり": "Hanasato_Minori",
                            "桐谷遥": "Kiritani_Haruka",
                            "桃井愛莉": "Momoi_Airi",
                            "日野森雫": "Hinomori_Shizuku",

                            "星乃一歌": "Hoshino_Ichika",
                            "天馬咲希": "Tennma_Saki",
                            "望月穂波": "Mochizuki_Honami",
                            "日野森志歩": "Hinomori_Shiho",
                            }

# 画像保存用ディレクトリ
output_dir = "./make_dataset/stamps"
os.makedirs(output_dir, exist_ok=True)

# 対象のURL
url = "https://pjsekai.com/?b4db77bdb9"


# ウェブページを取得
response = requests.get(url)
response.raise_for_status()  # エラーがあれば停止
soup = BeautifulSoup(response.text, "html.parser")

# テーブル内のデータ行を取得（ヘッダー行を除外）
rows = soup.select("table.style_table tr")[1:]  # 最初の行（ヘッダー）をスキップ

char_name_dict = {}

for row in rows:
    try:
        # <img>タグから画像URLを取得
        img_tag = row.find("img")
        if img_tag and "data-src" in img_tag.attrs:
            img_url = img_tag["data-src"]  # data-src を取得
            # 相対パスを完全なURLに変換
            if img_url.startswith("./"):
                img_url = img_url.replace("./", "https://pjsekai.com/")

            # キャラクター名を取得（2列目）
            td_tags = row.find_all("td")
            if len(td_tags) > 1:
                char_name = td_tags[1].get_text(strip=True)

                # 画像をダウンロード
                img_response = requests.get(img_url)
                img_response.raise_for_status()


                # Convert to Romaji
                if char_name in prsk_charactor_name_dict:
                    chara_name_Romaji = prsk_charactor_name_dict[char_name]

                    if chara_name_Romaji in char_name_dict:
                        char_name_dict[chara_name_Romaji] += 1
                        file_name = f"{chara_name_Romaji}_{char_name_dict[chara_name_Romaji]:04d}.png"

                    else: # not in dict
                        os.makedirs(os.path.join(output_dir, chara_name_Romaji), exist_ok=True)

                        char_name_dict[chara_name_Romaji] = 1
                        file_name = f"{chara_name_Romaji}_0000.png"


                    # file_name = f"{char_name}.png"
                    file_path = os.path.join(output_dir,chara_name_Romaji, file_name)

                    # ファイル保存
                    #resize to 80x80 and add white background and save
                    img = Image.open(BytesIO(img_response.content))

                    # aspect not 1:1 then add in longer side in alpha = 0 and make longer side 1:1 aspect
                    if img.width != img.height:
                        width, height = img.size
                        white_canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
                        white_canvas.paste(img, (0, 0), img) 
                        new_size = max(width, height)
                        square_canvas = Image.new("RGBA", (new_size, new_size), (255, 255, 255, 255))
                        x_offset = (new_size - width) // 2
                        y_offset = (new_size - height) // 2
                        square_canvas.paste(white_canvas, (x_offset, y_offset))
                        img = square_canvas 
                    else:
                        pass

                    img = img.resize((80, 80), Image.LANCZOS)
                    img = img.convert("RGBA")
                    img_new = Image.new("RGBA", img.size, (255, 255, 255, 255)) 
                    img_new.paste(img, (0, 0), img)
                    # save in PNG
                    with open(file_path, "wb") as f:
                        img_new.save(f, format="PNG")



                    # with open(file_path, "wb") as f:
                    
                    #     f.write(img_response.content)

                    print(f"保存完了: {file_name}")
                
                else:# 複数キャラクターはスキップ
                    print(f"キャラクター名が見つかりません: {char_name}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")



