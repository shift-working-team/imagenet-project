import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# 1. 수집 설정
IMAGES_PER_CLASS = 60
MIN_RESOLUTION = 256
SAVE_DIR = "./data/raw"

# 2. 2026년 기준 가장 안정적인(Parquet 지원) 데이터셋 설정
dataset_configs = [
    # {
    #     "path": "ethz/food101",        # 음식 도메인 (매우 안정적)
    #     "split": "train",
    #     "label_key": "label",
    #     "image_key": "image",
    #     "classes": [
    #         'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    #         'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    #         'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    #         'ceviche', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings'
    #     ]
    # },
    {
        "path": "timm/oxford-iiit-pet",     # 동물 도메인 (공식 데이터셋 ID)
        "split": "train",
        "label_key": "label",
        "image_key": "image",
        "classes": [
            'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 
            'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
            'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound'
        ]
    },
    {
        "path": "dpdl-benchmark/oxford_flowers102", # 식물 도메인 (timm 라이브러리 관리)
        "split": "train",
        "label_key": "label",
        "image_key": "image",
        "classes": None # 아래에서 상위 15개 자동 추출
    }
]

def collect_stable_images():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for config in dataset_configs:
        print(f"\n📦 데이터셋 로딩 중: {config['path']}")
        try:
            # trust_remote_code 에러를 피하기 위해 최신 데이터셋 구조 활용
            # 만약 에러가 발생하면 trust_remote_code=True를 추가하거나 
            # 아래의 'Dataset Card' 확인 팁을 참고하세요.
            ds = load_dataset(config['path'], split=config['split'])
            label_names = ds.features[config['label_key']].names
            
            target_classes = config['classes'] if config['classes'] else label_names[:15]
            
            for target_cls in target_classes:
                if target_cls not in label_names: continue
                
                target_idx = label_names.index(target_cls)
                class_save_path = os.path.join(SAVE_DIR, target_cls.replace(" ", "_"))
                os.makedirs(class_save_path, exist_ok=True)
                
                print(f"🔍 {target_cls} 수집 (해상도 {MIN_RESOLUTION}px+ 필터링)")
                count = 0
                
                # 해당 클래스만 필터링
                class_ds = ds.filter(lambda x: x[config['label_key']] == target_idx)
                
                for item in tqdm(class_ds, desc=f"Saving {target_cls}", leave=False):
                    if count >= IMAGES_PER_CLASS: break
                    
                    img = item[config['image_key']]
                    
                    # 해상도 체크 (가로, 세로 모두 256 이상)
                    if img.width >= MIN_RESOLUTION and img.height >= MIN_RESOLUTION:
                        img = img.convert("RGB")
                        save_name = f"hf_{target_cls}_{count:04d}.jpg"
                        img.save(os.path.join(class_save_path, save_name))
                        count += 1
                
                if count < IMAGES_PER_CLASS:
                    print(f"⚠️ {target_cls}: 만족하는 이미지가 {count}장뿐입니다.")
                    
        except Exception as e:
            print(f"❌ {config['path']} 처리 중 에러 발생: {e}")

if __name__ == "__main__":
    collect_stable_images()