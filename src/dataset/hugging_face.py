import os
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. 수집 기본 설정
# ==========================================
TARGET_COUNT = 60
MIN_RES = 256
PREFIX = "hf"
BASE_DIR = "./data/raw"

# ==========================================
# 2. 아주 단순해진 50개 클래스 설정
# ==========================================
# 도메인별로 '메인 데이터셋'과 '여분 데이터셋(fallback)'을 하나씩만 둡니다.
DOMAIN_CONFIGS = {
    "food": {
        "dataset": "ethz/food101",
        "fallback_dataset": "Kaludi/data-food-category-classification", # 메인에서 못 채우면 여기서 찾음
        "mapping": {
            'pizza': 'pizza', 'hamburger': 'hamburger', 'sushi': 'sushi',
            'pasta': 'spaghetti_carbonara', 'salad': 'caesar_salad', 
            'steak': 'steak', 'cake': 'chocolate_cake', 'sandwich': 'club_sandwich',
            'fried chicken': 'fried_chicken', 'bread': 'garlic_bread', 
            'apple': 'apple_pie', 'banana': 'frozen_yogurt', 
            'strawberry': 'strawberry_shortcake', 'orange': 'macarons', 
            'carrot': 'carrot_cake'
        }
    },
    "animal": {
        "dataset": "AlvaroVasquezAI/Animal_Image_Classification_Dataset",
        "fallback_dataset": "Prgckwb/fake-animals",
        "mapping": {
            'golden retriever': 'golden retriever', 
            'bulldog': 'French bulldog',
            'siamese cat': 'Siamese cat', 
            'persian cat': 'Persian cat',
            'eagle': 'bald eagle', 
            'owl': 'great grey owl', 
            'lion': 'lion',
            'elephant': 'African elephant', 
            'zebra': 'zebra', 
            'giraffe': 'giraffe'
        }
    },
    "plant": {
        "dataset": "anhaltai/plantNaturalist500k",
        "fallback_dataset": "DeadPixels/DPhi_Sprint_25_Flowers",
        "mapping": {
            'rose': 'hip, rose hip, rosehip', 
            'sunflower': 'daisy', 
            'daisy': 'daisy', 
            'tulip': 'tulip', 
            'palm tree': 'palmtree',
            'pine tree': 'pine', 
            'maple tree': 'buckeye', 
            'bamboo': 'bamboo'
        }
    },
    "object": {
        "dataset": "timm/objectnet",
        "fallback_dataset": "ILSVRC/imagenet-1k", # 사물 여분 데이터셋 예시
        "mapping": {
            'laptop': 'notebook, notebook computer', 'watch': 'digital watch',
            'camera': 'reflex camera', 'chair': 'folding chair',
            'clock': 'wall clock', 'microwave': 'microwave, microwave oven',
            'refrigerator': 'refrigerator, icebox'
        }
    },
    "vehicle": {
        "dataset": "DrBimmer/vehicle-classification",
        "fallback_dataset": "ILSVRC/imagenet-1k",
        "mapping": {
            'car': 'sports car', 'bicycle': 'mountain bike, all-terrain bike, off-roader',
            'motorcycle': 'motor scooter, scooter', 'airplane': 'airliner',
            'bus': 'school bus'
        }
    },
    "fashion": {
        "dataset": "zalando-datasets/fashion_mnist",
        "fallback_dataset": "ILSVRC/imagenet-1k",
        "mapping": {
            'backpack': 'backpack, back pack, knapsack, packsack, rucksack, haversack',
            'sneakers': 'running shoe', 'umbrella': 'umbrella',
            'glasses': 'sunglass', 'hat': 'cowboy hat, ten-gallon hat'
        }
    }
}

# 로드 속도를 높이기 위한 메모리 캐싱
loaded_datasets = {}

def get_hf_dataset(dataset_path):
    if dataset_path not in loaded_datasets:
        print(f"\n⏳ 데이터셋 로드 중: {dataset_path} ...")
        loaded_datasets[dataset_path] = load_dataset(dataset_path, split="train")
    return loaded_datasets[dataset_path]

# ==========================================
# 3. 추가 수집(Fallback) 로직이 포함된 함수
# ==========================================
def fetch_images_from_dataset(ds_path, master_name, target_labels, current_count, save_path):
    """특정 데이터셋에서 이미지를 가져오는 핵심 함수 (메인, 여분 모두 이 함수를 거침)"""
    if current_count >= TARGET_COUNT:
        return current_count
        
    try:
        ds = get_hf_dataset(ds_path)
        hf_label_names = ds.features['label'].names
        
        # 1. 딕셔너리에 적은 매핑 이름으로 먼저 시도, 없으면 2. 내 마스터 이름으로 시도
        search_labels = [target_labels, master_name] 
        found_label = None
        
        for label in search_labels:
            if label in hf_label_names:
                found_label = label
                break
                
        if not found_label:
            return current_count # 이 데이터셋에는 해당 클래스가 없음

        target_idx = hf_label_names.index(found_label)
        class_ds = ds.filter(lambda x: x['label'] == target_idx)
        
        for item in tqdm(class_ds, desc=f"Saving {master_name} from {ds_path.split('/')[-1]}", leave=False):
            if current_count >= TARGET_COUNT:
                break
            
            img = item['image']
            if img.width >= MIN_RES and img.height >= MIN_RES:
                img = img.convert("RGB")
                img_number = str(current_count + 1).zfill(3)
                img_name = f"{PREFIX}_{master_name.replace(' ', '_')}_{img_number}.jpg"
                img.save(os.path.join(save_path, img_name), "JPEG", quality=95)
                current_count += 1
                
    except Exception as e:
        print(f"  -> ⚠️ {ds_path} 검색 중 에러 발생: {e}")
        
    return current_count

# ==========================================
# 4. 메인 실행부
# ==========================================
def collect_hf_images():
    os.makedirs(BASE_DIR, exist_ok=True)
    total_report = {}
    insufficient_classes = []

    for domain_name, config in DOMAIN_CONFIGS.items():
        print(f"\n" + "="*50)
        print(f"🚀 [{domain_name.upper()}] 도메인 수집 시작")
        print("="*50)

        main_ds = config['dataset']
        fallback_ds = config.get('fallback_dataset')

        for master_name, hf_label in config['mapping'].items():
            folder_name = master_name.lower().replace(" ", "_")
            save_path = os.path.join(BASE_DIR, folder_name)
            os.makedirs(save_path, exist_ok=True)
            
            print(f"\n🔍 '{folder_name}' 수집 진행 중...")
            
            # 1. 메인 데이터셋에서 먼저 수집 시도
            count = fetch_images_from_dataset(main_ds, folder_name, hf_label, 0, save_path)
            
            # 2. 목표 장수(60장)를 못 채웠고, 여분 데이터셋이 있다면 추가 수집 시도
            if count < TARGET_COUNT and fallback_ds:
                print(f"  -> 🔄 목표 미달({count}/{TARGET_COUNT}). 여분 데이터셋({fallback_ds})에서 추가 수집을 시도합니다.")
                count = fetch_images_from_dataset(fallback_ds, folder_name, hf_label, count, save_path)

            # 최종 장수 기록
            total_report[folder_name] = count
            if count < TARGET_COUNT:
                insufficient_classes.append((folder_name, count))

    # ==========================================
    # 최종 결과 리포트 출력
    # ==========================================
    print("\n" + "#"*50)
    print("📊 [전체 클래스 수집 완료 리포트]")
    print("#"*50)
    for cls, cnt in total_report.items():
        print(f"- {cls}: {cnt}장 수집")
        
    if insufficient_classes:
        print("\n🚨 [목표 미달 클래스 알림] (60장 미만)")
        for cls, cnt in insufficient_classes:
            print(f"  👉 {cls}: {cnt}장 (부족분: {TARGET_COUNT - cnt}장)")
    else:
        print("\n✅ 50개 모든 클래스가 목표치(60장)를 성공적으로 달성했습니다!")

if __name__ == "__main__":
    collect_hf_images()