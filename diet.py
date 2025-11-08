import os
import sys
import datetime
import pickle
import traceback

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None

try:
    import cv2
except Exception:
    cv2 = None

import requests

API_ID = "f3ea175a"
API_KEY = "eaa480a05e96324249b5ad419fd89288"
DETAILS_FILE = "details.xlsx"
NUTRITION_FILE = "nutrition.xlsx"
MODEL_FILE = "model.pickle"

def safe_input(prompt, cast=str, default=None):
    try:
        val = input(prompt)
        if val is None or val.strip() == "":
            return default
        return cast(val)
    except Exception:
        return default

def camera_capture_and_crop(save_crops=False, crop_size=(224, 224), max_crops=5):

    if cv2 is None:
        print("OpenCV is not installed. Camera disabled.")
        return []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera (device busy or not present)")
        return []

    cropping = False
    ix = iy = ex = ey = -1
    rectangles = []
    frame = None

    def crop(event, x, y, flags, params):
        nonlocal cropping, ix, iy, ex, ey, frame, rectangles
        if event == cv2.EVENT_LBUTTONDOWN:
            cropping = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping and frame is not None:
                temp = frame.copy()
                cv2.rectangle(temp, (ix, iy), (x, y), (0, 0, 255), 2)
                cv2.imshow("window", temp)
        elif event == cv2.EVENT_LBUTTONUP:
            cropping = False
            ex, ey = x, y
            rectangles.append([ix, iy, ex, ey])

    cv2.namedWindow("window")
    cv2.setMouseCallback("window", crop)

    print("Camera opened. Draw rectangles with left mouse button. Press 'x' to finish, 'r' to reset rectangles")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera")
            break

        display = frame.copy()
        for r in rectangles:
            try:
                cv2.rectangle(display, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)
            except Exception:
                continue

        cv2.imshow("window", display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('x'):
            break
        if k == ord('r'):
            rectangles = []
            print("Rectangles cleared")
        if len(rectangles) >= max_crops:
            print(f"Reached max crops: {max_crops}")
            break

    cap.release()
    cv2.destroyAllWindows()

    crops = []
    for idx, r in enumerate(rectangles[:max_crops]):
        try:
            x1, y1, x2, y2 = r
            x1, x2 = sorted([int(x1), int(x2)])
            y1, y2 = sorted([int(y1), int(y2)])
            h, w = frame.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cropped = frame[y1:y2, x1:x2].copy()
            resized = cv2.resize(cropped, crop_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 255.0
            crops.append(arr)
            if save_crops:
                fname = f"crop_{idx}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, cropped)
        except Exception:
            traceback.print_exc()
            continue

    return crops

class DummyModel:
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            batch = x.shape[0]
        else:
            try:
                batch = x.size(0)
            except Exception:
                batch = 1
        return np.random.randn(batch, 10)


def load_model(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded model from {path}")
            return model
        except Exception as e:
            print("Failed to load model.pickle:", e)
            print("Using DummyModel instead.")
            return DummyModel()
    else:
        print("model.pickle not found. Using DummyModel for predictions.")
        return DummyModel()


def compute_calcium(gender, age):
    if gender == 1:
        if age < 18:
            return 1300
        elif age <= 70:
            return 1000
        else:
            return 1300
    elif gender == 2:
        if age < 18 or age > 50:
            return 1300
        else:
            return 1000
    else:
        return 1000


def read_details():
    if os.path.exists(DETAILS_FILE):
        try:
            df = pd.read_excel(DETAILS_FILE)
            if df.empty or df.isnull().values.any():
                return None
            return df.iloc[0].to_dict()
        except Exception:
            return None
    return None


def write_details(personal):
    df = pd.DataFrame([personal])
    df.to_excel(DETAILS_FILE, index=False)


def ensure_today_row():
    today = datetime.date.today().strftime('%Y-%m-%d')
    if os.path.exists(NUTRITION_FILE):
        try:
            nutrit = pd.read_excel(NUTRITION_FILE)
            if nutrit.shape[0] == 0 or str(nutrit.iloc[0].get('date', '')) != today:
                entry = {
                    'date': today,
                    'water': 0.0,
                    'protein': 0.0,
                    'carbs': 0.0,
                    'fat': 0.0,
                    'calories': 0.0,
                    'calcium': 0.0
                }
                nutrit = pd.concat([pd.DataFrame([entry]), nutrit], ignore_index=True)
                nutrit.to_excel(NUTRITION_FILE, index=False)
                return entry
            else:
                row = nutrit.iloc[0]
                return {
                    'date': row['date'],
                    'water': float(row.get('water', 0.0)),
                    'protein': float(row.get('protein', 0.0)),
                    'carbs': float(row.get('carbs', 0.0)),
                    'fat': float(row.get('fat', 0.0)),
                    'calories': float(row.get('calories', 0.0)),
                    'calcium': float(row.get('calcium', 0.0))
                }
        except Exception:
            traceback.print_exc()

    entry = {
        'date': today,
        'water': 0.0,
        'protein': 0.0,
        'carbs': 0.0,
        'fat': 0.0,
        'calories': 0.0,
        'calcium': 0.0
    }
    pd.DataFrame([entry]).to_excel(NUTRITION_FILE, index=False)
    return entry


def save_today_row(entry):
    try:
        if os.path.exists(NUTRITION_FILE):
            nutrit = pd.read_excel(NUTRITION_FILE)
            if nutrit.shape[0] == 0:
                nutrit = pd.DataFrame([entry])
            else:
                nutrit.iloc[0] = pd.Series(entry)
            nutrit.to_excel(NUTRITION_FILE, index=False)
        else:
            pd.DataFrame([entry]).to_excel(NUTRITION_FILE, index=False)
        return True
    except Exception:
        traceback.print_exc()
        return False


def query_nutritionix_natural(query_text):
    url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
    headers = {
        'x-app-id': API_ID,
        'x-app-key': API_KEY,
        'Content-Type': 'application/json'
    }
    data = {'query': query_text}
    try:
        res = requests.post(url, json=data, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
        else:
            print('Nutritionix API error:', res.status_code, res.text)
            return None
    except Exception as e:
        print('Nutritionix request failed:', e)
        return None


def suggest_meals(remaining_protein, remaining_carbs, remaining_fat, dish_type):
    query = f"{dish_type} food with approx {int(remaining_protein)}g protein, {int(remaining_carbs)}g carbs, {int(remaining_fat)}g fat"
    url = 'https://trackapi.nutritionix.com/v2/search/instant'
    headers = {
        'x-app-id': API_ID,
        'x-app-key': API_KEY,
        'Content-Type': 'application/json'
    }
    try:
        res = requests.get(url, headers=headers, params={'query': query}, timeout=8)
        if res.status_code == 200:
            return res.json().get('common', [])[:5]
        else:
            print('Meal suggestion error:', res.status_code)
            return []
    except Exception as e:
        print('Suggestion request failed:', e)
        return []


def main():
    try:
        personal = read_details()
        if personal is None:
            print('Enter your details — saved to details.xlsx')
            personal = {
                'name': safe_input('Enter your name: ', str, 'User'),
                'age': int(safe_input('Enter your age: ', int, 25)),
                'height': int(safe_input('Enter your height in cm: ', int, 170)),
                'weight': int(safe_input('Enter your weight in kg: ', int, 70)),
                'gender': int(safe_input('Gender (1 Male, 2 Female): ', int, 1)),
                'type': int(safe_input('Diet type (1 veg,2 eggetarian,3 non-veg): ', int, 1)),
                'disease': int(safe_input('Disease (1 heart,2 diabetes,3 none): ', int, 3)),
                'exercise': int(safe_input('Exercise level (1 intense..4 light): ', int, 3))
            }
            write_details(personal)
            print('Details saved')
        else:
            print(f"Welcome back, {personal.get('name','User')}!")

        if personal['gender'] == 1:
            calorie_in = int((10 * personal['weight']) + (6.25 * personal['height']) - (5 * personal['age']) + 5)
        elif personal['gender'] == 2:
            calorie_in = int((10 * personal['weight']) + (6.25 * personal['height']) - (5 * personal['age']) - 161)
        else:
            calorie_in = int((10 * personal['weight']) + (6.25 * personal['height']) - (5 * personal['age']))

        protein_goal = personal['weight'] * 0.75
        carbs_goal = (calorie_in / 100) * 45.65
        fat_goal = (calorie_in / 100) * 30
        water_goal = personal['weight'] * 0.035
        calcium_goal = compute_calcium(personal['gender'], personal['age'])

        today_row = ensure_today_row()
        water_take = float(today_row['water'])
        protein_take = float(today_row['protein'])
        carbs_take = float(today_row['carbs'])
        fat_take = float(today_row['fat'])
        calorie_take = float(today_row['calories'])
        calcium_take = float(today_row['calcium'])

        model = load_model(MODEL_FILE)

        while True:
            print('--- Menu ---')
            print('1. I ate something (manual)')
            print('2. I drank water')
            print('3. Show recommended meals to complete goal')
            print('4. Show total progress')
            print('5. Exit')
            print('6. Camera detect + add (if you have model)')

            try:
                choice = int(safe_input('Choose: ', int, 5))
            except Exception:
                choice = 5

            if choice == 1:
                eat = safe_input("Enter your meal (e.g. '2 eggs' or '200g rice'): ", str, '100g boiled potato')
                nutr = query_nutritionix_natural(eat)
                if nutr is not None and 'foods' in nutr:
                    for item in nutr['foods']:
                        nf_cal = float(item.get('nf_calories', 0.0))
                        nf_pro = float(item.get('nf_protein', 0.0))
                        nf_carbs = float(item.get('nf_total_carbohydrate', 0.0))
                        nf_fat = float(item.get('nf_total_fat', 0.0))
                        print(f"{item.get('food_name','Food').title()} - {item.get('serving_weight_grams',0)}g: {nf_cal} kcal, P:{nf_pro}g, C:{nf_carbs}g, F:{nf_fat}g")

                        calorie_take += nf_cal
                        protein_take += nf_pro
                        carbs_take += nf_carbs
                        fat_take += nf_fat
                else:
                    print('Could not fetch nutrition info. Check API keys or network.')

            elif choice == 2:
                water_take += 0.20
                print(f"Added 0.20L water. Total: {water_take:.2f}L")

            elif choice == 3:
                rem_pro = max(0.0, protein_goal - protein_take)
                rem_carbs = max(0.0, carbs_goal - carbs_take)
                rem_fat = max(0.0, fat_goal - fat_take)
                rem_cal = max(0.0, calorie_in - calorie_take)
                print(f"Remaining - Protein: {rem_pro:.2f}g, Carbs: {rem_carbs:.2f}g, Fat: {rem_fat:.2f}g, Calories: {rem_cal:.2f} kcal")
                suggestions = suggest_meals(rem_pro, rem_carbs, rem_fat, ("vegetarian" if personal['type']==1 else "non-veg"))
                if suggestions:
                    print("Suggested meals:")
                    for s in suggestions:
                        print('-', s.get('food_name','Unknown'))
                else:
                    print('No suggestions available or API failed.')

            elif choice == 4:
                print('--- Progress ---')
                print(f"Calories: {calorie_take:.2f} kcal / {calorie_in} kcal ({(calorie_take / calorie_in) * 100 if calorie_in>0 else 0:.2f}%)")
                print(f"Protein: {protein_take:.2f} g / {protein_goal:.2f} g ({(protein_take / protein_goal) * 100 if protein_goal>0 else 0:.2f}%)")
                print(f"Carbs: {carbs_take:.2f} g / {carbs_goal:.2f} g ({(carbs_take / carbs_goal) * 100 if carbs_goal>0 else 0:.2f}%)")
                print(f"Fat: {fat_take:.2f} g / {fat_goal:.2f} g ({(fat_take / fat_goal) * 100 if fat_goal>0 else 0:.2f}%)")
                print(f"Water: {water_take:.2f} L / {water_goal:.2f} L ({(water_take / water_goal) * 100 if water_goal>0 else 0:.2f}%)")
                print(f"Calcium: {calcium_take:.2f} mg / {calcium_goal} mg ({(calcium_take / calcium_goal) * 100 if calcium_goal>0 else 0:.2f}%)")

            elif choice == 5:
                print('Exiting. Stay healthy.')
                break

            elif choice == 6:
                if cv2 is None:
                    print('OpenCV not installed. Install opencv-python to use camera detection.')
                else:
                    crops = camera_capture_and_crop(save_crops=False, crop_size=(224,224), max_crops=3)
                    if not crops:
                        print('No crops captured or camera failed.')
                    else:
                        arr = np.stack(crops, axis=0)  # (N,H,W,C)
                        arr = np.transpose(arr, (0,3,1,2))  # (N,C,H,W)

                        try:
                            if torch is not None and isinstance(model, torch.nn.Module):
                                t = torch.from_numpy(arr).float()
                                if torch.cuda.is_available():
                                    t = t.cuda()
                                    model = model.cuda()
                                model.eval()
                                with torch.no_grad():
                                    out = model(t)
                                    if isinstance(out, torch.Tensor):
                                        probs = F.softmax(out, dim=1).cpu().numpy()
                                    else:
                                        probs = np.array(out)
                            else:
                                out = model(arr)
                                probs = np.array(out)

                            for i, p in enumerate(probs):
                                idx = int(np.argmax(p))
                                print(f"Crop {i}: predicted class {idx}")
                                guessed_food = f"class_{idx}"
                                nutr = query_nutritionix_natural(guessed_food)
                                if nutr is not None and 'foods' in nutr:
                                    for item in nutr['foods']:
                                        nf_cal = float(item.get('nf_calories', 0.0))
                                        nf_pro = float(item.get('nf_protein', 0.0))
                                        nf_carbs = float(item.get('nf_total_carbohydrate', 0.0))
                                        nf_fat = float(item.get('nf_total_fat', 0.0))
                                        print(f"{item.get('food_name','Food').title()} -> {nf_cal} kcal")
                                        calorie_take += nf_cal
                                        protein_take += nf_pro
                                        carbs_take += nf_carbs
                                        fat_take += nf_fat
                                else:
                                    print('Nutrition lookup failed for predicted class.')

                        except Exception as e:
                            print('Error during model inference:', e)
                            traceback.print_exc()

            entry = {
                'date': datetime.date.today().strftime('%Y-%m-%d'),
                'water': water_take,
                'protein': protein_take,
                'carbs': carbs_take,
                'fat': fat_take,
                'calories': calorie_take,
                'calcium': calcium_take
            }
            save_today_row(entry)

    except KeyboardInterrupt:
        print('Interrupted by user. Exiting.')
    except Exception as e:
        print('Fatal error:', e)
        traceback.print_exc()

if __name__ == '__main__':
    main()
