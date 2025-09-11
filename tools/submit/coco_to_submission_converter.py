import json
import argparse

def get_image_Id(img_name_without_extension):
    # 함수는 '.png'가 없는 이름을 받도록 수정되었으므로, .png를 추가해줍니다.
    img_name = img_name_without_extension
    if not img_name.endswith('.png'):
        img_name += '.png'
    img_name = img_name.split('.png')[0] # 이 부분은 사실상 중복이나, 원본 함수 로직 유지를 위해 둡니다.
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

def convert_to_submission_format(input_file_path, submission_file_path):
    with open(input_file_path, 'r') as f:
        current_data = json.load(f)

    submission_data = []

    for item in current_data:
        image_name_str = item.get('image_id') # 예: "camera23_A_55"
        category_id = item.get('category_id')
        bbox = item.get('bbox')
        score = item.get('score')

        if image_name_str is None or category_id is None or bbox is None or score is None:
            print(f"Skipping item due to missing fields: {item}")
            continue
        
        try:
            submission_image_id = get_image_Id(image_name_str)
        except Exception as e:
            print(f"Error processing image name string {image_name_str}: {e}. Skipping this item.")
            continue

        submission_data.append({
            "image_id": submission_image_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": score
        })

    with open(submission_file_path, 'w') as f:
        json.dump(submission_data, f, indent=2)

    print(f"Conversion complete. Submission file saved to: {submission_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert COCO-like prediction JSON to submission format.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input COCO-like JSON file.")
    parser.add_argument('-o', '--output', type=str, default="./submission_converted.json", help="Path to save the converted submission JSON file. Defaults to 'submission_converted.json'.")
    
    args = parser.parse_args()
    
    # 명령줄 인자로 받은 경로를 사용
    convert_to_submission_format(args.input, args.output)
    print(f"Script finished. Input: {args.input}, Output: {args.output}")