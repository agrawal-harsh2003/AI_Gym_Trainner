import json
from collections import defaultdict

def get_default_object():
    return {
        'bbox': [],
        'keypoints': [],
        'num_keypoints': [],
        'area': [],
        'iscrowd': [],
        'category_id': [],
        'face_box': [],
        'lefthand_box': [],
        'righthand_box': [],
        'lefthand_kpts': [],
        'righthand_kpts': [],
        'face_kpts': [],
        'face_valid': [],
        'lefthand_valid': [],
        'righthand_valid': [],
        'foot_valid': [],
        'foot_kpts': []
    }

def process_coco_annotations(annotations_path, output_path):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    image_annotations = defaultdict(get_default_object)

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        
        for key in image_annotations[image_id].keys():
            if key in ann:
                image_annotations[image_id][key].append(ann[key])
            else:
                image_annotations[image_id][key].append(None)

    with open(output_path, 'w') as outfile:
        for img in coco_data['images']:
            img_id = img['id']
            
            entry = {
                'file_name': img['file_name'],
                'image_id': img_id,
                'height': img['height'],
                'width': img['width'],
                'license': img.get('license', 0),
                'coco_url': img.get('coco_url', ''),
                'date_captured': img.get('date_captured', ''),
                'flickr_url': img.get('flickr_url', ''),
                'objects': image_annotations[img_id] if image_annotations[img_id] else get_default_object()
            }
            json.dump(entry, outfile)
            outfile.write('\n')

    print(f'Metadata saved to {output_path}')

process_coco_annotations('/Users/harshagrawal/Downloads/coco_wholebody_val_v1.0.json', 
    '/Users/harshagrawal/Downloads/Dataset/val2017/metadata.jsonl')

process_coco_annotations('/Users/harshagrawal/Downloads/coco_wholebody_train_v1.0.json', 
    '/Users/harshagrawal/Downloads/Dataset/train2017/metadata.jsonl')