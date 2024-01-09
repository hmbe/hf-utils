import copy
import cv2
from PIL import Image

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


### make 'id', 'tokens', 'bboxes', 'ner_tags', 'image'
### ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
### apply normalize bboxes
def make_dataset_dict(json_dict, id = -1, image_path = None):
    image, size = load_image(image_path)
    dataset_dict = {'id': id, 'tokens':[], 'bboxes':[], 'ner_tags':[], 'image': image}

    # print(json_dict['form'][0].keys())
    # print(json_dict['form'][0]['words'])

    for tag_dict in json_dict['form']:
        for i, word_dict in enumerate(tag_dict['words']):
            ner_tag = None
            bbox = None
            token = None
            try:
                if tag_dict['label'] == 'question':
                    ner_tag = 'B-QUESTION' if i == 0 else 'I-QUESTION'
                elif tag_dict['label'] == 'answer':
                    ner_tag = 'B-ANSWER' if i == 0 else 'I-ANSWER'
                elif tag_dict['label'] == 'header':
                    ner_tag = 'B-HEADER' if i == 0 else 'I-HEADER'
                elif tag_dict['label'] == 'other':
                    ner_tag = 'O'

                bbox = word_dict['box']
                token = word_dict['text']
                normalized_bbox = normalize_bbox(bbox, size)

                assert(ner_tag != None and bbox != None and token != None)
                assert(normalized_bbox[0] <= 1000 and normalized_bbox[1] <= 1000 and normalized_bbox[2] <= 1000 and normalized_bbox[3] <= 1000)

                dataset_dict['bboxes'].append(normalized_bbox)
                dataset_dict['tokens'].append(token)
                dataset_dict['ner_tags'].append(ner_tag)

            except AssertionError as e:

                print(f'exception {e}, size: {size}, bbox: {bbox}, normalized_bbox: {normalized_bbox}')
                print(f'ner_tag: {ner_tag}, token: {token}')
                print(f'image_path: {image_path}')

    return dataset_dict

### insert dataset type in ['train', 'test']
def load_dataset_manually(json_dict_wtype, img_list_dict_wtype):
    
    ### make 'id', 'tokens', 'bboxes', 'ner_tags', 'image'
    train_test_dict = {
        'train':{'id': [], 'tokens':[], 'bboxes':[], 'ner_tags':[], 'image': []},
        'test':{'id': [], 'tokens':[], 'bboxes':[], 'ner_tags':[], 'image': []}
    }

    ### fix output dict to list
    dataset_keys = ['id', 'tokens', 'bboxes', 'ner_tags', 'image']

    for dataset_type in ['train', 'test']:
        target_json_dict = json_dict_wtype[dataset_type]
        target_img_list = img_list_dict_wtype[dataset_type]
        target_base_name_list = [x.split('/')[-1].split('.')[0] for x in target_img_list]

        for i, (json_name, json_dict) in enumerate(target_json_dict.items()):
            ### search and load images
            base_name = json_name.split('.')[0]
            try:
                train_img_path = target_img_list[target_base_name_list.index(base_name)]
                dataset_dict = make_dataset_dict(json_dict, id=i, image_path=train_img_path)
                for dk in dataset_keys:
                    train_test_dict[dataset_type][dk].append(dataset_dict[dk])

            except Exception as e:
                print(f'{base_name} is not found in img_path_list or train_base_name_list, exception {e}')

    return train_test_dict