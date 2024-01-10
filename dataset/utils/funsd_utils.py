from PIL import ImageDraw, ImageFont
from dataset.utils.aihub_bank_dataset_generator import unnormalize_box

label_list = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
id2label = {id:label for id, label in enumerate(label_list)}
label2id = {label:id for id, label in enumerate(label_list)}

def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label

def visualize_funsd(image, true_predictions, true_boxes):
    ### dependency on funsd
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # def unnormalize_box(bbox, width, height):
    #      return [
    #          width * (bbox[0] / 1000),
    #          height * (bbox[1] / 1000),
    #          width * (bbox[2] / 1000),
    #          height * (bbox[3] / 1000),
    #      ]

    label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

    width, height = image.size

    for prediction, box in zip(true_predictions, true_boxes):
        # predicted_label = iob_to_label(prediction).lower()
        predicted_label = label_list[int(prediction)] if prediction != -100 else 'other'
        predicted_label = iob_to_label(predicted_label).lower()
        box = unnormalize_box(box, width, height)

        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)
    
    return image