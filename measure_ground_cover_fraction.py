import os
import cv2
import numpy as np
import pandas as pd

def read_in_files(aom_dir_path):
    folders_in_directory = sorted(os.listdir(aom_dir_path))

    aom_sets = dict()

    for dir_name in folders_in_directory:

        print('[INFO] Reading in sets for: {0}'.format(dir_name))

        single_directory_path = os.path.join(aom_dir_path, dir_name)
        aom_names_in_directory = [s for s in os.listdir(single_directory_path) if s.endswith('.tif')]
        aom_paths = [os.path.join(single_directory_path, aom_name) for aom_name in aom_names_in_directory]

        image_set = []

        for (aom_path, aom_name) in zip(aom_paths, aom_names_in_directory):
            image = cv2.imread(aom_path)
            image_set.append((image, aom_name, aom_path))

        aom_sets[dir_name] = image_set

    return aom_sets


def extract_pixel_counts(aom_set_dict):

    layer_keys = list(aom_set_dict.keys())

    aom_count_mask_dict = dict()

    for layer_key in layer_keys:
        aoms_ls = aom_set_dict[layer_key]

        mask_and_count_img = dict()

        for (image, aom_name, aom_path) in aoms_ls:

            print('[INFO] Measuring AOM: {0}'.format(aom_path))

            b, g, r = cv2.split(image)

            mask = (b < g) & (r < g)
            mask = mask.astype('uint8')

            masked_image = cv2.bitwise_and(image, image, mask=mask)

            pixel_count = len(mask[mask > 0])

            print('[INFO] Plant pixels counted: {0}'.format(pixel_count))

            mask_and_count_img[aom_name] = (image, masked_image, pixel_count, layer_key)

        aom_count_mask_dict[layer_key] = mask_and_count_img

    return aom_count_mask_dict


def create_dict_by_aom(aom_set_dict, aom_names):

    layer_keys = list(aom_set_dict.keys())

    extraction_data_by_aom_dict = dict()

    for aom in aom_names:
        stack_data = []
        for layer_id in layer_keys:
            extraction_data_dict = aom_set_dict[layer_id]
            image, masked_image, pixel_count, layer_key = extraction_data_dict[aom]
            stack_data.append((image, masked_image, pixel_count, layer_key))

        extraction_data_by_aom_dict[aom] = stack_data

     return extraction_data_by_aom_dict


def create_image_stacks_by_aom(extraction_by_aom_dict, aom_names, out_path):

    image_stack_dict = dict()

    for aom in aom_names:
        extraction_data = extraction_by_aom_dict[aom]
        images = [i[0] for i in extraction_data]

        image_stack_dict[aom] = (images, aom)

        mean_height = int(np.mean([i.shape[0] for i in images]))
        mean_width = int(np.mean([i.shape[0] for i in images]))

        resized_images = []

        for img in images:
            resized_image = cv2.resize(img, (mean_height, mean_width), interpolation = cv2.INTER_AREA)
            resized_images.append(resized_image)

        image_stack = np.concatenate(resized_images)

        cv2.imwrite(os.path.join(out_path, aom), image_stack)

        print('[INFO] AOM stack complete: {0}'.format(aom))

        image_stack_dict[aom] = image_stack

    return image_stack_dict


def make_data_frame_of_counts(aom_count_mask_dict):

    layer_keys = list(aom_count_mask_dict.keys())
    layer_keys = sorted(layer_keys)

    data_ls = []

    for layer_key in layer_keys:

        print('[INFO] Collecting data for layer: {0}'.format(layer_key))

        extraction_data_dict = aom_count_mask_dict[layer_key]

        for (aom_name, (image, masked_image, pixel_count, layer_key)) in extraction_data_dict.items():
            data_ls.append((aom_name, pixel_count, layer_key))

    df = pd.DataFrame(data_ls)
    df.columns = ['aom_name', 'pixel_count', 'layer_key']
    df.loc[:, 'date'] = df.loc[:, 'layer_key'].map(lambda x: x.split('_')[0])
    df.loc[:, 'date'] = pd.to_datetime(df.loc[:, 'date'], format='%Y-%m-%d')

    return df


if __name__=='__main__':

    extraction_directory_path = '/home/will/peanut_growth_curves/extracted_aoms'
    aom_dict = read_in_files(extraction_directory_path)

    aom_count_mask_dict = extract_pixel_counts(aom_dict)

    layer_ids = list(aom_count_mask_dict.keys())
    aom_names_ls = [i[1] for i in aom_dict[layer_ids[0]]]

    df_counts = make_data_frame_of_counts(aom_count_mask_dict)
    df_counts.to_csv('/home/will/peanut_growth_curves/growth_curve_data_sets/growth_curve_data.csv')
    aom_ids = df_counts.loc[:, 'aom_name'].unique()

    extraction_data_by_aom_dict = create_dict_by_aom(aom_set_dict=aom_count_mask_dict, aom_names=aom_ids)

    image_stacks_dir = '/home/will/peanut_growth_curves/aom_photo_logs'
    image_stacks_dict = create_image_stacks_by_aom(extraction_data_by_aom_dict, aom_names_ls, out_path=image_stacks_dir)
