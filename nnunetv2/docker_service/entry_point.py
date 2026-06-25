import argparse
import glob
import json
import os
import subprocess

import numpy as np
import SimpleITK as sitk
import torch.accelerator
from tqdm import tqdm


def run_command(command, envs):
    command = command.strip()
    try:
        subprocess.run(command, env=envs, check=True, shell=True)
        return True
    except subprocess.CalledProcessError:
        print("Error for command:", command)
        return False


def try_inference(input_path: str, output_path: str, fold: str, use_cuda: bool = False, device_index: int = 0,
                  use_best: bool = False):
    if use_cuda:
        print(f"Using cuda device {device_index} for inference!")
    else:
        print(f"Using cpu for inference")

    envs = {**os.environ}
    if use_cuda:
        envs["CUDA_VISIBLE_DEVICES"] = str(device_index)

    checkpoint = "checkpoint_best.pth" if use_best else "checkpoint_final.pth"

    command = (
        f"nnUNetv2_predict "
        f"-i {input_path} "
        f"-o {output_path} "
        f"-d {envs['nnUNet_dataset']} "
        f"-p {envs['nnUNet_plans']} "
        f"-tr {envs['nnUNet_trainer']} "
        f"-c {envs['nnUNet_conf']} "
        f"-f {fold} "
        f"-step_size {envs['nnUNet_step_size']} "
        f"-chk {checkpoint} "
        f"-npp 0 "
        f"-nps 0 "
    )
    if use_cuda:
        command += "-device cuda "
    else:
        command += "-device cpu "
    if envs["nnUNet_disable_tta"] == "1":
        command += "--disable_tta "
    return run_command(command, envs)


def inference(args):
    model_path = "/app/nnUNet_results/Dataset100_A/nnUNetTrainerMuon3en4__nnUNetResEncUNetLPlans_torchres__3d_fullres"
    fold_path = os.path.join(model_path, f"fold_{args.fold}")
    model_checkpoint = os.path.join(fold_path, f"checkpoint_final.pth")
    use_best = False
    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Folder {args.input} is not available")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Folder {model_path} is not available")
    if not os.path.isdir(fold_path):
        raise FileNotFoundError(f"Fold {fold_path} not available, please train the model first")
    if not os.path.isfile(model_checkpoint):
        print(f"Model checkpoint{model_checkpoint} not available")
        model_checkpoint = os.path.join(fold_path, f"checkpoint_best.pth")
        use_best = True
        if not os.path.isfile(model_checkpoint):
            raise FileNotFoundError(f"Model checkpoint{model_checkpoint} not available, please train the model first")

    files = glob.glob(os.path.join(args.input, "*_0000.nii.gz"))
    if len(files) == 0:
        raise FileNotFoundError(f"Found no _0000.nii.gz files in {args.input}")
    if not os.path.isdir(args.output):
        raise FileNotFoundError(f"Folder {args.output} is not available")

    print(f"Cuda Available: {torch.cuda.is_available()}")
    succeeded = try_inference(args.input, args.output, args.fold, use_cuda=True, device_index=args.device,
                              use_best=use_best)
    if not succeeded:
        succeeded = try_inference(args.input, args.output, use_cuda=False, use_best=use_best)
        if not succeeded:
            print("Inference failed. Check the logs for the error")
            raise RuntimeError("Inference failed")
        print("Inference succeeded")


def remap_segmentation_labels(
        input_path,
        label_mapping,
        output_path,
):
    img = sitk.ReadImage(input_path)
    arr = sitk.GetArrayFromImage(img)
    remapped_arr = np.zeros_like(arr)

    for old_label, new_label in label_mapping.items():
        remapped_arr[arr == old_label] = new_label

    remapped_img = sitk.GetImageFromArray(remapped_arr)
    remapped_img.CopyInformation(img)
    sitk.WriteImage(remapped_img, output_path)


def pre_preprocess(labels, original_labels, labels_tr):
    os.makedirs(labels_tr, exist_ok=True)
    labels = labels.copy()
    del labels["background"]

    files = [x for x in os.listdir(original_labels) if x.endswith(".nii.gz")]

    for f in tqdm(files, desc="Preprocessing masks"):
        mask = os.path.join(original_labels, f)
        mask_labels = mask.replace(".nii.gz", ".json")
        if not os.path.exists(mask_labels):
            raise FileNotFoundError(f"mask_labels {mask_labels} not available!")
        new_mask = os.path.join(labels_tr, f)

        with open(mask_labels, "r") as fp:
            mask_labels = json.load(fp)

        mapping = {}
        for k, v in labels.items():
            if k in mask_labels:
                mapping[mask_labels[k]] = v

        remap_segmentation_labels(mask, mapping, new_mask)


def preprocess(args):
    raw_path = "/app/nnUNet_raw/Dataset100_A"
    images_tr = os.path.join(raw_path, "imagesTr")
    original_labels = os.path.join(raw_path, "labels")
    preprocess_path = "/app/nnUNet_preprocessed/Dataset100_A"
    labels_json = os.path.join(raw_path, "labels.json")
    dataset_json = os.path.join(raw_path, "dataset.json")

    if not os.path.isdir(raw_path):
        raise FileNotFoundError(f"Folder {raw_path} is not available")
    if not os.path.isdir(images_tr):
        raise FileNotFoundError(f"Folder {images_tr} is not available")
    if not os.path.isdir(original_labels):
        raise FileNotFoundError(f"Folder {original_labels} is not available")
    if not os.path.isdir(preprocess_path):
        raise FileNotFoundError(f"Folder {preprocess_path} is not available")

    def get_case_names(paths):
        ret = []
        for p in paths:
            p = os.path.basename(p)
            p = p.replace(".nii.gz", "")
            p = p.replace("_0000", "")
            ret.append(p)
        return ret

    train_files = get_case_names(glob.glob(os.path.join(images_tr, "*_0000.nii.gz")))
    label_files = get_case_names(glob.glob(os.path.join(original_labels, "*.nii.gz")))

    if len(train_files) < 5:
        raise FileNotFoundError(f"Found less than 5 _0000.nii.gz files in {images_tr}")

    if set(train_files) != set(label_files):
        raise RuntimeError(
            "We must have the same number of training images and training labels\n"
            f"Available training images: {train_files}\n"
            f"Available training labels: {label_files}"
        )

    error_msg = (
        f"File {labels_json} is not available or is corrupted.\n"
        """
        Example:
        {	
            "background": 0,
            "bone": 1, 
            "artery": 2, 
            "calcification": 3, 
            "thrombosis": 4
        }
        """
    )
    if not os.path.isfile(labels_json):
        raise FileNotFoundError(error_msg)
    try:
        with open(labels_json, "r") as f:
            labels = json.load(f)
    except Exception as e:
        raise RuntimeError(error_msg) from e

    labels = {
        "background": 0,
        **{label: i + 1 for i, label in enumerate(labels)},
    }
    dataset = {
        "channel_names": {
            "0": "NIFTI"
        },
        "labels": labels,
        "numTraining": len(train_files),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    with open(dataset_json, "w") as f:
        json.dump(dataset, f)
    labels_tr = os.path.join(raw_path, "labelsTr")
    pre_preprocess(labels, original_labels, labels_tr)

    envs = {**os.environ}
    command = (
        "nnUNetv2_plan_and_preprocess "
        f"-d {envs['nnUNet_dataset']} "
        f"-pl {envs['nnUNet_plans']} "
        f"-np {args.num_workers} "
        f"-c {envs['nnUNet_conf']} "
    )
    succeeded = run_command(command, envs)
    if not succeeded:
        print("Preprocess failed. Check the logs for the error")
        raise RuntimeError("Preprocess failed")
    command = (
        "nnUNetv2_train "
        f"-p {envs['nnUNet_plans']} "
        f"-tr {envs['nnUNet_trainer']} "
        f"--c "
        f"{envs['nnUNet_dataset']} "
        f"{envs['nnUNet_conf']} "
        f"{0} "
        "-device cpu"
    )
    envs["EXIT_AFTER_SPLIT"] = "1"
    envs["nnUNet_compile"] = "0"
    succeeded = run_command(command, envs)
    if not succeeded:
        print("Split creation failed. Check the logs for the error")
        raise RuntimeError("Split creation failed")


def train(args):
    preprocess_path = "/app/nnUNet_preprocessed/Dataset100_A"
    dataset_json = os.path.join(preprocess_path, "dataset.json")

    if not os.path.isdir(preprocess_path):
        raise FileNotFoundError(f"Folder {preprocess_path} is not available. Run preprocessing first!")
    if not os.path.isfile(dataset_json):
        raise FileNotFoundError(f"File {dataset_json} is not available. Run preprocessing first")

    envs = {**os.environ}
    envs["CUDA_VISIBLE_DEVICES"] = str(args.device)
    command = (
        "nnUNetv2_train "
        f"-p {envs['nnUNet_plans']} "
        f"-tr {envs['nnUNet_trainer']} "
        f"--c "
        f"{envs['nnUNet_dataset']} "
        f"{envs['nnUNet_conf']} "
        f"{args.fold} "
    )

    succeeded = run_command(command, envs)
    if not succeeded:
        print("Training failed. Check the logs for the error")
        raise RuntimeError("Training failed")


def props(_):
    properties = {
        "DATASET_PATH": os.getenv("cont_data_path"),
        "PREPROCESSED_PATH": os.getenv("cont_preproc_path"),
        "MODEL_PATH": os.getenv("cont_model_path"),
        "INPUT": os.getenv("cont_input_path"),
        "OUTPUT": os.getenv("cont_output_path"),
    }
    print(json.dumps(properties, indent=4))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands", required=True)

    parser_props = subparsers.add_parser("props", help="Properties")
    parser_props.set_defaults(func=props)

    parser_inference = subparsers.add_parser("inference", help="Do inference")
    parser_inference.add_argument("-fold", type=str, help="Fold", default="0")
    parser_inference.add_argument("-device", type=int, help="CUDA device index", default=0)
    parser_inference.set_defaults(func=inference, input="/app/input", output="/app/output")

    parser_preprocess = subparsers.add_parser("preprocess", help="Do preprocessing")
    parser_preprocess.add_argument("-num_workers", type=int, default=4,
                                   help="Number of parallel processes for preprocessing")
    parser_preprocess.set_defaults(func=preprocess)

    parser_train = subparsers.add_parser("train", help="Do training")
    parser_train.add_argument("-fold", type=str, help="Fold", default="0")
    parser_train.add_argument("-device", type=int, help="CUDA device index", default=0)
    parser_train.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
