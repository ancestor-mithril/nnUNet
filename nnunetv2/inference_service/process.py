import argparse
import glob
import os
import subprocess

import torch.accelerator


def try_inference(input_path: str, output_path: str, use_cuda: bool = False, device_index: int = 0):
    if use_cuda:
        print(f"Using cuda device {device_index} for inference!")
    else:
        print(f"Using cpu for inference")

    envs = {**os.environ}
    if use_cuda:
        envs["CUDA_VISIBLE_DEVICES"] = str(device_index)

    command = (
        f"nnUNetv2_predict "
        f"-i {input_path} "
        f"-o {output_path} "
        f"-d {envs['nnUNet_dataset']} "
        f"-p {envs['nnUNet_plans']} "
        f"-tr {envs['nnUNet_trainer']} "
        f"-c {envs['nnUNet_conf']} "
        f"-f {envs['nnUNet_fold']} "
        f"-step_size {envs['nnUNet_step_size']} "
        f"-npp 0 "
        f"-nps 0 "
    )
    if use_cuda:
        command += "-device cuda "
    else:
        command += "-device cpu "
    if envs["nnUNet_disable_tta"] == "1":
        command += "--disable_tta "
    command = command.strip()
    try:
        subprocess.run(command, env=envs, check=True, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Folder {args.input} is not available")
    files = glob.glob(os.path.join(args.input, "*_0000.nii.gz"))
    if len(files) == 0:
        raise FileNotFoundError(f"Found no _0000.nii.gz files in {args.input}")
    if not os.path.isdir(args.output):
        raise FileNotFoundError(f"Folder {args.output} is not available")

    print(f"Cuda Available: {torch.cuda.is_available()}")
    device_count = torch.cuda.device_count()
    print(f"Number of available devices: {device_count}")

    succeeded = False
    for i in range(device_count):
        succeeded = try_inference(args.input, args.output, use_cuda=True, device_index=i)
        if succeeded:
            break
    if not succeeded:
        succeeded = try_inference(args.input, args.output, use_cuda=False)
    if not succeeded:
        print("Inference failed. Check the logs for the error")
        raise RuntimeError("Inference failed")
    print("Inference succeeded")


if __name__ == "__main__":
    main()
