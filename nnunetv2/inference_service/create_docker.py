import argparse
import os
import shutil

from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.file_path_utilities import get_output_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-docker_app", type=str, required=True,
                        help="Folder in which to create the docker app")
    parser.add_argument("-base_docker_image", type=str, required=False, default="cuda12.9:py3.13_torch2.8.0",
                        help="Base docker image")
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', type=str, required=False, default="0",
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: 0')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    args = parser.parse_args()

    if os.path.isdir(args.docker_app) and len(os.listdir(args.docker_app)) != 0:
        raise RuntimeError(f"Folder {args.docker_app} is not empty")
    os.makedirs(args.docker_app, exist_ok=True)

    args.f = args.f if args.f == 'all' else int(args.f)
    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)
    docker_model_folder = model_folder.replace(nnUNet_results, "").strip("/")
    dataset = os.path.join(model_folder, 'dataset.json')
    plans = os.path.join(model_folder, 'plans.json')
    model_checkpoint = os.path.join(model_folder, f"fold_{args.f}", args.chk)

    shutil.copy(model_checkpoint, os.path.join(args.docker_app, "checkpoint_final.pth"))
    shutil.copy(plans, os.path.join(args.docker_app, "plans.json"))
    shutil.copy(dataset, os.path.join(args.docker_app, "dataset.json"))
    shutil.copy(os.path.join(os.path.dirname(__file__), "process.py"), os.path.join(args.docker_app, "process.py"))

    dockerfile = f"""
FROM {args.base_docker_image}

ENV nnUNet_dataset={args.d}
ENV nnUNet_plans={args.p}
ENV nnUNet_trainer={args.tr}
ENV nnUNet_conf={args.c}
ENV nnUNet_fold={args.f}
ENV nnUNet_step_size={args.step_size}
ENV nnUNet_disable_tta={'1' if args.disable_tta else '0'}

ENV nnUNet_raw=/app/nnUNet_raw
ENV nnUNet_results=/app/nnUNet_results
ENV nnUNet_preprocessed=/app/nnUNet_preprocessed

WORKDIR /app

RUN mkdir -p /app/input /app/output /app/nnUNet_raw /app/nnUNet_results /app/nnUNet_preprocessed && \
    chmod 777 /app/output && \
    pip install timed-decorator && \
    pip uninstall nnunetv2 --yes && \
    pip install 'git+https://github.com/ancestor-mithril/nnUNet.git@new' --no-cache-dir

COPY checkpoint_final.pth /app/nnUNet_results/{docker_model_folder}/fold_{args.f}/
COPY dataset.json plans.json /app/nnUNet_results/{docker_model_folder}/
COPY process.py /app

ENTRYPOINT ["python", "process.py"]
    """
    with open(os.path.join(args.docker_app, "Dockerfile"), "w") as f:
        f.write(dockerfile.strip() + "\n")


if __name__ == "__main__":
    main()
