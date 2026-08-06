import argparse
import os
import shutil


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("-docker_app", type=str, required=True,
                        help="Folder in which to create the docker app")
    parser.add_argument("-base_docker_image", type=str, required=False, default="cuda12.9:py3.13_torch2.8.0",
                        help="Base docker image")
    args = parser.parse_args()

    if os.path.isdir(args.docker_app) and len(os.listdir(args.docker_app)) != 0:
        raise RuntimeError(f"Folder {args.docker_app} is not empty")

    os.makedirs(args.docker_app, exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(__file__), "entry_point.py"), os.path.join(args.docker_app, "entry_point.py"))

    dockerfile = f"""
FROM {args.base_docker_image}

ENV nnUNet_dataset=100
ENV nnUNet_plans=nnUNetResEncUNetLPlans_torchres
ENV nnUNet_trainer=nnUNetTrainerMuon3en4
ENV nnUNet_conf=3d_fullres
ENV nnUNet_step_size=0.5
ENV nnUNet_disable_tta=0

ENV copy_elision=1
ENV IGNORE_INF=1
ENV PERF_LOGGER=1

ENV nnUNet_raw=/app/nnUNet_raw
ENV nnUNet_results=/app/nnUNet_results
ENV nnUNet_preprocessed=/app/nnUNet_preprocessed
ENV cont_data_path=${{nnUNet_raw}}/Dataset100_A
ENV cont_preproc_path=${{nnUNet_preprocessed}}/Dataset100_A
ENV cont_model_path=${{nnUNet_results}}/Dataset100_A/${{nnUNet_trainer}}__${{nnUNet_plans}}__${{nnUNet_conf}}
ENV cont_input_path=/app/input
ENV cont_output_path=/app/output
ENV nnUNet_stop_first_epoch=0

WORKDIR /app

RUN mkdir -p /app/input /app/output ${{cont_data_path}} ${{cont_model_path}} ${{cont_preproc_path}} /home/jovyan && \
    chmod 777 /app/output && \
    chmod 777 /home/jovyan && \
    chmod 777 ${{cont_preproc_path}} && \
    chmod 777 ${{cont_model_path}} && \
    pip install timed-decorator && \
    pip uninstall nnunetv2 --yes && \
    pip install 'git+https://github.com/ancestor-mithril/nnUNet.git@new' --no-cache-dir

COPY entry_point.py /app

ENTRYPOINT ["python", "entry_point.py"]
    """
    with open(os.path.join(args.docker_app, "Dockerfile"), "w") as f:
        f.write(dockerfile.strip() + "\n")


if __name__ == "__main__":
    entry_point()
