

Cum o sa fie folosit: 

0. Proprietati:


```
docker run --rm IMG_NAME props
```

Comanda props returneaza path-urile folosite in container, de forma:
```
{
    "DATASET_PATH": "...",
    "PREPROCESSED_PATH": "...",
    "MODEL_PATH": "..."
}
```

Toate acestea vor fi folosite in urmatoarele comenzi sub forma $DATASET_PATH, $PREPROCESSED_PATH, $MDOEL_PATH

1. Preprocessing

Pentru preprocessing sunt necesare:

```
export PATH_TO_DATASET_SERVER = ... (path to dataset folder on server)
export PATH_TO_PREPROCESS_SERVER = ... (path to preprocess folder on server)

docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v $PATH_TO_DATASET_SERVER:$DATASET_PATH \
  -v $PATH_TO_PREPROCESS_SERVER:$PREPROCESSED_PATH \
  IMG_NAME preprocess
```

PATH_TO_DATASET_SERVER nu trebuie sa fie read-only, pentru ca in timpul preprocessing-ului containerul scrie in dataset:
* dataset.json
* folderul labelsTr
* mastile remapate in labelsTr




2. Training

Trainingul trebuie facut dupa ce Preprocessing e gata pt path-ul $PATH_TO_PREPROCESS_SERVER.

```
export PATH_TO_DATASET_SERVER=...       # path to dataset folder on server
export PATH_TO_PREPROCESS_SERVER=...    # path to preprocess folder on server
export PATH_TO_MODEL_SERVER=...         # path unde se va salva modelul antrenat

export FOLD=1                           # poate fi 0, 1, 2, 3, 4 sau all
export DEVICE=0                         # index GPU, de exemplu 0 sau 1


docker run --rm --gpus all \
  --shm-size=36g \
  --user $(id -u):$(id -g) \
  -v $PATH_TO_DATASET_SERVER:$DATASET_PATH:ro \
  -v $PATH_TO_PREPROCESS_SERVER:$PREPROCESSED_PATH:ro \
  -v $PATH_TO_MODEL_SERVER:$MODEL_PATH \
  IMG_NAME train -fold $FOLD -device $DEVICE
```


3. Inference

Inferenta trebuie facuta dupa ce Trainingul este gata pt path-ul "PATH_TO_MODEL_SERVER" si fold-ul "FOLD"

```
export PATH_TO_MODEL_SERVER=...     # path unde exista modelul antrenat
export PATH_TO_INPUT_SERVER=...     # path catre folderul cu input-uri Nifti
export PATH_TO_OUTPUT_SERVER=...    # path catre folderul unde se vor scrie predictiile

export FOLD=1                       # poate fi 0, 1, 2, 3, 4 sau all
export DEVICE=0                     # index GPU, de exemplu 0 sau 1


docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v $PATH_TO_INPUT_SERVER:/app/input:ro \
  -v $PATH_TO_OUTPUT_SERVER:/app/output \
  -v $PATH_TO_MODEL_SERVER:$MODEL_PATH:ro \
  IMG_NAME inference -fold $FOLD -device $DEVICE
```

INPUT trebuie sa fie un folder cu pacienti in format Nifti.  eg: 4_0000.nii.gz
OUTPUT este folderul in care o sa fie scrise mastile prezise de model.
