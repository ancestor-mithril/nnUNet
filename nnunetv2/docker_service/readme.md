

Cum o sa fie folosit: 

0. Proprietati:


```
docker run --rm IMG_NAME props
```

Comanda props returneaza informatiile utile:
```
{
    "DATASET_PATH": "...",
    "PREPROCESSED_PATH": "...",
    "MODEL_PATH": "...",
    "INPUT": "...",
    "OUTPUT": "...",
    "OUTPUT": "...",
    "NUM_EPOCHS_RANGE": "...",
    "MODEL_SIZES_RANGE": "...",
    "COMMIT_HASH": "...",
    "FALLBACK_MODEL": "..."
}
```

Toate acestea vor fi folosite in urmatoarele comenzi sub forma DATASET_PATH, PREPROCESSED_PATH, MODEL_PATH

1. Preprocessing

Pentru preprocessing sunt necesare:

```
PATH_TO_DATASET_SERVER = ... (path to dataset folder on server)
PATH_TO_PREPROCESS_SERVER = ... (path to preprocess folder on server)

docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v PATH_TO_DATASET_SERVER:DATASET_PATH \
  -v PATH_TO_PREPROCESS_SERVER:PREPROCESSED_PATH \
  IMG_NAME preprocess
```

PATH_TO_DATASET_SERVER nu trebuie sa fie read-only, pentru ca in timpul preprocessing-ului containerul scrie in dataset:
* dataset.json
* folderul labelsTr
* mastile remapate in labelsTr




2. Training

Trainingul trebuie facut dupa ce Preprocessing e gata pt path-ul PATH_TO_PREPROCESS_SERVER.

```
PATH_TO_DATASET_SERVER=...       # path to dataset folder on server
PATH_TO_PREPROCESS_SERVER=...    # path to preprocess folder on server
PATH_TO_MODEL_SERVER=...         # path unde se va salva modelul antrenat

FOLD=1                           # poate fi 0, 1, 2, 3, 4 sau all
DEVICE=0                         # index GPU, de exemplu 0 sau 1


docker run --rm --gpus all \
  --shm-size=36g \
  --user $(id -u):$(id -g) \
  -v PATH_TO_DATASET_SERVER:DATASET_PATH:ro \
  -v PATH_TO_PREPROCESS_SERVER:PREPROCESSED_PATH:ro \
  -v PATH_TO_MODEL_SERVER:MODEL_PATH \
  IMG_NAME train -fold FOLD -device DEVICE
```

3. Validation

Validation poate fi facut dupa fiecare training s-a terminat. Validation foloseste exact aceiasi parametrii ca si training-ul, si se face la nivel de fold.

```
PATH_TO_DATASET_SERVER=...       # path to dataset folder on server
PATH_TO_PREPROCESS_SERVER=...    # path to preprocess folder on server
PATH_TO_MODEL_SERVER=...         # path unde a fost salvat modelul antrenat

FOLD=1                           # poate fi 0, 1, 2, 3, 4 sau all
DEVICE=0                         # index GPU, de exemplu 0 sau 1


docker run --rm --gpus all \
  --shm-size=36g \
  --user (id -u):(id -g) \
  -v PATH_TO_DATASET_SERVER:DATASET_PATH:ro \
  -v PATH_TO_PREPROCESS_SERVER:PREPROCESSED_PATH:ro \
  -v PATH_TO_MODEL_SERVER:MODEL_PATH \
  IMG_NAME validate -fold FOLD -device DEVICE
```

Validarea va scrie un fisier, "PATH_TO_MODEL_SERVER/fold_FOLD/validation/results.json". 
Acest fisier contine metricile pentru modelul respectiv, fold-ul respectiv.

4. Cross-Validation

Cross-Validation poate fi facut doar daca s-a terminat deja validarea FOLD-urilor 0, 1, 2, 3, 4. FOLD-ul "all" este ignorat (nu poate fi inclus in cross-validation).
Foloseste aceiasi parametrii ca si validation, in afara de FOLD, si nu mai are nevoie de PREPROCESSED_PATH sau DATASET_PATH.

```
PATH_TO_MODEL_SERVER=...         # path unde a fost salvat modelul antrenat

DEVICE=0                         # index GPU, de exemplu 0 sau 1


docker run --rm --gpus all \
  --shm-size=36g \
  --user $(id -u):$(id -g) \
  -v PATH_TO_MODEL_SERVER:MODEL_PATH \
  IMG_NAME cross_validate -device DEVICE
```

Cross-validarea va scrie un fisier, "PATH_TO_MODEL_SERVER/final_results.json". 
Aici se vor afla metricile de cross-validare.

5. Inference

Inferenta trebuie facuta dupa ce Trainingul este gata pt path-ul "PATH_TO_MODEL_SERVER" si fold-ul "FOLD".
Inference poate primi si varianta "ensemble" ca si fold, doar ca are nevoie de trainingul gata pentru toate fold-urile 0, 1, 2, 3, 4

```
PATH_TO_MODEL_SERVER=...     # path unde exista modelul antrenat
PATH_TO_INPUT_SERVER=...     # path catre folderul cu input-uri Nifti
PATH_TO_OUTPUT_SERVER=...    # path catre folderul unde se vor scrie predictiile

FOLD=1                       # poate fi 0, 1, 2, 3, 4, all sau ensemble
DEVICE=0                     # index GPU, de exemplu 0 sau 1


docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v PATH_TO_INPUT_SERVER:/app/input:ro \
  -v PATH_TO_OUTPUT_SERVER:/app/output \
  -v PATH_TO_MODEL_SERVER:MODEL_PATH:ro \
  IMG_NAME inference -fold FOLD -device DEVICE
```

Inferenta cu "all" => doar atunci cand "all" a fost antrenat
```
FOLD=all
DEVICE=0


docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v PATH_TO_INPUT_SERVER:/app/input:ro \
  -v PATH_TO_OUTPUT_SERVER:/app/output \
  -v PATH_TO_MODEL_SERVER:MODEL_PATH:ro \
  IMG_NAME inference -fold FOLD -device DEVICE
```

Inferenta de tip ansamblu => doar atunci cand au fost antrenate toate foldurile: "0", "1", "2", "3", "4"
```
FOLD=ensemble
DEVICE=0


docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v PATH_TO_INPUT_SERVER:/app/input:ro \
  -v PATH_TO_OUTPUT_SERVER:/app/output \
  -v PATH_TO_MODEL_SERVER:MODEL_PATH:ro \
  IMG_NAME inference -fold FOLD -device DEVICE
```

INPUT trebuie sa fie un folder cu pacienti in format Nifti.  eg: 4_0000.nii.gz
OUTPUT este folderul in care o sa fie scrise mastile prezise de model.
