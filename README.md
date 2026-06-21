# PINN_PPO

| 항목 | 버전 |
|---|---|
| Unity Editor | 2023.2.22f1 |
| Unity ML-Agents Package | 3.0.0 |
| Python | 3.10.12 |
| ml-agents | 1.1.0 |
| ml-agents-envs | 1.1.0 |
| PyTorch | 2.2.1+cu121 |
| ONNX | 1.15.0 |
| setuptools | 80.9.0 |
| CUDA PyTorch Index | cu121 |

Unity ML-Agents 기반 PINN + PPO 프로젝트입니다.

이 문서는 처음 프로젝트를 받는 사람이 GitHub에서 clone한 뒤 Python 가상환경 설정

ML-Agents editable 설치, Unity 프로젝트 설정까지 진행하는 순서를 설명합니다.

--------------------------------------------------------------------------------------------------

## 1. Repository Clone

먼저 원하는 위치에서 GitHub 저장소를 clone합니다.

```bash
git clone https://github.com/HB8514/PINN_PPO.git
```

clone한 폴더로 이동합니다.

```bash
cd PINN_PPO
```

--------------------------------------------------------------------------------------------------

## 2. Python 가상환경 생성

Conda를 사용하는 경우 아래처럼 가상환경을 생성합니다.

```bash
conda create -n mlagents python=3.10.12
```

가상환경을 활성화합니다.

```bash
conda activate mlagents
```

--------------------------------------------------------------------------------------------------

## 3. Python 패키지 설치

프로젝트 루트 폴더에서 아래 명령어를 실행합니다.

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 안에는 ML-Agents 패키지가 editable 모드로 연결되어 있습니다.

```txt
-e ./ml-agents-release_22/ml-agents
-e ./ml-agents-release_22/ml-agents-envs
```

따라서 별도로 `pip install -e` 명령어를 다시 입력할 필요 없이, `requirements.txt` 설치만으로 로컬 ML-Agents 코드가 연결됩니다.

설치가 끝난 뒤 아래 명령어로 확인할 수 있습니다.

```bash
python -m pip show mlagents
python -m pip show mlagents-envs
```

정상적으로 설치되었다면 `Editable project location`이 clone한 프로젝트 내부 경로를 가리켜야 합니다.

예시:

```txt
Editable project location: .../PINN_PPO/ml-agents-release_22/ml-agents
Editable project location: .../PINN_PPO/ml-agents-release_22/ml-agents-envs
```

추가로 import 확인을 하려면 아래 명령어를 실행합니다.

```bash
python -c "import mlagents, mlagents_envs; print(mlagents.__file__); print(mlagents_envs.__file__)"
```

--------------------------------------------------------------------------------------------------

## 4. Unity 프로젝트 열기

Unity Hub를 실행한 뒤 clone한 프로젝트 폴더를 추가합니다.

```txt
Unity Hub → Add → clone한 PINN_PPO 폴더 선택
```

프로젝트를 열었을 때 Safe Mode로 진입하라는 메시지가 나오면 Safe Mode로 들어갑니다.

--------------------------------------------------------------------------------------------------

## 5. Unity ML-Agents 패키지 연결

Unity 상단 메뉴에서 Package Manager를 엽니다.

```txt
Window → Package Manager
```

Package Manager 창에서 왼쪽 위의 `+` 버튼을 누른 뒤 아래 항목을 선택합니다.

```txt
Add package from disk...
```

그다음 clone한 프로젝트 폴더 안에서 아래 파일을 선택합니다.

```txt
ml-agents-release_22/com.unity.ml-agents/package.json
```

예시 경로:

```txt
PINN_PPO/ml-agents-release_22/com.unity.ml-agents/package.json
```

이 과정을 완료하면 Unity 프로젝트에 ML-Agents 패키지가 로컬 패키지로 연결됩니다.

--------------------------------------------------------------------------------------------------

## 6. Unity UI 패키지 설치

Package Manager 창에서 왼쪽 위의 패키지 표시 범위를 변경합니다.

```txt
Packages: In Project → Unity Registry
```

검색창에 아래 키워드를 입력합니다.

```txt
Unity UI
```

검색 결과에서 `Unity UI` 패키지를 선택한 뒤 `Install` 버튼을 눌러 설치합니다.

--------------------------------------------------------------------------------------------------

## 7. 실행 전 확인 사항

Python 쪽에서 가상환경이 활성화되어 있는지 확인합니다.

```bash
conda activate mlagents
```

ML-Agents 명령어가 정상적으로 실행되는지 확인합니다.

```bash
mlagents-learn --help
```

PyTorch CUDA 사용 가능 여부를 확인하려면 아래 명령어를 실행합니다.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`True`가 출력되면 PyTorch에서 GPU를 사용할 수 있는 상태입니다.

--------------------------------------------------------------------------------------------------

## 8. 전체 설치 요약

```bash
git clone https://github.com/HB8514/PINN_PPO.git
cd PINN_PPO

conda create -n mlagents python=3.10.12
conda activate mlagents

python -m pip install -r requirements.txt
```

그다음 Unity Hub에서 프로젝트를 열고, Package Manager에서 아래 패키지를 연결합니다.

```txt
Add package from disk:
ml-agents-release_22/com.unity.ml-agents/package.json
```

그리고 Unity Registry에서 `Unity UI` 패키지를 설치합니다.
