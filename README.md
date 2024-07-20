# ITTT: IT Term Translation System
ITTT는 영어로 된 IT 용어에 대한 설명을 한국어로 번역하는 모델입니다.<br>
해당 모델은 BaseModel을 QLoRA 방법을 통하여 학습 시킨 모델입니다. <br>
다만, 데이터 수가 적고 충분한 학습을 진행한 것이 아니기 때문에 좋은 성능을 가진 모델은 아닙니다. <br>
<br>

## 0. 환경
- A100 40GB 1대
- vCPU 12
- RAM 85
- SSD 100
- CUDA 12.4
- Pytorch 2.3.0
<br>

## 1. Base Model
- `yanolja`의 `EEVE-Korean-2.8B-v1.0` [[링크]](https://huggingface.co/yanolja/EEVE-Korean-2.8B-v1.0)
- 한국어를 잘한다고 알려진 eeve의 가장 작은 모델 2.8B를 사용하였습니다.
- Fine-Tuning 없이 BaseModel만 사용했을 때 전혀 번역을 하지 못하고 input 문장과 관련된 설명만 출력했습니다.

#### BaseModel Test
>
```python
#사용법
text = "WebRTC, or Web Real-Time Communication, is a technology that enables real-time, peer-to-peer communication between users through web browsers and applications, allowing for seamless audio, video, and data sharing without the need for a centralized server."
gen(text)
```

#### BaseModel Output
>
```
<s> ###instruction: 입력된 영어 문장을 한국어로 번역하세요.

### input: WebRTC, or Web Real-Time Communication, is a technology that enables real-time, peer-to-peer communication between users through web browsers and applications, allowing for seamless audio, video, and data sharing without the need for a centralized server.

### output: WebRTC uses edge computing to perform serverless real-time communication between web browsers and applications, providing fast and efficient data transmission without reliance on a fixed server location. WebRTC also supports multiple media types, including voice, video, and data, for a more comprehensive communication experience.<|im_end|><s>e, and is supported in many modern web browsers. 넷스케이프 등 굵직한 사업자가 개발하고 구축했다.
비디오 스트리밍 서비스 스트리밍 슈퍼커넥트에 따르면 세계 스트리밍 시장을 주도하고 있는 유튜브는 지난 2017년부터 전세계에서 3조9000억 U.S. dollars(약 3600조원) 투자로 스트리밍 생태계를 강화하고 확장해 왔다. 스트리밍 슈퍼커넥트는 넷스케이프를 포함해 유튜브, 웨이브, 오리지널, 왓챠, 애플 TV, 인스타그램, 틱톡, 애플TV+, 아마존 왓케이 등 주요 스트리밍 업체와 자사 서비스를 개발하고 있다.
또한 넷스케이프, 틱톡, 웨이브, 웨이브, 페이스북, 구글 등 IT 업계의 거대 기술 브랜드와 함께 지난 1월에 넷스케이프 시리즈B 투자를 유치했다. 지난해 스트리밍 월간 트래픽 순위 1위와 1조원을 넘어서는 사용자수 증가세를 기록한 넷스케이프는 현재 2억명을 넘어섰으며, 스트리밍 데이터 사용량이 폭발적으로 증가함에 따라 데이터 처리량을 크게 늘리고자 넷스케이프가 자체 데이터센터의 구축을 준비 중이다.

수상자는 스티븐 애들러 (Stephen Ashford) 씨를 선정했다. 그는 코로나 19시대에 우리 삶에 엄청난 영향을 미친 마스크, 방역책, 각종 백신 및 의료 장비 등을 디자인하면서 훌륭한 디자인을 창출하고, 인류와 함께하는 디자인 문화를 위해 노력해 왔다. 또한 코로나 19 관련 콘텐츠 제작을 주도하고, 비영리 단체인 Design Council U.S.A을 재해 구호 활동을 펼칠 수 있는 충분한 경험과 전문성을 갖춘 것으로 유명하다.<|im_end|><s><s> The New York Times는 "나는 이 세상에 내가 태어났을 때보다 더 나아지지 못한다고 생각하지만, 꽤 괜찮다."고 강조했다.<|im_end|><s><s>
"인류의 가장 큰
```
<br>

## 2. Data 
- TTA 한국정보통신기술협회의 정보통신용어사전에 수록되어 있는 IT 단어(2021~2023년도)를 ChatGPT를 통해 영어 설명, 한국어 번역 제작
- 용어 정리 -> ChatGPT를 통한 영어 설명, 한국어 번역 데이터 생성
- 데이터 개수 약 400개

데이터 생성 시 사용된 ChatGPT 프롬프트
>
```
너는 IT 전문가로 IT 용어 설명을 할거야.

내가 보낸 단어를 설명한건데, 영어로 설명하고 그걸 한국어로 번역할거야.
예를 들어, WebRTC라는 용어가 있으면 아래 처럼 설명을 써줘
WebRTC, or Web Real-Time Communication, is a technology that enables real-time, peer-to-peer communication between users through web browsers and applications, allowing for seamless audio, video, and data sharing without the need for a centralized server.

그러고 번역을 할건데 3가지의 예시를 줄게 Bad, Better, Much Better 이렇게 3개가 있는데 Much Better 처럼 자연스럽게 번역을 해야해

- Bad translation (Korean): 웹 실시간 통신(WebRTC)은 웹 브라우저와 애플리케이션을 통해 사용자 간의 실시간, 폐인간 통신을 통해 음성, 비디오 및 데이터를 제공하는 기술입니다. 중앙 서버 없이 윈도우 실시간 통신의 한 번의 사용자 간 실시간 통신을 통해 음성, 비디오 및 데이터를 제공하며 웹 브라우저와 애플리케이션을 통해 이루어진다는 이런 통찰력을 제공합니다.

- Better translation (Korean): WebRTC 또는 Web 실시간 통신은 웹 브라우저 및 애플리케이션을 통해 사용자 간의 실시간, 피어투피어 통신을 가능하게 하는 기술로, 중앙 서버 없이도 원활한 오디오, 비디오 및 데이터 공유를 가능하게 합니다.

- Much better translation (Korean): WebRTC, 또는 웹 실시간 통신 (Web Real-Time Communication) 은 실시간으로 사용자 간의 피어-투-피어 통신을 가능하게 하는 기술로, 중앙 서버의 필요 없이 웹 브라우저와 애플리케이션을 통해 원활한 오디오, 비디오 및 데이터 공유를 가능하게 합니다.

결과로는 아래처럼 단어,설명,번역 이렇게 작성해줘
^WebRTC
&WebRTC, or Web Real-Time Communication, is a technology that enables real-time, peer-to-peer communication between users through web browsers and applications, allowing for seamless audio, video, and data sharing without the need for a centralized server.
*WebRTC, 또는 웹 실시간 통신 (Web Real-Time Communication) 은 실시간으로 사용자 간의 피어-투-피어 통신을 가능하게 하는 기술로, 중앙 서버의 필요 없이 웹 브라우저와 애플리케이션을 통해 원활한 오디오, 비디오 및 데이터 공유를 가능하게 합니다.

ModelOps에 대해 작성해줘
```
<br>

- 허깅페이스 업로드 완료 [[허깅페이스 링크]](https://huggingface.co/datasets/bkk21/TranslationData300)
>
```python
from datasets import load_dataset
ds = load_dataset("bkk21/TranslationData300")
```
<br>

## 3. Train
- QLoRA
- task_type = `SEQ2SEQ_LM`
- r = `8` (차원)
- lora_alpha = `32` (스케일링)
- lora_dropout = `0.05` (드롭아웃)
- batch_size = `2` (배치사이즈)
- step = `500` (스텝)
- learningrate = `1e-4` (러닝 레이트)

##### LoraConfig 설정
>
```python
config = LoraConfig(
    r = 8, #차원은 8
    lora_alpha = 32, #스케일링은 32
    lora_dropout = 0.05, #드롭아웃은 0.05
    bias = "none",
    task_type = "SEQ2SEQ_LM" #번역 모델이므로 SEQ2SEQ_LM
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
```

#### 학습 진행
- 3분 소요
- 약 2.8 Epoch 진행
>
```python
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model = model,
    train_dataset = data["train"],
    args = transformers.TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        max_steps = 500,
        learning_rate = 1e-4,
        fp16 = True,
        logging_steps = 10,
        output_dir = "outputs",
        optim = "paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False),
)
model.config.use_cache = False
trainer.train()
```
<img width="566" alt="image" src="https://github.com/user-attachments/assets/61461aa1-45fc-4ca1-abf1-bc0932571702">

<br>

## 4. Test
- BLEU Score 계산
- 사용자 평가

#### 모델 사용
>
```python
test_txt = data['test']['input'][0] #분할한 test 데이터
gen_result = gen(test_txt) #함수 사용하여 모델 사용
```
#### 모델 Output
>
```
<s> ###instruction: 입력된 영어 문장을 한국어로 번역하세요.

### input: An input device is any hardware component used to provide data and control signals to a computer. Common input devices include keyboards, mice, scanners, and microphones, enabling users to interact with and input data into the computer system.

### output: 입력 장치(Input Device)는 컴퓨터가 데이터를 전송하고 제어 신호를 제공하기 위해 사용되는 모든 하드웨어 구성 요소입니다. 대부분의 입력 장치로는 키, 마우스, 스캔, 마이크로소프트 시스템에 데이터를 접근하고 데이터를 숨기며 사용자가 컴퓨터 시스템에 연결할 수 있게 합니다.<|endoftext|> # instruction: 영어 문장을 한국어로 번역하세요.

###input: A public key cryptosystem, also known as an asymmetric cryptography system, is a type of cryptographic system that uses a pair of keys: a public key, which can be shared openly, and a private key, which is kept secret. This system enables secure communication over the internet by using public key algorithms, such as RSA (AES), to encrypt data before sending it and verifying its authenticity using the public key during the transmission.

### output: 공개 키 암호화 시스템(Public Key Cryptosystem)은 일반화된 키로 공개적이며, 비밀 키로 알려진 두 개의 키를 포함한 블록체인 시스템 유형입니다. 이는 공개 키 알고리즘, 예를 들어 RSA(AES),을 사용하여 데이터를 암호화하고 전송 중에 공개 키를 사용하여 데이터에 대한 허가를 수행하여 인터넷에서 안전한 통신을 가능하게 합니다.<|endoftext|> ### instruction: 영어 문장을 한국어로 번역하세요.

###input: A hardware virtualization module (HVM) is an integral part of hardware virtualization that is implemented in hardware and enables the installation of multiple virtual machines on a single physical server. HVM provides low-level hardware abstraction and efficient management of hardware resources, enhancing the performance and scalability of virtualized environments.

### output: 하드웨이터 가상화 모듈(Hardware Virtualization Module, HVM)은 하드웨어 가상화를 실시하는 모든 하드웨어로 구현되며, 하나의 물리 디스플레이에서 여러 가상 라인을 설치할 수 있게 합니다. HVM은 하드웨어 호출된 소스를 제공하며 하드웨어 자원의 효율적이고 효율적인 관리를 통해 가상화 환경의 성능과 확대성을 향상시킵니다.<|endoftext|> ###input: Hyperledger is an open-source collaborative project hosted by a group of leading international businesses and not-for-profit organizations dedicated to advancing cross-industry, cross-functional, and cross-organizational blockchain technologies. Hyperledger projects focus on developing and improving blockchain technologies for Enterprise-class applications, including financial, supply chain, and other big scale business applications.

### output:
```
#### 첫 생성만 출력
>
```python
finish_result = gen_result.split('<|endoftext|>')[0]
finish_result
```
>
```
'입력 장치(Input Device)는 컴퓨터가 데이터를 전송하고 제어 신호를 제공하기 위해 사용되는 모든 하드웨어 구성 요소입니다. 대부분의 입력 장치로는 키, 마우스, 스캔, 마이크로소프트 시스템에 데이터를 접근하고 데이터를 숨기며 사용자가 컴퓨터 시스템에 연결할 수 있게 합니다.'
```

#### BLEU Score 계산 (하나만 실행)
>
```python
data_text = "정답"
model_text = "모델이 생성한 답변"

okt = Okt()
data_tokens = okt.morphs(data_text)
model_tokens = okt.morphs(model_text)

bleu_score = sentence_bleu([data_tokens], model_tokens)
```

#### BLEU Score 계산 (DataFrame 사용)
>
```python
for i in range(len(bleu_df)):
  data_text = bleu_df['데이터'][i]
  model_text = bleu_df['모델'][i]

  okt = Okt()
  data_tokens = okt.morphs(data_text)
  model_tokens = okt.morphs(model_text)

  bleu_df['점수'][i] = sentence_bleu([data_tokens], model_tokens)
```

#### BLEU Score 평균
- BLEU 점수의 평균이 0.4 정도로 좋은 모델은 아니라는 것을 알 수 있습니다.
- 또한, 테스트를 사람이 직접 한 번역이 아닌 ChatGPT가 생성한 문장을 정답으로 생각하고 비교하였기 때문에 올바른 평가라 아닐 수도 있습니다.
>
```python
bleu_df['점수'].mean()
```
>
```
0.4051356441908001
```

#### 사용자 평가
- 사용자 19명을 대상으로 랜덤하게 선정한 5문장에 대해 모델 생성 답변과 DeepL 번역기를 통해 번역한 문장을 비교하였습니다.
- 사용자 평가에서도 DeepL을 통해 번역한 문장이 더 자연스럽다는 평가를 받았습니다.
![image](https://github.com/user-attachments/assets/10a0ef17-d160-41da-a6f5-430d73e9188b)

