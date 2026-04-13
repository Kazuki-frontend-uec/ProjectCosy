# 日本語声質制御TTS：Japanese Style-Controlled TTS (CosyVoice-based)

このプロジェクトは自然言語文により感情・韻律・話者といった声質を詳細に制御できるTTSを目指す。
システム構成は CosyVoice2 (https://github.com/Render-AI-Team/CosyVoice2) を着想を得ている。
最終的な目標は以下の様な声質表現文により、感情・韻律・話者といった声質の一括制御、Zero-shotでの音声模倣等の豊かな声質制御性を持つTTSシステムである。

声質表現文
(Coco-Nut Corpus: https://sites.google.com/site/shinnosuketakamichi/research-topics/coconut_corpus)
* 「30代くらいの男性の声。ゆっくりと穏やかな話し方でした。苦悩に満ちた、けだるそうな声でした。」
* 「元気な男性が明るい声で、テンション高く発表をするように喋っている。」
* 「若い女性が、抑揚のある声で、ゆっくりと喋っている」

This project aims to develop a **Japanese text-to-speech (TTS) system that allows voice control using natural language descriptions**.

The system is inspired by CosyVoice2 and extends its framework to support **Japanese speech generation and voice-quality control through textual descriptions**.

The long-term goal is to create a TTS system where users can specify **both speaking style and speaker characteristics using natural language**.

---

# 概要：Overview

近年のTTSではdiffusion-modelの台頭や計算資源の充実により自然な韻律が可能になっている。
しかしながら、話者や韻律の制御については未だ話者IDや感情ラベルに頼った方法が主流である。
数秒のサンプルによる話者と韻律のZero-shot模倣はあるが、それもサンプル音源に依存する問題がある。

本プロジェクトでは、ユーザーが自然言語の表現を用いて声の特徴を制御できる、より直感的なインターフェースを探求する。具体的には声質表現文を学習済みLM-TTSに条件付けて追加学習を行う。
本研究では、こうした**声質を表す自然言語表現が、音声合成システムにおける制御インターフェースとして機能するか**を検証する。

また、本システムは**日本語音声生成**に特化して設計している。

Recent text-to-speech systems can generate highly natural speech.
However, controlling **voice style or speaker characteristics** typically requires predefined speaker IDs or style tokens.

This project explores a more intuitive interface where users can control voice characteristics using natural language expressions such as:

* "calm voice"
* "energetic voice"
* "soft and gentle voice"

The research investigates whether **voice-quality descriptions can function as a controllable interface for speech synthesis systems**.

The system is designed specifically for **Japanese speech generation**.

---

# 提案システム：System Pipeline

本研究では、日本語音声生成における自然言語による声質制御を実現するための基盤として、**音声を離散トークン列として扱う音声生成フレームワーク**の構築を目指す。

システムは大きく以下の2段階から構成される。

1. 音声のトークン化（Acoustic Tokenization）
1. 言語モデルによる音声トークン生成（Speech Token Language Modeling）

The proposed system consists of the following stages.

## 1. 音声トークン化：Acoustic Tokenization

まず、音声信号を離散トークンへ変換する技術を確立する。
CosyVoice に準拠する場合、学習済みASRモデルのエンコーダ出力を量子化する手法が採用されている。
具体的には、ASRエンコーダの出力特徴量に対して以下の量子化手法を適用する。

* VQ (Vector Quantization)
* FSQ (Finite Scalar Quantization)

これにより、連続的な音響表現を離散トークン列へと変換する。

### この手法の利点

この方法には以下の利点がある。

* ASRの中間表現を利用するため、話者性がある程度除去される
* 意味情報が強調された表現になる
* コードブックサイズを柔軟に調整できる

特に、ASRの中間表現は言語内容を強く反映するため、
音声生成モデルにおいて テキストとの対応を学習しやすい特徴量になる。

### 課題

一方で、この方法には以下の課題もある。

* 話者性が完全には除去されない
* トークンの情報が 意味と話者の中間的表現になりやすい
* システム設計によっては 中途半端な表現になってしまう可能性がある

そのため、トークン設計の段階で 話者性・意味情報・音響情報のバランスを検討する必要がある。

### 代替アプローチ

代替案として、音声コーデック型のトークナイザの利用も検討している。
例として WavTokenizer などがある。
このアプローチでは、音声コーデックモデルを用いて
音声信号を離散トークンへ変換する。

音声コーデックを用いる方法には次の利点がある。

* トークナイザの学習時に デコーダも同時に学習される
* 音声復元（デコード）が容易
* トークンが 音響再現性を強く保持する

特に、音声生成システムでは最終的に波形復元が必要になるため、
デコードが容易である点は大きな利点となる。


Intermediate representations from a pretrained TTS model are discretized using quantization methods.

Possible approaches include:

* FSQ (Finite Scalar Quantization)
* VQ (Vector Quantization)

Additionally, the project is exploring the use of:

* WavTokenizer

for more effective speech tokenization.

This step converts continuous acoustic representations into **discrete speech tokens**.

---

## 2. 言語モデルによる音声生成：Language Model Training
言語モデルによる音声生成

次のステップでは、音声トークン列を生成する言語モデルを学習する。

ここでは

* テキストアノテーション
* 音声トークン列

のペアを用いて、言語モデルにトークン列生成を学習させる。
入力形式は以下のようになる。

[Text Annotation]
<speech_start>
[token1 token2 token3 ...]
<speech_end>

音声トークンの開始と終了を示す 特殊トークンを導入することで、
言語モデルが音声トークン列の生成構造を学習できるようにする。

CosyVoice では、ベースとなる言語モデルとして Qwen (0.5B) が使用されている。

本研究では、日本語音声生成を対象とするため

* 日本語対応
* 適切なモデルサイズ
* 追加トークンへの対応

といった観点から、以下の言語モデルの利用を検討している。

* Qwen 系モデル
* Rakuten LM
* GPT-oss
* その他日本語対応LLM


The extracted speech tokens are added to the vocabulary of a pretrained language model.

The model is then trained to generate **speech tokens conditioned on text input**.

Training datasets include:

* ReazonSpeech
* YouTube speech data

The pipeline used to collect YouTube speech data is available in a separate repository:

https://github.com/Kazuki-frontend-uec/YouTubeData

---

## 3. Token-to-MelSpectrogram Generation

A **diffusion-based model** is trained to convert acoustic tokens into Mel-spectrograms.

```
Speech Tokens → Diffusion Model → Mel Spectrogram
```

This module generates acoustic features which are later converted to waveform audio by a vocoder.

---

## 4. Voice Style Conditioning

Voice-quality descriptions are incorporated into the input sequence of the language model.

The model is then fine-tuned using the voice description dataset:

* Coco-Nut

This dataset contains pairs of:

* speech recordings
* textual voice-quality descriptions

The language model learns the relationship between **textual voice descriptions and acoustic speech characteristics**.

---

# Research Contributions

Compared to CosyVoice2, this project introduces several research directions.

## Natural Language Voice Control

Voice-quality descriptions are introduced as an additional conditioning signal.

Example:

```
Input:
"やさしい声で穏やかに喋っています"
```

This allows intuitive voice control without manually selecting speaker IDs or style tokens.

---

## Speaker Control via Text (Future Goal)

In the long term, voice descriptions may control not only **speaking style** but also **speaker identity**.

Example concept:

```
"深みのある男性の声でゆっくり話す"
```

This approach could unify **style control and speaker specification** using natural language.

---

## Alternative Acoustic Tokenization

The project also explores using:

* WavTokenizer

for improved speech token representation compared to standard quantization approaches.

---

# Current Progress

The project is currently in **Stage 2 (Language Model Training)**.

Completed:

* Acoustic token extraction
* Token integration into the language model

In progress:

* Large-scale language model training using speech tokens

Future stages include:

* Diffusion Token-to-Mel model training
* Style-conditioned fine-tuning
* Full pipeline evaluation

---

# TODO

## Model Engineering

* Hack the tokenization component of CosyVoice2 using ONNX export

## Acoustic Tokenization Research

* Investigate optimal **ASR-based architectures** for speech tokenization

## LLM Adaptation

Improve Japanese output support for the GPT implementation in:

* unthroth library (gpt-oss)

---

# Datasets

This project utilizes several datasets:

* ReazonSpeech
* YouTube speech dataset (self-collected)
* Coco-Nut

---

# Research Motivation

Current speech synthesis systems can generate natural audio but lack **flexible and interpretable control mechanisms**.

This research explores whether **natural language descriptions of voice characteristics** can serve as a universal interface for speech synthesis control.

If successful, users could control speech generation using intuitive language rather than predefined speaker or style parameters.

---

# Status

🚧 Work in Progress

---

# Author

Graduate student researching speech synthesis, speech generation, and style-controllable TTS systems.
