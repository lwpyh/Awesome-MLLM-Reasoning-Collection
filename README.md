# Awesome-MLLM-Reasoning-Collection
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

👏 Welcome to the Awesome-MLLM-Reasoning-Collections repository! This repository is a carefully curated collection of papers, code, datasets, benchmarks, and resources focused on reasoning within Multi-Modal Large Language Models (MLLMs).

Feel free to ⭐ star and fork this repository to keep up with the latest advancements and contribute to the community.
### Table of Contents
- [Papers and Projects](#PapersandProjects)
  * [Commonsense Reasoning](#VQA)
  * [Reasoning Segmentation and Detection](#ReasoningSegmentation)
  * [Spatial and Temporal Grounding and Understanding](#Spatio-TemporalReasoning)
  * [Math Reasoning](#MathReasoning)
  * [Chart Rasoning](#ChartRasoning)
  * [Visual Generation](#VisualGeneration)
  * [Others](#others)
- [Benchmarks](#benchmarks)
- [Open-source projects](#Open-sourceprojects)
- [Contibuting](#Contributiong)


<a name="PapersandProjects"></a>
## Papers and Projects 📄

<a name="VQA"></a>
### Commonsense Reasoning
#### Image MLLM
* 25.03 [VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](https://arxiv.org/pdf/2503.07523) | [Paper📑](https://arxiv.org/pdf/2503.07523) [Code🖥️](https://github.com/zhangquanchen/VisRL) 
  - The first framework that applies reinforcement learning (RL) to the problem of intention-driven visual perception | Task: VQA
* 25.03  [Unified Reward Model for Multimodal Understanding and Generation](https://arxiv.org/abs/2503.05236) | [Paper📑](https://arxiv.org/abs/2503.05236) [Code🖥️](https://codegoat24.github.io/UnifiedReward/) [Dataset🤗](https://huggingface.co/collections/CodeGoat24/unifiedreward-training-data-67c300d4fd5eff00fa7f1ede)
  -  Improve MLLM's understanding and generation ability with DPO  | Task: VQA & Generation
* 25.02 [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923) | [Paper📑](https://arxiv.org/pdf/2502.13923) [Code🖥️](https://github.com/QwenLM/Qwen2.5-VL) [Huggingface🤗](https://huggingface.co/Qwen)
   - The latest flagship model of Qwen vision-language series for various multimodal tasks | Task: Reasoning & Understainding
* 25.02 [MM-RLHF: The Next Step Forward in Multimodal LLM Alignment](https://arxiv.org/abs/2502.10391) | [Paper📑](https://arxiv.org/abs/2502.10391) [Project🌐](https://mm-rlhf.github.io/)
  - A comprehensive project for aligning MlLMs with human preferences | Task: Reward & VQA
* 25.01 [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599) (MoonshotAI) | [Project🌐](https://github.com/MoonshotAI/Kimi-k1.5)
  - The latest flagship model of Kimi series for various multimodal tasks | Task: Reasoning & Understainding
* 25.01 [InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model](https://arxiv.org/abs/2501.12368) | [Paper📑](https://arxiv.org/abs/2501.12368) [Code🖥️](https://github.com/InternLM/InternLM-XComposer)
  - A simple yet effective multi-modal reward model that aligns MLLMs with human preferences | Reward & VQA  
* 25.01 [LlamaV-o1: Rethinking Step-By-Step Visual Reasoning in LLMs](https://arxiv.org/abs/2501.06186) | [Paper📑](https://arxiv.org/abs/2501.06186) [Code🖥️](https://github.com/mbzuai-oryx/LlamaV-o1)
  - A combined multi-step curriculum learning and beam search multimodal reasoning model |  VQA
* 25.01 [ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding](https://arxiv.org/pdf/2501.05452) | [Paper📑](https://arxiv.org/pdf/2501.05452) [Code🖥️](https://github.com/zeyofu/ReFocus_Code) [Model🤗](https://huggingface.co/Fiaa/ReFocus)
  - Perform visual chain of thought via input-image editing to help multimodal reasoning. | Task: VQA
* 24.12 [Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search](https://arxiv.org/abs/2412.18319) | [Paper📑](https://arxiv.org/abs/2412.18319) [Code🖥️](https://github.com/HJYao00/Mulberry) [Dataset🤗](https://huggingface.co/datasets/HuanjinYao/Mulberry-SFT)
  - Improve MLLM reasoning ability via collective monte carlo tree search | VQA 
* 24.11  [LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](https://arxiv.org/abs/2411.10440) | [Paper📑](https://arxiv.org/abs/2411.10440) [Code🖥️](https://github.com/PKU-YuanGroup/LLaVA-CoT) [Model🤗](https://huggingface.co/Xkev/Llama-3.2V-11B-cot)
  -  A novel MLLM designed to conduct autonomous multistage reasoning. | VQA
* 24.11 [Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](https://arxiv.org/abs/2411.14432) | [Paper📑](https://arxiv.org/abs/2411.14432) [Code🖥️](https://github.com/dongyh20/Insight-V) [Model🤗](https://huggingface.co/collections/THUdyh/insight-v-673f5e1dd8ab5f2d8d332035)
  - Explore long-chain visual reasoning with MLLMs  | VQA  
* 24.11 [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://arxiv.org/abs/2411.10442) | [Paper📑](https://arxiv.org/abs/2411.10442) [Code🖥️](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl2.0_mpo) [Model🤗](https://huggingface.co/OpenGVLab/InternVL2-8B-MPO)
  - A preference optimization (PO) process to enhance the multimodal reasoning capabilities of MLLMs.  | VQA                
* 24.10 [Improve Vision Language Model Chain-of-thought Reasoning](https://arxiv.org/pdf/2410.16198) | [Paper📑](https://arxiv.org/pdf/2410.16198) [Code🖥️](https://github.com/RifleZhang/LLaVA-Reasoner-DPO)
  - Apply reinforcement learning on 193k CoT sft data for reasoning | VQA    
* 24.03  (NeurIPS24)[Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning](https://proceedings.neurips.cc/paper_files/paper/2024/file/0ff38d72a2e0aa6dbe42de83a17b2223-Paper-Datasets_and_Benchmarks_Track.pdf) | [Paper📑](https://proceedings.neurips.cc/paper_files/paper/2024/file/0ff38d72a2e0aa6dbe42de83a17b2223-Paper-Datasets_and_Benchmarks_Track.pdf) [Code🖥️](https://github.com/deepcs233/Visual-CoT) 
 [Dataset🤗](https://huggingface.co/datasets/deepcs233/Visual-CoT)
  - Visual CoT for improve MLLMs' reasoning ability | VQA
* 23.02 [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923) | [Paper📑](https://arxiv.org/abs/2302.00923) [Code🖥️](https://github.com/amazon-science/mm-cot)
  - Visual CoT for MLLM reasoning | VQA

#### Video MLLM
* 25.02 [video-SALMONN-o1: Reasoning-enhanced Audio-visual Large Language Model](https://arxiv.org/abs/2502.11775) | [Paper📑](https://arxiv.org/abs/2502.11775) [Demo🖥️](https://github.com/BriansIDP/video-SALMONN-o1)
  - A open-source reasoning-enhanced audio-visual LLM designed for general video understanding tasks.  | Task: Video QA  
* 25.02 [Open-R1-Video]((https://github.com/Wang-Xiaodong1899/Open-R1-Video)) | [Code🖥️](https://github.com/Wang-Xiaodong1899/Open-R1-Video) [Dataset🤗](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k)
  - A open-source R1-style video understanding model | Task: Video QA
* 25.01 [Temporal Preference Optimization for Long-Form Video Understanding](https://arxiv.org/abs/2501.13919) | [Paper📑](https://arxiv.org/abs/2501.13919)[Code🖥️](https://ruili33.github.io/tpo_website/)
  - A novel post-training framework designed to enhance the temporal grounding capabilities of video-LMMs through preference learning | Task: Video QA
* 25.01 [Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding](https://arxiv.org/abs/2501.07888https://github.com/bytedance/tarsier) | [Paper📑](https://arxiv.org/abs/2501.07888) [Code🖥️](https://github.com/bytedance/tarsier?tab=readme-ov-file) 
 [Model🤗](https://huggingface.co/omni-research/Tarsier-34b)
  - A family of VLMs designed for high-quality video captioning and understanding | Task: Video captioning & QA 

<a name="ReasoningSegmentation"></a>
### Reasoning Segmentation and Detection
#### Image MLLM
* 25.03 [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785) | [Paper📑](https://arxiv.org/abs/2503.01785) [Code🖥️](https://github.com/Liuziyu77/Visual-RFT) 
 [Dataset🤗](https://huggingface.co/collections/laolao77/virft-datasets-67bc271b6f2833eccc0651df) 
  - Extend Reinforcement Fine-Tuning on visual tasks with GRPO   | Task: Detection & Grounding & Classification
* 25.03 [Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning](https://arxiv.org/pdf/2503.07065) | [Paper📑](https://arxiv.org/pdf/2503.07065)
  - Improve generalization and reasoning of VLMs with GRPO | Task: Detection & Classification & Math
* 25.03 [Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement](https://arxiv.org/abs/2503.06520) | [Paper📑](https://arxiv.org/abs/2503.06520) [Code🖥️](https://github.com/dvlab-research/Seg-Zero) [Model🤗](https://huggingface.co/Ricky06662/Seg-Zero-7B)
  - Address object detection and segmentation with GRPO | Task: Object Detection & Object Segmentation

<a name="Spatio-TemporalReasoning"></a>
### Spatial and Temporal Grounding and Understanding
#### Image MLLM
* 25.03 [VisualThinker-R1-Zero](https://arxiv.org/abs/2503.05132)  [Paper📑](https://arxiv.org/abs/2503.05132) | [Code💻](https://github.com/turningpoint-ai/VisualThinker-R1-Zero)
  - R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model | Task: Counting & Reasoning & 3D Understanding (CV-Bench)
* 25.03 (CVPR2025)[GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks](https://arxiv.org/pdf/2503.06514) | [Paper📑](https://arxiv.org/pdf/2503.06514)
  - Fine-tune VLMs using GFlowNet to promote generation of diverse solutions.|  Task: NumberLine (NL) & BlackJack (BJ)
* 25.02 [R1-V: Reinforcing Super Generalization Ability in Vision Language Models with Less Than $3](https://github.com/Deep-Agent/R1-V) |  [Code🖥️](https://github.com/Deep-Agent/R1-V)
  - A open-source project for VLM reasoning with GRPO | Task: Counting, Number Related Reasoning and Geometry Reasoning
* 25.01 [Imagine while Reasoning in Space: Multimodal Visualization-of-Thought] | [Paper📑](https://arxiv.org/pdf/2501.07542)
  - Enables visual thinking in MLLMs by generating image visualizations of their reasoning traces.  | Task: Spatial Reasoning
#### Video MLLM
* 25.03 [MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse](https://github.com/PzySeere/MetaSpatial) | [Code🖥️](https://github.com/PzySeere/MetaSpatial)
  - Enhance spatial reasoning in VLMs using GRPO  | Task: 3D Spatial Reasoning
* 25.02 [Video-R1: Towards Super Reasoning Ability in Video Understanding](https://github.com/tulerfeng/Video-R1) | [Code🖥️](https://github.com/tulerfeng/Video-R1)
  - Integrate deep thinking capabilities into video understanding tasks through the R1 paradigm | Task:  Video Counting 
* 24.12 [TIMEREFINE: Temporal Grounding with Time Refining Video LLM](https://arxiv.org/pdf/2412.09601) | [Paper📑](https://arxiv.org/pdf/2412.09601) | [Code🖥️](https://github.com/SJTUwxz/TimeRefine)
  * Enhance Video LLMs to handle the temporal grounding task by modifying the learning objective
* 24.11 (CVPR2025) [Number it: Temporal Grounding Videos like Flipping Manga](https://arxiv.org/pdf/2411.10332) | [Paper📑](https://arxiv.org/pdf/2411.10332) | [Code💻](https://github.com/yongliang-wu/NumPro)
  * Enhances Video-LLMs by overlaying frame numbers onto video frames
* 24.11 [TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability](https://arxiv.org/abs/2411.18211) | [Paper📑](https://arxiv.org/pdf/2411.18211) | [Code💻](https://github.com/TimeMarker-LLM/TimeMarker/)
  * A versatile Video-LLM featuring robust temporal localization abilities
* 24.08 (AAAI2025) [Grounded Multi-Hop VideoQA in Long-Form Egocentric Videos](https://arxiv.org/abs/2408.14469) | [Paper📑](https://arxiv.org/pdf/2408.14469) | [Code💻](https://github.com/qirui-chen/MultiHop-EgoQA)
  * Leverage the world knowledge reasoning capabilities of MLLMs to retrieve temporal evidence in the video with flexible grounding tokens.
* 24.08 (ICLR2025) [TRACE: Temporal Grounding Video LLM via Casual Event Modeling](https://arxiv.org/abs/2410.05643) | [Paper📑](https://arxiv.org/pdf/2410.05643) | [Code💻](https://github.com/gyxxyg/TRACE)
  * Tailored to implement the causal event modeling framework through timestamps, salient scores, and textual captions.

<a name="MathReasoning"></a>

### Math Reasoning
#### Image MLLM
* 25.03 [LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL](https://arxiv.org/pdf/2503.07536) | [Paper📑](https://arxiv.org/pdf/2503.07536) [Code🖥️](https://github.com/TideDra/lmm-r1) [Dataset🤗](https://huggingface.co/datasets/VLM-Reasoner/VerMulti)
  - A two-stage rule-based RL framework that efficiently enhances reasoning capabilities | Task: Math & Sokoban
* 25.03 [VisualPRM: An Effective Process Reward Model for Multimodal Reasoning](https://arxiv.org/abs/2503.10291) | [Paper📑](https://arxiv.org/abs/2503.10291) [Code🖥️](https://github.com/OpenGVLab/InternVL) [Dataset🤗](https://huggingface.co/datasets/OpenGVLab/VisualProcessBench)
  - Improve the reasoning abilities of existing MLLMs with Best-of-N evaluation strategies | Task: Math & MMMU  
* 25.03 [R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](https://arxiv.org/pdf/2503.10615) | [Paper📑](https://arxiv.org/pdf/2503.10615) [Code🖥️](https://github.com/Fancy-MLLM/R1-Onevision) [Dataset🤗](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision)
  - A multimodal reasoning model bridged the gap between multimodal capabilities and reasoning abilities with GRPO | Task: Math
* 25.03 [MMR1: Advancing the Frontiers of Multimodal Reasoning](https://github.com/LengSicong/MMR1) | [Code🖥️](https://github.com/LengSicong/MMR1)
  - a Large Multimodal Model specialized in mathematical tasks using GRPO | Task: Math
* 25.03 [Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning](https://arxiv.org/pdf/2503.07065) | [Paper📑](https://arxiv.org/pdf/2503.07065)
  - Improve generalization and reasoning of VLMs with GRPO | Task: Detection & Classification & Math
* 25.03 [Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](https://arxiv.org/abs/2503.06749) | [Paper📑](https://arxiv.org/abs/2503.06749)[Code🖥️](https://github.com/Osilly/Vision-R1)
  - Improve reasoning ability of MLLM with GRPO                                                         | Task: Math
* 25.03 [MM-EUREKA: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning](https://arxiv.org/abs/2503.07365) | [Paper📑](https://arxiv.org/abs/2503.07365) [Code🖥️](https://github.com/ModalMinds/MM-EUREKA) [Dataset🤗](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)
  - Extend large-scale rule-based reinforcement learning to multimodal reasoning                              | Task: Math
* 25.03 [EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework] [Code🖥️](https://github.com/hiyouga/EasyR1)
  - A Multimodal GRPO training framework              | Task: Math
* 25.02 [Qwen2.5-VL] [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923) | [Paper📑](https://arxiv.org/pdf/2502.13923) [Code🖥️](https://github.com/QwenLM/Qwen2.5-VL) [Huggingface🤗](https://huggingface.co/Qwen)
   - The latest flagship model of Qwen vision-language series for various multimodal tasks | Task: Reasoning & Understainding               * 25.02    [Multimodal Open R1]((https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)) | [Code🖥️](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) 
  - A open-source database for video R1 reproduce.    | Task: Math                    
* 25.02 [Boosting Multimodal Reasoning with MCTS-Automated Structured Thinking](https://arxiv.org/pdf/2502.02339) | [Paper📑](https://arxiv.org/pdf/2502.02339)
  - An automated structured thinking paradigm for multimodal reasoning via Monte Carlo Tree Search | Task: Math
* 25.02 [MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](https://www.arxiv.org/pdf/2502.13383) | [Paper📑](https://www.arxiv.org/pdf/2502.13383) [Code🖥️](https://github.com/Aurora-slz/MM-Verify)
  - Enhance multimodal reasoning through longer inference and more robust verification. | Task: Math
* 25.01 [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599) (MoonshotAI) | [Project🌐](https://github.com/MoonshotAI/Kimi-k1.5)
  - The latest flagship model of Kimi series for various multimodal tasks | Task: Reasoning & Understainding                  
* 25.01 [Virgo: A Preliminary Exploration on Reproducing o1-like MLLM](https://arxiv.org/abs/2501.01904) | [Paper📑](https://arxiv.org/abs/2501.01904) [Code🖥️](https://github.com/RUCAIBox/Virgo) [Model🤗](https://huggingface.co/RUC-AIBOX/Virgo-72B)
  - A o1-like MLLM for multimodal reasoning    |Task: Math & MMMU      

<a name="ChartRasoning"></a>
### Chart Rasoning

* 24.09 [ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning](https://arxiv.org/abs/2402.12185) | [Project🌐](https://unimodal4reasoning.github.io/DocGenome_page/)  [Paper📑](https://arxiv.org/abs/2402.12185) [Code🖥️](https://github.com/Alpha-Innovator/ChartVLM)
  - Offer a new perspective on handling chart reasoning tasks that strongly depend on interpretable patterns. | Task: Chart Reasoning
* 24.02 (ACL24) [ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning](https://arxiv.org/abs/2401.02384) | [Paper📑](https://arxiv.org/abs/2401.02384) [Code🖥️](https://github.com/OpenGVLab/ChartAst) [Dataset🤗]
  - Use large-scale chart data to align and instruction tuning | Task: Chart Reasoning

<a name="VisualGeneration"></a>
### Visual Generation
#### Image MLLM
* 25.03 [GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/pdf/2503.10639) | [Paper📑](https://arxiv.org/pdf/2503.10639) 
  - A paradigm that enables generation and editing through an explicit language reasoning process before outputting images   | Task: Image Generation
* 25.03  [Unified Reward Model for Multimodal Understanding and Generation](https://arxiv.org/abs/2503.05236) | [Paper📑](https://arxiv.org/abs/2503.05236) [Code🖥️](https://codegoat24.github.io/UnifiedReward/) [Dataset🤗](https://huggingface.co/collections/CodeGoat24/unifiedreward-training-data-67c300d4fd5eff00fa7f1ede)
  -  Improve MLLM's understanding and generation ability with DPO  | Task: VQA & Generation
* 25.01 (CVPR25) [Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step](https://arxiv.org/pdf/2501.13926) | [Paper📑](https://arxiv.org/pdf/2501.13926) [Code🖥️](https://github.com/ZiyuGuo99/Image-Generation-CoT) [Model🤗](https://huggingface.co/ZiyuG/Image-Generation-CoT)
  - The first comprehensive investigation of the potential of CoT reasoning to enhance autoregressive image generation. | Task: Image Generation
* 24.12 [EVLM: Self-Reflective Multimodal Reasoning for Cross-Dimensional Visual Editing](https://arxiv.org/pdf/2412.10566) | [Paper📑](https://arxiv.org/pdf/2412.10566)
  - A system designed to interpret such instructions in conjunction with reference visuals, producing precise and context-aware editing prompts.  | Task: Image Generation
#### Video MLLM 
* 25.02 [C-Drag:Chain-of-Thought Driven Motion Controller for Video Generation](https://arxiv.org/pdf/2502.19868) | [Paper📑](https://arxiv.org/pdf/2502.19868) [Code🖥️](https://github.com/WesLee88524/C-Drag-Official-Repo) [Dataset🤗](https://drive.google.com/file/d/1L2SYadeqZPObvSj9Mb6fK-KHtR0n-DKk/view)
  - A Chain-of-Thought-based motion controller for controllable video generation | Task: Video Generation


<a name="others"></a>
### Others
#### Image MLLM
* 25.03   [VLM-R1: A stable and generalizable R1-style Large Vision-Language Model](https://github.com/om-ai-lab/VLM-R1/tree/main?tab=readme-ov-file) [Code🖥️](https://github.com/om-ai-lab/VLM-R1/tree/main?tab=readme-ov-file) [Dataset🤗](https://huggingface.co/datasets/omlab/VLM-R1)  [Model🤗](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps/tree/main)
  - A reproduced R1-style VLM | Task: Referring Expression Comprehension
* 25.02 [MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning](https://arxiv.org/pdf/2502.19634)| [Paper📑](https://arxiv.org/pdf/2502.19634)
  - A MLLM trained with GRPO for medical image VQA.| Task: Medical Image VQA
#### Video MLLM
* 25.03 [R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning](https://arxiv.org/abs/2503.05379) | [Paper📑](https://arxiv.org/abs/2503.05379)[Code🖥️](https://github.com/HumanMLLM/R1-Omni) [Model🤗](https://huggingface.co/StarJiaxing/R1-Omni-0.5B/tree/main)
  - Impove reasoning capability, emotion recognition accuracy, and generalization ability with RLVR.  | Task: Emotion recognition


<a name="benchmarks"></a>
## Benchmarks 📊

| Date  | Project                                                      | Task                                     | Links                                                        |
| ----- | ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------ |
| 25.03 | V-STaR : Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning | Spatio-temporal Reasoning                | [[🌐 Project]](https://v-star-bench.github.io/) [[📑 Paper]](https://arxiv.org/abs/2311.17982) [[🤗 Data]](https://huggingface.co/datasets/V-STaR-Bench/V-STaR) |
| 25.03 | Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning | 3D-CoT                                   | [[📑 Paper]](https://arxiv.org/pdf/2503.06232) [[🤗 Data]](https://huggingface.co/datasets/Battam/3D-CoT) |
| 25.02 | MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models | MM-IQ                                    | [[📑 Paper]](https://arxiv.org/pdf/2502.00698) [[💻 Code]](https://github.com/AceCHQ/MMIQ) |
| 25.02 | MM-RLHF: The Next Step Forward in Multimodal LLM Alignment   | MM-RLHF-RewardBench, MM-RLHF-SafetyBench | [[📑 Paper]](https://arxiv.org/abs/2502.10391)                |
| 25.02 | ZeroBench: An Impossible* Visual Benchmark for Contemporary Large Multimodal Models | ZeroBench                                | [[🌐 Project]](https://zerobench.github.io/) [[🤗 Dataset]](https://huggingface.co/datasets/jonathan-roberts1/zerobench) [[💻 Code]](https://github.com/jonathan-roberts1/zerobench/) |
| 25.02 | MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency | MME-CoT                                  | [[📑 Paper]](https://arxiv.org/pdf/2502.09621) [[💻 Code]](https://github.com/CaraJ7/MME-CoT) |
| 25.02 | OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference | MM-AlignBench                            | [[📑 Paper]](https://arxiv.org/abs/2502.18411) [[💻 Code]](https://github.com/PhoenixZ810/OmniAlign-V) |
| 25.01 | LlamaV-o1: Rethinking Step-By-Step Visual Reasoning in LLMs  | VRCBench                                 | [[📑 Paper]](https://arxiv.org/abs/2501.06186) [[💻 Code]](https://github.com/mbzuai-oryx/LlamaV-o1) |
| 24.12 | Online Video Understanding: A Comprehensive Benchmark and  Memory-Augmented Method | VideoChat-Online                         | [[Paper📑]](https://arxiv.org/abs/2501.00584) [[Code💻]](https://github.com/qirui-chen/MultiHop-EgoQA) |
| 24.11 | VLRewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models | VLRewardBench                            | [[📑 Paper]](https://arxiv.org/abs/2411.17451)                |
| 24.11 | Grounded Multi-Hop VideoQA in Long-Form Egocentric Videos    | MH-VidQA                                 | [[Paper📑]](https://arxiv.org/pdf/2408.14469) [[Code💻]](https://github.com/MCG-NJU/VideoChat-Online) |
| 24.07 | REXTIME: A Benchmark Suite for Reasoning-Across-Time in Videos | REXTIME                                  | [[Paper📑]](https://arxiv.org/abs/2406.19392) [[Code💻]](https://github.com/ReXTime/ReXTime) |
| 24.06 | ChartMimic: Evaluating LMM’s Cross-Modal Reasoning Capability via Chart-to-Code Generation | ChartBench | [[Project🌐]](https://chartmimic.github.io/) [[Paper📑]](https://arxiv.org/abs/2406.09961) [[Code🖥️]](https://github.com/ChartMimic/ChartMimic) |
| 24.05 | M3CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought | M3CoT                                    | [[📑 Paper]](https://arxiv.org/html/2405.16473v1)             |

<a name="Open-sourceprojects"></a>

## Open-source Projects 
* [EasyR1 💻](https://github.com/hiyouga/EasyR1)  ![EasyR1](https://img.shields.io/github/stars/hiyouga/EasyR1) (An Efficient, Scalable, Multi-Modality RL Training Framework)

* [Multimodal Open R1 💻](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)  ![Multimodal Open R1](https://img.shields.io/github/stars/EvolvingLMMs-Lab/open-r1-multimodal) [Model 🤗](https://huggingface.co/lmms-lab/Qwen2-VL-2B-GRPO-8k) [Dataset 🤗](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified)
  
* [LMM-R1 💻](https://github.com/TideDra/lmm-r1) ![LMM-R1](https://img.shields.io/github/stars/TideDra/lmm-r1) [Code 💻](https://github.com/TideDra/lmm-r1)

* [MMR1 💻](https://github.com/LengSicong/MMR1) ![LengSicong/MMR1](https://img.shields.io/github/stars/LengSicong/MMR1) [Code 💻](https://github.com/LengSicong/MMR1) [Model 🤗](https://huggingface.co/MMR1/MMR1-Math-v0-7B) [Dataset 🤗](https://huggingface.co/datasets/MMR1/MMR1-Math-RL-Data-v0) 

* [R1-Multimodal-Journey 💻](https://github.com/FanqingM/R1-Multimodal-Journey) ![R1-Multimodal-Journey](https://img.shields.io/github/stars/FanqingM/R1-Multimodal-Journey) (Latest progress at [MM-Eureka](https://github.com/ModalMinds/MM-EUREKA))

* [R1-V 💻](https://github.com/Deep-Agent/R1-V)  ![R1-V](https://img.shields.io/github/stars/Deep-Agent/R1-V) [Blog 🎯](https://deepagent.notion.site/rlvr-in-vlms) [Datasets 🤗](https://huggingface.co/collections/MMInstruction/r1-v-67aae24fa56af9d2e2755f82)
  
* [VLM-R1 💻](https://github.com/om-ai-lab/VLM-R1)  ![VLM-R1](https://img.shields.io/github/stars/om-ai-lab/VLM-R1) [Model 🤗](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps)  [Dataset 🤗](https://huggingface.co/datasets/omlab/VLM-R1) [Demo 🤗](https://huggingface.co/spaces/omlab/VLM-R1-Referral-Expression)

* [R1-Vision 💻](https://github.com/yuyq96/R1-Vision) ![R1-Vision](https://img.shields.io/github/stars/yuyq96/R1-Vision) [Cold-Start Datasets 🤗](https://huggingface.co/collections/yuyq96/r1-vision-67a6fb7898423dca453efa83)

* [R1-Onevision 💻](https://github.com/Fancy-MLLM/R1-Onevision)  ![R1-Onevision](https://img.shields.io/github/stars/Fancy-MLLM/R1-Onevision) [Model 🤗](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B)  [Dataset 🤗](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision) [Demo 🤗](https://huggingface.co/spaces/Fancy-MLLM/R1-Onevision) [Report 📝](https://yangyi-vai.notion.site/r1-onevision)

* [Open R1 Video 💻](https://github.com/Wang-Xiaodong1899/Open-R1-Video) ![Open R1 Video](https://img.shields.io/github/stars/Wang-Xiaodong1899/Open-R1-Video) [Models 🤗](https://huggingface.co/Xiaodong/Open-R1-Video-7B)  [Datasets 🤗](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k) [Datasets 🤗](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k)

* [Video-R1 💻](https://github.com/tulerfeng/Video-R1) ![Video-R1](https://img.shields.io/github/stars/tulerfeng/Video-R1) [Code 💻](https://github.com/tulerfeng/Video-R1)
 [Dataset 🤗](https://huggingface.co/datasets/Video-R1/DVD-counting)

<a name="Contributiong"></a>
## Contributing 
If you are interested in contributing, please refer to [HERE](contribution.md) for instructions in contribution.
