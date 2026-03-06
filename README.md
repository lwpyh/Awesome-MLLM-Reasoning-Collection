# Awesome-MLLM-Reasoning-Collection
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

👏 Welcome to the Awesome-MLLM-Reasoning-Collections repository! This repository is a carefully curated collection of papers, code, datasets, benchmarks, and resources focused on reasoning within Multimodal Large Language Models (MLLMs).

Feel free to ⭐ star and fork this repository to keep up with the latest advancements and contribute to the community.
### Table of Contents
- [Awesome-MLLM-Reasoning-Collection](#awesome-mllm-reasoning-collection)
    - [Table of Contents](#table-of-contents)
  - [Papers and Projects 📄](#papers-and-projects-)
    - [Commonsense Reasoning](#commonsense-reasoning)
      - [Image MLLM](#image-mllm)
      - [Video MLLM](#video-mllm)
      - [Audio MLLM](#audio-mllm)
      - [Omni MLLM](#omni-mllm)
    - [Reasoning Segmentation and Detection](#reasoning-segmentation-and-detection)
      - [Image MLLM](#image-mllm-1)
      - [Video MLLM](#video-mllm-1)
      - [Audio MLLM](#audio-mllm-1)
      - [Omni MLLM](#omni-mllm-1)
    - [Spatial and Temporal Grounding and Understanding](#spatial-and-temporal-grounding-and-understanding)
      - [Image MLLM](#image-mllm-2)
      - [Video MLLM](#video-mllm-2)
      - [Audio MLLM](#audio-mllm-2)
      - [Omni MLLM](#omni-mllm-2)
    - [Math Reasoning](#math-reasoning)
      - [Image MLLM](#image-mllm-3)
    - [Chart Rasoning](#chart-rasoning)
    - [Visual-AUdio Generation](#visual-generation)
      - [Image MLLM](#image-mllm-4)
      - [Video MLLM](#video-mllm-3)
      - [Audio MLLM](#audio-mllm-3)
    - [Reasoning with Agent/Tool](#reasoning-with-agent) 
    - [Medical Reasoning](#medical-reasoning)
      - [Audio MLLM](#audio-mllm-4)
      - [Omni MLLM](#omni-mllm-3)
    - [Embodied Reasoning](#embodied-reasoning)
    - [Others](#others)
      - [Image MLLM](#image-mllm-5)
      - [Video MLLM](#video-mllm-4)
      - [Audio MLLM](#audio-mlllm-5)
      - [Omni MLLM](#omni-mllm-4)
  - [Benchmarks 📊](#benchmarks-)
  - [Open-source Projects](#open-source-projects)
  - [Contributing](#contributing)


<a name="PapersandProjects"></a>
## Papers and Projects 📄

<a name="VQA"></a>
### Commonsense Reasoning
#### Image MLLM
* 26.02 [From Blind Spots to Gains: Diagnostic-Driven Iterative Training for Large Multimodal Models](https://arxiv.org/abs/2602.22859) | [Paper📑](https://arxiv.org/abs/2602.22859) [Code🖥️](https://github.com/hongruijia/DPE) [Model🤗](https://huggingface.co/hongruijia/Qwen3_VL_8B_Instruct_DPE_v3)
  - Spiral-loop framework diagnosing capability gaps in MLLMs and generating targeted data and RL training to close them iteratively. | Task: Reasoning & Understanding
* 26.02 [Imagination Helps Visual Reasoning, But Not Yet in Latent Space](https://arxiv.org/abs/2602.22766) | [Paper📑](https://arxiv.org/abs/2602.22766) [Code🖥️](https://github.com/AI9Stars/CapImagine)
  - CapImagine proposes text-based explicit imagination outperforming latent-space baselines on vision-centric benchmarks via causal mediation analysis. | Task: Reasoning & Understanding
* 26.02 [NoLan: Mitigating Object Hallucinations in Large Vision-Language Models via Dynamic Suppression of Language Priors](https://arxiv.org/abs/2602.22144) | [Paper📑](https://arxiv.org/abs/2602.22144) [Code🖥️](https://github.com/lingfengren/NoLan)
  - Training-free decoding dynamically suppressing language priors by comparing multimodal vs. text-only output distributions, achieving +6.45/+7.21 accuracy on POPE. | Task: Reasoning & Understanding
* 26.02 [Selective Training for Large Vision Language Models via Visual Information Gain](https://arxiv.org/abs/2602.17186) | [Paper📑](https://arxiv.org/abs/2602.17186)
  - Visual Information Gain (VIG) metric quantifying how much visual input reduces prediction uncertainty for improved visual grounding and reduced language bias. | Task: Reasoning & Understanding
* 26.02 [MetaphorStar: Image Metaphor Understanding and Reasoning with End-to-End Visual Reinforcement Learning](https://arxiv.org/abs/2602.10575) | [Paper📑](https://arxiv.org/abs/2602.10575) [Code🖥️](https://github.com/MING-ZCH/MetaphorStar) [Model🤗](https://huggingface.co/MING-ZCH/MetaphorStar-7B)
  - End-to-end visual RL framework for image metaphor comprehension with TFQ-GRPO method, achieving 82.6% average improvement on image implication benchmarks. | Task: Reasoning & Understanding
* 26.02 [Learning Self-Correction in Vision-Language Models via Rollout Augmentation](https://arxiv.org/abs/2602.08503) | [Paper📑](https://arxiv.org/abs/2602.08503) [Model🤗](https://huggingface.co/Tuwhy/Octopus-8B)
  - Octopus synthesizes dense self-correction examples for VLMs via RL, achieving SOTA among open-source VLMs on 7 benchmarks. | Task: Reasoning & Understanding
* 26.02 [SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs](https://arxiv.org/abs/2602.06566) | [Paper📑](https://arxiv.org/abs/2602.06566)
  - Decouples visual perception from reasoning in VLMs via a two-stage pipeline, enabling efficient test-time scaling with 200× lower token budget. | Task: Reasoning & Understanding
* 26.02 [Modality Gap-Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models](https://arxiv.org/abs/2602.07026) | [Paper📑](https://arxiv.org/abs/2602.07026) [Code🖥️](https://github.com/Yu-xm/ReVision)
  - Fixed-frame Modality Gap Theory with training-free ReAlign alignment and scalable ReVision pretraining using unpaired data to bridge the modality gap. | Task: Reasoning & Understanding
* 26.02 [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276) | [Paper📑](https://arxiv.org/abs/2602.02276) [Code🖥️](https://github.com/MoonshotAI/Kimi-K2.5) [Model🤗](https://huggingface.co/moonshotai/Kimi-K2.5)
  - Open-source multimodal agentic model achieving SOTA across coding, vision, reasoning, and agentic tasks via joint text-vision RL and Agent Swarm parallel execution. | Task: Reasoning & Understanding
* 26.02 [Toward Cognitive Supersensing in Multimodal Large Language Model](https://arxiv.org/abs/2602.01541) | [Paper📑](https://arxiv.org/abs/2602.01541) [Code🖥️](https://github.com/PediaMedAI/Cognition-MLLM) [Model🤗](https://huggingface.co/PediaMedAI/CogSense-8B) [Dataset🤗](https://huggingface.co/datasets/PediaMedAI/CogSense-Bench)
  - Trains MLLMs to generate internal visual imagery sequences for abstract visual reasoning, evaluated on CogSense-Bench spanning five cognitive dimensions. | Task: Reasoning & Understanding
* 26.02 [Thinking with Comics: Enhancing Multimodal Reasoning through Structured Visual Storytelling](https://arxiv.org/abs/2602.02453) | [Paper📑](https://arxiv.org/abs/2602.02453) [Code🖥️](https://github.com/andongBlue/Think-with-Comics)
  - Uses comics as a visual medium to improve multimodal reasoning efficiency while preserving temporal structure and narrative coherence. | Task: Reasoning & Understanding
* 26.02 [SwimBird: Eliciting Switchable Reasoning Mode in Hybrid Autoregressive MLLMs](https://arxiv.org/abs/2602.06040) | [Paper📑](https://arxiv.org/abs/2602.06040) [Code🖥️](https://github.com/Accio-Lab/SwimBird) [Dataset🤗](https://huggingface.co/datasets/Accio-Lab/SwimBird-SFT-92K)
  - Hybrid autoregressive MLLM dynamically switching between text-only, vision-only, and interleaved vision-text reasoning modes based on input queries. | Task: Reasoning & Understanding
* 26.01 [MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Methods](https://arxiv.org/abs/2601.21821) | [Paper📑](https://arxiv.org/abs/2601.21821) [Model🤗](https://huggingface.co/OpenDataArena/MMFineReason-8B) [Dataset🤗](https://huggingface.co/datasets/OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking)
  - A 1.8M-sample multimodal reasoning dataset with high-quality CoT annotations; the 8B model approaches Qwen3-VL-32B-Thinking performance. | Task: Reasoning & Understanding
* 26.01 [DiffThinker: Towards Generative Multimodal Reasoning with Diffusion Models](https://arxiv.org/abs/2512.24165) | [Paper📑](https://arxiv.org/abs/2512.24165) [Code🖥️](https://github.com/lcqysl/DiffThinker) [Model🤗](https://huggingface.co/yhx12/DiffThinker) [Dataset🤗](https://huggingface.co/datasets/yhx12/DiffThinker_Eval)
  - Reformulates multimodal reasoning as a native image-to-image generative task using diffusion models. | Task: Reasoning & Understanding
* 26.01 [Visual Generation Unlocks Human-Like Reasoning through Multimodal World Models](https://arxiv.org/abs/2601.19834) | [Paper📑](https://arxiv.org/abs/2601.19834) [Code🖥️](https://github.com/thuml/Reasoning-Visual-World) [Dataset🤗](https://huggingface.co/datasets/thuml/VisWorld-Eval)
  - Proposes the visual superiority hypothesis: visual generation serves as a more natural world model for physical/spatial reasoning tasks. | Task: Reasoning & Understanding
* 26.01 [VTC-R1: Vision-Text Compression for Efficient Long-Context Reasoning](https://arxiv.org/abs/2601.22069) | [Paper📑](https://arxiv.org/abs/2601.22069) [Code🖥️](https://github.com/w-yibo/VTC-R1)
  - Compresses textual reasoning traces into compact images as "optical memory" for VLMs, achieving 3.4x token compression. | Task: Reasoning & Understanding
* 26.01 [UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision](https://arxiv.org/abs/2601.03193) | [Paper📑](https://arxiv.org/abs/2601.03193) [Code🖥️](https://github.com/Hungryyan1/UniCorn) [Model🤗](https://huggingface.co/CostaliyA/UniCorn)
  - Self-improvement framework partitioning a single model into Proposer/Solver/Judge roles via self-play to improve comprehension and generation. | Task: Reasoning & Understanding
* 26.01 [LaViT: Aligning Latent Visual Thoughts for Multi-modal Reasoning](https://arxiv.org/abs/2601.10129) | [Paper📑](https://arxiv.org/abs/2601.10129) [Code🖥️](https://github.com/Svardfox/LaViT) [Model🤗](https://huggingface.co/Svard/LaViT-3B) [Dataset🤗](https://huggingface.co/datasets/Svard/LaViT-15k)
  - Addresses the "Perception Gap" by aligning latent visual thoughts (attention trajectories) between teacher and student models. | Task: Reasoning & Understanding
* 26.01 [STEP3-VL-10B Technical Report](https://arxiv.org/abs/2601.09668) | [Paper📑](https://arxiv.org/abs/2601.09668) [Code🖥️](https://github.com/stepfun-ai/Step3-VL-10B) [Model🤗](https://huggingface.co/stepfun-ai/Step3-VL-10B)
  - A 10B multimodal foundation model with Parallel Coordinated Reasoning (PaCoRe) for test-time compute scaling. | Task: Reasoning & Understanding
* 26.01 [Generation Enhances Understanding in Unified Multimodal Models via Multi-Representation Generation](https://arxiv.org/abs/2601.21406) | [Paper📑](https://arxiv.org/abs/2601.21406) [Code🖥️](https://github.com/Sugewud/UniMRG)
  - Trains unified multimodal models to generate pixel, depth, and segmentation representations alongside understanding. | Task: Reasoning & Understanding
* 26.01 [What Matters in Data Curation for Multimodal Reasoning? Insights from the DCVLR Challenge](https://arxiv.org/abs/2601.10922) | [Paper📑](https://arxiv.org/abs/2601.10922)
  - First-place NeurIPS 2025 DCVLR challenge submission revealing difficulty-based example selection as dominant driver in data curation. | Task: Reasoning & Understanding
* 26.01 [MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models](https://arxiv.org/abs/2601.21181) | [Paper📑](https://arxiv.org/abs/2601.21181)
  - Modality-adaptive decoding to mitigate cross-modal hallucinations in MLLMs by dynamically adjusting decoding. | Task: Reasoning & Understanding
* 25.12 [OneThinker: All-in-one Reasoning Model for Image and Video](https://arxiv.org/abs/2512.03043) | [Paper📑](https://arxiv.org/abs/2512.03043)
  - Unifies image and video understanding across diverse visual tasks using RL with EMA-GRPO technique. | Task: Reasoning & Understanding
* 25.12 [Puzzle Curriculum GRPO for Vision-Centric Reasoning](https://arxiv.org/abs/2512.14944) | [Paper📑](https://arxiv.org/abs/2512.14944)
  - Supervision-free RL method enhancing visual reasoning in VLMs through self-supervised puzzle environments. | Task: Reasoning & Understanding
* 25.12 [Robust-R1: Degradation-Aware Reasoning for Robust Visual Understanding](https://arxiv.org/abs/2512.17532) | [Paper📑](https://arxiv.org/abs/2512.17532)
  - Enhances MLLM robustness to visual degradations by modeling degradation parameters through structured reasoning chains. | Task: Reasoning & Understanding
* 25.12 [See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning](https://arxiv.org/abs/2512.22120) | [Paper📑](https://arxiv.org/abs/2512.22120)
  - Improves VLM multimodal reasoning via paired masked views to enforce fine-grained visual reliance. | Task: Reasoning & Understanding
* 25.11 [OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe](https://arxiv.org/abs/2511.16334) | [Paper📑](https://arxiv.org/abs/2511.16334)
  - Open general-purpose framework for advancing multimodal reasoning. | Task: Reasoning & Understanding
* 25.11 [ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning](https://arxiv.org/abs/2510.27492) | [Paper📑](https://arxiv.org/abs/2510.27492)
  - Studies emergent properties in multimodal interleaved chain-of-thought reasoning. | Task: Reasoning & Understanding
* 25.11 [TiDAR: Think in Diffusion, Talk in Autoregression](https://arxiv.org/abs/2511.08923) | [Paper📑](https://arxiv.org/abs/2511.08923)
  - Combines diffusion-based thinking with autoregressive generation for multimodal reasoning. | Task: Reasoning & Understanding
* 25.10 [TTRV: Test-Time Reinforcement Learning for Vision Language Models](https://arxiv.org/abs/2510.06783) | [Paper📑](https://arxiv.org/abs/2510.06783)
  - Test-time reinforcement learning applied to vision-language models for improved reasoning. | Task: Reasoning & Understanding
* 25.10 [VLM-FO1: Bridging the Gap Between High-Level Reasoning and Fine-Grained Perception in VLMs](https://arxiv.org/abs/2509.25916) | [Paper📑](https://arxiv.org/abs/2509.25916)
  - Improves VLMs' ability to combine high-level reasoning with detailed visual perception. | Task: Reasoning & Understanding
* 25.10 [ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping](https://arxiv.org/abs/2510.08457) | [Paper📑](https://arxiv.org/abs/2510.08457)
  - Adaptive reasoning for multimodal models using entropy shaping. | Task: Reasoning & Understanding
* 25.09 [R-4B: Incentivizing General-Purpose Auto-Thinking Capability in MLLMs via Bi-Mode Annealing and RL](https://arxiv.org/abs/2508.21113) | [Paper📑](https://arxiv.org/abs/2508.21113)
  - Training method using RL and annealing to improve auto-thinking and reasoning in multimodal LLMs. | Task: Reasoning & Understanding
* 25.09 [LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training](https://arxiv.org/abs/2509.23661) | [Paper📑](https://arxiv.org/abs/2509.23661)
  - Open-source framework for training multimodal vision-language models. | Task: Reasoning & Understanding
* 25.08 [Thyme: Think Beyond Images](https://arxiv.org/abs/2508.11630) | [Paper📑](https://arxiv.org/abs/2508.11630)
  - Multimodal reasoning system that extends beyond surface-level image understanding to higher-level thinking. | Task: Reasoning & Understanding
* 25.08 [Controlling Multimodal LLMs via Reward-guided Decoding](https://arxiv.org/abs/2508.11616) | [Paper📑](https://arxiv.org/abs/2508.11616)
  - Controls MLLM reasoning outputs through reward-based generation guidance at decoding time. | Task: Reasoning & Understanding
* 25.08 [Self-Rewarding Vision-Language Model via Reasoning Decomposition](https://arxiv.org/abs/2508.15882) | [Paper📑](https://arxiv.org/abs/2508.15882)
  - VLM that uses reasoning decomposition and self-reward to improve visual reasoning quality. | Task: Reasoning & Understanding
* 25.08 [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471) | [Paper📑](https://arxiv.org/abs/2508.06471)
  - Foundation model with strong agentic, reasoning, and coding capabilities across modalities. | Task: Reasoning & Understanding
* 25.07 [GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning](https://arxiv.org/abs/2507.01006) | [Paper📑](https://arxiv.org/abs/2507.01006) [Code🖥️](https://github.com/THUDM/GLM-4.1V-Thinking)
  - A reasoning-centric training framework for general-purpose multimodal reasoning. | Task: Reasoning & Understainding
* 25.07 [MiCo: Multi-image Contrast for Reinforcement Visual Reasoning](https://arxiv.org/abs/2506.22434) | [Paper📑](https://arxiv.org/abs/2506.22434)
   - Construct image triplets comprising two augmented views of the same image and a third, similar but distinct image. | Task: Reasoning & Understainding
* 25.06 [Vision Matters: Simple Visual Perturbations Can Boost Multimodal Math Reasoning](https://arxiv.org/abs/2506.09736) | [Paper📑](https://arxiv.org/abs/2506.09736) [Code🖥️](https://github.com/YutingLi0606/Vision-Matters) [Model🤗](https://huggingface.co/collections/Yuting6/vision-matters-684801dd1879d3e639a930d1)
  - Simple visual perturbation framework that can be easily integrated into existing post-training pipelines including SFT, DPO, and GRPO. | Task: Reasoning & Understainding
* 25.05 [Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning](https://arxiv.org/pdf/2505.14677) | [Paper📑](https://arxiv.org/pdf/2505.14677) [Code🖥️](https://github.com/maifoundations/Visionary-R1) [Model🤗](https://huggingface.co/maifoundations/Visionary-R1)
* 25.05 [Sherlock: Self-Correcting Reasoning in Vision-Language Models](http://arxiv.org/pdf/2505.22651) | [Paper📑](http://arxiv.org/pdf/2505.22651) [Code🖥️](https://github.com/DripNowhy/Sherlock) [Model🤗](https://huggingface.co/collections/Tuwhy/sherlock-6835f46e450a48f228f7e80d)
  - Explore self-correction as a strategy to enhance reasoning VLMs | Task: Reasoning & Understainding
* 25.05 [EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.04623) | [Paper📑](https://arxiv.org/pdf/2505.04623) [Code🖥️](https://github.com/HarryHsing/EchoInk) [Model🤗](https://huggingface.co/harryhsing/EchoInk-R1-7B)
  - The first general framework for unified audio-visual reasoning via reinforcement learning | Task: Reasoning & Understainding
* 25.03 [Skywork-R1V: Pioneering Multimodal Reasoning with CoT](https://github.com/SkyworkAI/Skywork-R1V/blob/main/Skywork_R1V.pdf) | [Paper📑](https://github.com/SkyworkAI/Skywork-R1V/blob/main/Skywork_R1V.pdf) [Code🖥️](https://github.com/SkyworkAI/Skywork-R1V) [Model🤗](https://huggingface.co/Skywork/Skywork-R1V-38B)
  - The first industry open-sourced multimodal reasoning model with advanced visual chain-of-thought capabilities | Task: Reasoning & Understainding
* 25.03 [CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](https://arxiv.org/pdf/2503.05255) | [Paper📑](https://arxiv.org/pdf/2503.05255)
  - Mimic human-like ”slow thinking” for multi-image understanding. | Task: VQA
* 25.03 [DAPO: an Open-Source LLM Reinforcement Learning System at Scale](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf) | [Paper📑](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf) [Code🖥️](https://github.com/BytedTsinghua-SIA/DAPO) [Data🤗](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
  - Propose the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm. | Task: Math
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
* 26.02 [A Very Big Video Reasoning Suite](https://arxiv.org/abs/2602.20159) | [Paper📑](https://arxiv.org/abs/2602.20159) [Model🤗](https://huggingface.co/Video-Reason/VBVR-Wan2.2) [Dataset🤗](https://huggingface.co/datasets/Video-Reason/VBVR-Dataset)
  - 1M+ video clip dataset spanning 200 reasoning tasks (VBVR) with VBVR-Bench for verifiable evaluation, enabling emergent generalization via large-scale scaling. | Task: Video Understanding & Reasoning
* 26.02 [Thinking in Frames: How Visual Context and Test-Time Scaling Empower Video Reasoning](https://arxiv.org/abs/2601.21037) | [Paper📑](https://arxiv.org/abs/2601.21037) [Project🌐](https://thinking-in-frames.github.io/)
  - Video generation models achieve zero-shot generalization for visual reasoning by using generated frames as intermediate reasoning steps with a visual test-time scaling law. | Task: Video Understanding & Reasoning
* 26.02 [Multimodal Fact-Level Attribution for Verifiable Reasoning](https://arxiv.org/abs/2602.11509) | [Paper📑](https://arxiv.org/abs/2602.11509) [Code🖥️](https://github.com/meetdavidwan/murgat)
  - MuRGAt benchmark requiring MLLMs to provide precise fact-level citations across video, audio, and modalities, finding strong models frequently hallucinate citations despite correct reasoning. | Task: Video Understanding & Reasoning
* 26.01 [Taming Hallucinations: Boosting MLLMs' Video Understanding via Counterfactual Video Generation](https://arxiv.org/abs/2512.24271) | [Paper📑](https://arxiv.org/abs/2512.24271)
  - Uses counterfactual video generation to reduce hallucinations and improve temporal reasoning in multimodal LLMs. | Task: Video Understanding & Reasoning
* 25.12 [Rethinking Chain-of-Thought Reasoning for Videos](https://arxiv.org/abs/2512.09616) | [Paper📑](https://arxiv.org/abs/2512.09616)
  - Proposes improved chain-of-thought reasoning strategies specifically designed for video understanding tasks. | Task: Video Understanding & Reasoning
* 25.12 [SAGE: Training Smart Any-Horizon Agents for Long Video Reasoning with RL](https://arxiv.org/abs/2512.13874) | [Paper📑](https://arxiv.org/abs/2512.13874)
  - RL-based framework training agents for long-horizon video reasoning across variable time spans. | Task: Video Understanding & Reasoning
* 25.11 [Video-R4: Reinforcing Text-Rich Video Reasoning with Visual Rumination](https://arxiv.org/abs/2511.17490) | [Paper📑](https://arxiv.org/abs/2511.17490)
  - Enhances reasoning over text-rich video content via visual rumination. | Task: Video Understanding & Reasoning
* 25.10 [Video-Thinker: Sparking "Thinking with Videos" via Reinforcement Learning](https://arxiv.org/abs/2510.23473) | [Paper📑](https://arxiv.org/abs/2510.23473)
  - Reasoning framework enabling models to think with video inputs via RL. | Task: Video Understanding & Reasoning
* 25.10 [StreamingVLM: Real-Time Understanding for Infinite Video Streams](https://arxiv.org/abs/2510.09608) | [Paper📑](https://arxiv.org/abs/2510.09608)
  - Real-time video stream understanding with multimodal LLMs. | Task: Video Understanding & Reasoning
* 25.09 [Video models are zero-shot learners and reasoners](https://arxiv.org/abs/2509.20328) | [Paper📑](https://arxiv.org/abs/2509.20328)
  - Demonstrates zero-shot reasoning capabilities in video models. | Task: Video Understanding & Reasoning
* 25.07 [Scaling RL to Long Videos](https://arxiv.org/abs/2507.07966)| [Paper📑](https://arxiv.org/pdf/2507.07966) [Model🤗](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B) [Code🖥️](https://github.com/NVlabs/Long-RL)
* 25.06 [DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](https://arxiv.org/abs/2506.07464)|[Paper📑](https://arxiv.org/pdf/2506.07464)
* 25.06 [VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](https://arxiv.org/abs/2505.12434) | [Paper📑](https://arxiv.org/abs/2505.12434) [Model🤗](https://huggingface.co/QiWang98/VideoRFT) [Code🖥️](https://github.com/QiWang98/VideoRFT)
  - Extend Reinforcement Fine-Tuning (RFT) to the video reasoning domain, a long-standing challenge. | Task: Video Understanding & Reasoning
* 25.06 [VersaVid-R1: A Versatile Video Understanding and Reasoning Model from Question Answering to Captioning Tasks](https://www.arxiv.org/abs/2506.09079)|[Paper📑](https://www.arxiv.org/pdf/2506.09079) [Model🤗](https://huggingface.co/VersaVid-R1/VersaVid-R1) [Code🖥️](https://github.com/VersaVid-R1/VersaVid-R1)
* 25.05 [SpaceR: Reinforcing MLLMs in Video Spatial  Reasoning](https://arxiv.org/abs/2504.01805v2)|[Paper📑](https://arxiv.org/pdf/2504.01805v2) [Model🤗](https://huggingface.co/RUBBISHLIKE/SpaceR) [Code🖥️](https://github.com/OuyangKun10/SpaceR)
* 25.05 [Video-R1: Reinforcing Video Reasoning in MLLMs](https://arxiv.org/abs/2503.21776) | [Paper📑](https://arxiv.org/pdf/2503.21776)[Model🤗](https://huggingface.co/Video-R1/Video-R1-7B)  [Code🖥️](https://github.com/tulerfeng/Video-R1)
* 25.04 [TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning](https://arxiv.org/abs/2504.09641) |  [Paper📑](https://arxiv.org/pdf/2504.09641) [Model🤗](https://huggingface.co/Zhang199/TinyLLaVA-Video-R1) [Code🖥️](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)
* 25.04 [Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning](https://arxiv.org/pdf/2505.03318) | [Paper📑](https://arxiv.org/pdf/2505.03318) [Project🌐](https://codegoat24.github.io/UnifiedReward/think) [Code🖥️](https://github.com/CodeGoat24/UnifiedReward)
  - The first unified multimodal CoT reward model, capable of step-by-step long-chain reasoning for visual understanding and generation reward tasks. | Task: Video Understanding and Feneration
* 25.04 [ViSMaP: Unsupervised Hour-long Video Summarisation by Meta-Prompting](https://arxiv.org/abs/2504.15921) | [Paper📑](https://arxiv.org/abs/2504.15921)
  - A system to summarise hour long videos with no-supervision. | Task: Video Summary
* 25.04 [TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning](https://arxiv.org/abs/2504.09641) | [Paper📑](https://arxiv.org/abs/2504.09641) [Code🖥️](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) | [Model🤗](https://huggingface.co/Zhang199/TinyLLaVA-Video-R1)
  - Present the small-scale video reasoning model TinyLLaVA-Video-R1 | Task: Video QA
* 25.04 [VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning](https://arxiv.org/abs/2503.13444) | [Paper📑](https://arxiv.org/abs/2503.13444) [Code🖥️](https://github.com/yeliudev/VideoMind) | [Dataset🤗](https://huggingface.co/datasets/yeliudev/VideoMind-Dataset)
  - A novel video-language agent designed for temporal-grounded video understanding. | Task: Video QA
* 25.04 [Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1](https://arxiv.org/pdf/2503.24376) | [Paper📑](https://arxiv.org/pdf/2503.24376) [Code🖥️](https://github.com/TencentARC/SEED-Bench-R1) | [Dataset🤗](https://huggingface.co/datasets/TencentARC/SEED-Bench-R1)
  - Reveals that RL enhances visual perception but often produces less logically coherent reasoning chains. | Task: Video QA
* 25.03 [VIDEOTREE: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos](https://arxiv.org/abs/2405.19209) |[Paper📑](https://arxiv.org/pdf/2405.19209) [Code🖥️](https://github.com/Ziyang412/VideoTree) 
* 25.02 [CoS: Chain-of-Shot Prompting for Long Video Understanding](https://arxiv.org/pdf/2502.06428) | [Paper📑](https://arxiv.org/pdf/2502.06428) [Code🖥️](https://github.com/lwpyh/CoS_codes1)
  - Approach long video understanding by optimising input video information to fully utilise MLLM’s ability to comprehend long videos. | Task: Video VQA
* 25.02 [video-SALMONN-o1: Reasoning-enhanced Audio-visual Large Language Model](https://arxiv.org/abs/2502.11775) | [Paper📑](https://arxiv.org/abs/2502.11775) [Demo🖥️](https://github.com/BriansIDP/video-SALMONN-o1)
  - A open-source reasoning-enhanced audio-visual LLM designed for general video understanding tasks.  | Task: Video QA  
* 25.02 [Open-R1-Video]((https://github.com/Wang-Xiaodong1899/Open-R1-Video)) | [Code🖥️](https://github.com/Wang-Xiaodong1899/Open-R1-Video) [Dataset🤗](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k)
  - A open-source R1-style video understanding model | Task: Video QA
* 25.01 [Temporal Preference Optimization for Long-Form Video Understanding](https://arxiv.org/abs/2501.13919) | [Paper📑](https://arxiv.org/abs/2501.13919)[Code🖥️](https://ruili33.github.io/tpo_website/)
  - A novel post-training framework designed to enhance the temporal grounding capabilities of video-LMMs through preference learning | Task: Video QA
* 25.01 [Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding](https://arxiv.org/abs/2501.07888https://github.com/bytedance/tarsier) | [Paper📑](https://arxiv.org/abs/2501.07888) [Code🖥️](https://github.com/bytedance/tarsier?tab=readme-ov-file)
  [Model🤗](https://huggingface.co/omni-research/Tarsier-34b)
  - A family of VLMs designed for high-quality video captioning and understanding | Task: Video captioning & QA
* 24.12 (ECCV24) [VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](https://arxiv.org/abs/2403.11481) | [Paper📑](https://arxiv.org/abs/2403.11481) [Code🖥️](https://github.com/YueFan1014/VideoAgent) [Project🌐](https://videoagent.github.io/)
  - Explore how reconciling several foundation models with a novel unified memory mechanism could tackle the challenging video understanding problem  | Task: Video captioning & QA

#### Audio MLLM
* 25.10 [UALM: Unified Audio Language Model for Understanding, Generation and Reasoning](https://arxiv.org/abs/2510.12000)  [Project🌐](https://research.nvidia.com/labs/adlr/UALM/)
* 25.09 [MiMo Audio: Audio Language Models are Few-Shot Learners](https://github.com/XiaomiMiMo/MiMo-Audio) [Project🌐](https://xiaomimimo.github.io/MiMo-Audio-Demo/)  [Code🖥️](https://github.com/XiaomiMiMo/MiMo-Audio)
* 25.07 [Audio Entailment: Assessing Deductive Reasoning for Audio Understanding](https://arxiv.org/abs/2407.18062)
* 25.07 [Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models](https://arxiv.org/abs/2507.08128)
* 25.05 [AudSemThinker: Enhancing Audio-Language Models Through Reasoning over Semantics of Sound](https://arxiv.org/abs/2505.14142)
* 25.05 [Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?](https://arxiv.org/abs/2505.09439)  
 - Utilizing GRPO to enhance audio reasoning performance
* 25.04 [SARI: Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning](https://arxiv.org/abs/2504.15900)
* 25.04 [Kimi-Audio Technical Report](https://arxiv.org/abs/2504.18425)  [Code🖥️](https://github.com/MoonshotAI/Kimi-Audio)
* 25.03 [Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering](https://arxiv.org/html/2503.11197v1)
* 25.03 [Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models](https://arxiv.org/pdf/2503.02318)  [Project🌐](https://xzf-thu.github.io/Audio-Reasoner/)
  - Utilizing CoT data for audio understanding tasks.
* 25.03 [Mellow: a small audio language model for reasoning](https://arxiv.org/pdf/2503.08540)  [Code🖥️](https://github.com/soham97/mellow)
  - Small audio-language model (167M) designed for audio understanding, audio entailment, audio difference and captioning.
* 25.03 [Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities](https://arxiv.org/pdf/2503.03983) [Project🌐](https://research.nvidia.com/labs/adlr/AF2/)
  - NVIDIA audio-language for various audio understanding and reasoning.
* 25.02 [Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction](https://arxiv.org/abs/2502.11946) [Code🖥️](https://github.com/stepfun-ai/Step-Audio)
* 25.01 [Audio-CoT: Exploring Chain-of-Thought Reasoning in Large Audio Language Model](https://arxiv.org/pdf/2501.07246)
  - Finetuning Qwen2-Audio with CoT data for audio understanding and retrieval tasks.
* 24.07 [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759) [Paper📑](https://arxiv.org/abs/2407.10759)  [Code🖥️](https://github.com/QwenLM/Qwen2-Audio)
  - Qwen audio-language series for various audio understanding tasks especially for speech.
* 24.07 (EMNLP2024) [GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities](https://arxiv.org/pdf/2406.11768)  [Project🌐](https://sreyan88.github.io/gamaaudio/)
  - NVIDIA audio-language for various audio understanding and reasoning.
* 24.02 (ICML2024)[Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities](https://arxiv.org/pdf/2402.01831) [Code🖥️](https://github.com/NVIDIA/audio-flamingo)
  - audio-language for various audio understanding and reasoning with Q-formers.
* 23.11 [Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models](https://arxiv.org/pdf/2311.07919) [Code🖥️](https://github.com/QwenLM/Qwen-Audio)
  - Qwen audio-language series for various audio understanding tasks in speech sound and music.
* 23.10 (ICLR2024) [SALMONN: Towards Generic Hearing Abilities for Large Language Models](https://arxiv.org/pdf/2310.13289) [Code🖥️](https://github.com/bytedance/SALMONN)
  - Bytedance audio-language for various audio understanding tasks especially for speech and sound with Q-former.
* 23.09 (NAACL2024) [MusiLingo: Bridging Music and Text with Pre-trained Language Models for Music Captioning and Query Response](https://arxiv.org/pdf/2309.08730)
  - Music-language for understanding and captioning tasks.

#### Omni MLLM
* 26.02 [OmniGAIA: Towards Native Omni-Modal AI Agents](https://arxiv.org/abs/2602.22897) | [Paper📑](https://arxiv.org/abs/2602.22897) [Code🖥️](https://github.com/RUC-NLPIR/OmniGAIA) [Model🤗](https://huggingface.co/RUC-NLPIR/OmniAtlas-Qwen3-30B-A3B) [Dataset🤗](https://huggingface.co/datasets/RUC-NLPIR/OmniGAIA)
  - OmniGAIA benchmark for omni-modal agent evaluation on cross-modal reasoning and tool-use, with OmniAtlas agent trained via hindsight-guided tree exploration and OmniDPO. | Task: Reasoning & Understanding
* 26.02 [Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device](https://arxiv.org/abs/2602.20161) | [Paper📑](https://arxiv.org/abs/2602.20161) [Code🖥️](https://github.com/Amshaker/Mobile-O) [Model🤗](https://huggingface.co/Amshaker/Mobile-O-0.5B-iOS)
  - Compact on-device unified multimodal model (~3s/512×512 on iPhone) outperforming Show-O and JanusFlow on generation and visual understanding benchmarks. | Task: Reasoning & Understanding
* 26.02 [OmniVideo-R1: Reinforcing Audio-visual Reasoning with Query Intention and Modality Attention](https://arxiv.org/abs/2602.05847) | [Paper📑](https://arxiv.org/abs/2602.05847)
  - Reinforced framework for omnivideo models improving mixed-modality reasoning by combining query-intensive grounding and modality-attentive fusion via contrastive learning. | Task: Reasoning & Understanding
* 25.12 [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) | [Paper📑](https://arxiv.org/abs/2511.21631)
  - Advanced VLM excelling in text and multimodal understanding supporting up to 256K tokens of interleaved text, images, and video. | Task: Reasoning & Understanding
* 25.10 [InteractiveOmni: A Unified Omni-modal Model for Audio-Visual Multi-turn Dialogue](https://arxiv.org/abs/2510.13747#:~:text=,visual%20interaction.%20To%20enable)
* 25.10 [Ming-Flash-Omni: A Sparse, Unified Architecture for Multimodal Perception and Generation](https://arxiv.org/abs/2510.24821) | [Paper📑](https://arxiv.org/abs/2510.24821)
  - Unified sparse architecture for multimodal perception and generation across modalities. | Task: Reasoning & Understanding
* 25.10 [OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM](https://arxiv.org/abs/2510.15870) | [Paper📑](https://arxiv.org/abs/2510.15870)
  - Multimodal LLM for comprehensive understanding across all modalities. | Task: Reasoning & Understanding
* 25.09 [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
* 25.09 [Lavida-O: Elastic Large Masked Diffusion Models for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2509.19244) | [Paper📑](https://arxiv.org/abs/2509.19244)
  - Unified model for multimodal understanding and generation across modalities. | Task: Reasoning & Understanding
* 25.07 [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](https://arxiv.org/abs/2507.06261)
* 25.05 [EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.04623)
* 25.03 [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
* 25.01 [OpenOmni: Advancing Open-Source Omnimodal Large Language Models with Progressive Multimodal Alignment and Real-Time Self-Aware Emotional Speech Synthesis](https://arxiv.org/abs/2501.04561)
* 24.10 [Baichuan-Omni Technical Report](https://arxiv.org/abs/2410.08565)
* 24.09 [MIO: A Foundation Model on Multimodal Tokens](https://arxiv.org/html/2409.17692v1)
* 24.08 [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](https://arxiv.org/abs/2408.01800) [[Code]](https://github.com/OpenBMB/MiniCPM-o)
* 24.02 [AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling](https://arxiv.org/html/2402.12226v2)
* 23.12 [Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action](https://arxiv.org/abs/2312.17172)

<a name="ReasoningSegmentation"></a>

### Reasoning Segmentation and Detection
#### Image MLLM
* 26.02 [Retrieve and Segment: Are a Few Examples Enough to Bridge the Supervision Gap in Open-Vocabulary Segmentation?](https://arxiv.org/abs/2602.23339) | [Paper📑](https://arxiv.org/abs/2602.23339) [Code🖥️](https://github.com/TilemahosAravanis/Retrieve-and-Segment)
  - Retrieval-augmented test-time adapter for open-vocabulary segmentation fusing textual prompts with pixel-annotated visual support features to narrow zero-shot vs. supervised gap. | Task: Reasoning Segmentation
* 26.02 [Seg-ReSearch: Segmentation with Interleaved Reasoning and External Search](https://arxiv.org/abs/2602.04454) | [Paper📑](https://arxiv.org/abs/2602.04454) [Code🖥️](https://github.com/iSEE-Laboratory/Seg-ReSearch) [Dataset🤗](https://huggingface.co/datasets/iSEE-Laboratory/OK_VOS)
  - Novel segmentation paradigm enabling interleaved reasoning and external search to overcome knowledge bottlenecks, with OK-VOS benchmark for open-knowledge video object segmentation. | Task: Reasoning Segmentation
* 26.01 [Urban Socio-Semantic Segmentation with Vision-Language Reasoning](https://arxiv.org/abs/2601.10477) | [Paper📑](https://arxiv.org/abs/2601.10477) [Code🖥️](https://github.com/AMAP-ML/SocioReasoner) [Model🤗](https://huggingface.co/vvangfaye/SocioReasoner-3B) [Dataset🤗](https://huggingface.co/datasets/vvangfaye/SocioSeg)
  - Vision-language reasoning framework for urban satellite segmentation identifying both physical and social categories via multi-stage reasoning. | Task: Reasoning Segmentation
* 26.01 [SAMTok: Representing Any Mask with Two Words](https://arxiv.org/abs/2601.16093) | [Paper📑](https://arxiv.org/abs/2601.16093)
  - Efficient mask tokenization representing arbitrary segmentation masks with just two tokens, enabling reasoning-driven segmentation. | Task: Reasoning Segmentation
* 26.01 [Towards Pixel-Level VLM Perception via Simple Points Prediction](https://arxiv.org/abs/2601.19228) | [Paper📑](https://arxiv.org/abs/2601.19228)
  - Enables pixel-level perception in VLMs through simple points prediction, bridging VLM reasoning and fine-grained spatial detection. | Task: Detection & Grounding
* 25.12 [ReVSeg: Incentivizing the Reasoning Chain for Video Segmentation with Reinforcement Learning](https://arxiv.org/abs/2512.02835) | [Paper📑](https://arxiv.org/abs/2512.02835)
  - Uses RL to incentivize reasoning chains for improved video segmentation. | Task: Reasoning Segmentation
* 25.12 [InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search](https://arxiv.org/abs/2512.18745) | [Paper📑](https://arxiv.org/abs/2512.18745)
  - Enhances multimodal models with generalized visual search for improved grounding. | Task: Detection & Grounding
* 25.11 [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719) | [Paper📑](https://arxiv.org/abs/2511.16719)
  - Advances segmentation with concept-based reasoning. | Task: Reasoning Segmentation
* 25.10 [Decomposed Attention Fusion in MLLMs for Training-Free Video Reasoning Segmentation](https://arxiv.org/abs/2510.19592) | [Paper📑](https://arxiv.org/abs/2510.19592)
  - Video reasoning and segmentation with multimodal models without training. | Task: Reasoning Segmentation
* 25.09 [RefAM: Attention Magnets for Zero-Shot Referral Segmentation](https://arxiv.org/abs/2509.22650) | [Paper📑](https://arxiv.org/abs/2509.22650)
  - Zero-shot referral segmentation using attention-based visual reasoning. | Task: Reasoning Segmentation
* 25.07 [UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence with Spatial Reasoning and Understanding](https://arxiv.org/abs/2506.23219) | [Paper📑](https://arxiv.org/abs/2506.23219) [Code🖥️](https://github.com/tsinghua-fib-lab/UrbanLLaVA) 
  - A multi-stage training framework that decouples spatial reasoning enhancement from domain knowledge learning, thereby improving performance across diverse urban tasks.   | Task: Urban tasks
* 25.07 [Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs](https://arxiv.org/abs/2506.21656) | [Paper📑](https://arxiv.org/abs/2506.21656)
  - A novel fine-grained preference optimization approach that significantly improves spatial reasoning capabilities in  VLMs | Task: Spatial Tasks
* 25.06 [Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning](https://arxiv.org/abs/2506.04034) | [Paper📑](https://arxiv.org/abs/2506.04034) [Code🖥️](https://rexthinker.github.io/) 
 [Model🤗](https://huggingface.co/IDEA-Research/Rex-Thinker-GRPO-7B) 
  - a grounded model reasons step-by-step—just like a human would   | Task: Detection & Grounding
* 25.03 [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785) | [Paper📑](https://arxiv.org/abs/2503.01785) [Code🖥️](https://github.com/Liuziyu77/Visual-RFT) 
 [Dataset🤗](https://huggingface.co/collections/laolao77/virft-datasets-67bc271b6f2833eccc0651df) 
  - Extend Reinforcement Fine-Tuning on visual tasks with GRPO   | Task: Detection & Grounding & Classification
* 25.03 [Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning](https://arxiv.org/pdf/2503.07065) | [Paper📑](https://arxiv.org/pdf/2503.07065)
  - Improve generalization and reasoning of VLMs with GRPO | Task: Detection & Classification & Math
* 25.03 [Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement](https://arxiv.org/abs/2503.06520) | [Paper📑](https://arxiv.org/abs/2503.06520) [Code🖥️](https://github.com/dvlab-research/Seg-Zero) [Model🤗](https://huggingface.co/Ricky06662/Seg-Zero-7B)
  - Address object detection and segmentation with GRPO | Task: Object Detection & Object Segmentation
* 24.08 (NeurIPS) [Leveraging Hallucinations to Reduce Manual Prompt Dependency in Promptable Segmentation](https://arxiv.org/abs/2408.15205) | [Paper📑](https://arxiv.org/abs/2408.15205) [Code🖥️](https://github.com/lwpyh/ProMaC_code)
  - Utilize hallucinations to mine task-related information from images and verify its accuracy for enhancing precision of the generated prompts. | Task: Reasoning Segmentation
* 24.07 (CVPR24) [Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs](https://openaccess.thecvf.com/content/CVPR2024/papers/Ranasinghe_Learning_to_Localize_Objects_Improves_Spatial_Reasoning_in_Visual-LLMs_CVPR_2024_paper.pdf) | [Paper📑](https://openaccess.thecvf.com/content/CVPR2024/papers/Ranasinghe_Learning_to_Localize_Objects_Improves_Spatial_Reasoning_in_Visual-LLMs_CVPR_2024_paper.pdf)
  - Explore how instruction fine-tuning objectives could inject spatial awareness into V-LLMs| | Task: Reasoning Localization
* 23.04 (AAAI24) [Relax Image-Specific Prompt Requirement in SAM: A Single Generic Prompt for Segmenting Camouflaged Objects](https://arxiv.org/abs/2312.07374) | [Paper📑](https://arxiv.org/abs/2312.07374) [Code🖥️](https://github.com/jyLin8100/GenSAM)
  - Employ Cross-modal Chains of Thought Prompting (CCTP) to reason visual prompts using the semantic information given by a generic text prompt. | Task: Reasoning segmentation
* 23.12 (CVPR24) [PixelLM:Pixel Reasoning with Large Multimodal Model](https://arxiv.org/abs/2312.02228) | [Paper📑](https://arxiv.org/pdf/2312.02228.pdf) [Code🖥️](https://github.com/MaverickRen/PixelLM)
  - An effective and efficient LMM for pixel-level reasoning and understanding | Task: Reasoning Segmentation
* 23.08 (CVPR24)[LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/abs/2308.00692) | [Paper📑](https://arxiv.org/abs/2308.00692) [Code🖥️](https://github.com/showlab/VideoLISA) [Dataset🤗](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy?usp=sharing)
  - Inherit the language generation capabilities of the MLLM while also possessing the ability to produce segmentation masks. | Task: Reasoning Segmentation
#### Video MLLM
* 26.02 [VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](https://arxiv.org/abs/2602.17807) | [Paper📑](https://arxiv.org/abs/2602.17807) [Code🖥️](https://github.com/tue-mps/videomt) [Model🤗](https://huggingface.co/tue-mps/videomt-dinov2-small-ytvis2019)
  - Lightweight encoder-only video segmentation on plain ViT with query propagation and fusion, achieving 160 FPS with ViT-L without dedicated tracking modules. | Task: Reasoning Segmentation
* 26.02 [Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](https://arxiv.org/abs/2602.18996) | [Paper📑](https://arxiv.org/abs/2602.18996) [Code🖥️](https://github.com/shannany0606/CCMP)
  - Conditional binary segmentation with cycle-consistency training for object-level correspondence across egocentric/exocentric viewpoints without ground-truth annotations (CVPR 2026). | Task: Reasoning Segmentation
* 24.08 (ECCV24)[VISA: Reasoning Video Object Segmentation via Large Language Model](http://arxiv.org/abs/2407.11325) | [Paper📑](http://arxiv.org/abs/2407.11325) [Code🖥️](https://github.com/cilinyan/VISA) [Dataset🤗](https://github.com/cilinyan/ReVOS-api)
   - Leverage the world knowledge reasoning capabilities of MLLMs while possessing the ability to segment and track objects in videos with a mask decoder | Task: Reasoning Segmentation
* 24.07 (NeruIPS24)[One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos](https://arxiv.org/abs/2409.19603) |  [Paper📑](https://arxiv.org/abs/2409.19603) [Code🖥️](https://github.com/dvlab-research/LISA) [Model🤗](https://huggingface.co/ZechenBai/VideoLISA-3.8B)
  - Integrating a Sparse Dense Sampling strategy into the video-LLM to balance temporal context and spatial detail within computational constraints |  Task: Reasoning Segmentation
* 24.01 (CVPR24) [OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding](https://arxiv.org/abs/2401.10229) | [Paper📑](https://arxiv.org/abs/2401.10229) [Code🖥️](https://github.com/lxtGH/OMG-Seg)
  - A transformer-based encoder-decoder architecture with task-specific queries and outputs for multiple tasks | Task: Reasoning Segmentation/Detection
#### Audio MLLM
* 24.10 [Can Large Audio-Language Models Truly Hear? Tackling Hallucinations with Multi-Task Assessment and Stepwise Audio Reasoning](https://arxiv.org/abs/2410.16130)

#### Omni MLLM
* 25.07 [Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](https://arxiv.org/abs/2507.22886)
* 24.08 [Unleashing the Temporal-Spatial Reasoning Capacity of GPT for Training-Free Audio and Language Referenced Video Object Segmentation](https://arxiv.org/abs/2408.15876)

<a name="Spatio-TemporalReasoning"></a>
### Spatial and Temporal Grounding and Understanding
#### Image MLLM
* 26.02 [GeoWorld: Geometric World Models](https://arxiv.org/abs/2602.23058) | [Paper📑](https://arxiv.org/abs/2602.23058)
  - Hyperbolic JEPA preserving latent state structures for improved long-horizon world model prediction and Geometric RL planning (CVPR 2026). | Task: Spatial Reasoning
* 26.02 [When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning](https://arxiv.org/abs/2602.08236) | [Paper📑](https://arxiv.org/abs/2602.08236) [Code🖥️](https://github.com/Yui010206/Adaptive-Visual-Imagination-Control)
  - AVIC adaptively invokes visual imagination via world models to match or outperform fixed imagination strategies for spatial reasoning with far fewer calls. | Task: Spatial Reasoning
* 26.02 [Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?](https://arxiv.org/abs/2602.07055) | [Paper📑](https://arxiv.org/abs/2602.07055) [Code🖥️](https://github.com/mll-lab-nu/Theory-of-Space) [Dataset🤗](https://huggingface.co/datasets/MLL-Lab/tos-data)
  - Evaluates VLMs' ability to construct spatial beliefs through active exploration, revealing Active-Passive Gap and Belief Inertia—VLMs fail to update stale spatial priors. | Task: Spatial Understanding
* 26.02 [SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](https://arxiv.org/abs/2602.03916) | [Paper📑](https://arxiv.org/abs/2602.03916) [Code🖥️](https://github.com/SpatiaLab-Reasoning/SpatiaLab) [Dataset🤗](https://huggingface.co/datasets/ciol-research/SpatiaLab)
  - Benchmark of 1,400 VQA pairs across six spatial reasoning categories revealing VLMs achieve only ~55% vs. 87.6% human accuracy (ICLR 2026). | Task: Spatial Reasoning
* 26.02 [LangMap: A Hierarchical Benchmark for Open-Vocabulary Goal Navigation](https://arxiv.org/abs/2602.02220) | [Paper📑](https://arxiv.org/abs/2602.02220) [Code🖥️](https://github.com/bo-miao/LangMap)
  - Multi-granularity open-vocabulary navigation task with 414 object categories and 18K+ navigation tasks across scene, room, region, and instance levels. | Task: Spatial Grounding & Navigation
* 26.01 [CoV: Chain-of-View Prompting for Spatial Reasoning](https://arxiv.org/abs/2601.05172) | [Paper📑](https://arxiv.org/abs/2601.05172) [Code🖥️](https://github.com/ziplab/CoV)
  - Training-free test-time reasoning framework transforming VLMs into active viewpoint reasoners through coarse-to-fine 3D exploration, +11.56% on OpenEQA. | Task: Spatial Reasoning
* 26.01 [Think3D: Thinking with Space for Spatial Reasoning](https://arxiv.org/abs/2601.13029) | [Paper📑](https://arxiv.org/abs/2601.13029)
  - Framework for spatial reasoning enabling models to reason in 3D space for improved visual understanding tasks. | Task: Spatial Reasoning & 3D Understanding
* 25.12 [SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL](https://arxiv.org/abs/2512.04069) | [Paper📑](https://arxiv.org/abs/2512.04069)
  - Tool-augmented spatial reasoning using double interactive reinforcement learning. | Task: Spatial Reasoning
* 25.12 [COOPER: A Unified Model for Cooperative Perception and Reasoning in Spatial Intelligence](https://arxiv.org/abs/2512.04563) | [Paper📑](https://arxiv.org/abs/2512.04563)
  - Unified model combining cooperative perception with spatial intelligence reasoning. | Task: Spatial Reasoning
* 25.11 [SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](https://arxiv.org/abs/2511.07403) | [Paper📑](https://arxiv.org/abs/2511.07403)
  - Uses reinforcement learning with spatial rewards to improve 3D reasoning in MLLMs. | Task: Spatial Reasoning & 3D Understanding
* 25.11 [G2VLM: Geometry Grounded Vision Language Model with Unified 3D Reconstruction and Spatial Reasoning](https://arxiv.org/abs/2511.21688) | [Paper📑](https://arxiv.org/abs/2511.21688)
  - Unifies 3D reconstruction and spatial reasoning in a geometry-grounded VLM. | Task: Spatial Reasoning & 3D Understanding
* 25.10 [SpaceVista: All-Scale Visual Spatial Reasoning from mm to km](https://arxiv.org/abs/2510.09606) | [Paper📑](https://arxiv.org/abs/2510.09606)
  - Spatial reasoning across multiple scales in visual understanding. | Task: Spatial Reasoning
* 25.08 [3D-R1: Enhancing Reasoning in 3D VLMs for Unified Scene Understanding](https://arxiv.org/abs/2507.23478) | [Paper📑](https://arxiv.org/abs/2507.23478)
  - Enhances reasoning capabilities of 3D vision-language models for unified 3D scene understanding. | Task: Spatial Reasoning & 3D Understanding
* 25.04 [Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](https://arxiv.org/pdf/2504.17207) | [Paper📑](https://arxiv.org/pdf/2504.17207) [Project🌐](https://apc-vlm.github.io/) [Code🖥️](https://github.com/KAIST-Visual-AI-Group/APC-VLM) 
  - A framework for perspective-aware reasoning in vision-language models (VLMs) through mental imagery simulation. | Task: Spatial Reasoning & Understanding
* 25.04 [Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning](https://arxiv.org/html/2503.20752v2) | [Paper📑](https://arxiv.org/html/2503.20752v2) [Project🌐](https://tanhuajie.github.io/ReasonRFT/) [Code🖥️](https://github.com/tanhuajie/Reason-RFT) [Dataset🤗](https://huggingface.co/datasets/tanhuajie2001/Reason-RFT-CoT-Dataset)
  - Introduce a combined RL and SFT training paradigm to enhance visual reasoning capabilities in multimodal models. | Task: Spatial Reasoning & Understanding
* 25.04 [InteractVLM: 3D Interaction Reasoning from 2D Foundational Models](https://arxiv.org/abs/2504.05303) | [Paper📑](https://arxiv.org/abs/2504.05303) [Code💻](https://github.com/saidwivedi/InteractVLM)
  - Harnesses the broad visual knowledge of large Vision-Language Models (VLMs), fine-tuned with limited 3D contact data. Task: 3D Reconstruction
* 25.03 [Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks](https://arxiv.org/abs/2503.21696) | [Paper📑](https://arxiv.org/abs/2503.21696) [Code💻](https://github.com/zwq2018/embodied_reasoner)  [Project🌐](https://embodied-reasoner.github.io/ ) [Dataset🤗](https://huggingface.co/datasets/zwq2018/embodied_reasoner)
  - A model that extends O1-style reasoning to interactive embodied tasks. | Task: Interactive Embodied Tasks
* 25.03 [VisualThinker-R1-Zero](https://arxiv.org/abs/2503.05132) | [Paper📑](https://arxiv.org/abs/2503.05132) [Code💻](https://github.com/turningpoint-ai/VisualThinker-R1-Zero)
  - R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model | Task: Counting & Reasoning & 3D Understanding (CV-Bench)
* 25.03 (CVPR2025)[GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks](https://arxiv.org/pdf/2503.06514) | [Paper📑](https://arxiv.org/pdf/2503.06514)
  - Fine-tune VLMs using GFlowNet to promote generation of diverse solutions.|  Task: NumberLine (NL) & BlackJack (BJ)
* 25.02 [R1-V: Reinforcing Super Generalization Ability in Vision Language Models with Less Than $3](https://github.com/Deep-Agent/R1-V) |  [Code🖥️](https://github.com/Deep-Agent/R1-V)
  - A open-source project for VLM reasoning with GRPO | Task: Counting, Number Related Reasoning and Geometry Reasoning
* 25.01 [Imagine while Reasoning in Space: Multimodal Visualization-of-Thought](https://arxiv.org/pdf/2501.07542) | [Paper📑](https://arxiv.org/pdf/2501.07542)
  - Enables visual thinking in MLLMs by generating image visualizations of their reasoning traces.  | Task: Spatial Reasoning
#### Video MLLM
* 26.02 [TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions](https://arxiv.org/abs/2602.08711) | [Paper📑](https://arxiv.org/abs/2602.08711) [Code🖥️](https://github.com/yaolinli/TimeChat-Captioner) [Model🤗](https://huggingface.co/yaolily/TimeChat-Captioner-GRPO-7B)
  - Omni Dense Captioning with six-dimensional structural schema generating time-aware audio-visual narratives with explicit timestamps, surpassing Gemini-2.5-Pro on the task. | Task: Temporal Grounding/Understanding
* 26.02 [4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere](https://arxiv.org/abs/2602.10094) | [Paper📑](https://arxiv.org/abs/2602.10094)
  - 4D dynamic scene reconstruction framework with conditional querying at arbitrary space-time locations for flexible spatiotemporal understanding of dynamic scenes. | Task: Spatial-Temporal Understanding
* 26.01 [VideoLoom: A Video Large Language Model for Joint Spatial-Temporal Understanding](https://arxiv.org/abs/2601.07290) | [Paper📑](https://arxiv.org/abs/2601.07290) [Code🖥️](https://github.com/JPShi/VideoLoom) [Model🤗](https://huggingface.co/JPShi/VideoLoom-8B)
  - Unified Video LLM for joint spatial-temporal understanding with LoomData-8.7k dataset and LoomBench benchmark. | Task: Spatial-Temporal Understanding
* 26.01 [VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice](https://arxiv.org/abs/2601.05175) | [Paper📑](https://arxiv.org/abs/2601.05175) [Code🖥️](https://github.com/IVUL-KAUST/VideoAuto-R1)
  - Video understanding framework with "reason-when-necessary" strategy using confidence-based reasoning activation, reducing response length 3.3x. | Task: Video Understanding & Reasoning
* 26.01 [Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding](https://arxiv.org/abs/2601.10611) | [Paper📑](https://arxiv.org/abs/2601.10611) [Code🖥️](https://github.com/allenai/molmo2)
  - Open-source video-language model family with point-driven grounding and video tracking capabilities surpassing Gemini 3 Pro on grounding. | Task: Spatial Understanding & Grounding
* 26.01 [PROGRESSLM: Towards Progress Reasoning in Vision-Language Models](https://arxiv.org/abs/2601.15224) | [Paper📑](https://arxiv.org/abs/2601.15224) [Code🖥️](https://github.com/ProgressLM/ProgressLM) [Model🤗](https://huggingface.co/Raymond-Qiancx/ProgressLM-3B-RL) [Dataset🤗](https://huggingface.co/datasets/Raymond-Qiancx/ProgressLM-Dataset)
  - Addresses task progress estimation in VLMs with Progress-Bench benchmark and ProgressLM-3B model. | Task: Temporal Reasoning & Understanding
* 26.01 [HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](https://arxiv.org/abs/2601.14724) | [Paper📑](https://arxiv.org/abs/2601.14724)
  - Efficient streaming video understanding via hierarchical KV cache memory enabling temporal reasoning over long videos. | Task: Temporal Reasoning
* 25.12 [4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation](https://arxiv.org/abs/2512.17012) | [Paper📑](https://arxiv.org/abs/2512.17012)
  - Region-level 4D (3D + temporal) understanding through perceptual distillation. | Task: Spatial-Temporal Understanding
* 25.12 [MMSI-Video-Bench: A Holistic Benchmark for Video-Based Spatial Intelligence](https://arxiv.org/abs/2512.10863) | [Paper📑](https://arxiv.org/abs/2512.10863)
  - Comprehensive benchmark for evaluating spatial intelligence in video understanding. | Task: Spatial-Temporal Understanding
* 25.11 [VLA-4D: Embedding 4D Awareness into Vision-Language-Action Models for SpatioTemporally Coherent Robotic Manipulation](https://arxiv.org/abs/2511.17199) | [Paper📑](https://arxiv.org/abs/2511.17199)
  - Incorporates 4D spatiotemporal awareness into VLA models for coherent robotic manipulation. | Task: Spatial-Temporal Understanding
* 25.10 [Trace Anything: Representing Any Video in 4D via Trajectory Fields](https://arxiv.org/abs/2510.13802) | [Paper📑](https://arxiv.org/abs/2510.13802)
  - 4D spatial-temporal representation learning from video. | Task: Spatial-Temporal Understanding
* 25.08 [VLM4D: Towards Spatiotemporal Awareness in Vision Language Models](https://arxiv.org/abs/2508.02095) | [Paper📑](https://arxiv.org/abs/2508.02095)
  - Extends VLMs with spatiotemporal reasoning for understanding spatial and temporal dynamics. | Task: Spatial-Temporal Understanding
* 25.05 [MUSEG: Reinforcing Video Temporal Understanding via Timestamp-Aware Multi-Segment Grounding](https://arxiv.org/abs/2505.20715) | [Paper📑](https://arxiv.org/abs/2505.20715) [Code💻](https://github.com/THUNLP-MT/MUSEG)
* 25.04 [VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning](https://arxiv.org/pdf/2504.06958) | [Paper📑](https://arxiv.org/pdf/2504.06958) [Code💻](https://github.com/OpenGVLab/VideoChat-R1)
  - A novel spatiao-temporal perception framework with GRPO | Task: Spatial Understanding and Grounding
* 25.04 [VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search](https://arxiv.org/html/2504.09130v1) | [Paper📑](https://arxiv.org/html/2504.09130v1) [Code💻](https://github.com/ekonwang/VisuoThink)
  - A novel framework that seamlessly integrates visuospatial and linguistic domains | Task: Geometry and Spatial Reasoning
* 25.04 [Improved Visual-Spatial Reasoning via R1-Zero-Like Training](https://arxiv.org/abs/2504.00883) | [Paper📑](https://arxiv.org/abs/2504.00883) [Code💻](https://github.com/zhijie-group/R1-Zero-VSI)
  - Incorporate GRPO training for improved visual-spatial reasoning, using the carefully curated VSI-100k dataset. | Task: Video Understanding
* 25.03 [Envolving Temporal Reasoning Capability into LMMs via Temporal Consistent Reward](https://github.com/appletea233/Temporal-R1) | [Code💻](https://github.com/appletea233/Temporal-R1) [Model🤗](https://huggingface.co/appletea2333)
  - Investigate the potential of GRPO in the video temporal grounding task, which demands precise temporal alignment between visual and linguistic modalities as well as advanced reasoning capabilities | Task: Temporal Grounding
* 25.03 [TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM](https://arxiv.org/abs/2503.13377) | [Paper📑](https://arxiv.org/abs/2503.13377) [Code💻](https://github.com/www-Ye/TimeZero) [Model🤗](https://huggingface.co/wwwyyy/TimeZero-Charades-7B)
  - A reasoning-guided MLLM for temporal video grounding, trained with GRPO. | Task: Temporal Grounding
* 25.03 [LLaVA-ST: A Multimodal Large Language Model for Fine-Grained Spatial-Temporal Understanding](https://arxiv.org/abs/2501.08282) | [Paper📑](https://arxiv.org/abs/2501.08282) [Code💻](https://github.com/appletea233/LLaVA-ST)
  - A MLLM for fine-grained spatial-temporal multimodal understanding. | Task: Spatial-Temporal Understanding
* 25.03 [MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse](https://github.com/PzySeere/MetaSpatial) | [Code🖥️](https://github.com/PzySeere/MetaSpatial)
  - Enhance spatial reasoning in VLMs using GRPO  | Task: 3D Spatial Reasoning
* 25.02 [Video-R1: Towards Super Reasoning Ability in Video Understanding](https://github.com/tulerfeng/Video-R1) | [Code🖥️](https://github.com/tulerfeng/Video-R1)
  - Integrate deep thinking capabilities into video understanding tasks through the R1 paradigm | Task:  Video Counting
* 24.12 [TIMEREFINE: Temporal Grounding with Time Refining Video LLM](https://arxiv.org/pdf/2412.09601) | [Paper📑](https://arxiv.org/pdf/2412.09601) | [Code🖥️](https://github.com/SJTUwxz/TimeRefine)
  * Enhance Video LLMs to handle the temporal grounding task by modifying the learning objective | Task: Temporal Grounding
* 24.11 (CVPR2025) [Number it: Temporal Grounding Videos like Flipping Manga](https://arxiv.org/pdf/2411.10332) | [Paper📑](https://arxiv.org/pdf/2411.10332) | [Code💻](https://github.com/yongliang-wu/NumPro)
  * Enhances Video-LLMs by overlaying frame numbers onto video frames | Task: Temporal Grounding
* 24.11 [TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability](https://arxiv.org/abs/2411.18211) | [Paper📑](https://arxiv.org/pdf/2411.18211) | [Code💻](https://github.com/TimeMarker-LLM/TimeMarker/)
  * A versatile Video-LLM featuring robust temporal localization abilities | Task: Temporal Grounding and Video QA
* 24.08 (AAAI2025) [Grounded Multi-Hop VideoQA in Long-Form Egocentric Videos](https://arxiv.org/abs/2408.14469) | [Paper📑](https://arxiv.org/pdf/2408.14469) | [Code💻](https://github.com/qirui-chen/MultiHop-EgoQA)
  * Leverage the world knowledge reasoning capabilities of MLLMs to retrieve temporal evidence in the video with flexible grounding tokens. | Task: Multi-Hop VideoQA
* 24.08 (ICLR2025) [TRACE: Temporal Grounding Video LLM via Casual Event Modeling](https://arxiv.org/abs/2410.05643) | [Paper📑](https://arxiv.org/pdf/2410.05643) | [Code💻](https://github.com/gyxxyg/TRACE)
  * Tailored to implement the causal event modeling framework through timestamps, salient scores, and textual captions. | Task: Temporal Grounding
#### Audio MLLM
* 25.07 [Towards Spatial Audio Understanding via Question Answering](https://arxiv.org/abs/2507.09195)
* 24.06 (InterSpeech 2024) [Can Large Language Models Understand Spatial Audio?](https://arxiv.org/abs/2406.07914) | 
* 24.02 (ICML 2024)[BAT: Learning to Reason about Spatial Sounds with Large Language Models](https://arxiv.org/abs/2402.01591) |

#### Omni MLLM
* 24.06 [VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs](https://arxiv.org/abs/2406.07476)

<a name="MathReasoning"></a>

### Math Reasoning
#### Image MLLM
* 26.02 [P1-VL: Bridging Visual Perception and Scientific Reasoning in Physics Olympiads](https://arxiv.org/abs/2602.09443) | [Paper📑](https://arxiv.org/abs/2602.09443) [Code🖥️](https://github.com/PRIME-RL/P1-VL) [Model🤗](https://huggingface.co/PRIME-RL/P1-VL-30B-A3B)
  - Open-source VLM family for advanced scientific reasoning using curriculum RL and agentic augmentation, achieving the first open-source model winning 12 gold medals at physics olympiad level. | Task: Math
* 26.02 [DeepVision-103K: A Visually Diverse, Broad-Coverage, and Verifiable Mathematical Dataset for Multimodal Reasoning](https://arxiv.org/abs/2602.16742) | [Paper📑](https://arxiv.org/abs/2602.16742) [Code🖥️](https://github.com/SKYLENAGE-AI/DeepVision-103K) [Dataset🤗](https://huggingface.co/datasets/skylenage/DeepVision-103K)
  - 103K-sample RLVR training dataset for multimodal K12 mathematical reasoning with diverse topics and rich visual elements, generalizing to general multimodal reasoning tasks. | Task: Math
* 26.02 [Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models](https://arxiv.org/abs/2601.22060) | [Paper📑](https://arxiv.org/abs/2601.22060) [Code🖥️](https://github.com/Osilly/Vision-DeepResearch)
  - Multimodal deep-research paradigm enabling multi-turn, multi-entity, multi-scale visual and textual search via cold-start supervision and RL. | Task: Math
* 26.01 [CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](https://arxiv.org/abs/2601.01874) | [Paper📑](https://arxiv.org/abs/2601.01874) [Project🌐](https://shchen233.github.io/cogflow/)
  - Cognitive-inspired three-stage framework (Perception-Internalization-Reasoning) for visual math with MathCog dataset of 120K+ annotations. | Task: Math
* 26.01 [MindWatcher: Toward Smarter Multimodal Tool-Integrated Reasoning](https://arxiv.org/abs/2512.23412) | [Paper📑](https://arxiv.org/abs/2512.23412)
  - Multimodal tool-integrated reasoning framework enhancing chain-of-thought with tool use for complex math/science problems. | Task: Math
* 26.01 [MMFormalizer: Multimodal Autoformalization in the Wild](https://arxiv.org/abs/2601.03017) | [Paper📑](https://arxiv.org/abs/2601.03017)
  - Framework for automatically formalizing multimodal mathematical content from images and text into formal representations. | Task: Math
* 25.11 [MathSE: Improving Multimodal Mathematical Reasoning via Self-Evolving Iterative Reflection and Reward-Guided Fine-Tuning](https://arxiv.org/abs/2511.06805) | [Paper📑](https://arxiv.org/abs/2511.06805)
  - Improves multimodal math reasoning via iterative self-evolution and reward-guided training. | Task: Math
* 25.10 [Training Vision-Language Process Reward Models for Test-Time Scaling in Multimodal Reasoning](https://arxiv.org/abs/2509.23250) | [Paper📑](https://arxiv.org/abs/2509.23250)
  - Process reward models for scaling multimodal reasoning at test time. | Task: Math
* 25.09 [BaseReward: A Strong Baseline for Multimodal Reward Model](https://arxiv.org/abs/2509.16127) | [Paper📑](https://arxiv.org/abs/2509.16127)
  - Strong baseline reward model for multimodal RL-based alignment. | Task: Math
* 25.08 [MathReal: A Real Scene Benchmark for Evaluating Math Reasoning in MLLMs](https://arxiv.org/abs/2508.06009) | [Paper📑](https://arxiv.org/abs/2508.06009)
  - Benchmark for evaluating multimodal math reasoning using real-world scene photographs. | Task: Math
* 25.11 [Perceptual-Evidence Anchored Reinforced Learning for Multimodal Reasoning](https://arxiv.org/abs/2511.18437) | [Paper📑](https://arxiv.org/abs/2511.18437) [Code🖥️](https://github.com/MiliLab/PEARL) [Model🤗](https://huggingface.co/Rex1090/PEARL-8B)
   - Introduce a perception checklist to anchor RL policy updates in verified visual evidence and prevent hallucinations. | Task: Math
* 25.11 [Metis-HOME: Hybrid Optimized Mixture-of-Experts for Multimodal Reasoning](https://arxiv.org/abs/2510.20519) | [Paper📑](https://arxiv.org/abs/2510.20519) [Code🖥️](https://github.com/MM-Thinking/Metis-HOME) [Model🤗](https://huggingface.co/mmthinking/Metis-HOME)
  - Use a mixture-of-experts framework with dynamic routing for balancing complex reasoning and general tasks. | Task: Math
* 25.10 [Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](https://arxiv.org/abs/2510.25801) | [Paper📑](https://arxiv.org/abs/2510.25801) [Code🖥️](https://github.com/Kwen-Chen/SPECS-VL)
  - Replace supervised fine-tuning with self-distilled, preference-based cold starts to improve RL generalization. | Task: Math
* 25.09 [DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](https://arxiv.org/abs/2509.25866) | [Paper📑](https://arxiv.org/abs/2509.25866) [Code🖥️](https://github.com/MiliLab/DeepSketcher)
  - Internalize visual reasoning by directly manipulating visual embeddings using code-rendered trajectories, bypassing external tools and reducing grounding noise. | Task: Math
* 25.07 [The Synergy Dilemma of Long-CoT SFT and RL: Investigating Post-Training Techniques for Reasoning VLMs](https://www.arxiv.org/abs/2507.07562) [Paper📑](https://www.arxiv.org/abs/2507.07562) [Code🖥️](https://github.com/JierunChen/SFT-RL-SynergyDilemma) 
  - a systematic investigation into the distinct roles and interplay of long-CoT SFT and RL across multiple multimodal reasoning benchmarks. | Task: Math
* 25.06 [Metis-RISE: RL Incentivizes and SFT Enhances Multimodal Reasoning Model Learning](https://arxiv.org/abs/2506.13056) | [Paper📑](https://arxiv.org/abs/2506.13056) [Code🖥️](https://github.com/MM-Thinking/Metis-RISE) [Model🤗](https://github.com/MM-Thinking/Metis-RISE)
  - Reverse the training pipeline by first using RL for reasoning exploration, then applying SFT with self-distilled and expert-augmented trajectories for stability and capability enhancement. | Task: Math
* 25.06 [SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis](https://arxiv.org/abs/2506.02096) [Paper📑](https://arxiv.org/abs/2506.02096) [Code🖥️](https://github.com/NUS-TRAIL/SynthRL) [Model🤗](https://huggingface.co/collections/Jakumetsu/synthrl-6839d265136fa9ca717105c5)
  - A novel framework that enhances the reasoning capabilities of multimodal large language models. | Task: Math
* 25.06 [SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](https://arxiv.org/abs/2506.01713) [Paper📑](https://arxiv.org/abs/2506.01713) [Code🖥️](https://github.com/SUSTechBruce/SRPO_MLLMs) [Model🤗](https://huggingface.co/datasets/SRPOMLLMs/srpo-sft-data)
  - scale the training data with correctness and distribution guarantees to achieve better performance. | Task: Math
* 25.05 [Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO](https://arxiv.org/pdf/2505.22453) [Paper📑](https://arxiv.org/pdf/2505.22453) [Code🖥️](https://github.com/waltonfuture/MM-UPT) 
  - A Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO. | Task: Math
* 25.05 [X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains](https://arxiv.org/abs/2505.03981) | [Paper📑](https://arxiv.org/abs/2505.03981) [Code🖥️](github.com/microsoft/x-reasoner) 
  - A training recipe that optimizes the reasoning capability of VLMs with SFT and RL on general-domain text-only data. | Task: Math
* 25.04 [NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](https://arxiv.org/pdf/2504.13055) | [Paper📑](https://arxiv.org/pdf/2504.13055) [Code🖥️](https://github.com/John-AI-Lab/NoisyRollout) [Model🤗](https://huggingface.co/collections/xyliu6/noisyrollout-67ff992d1cf251087fe021a2)
  - Introduces targeted rollout diversity by mixing rollouts from both clean and moderately distorted images, encouraging the model to learn more robust behaviors. | Task: Math
* 25.04 [VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning](https://arxiv.org/abs/2504.08837) | [Paper📑](https://arxiv.org/abs/2504.08837) [Code🖥️](https://github.com/TIGER-AI-Lab/VL-Rethinker/) [Model🤗](https://huggingface.co/TIGER-Lab/VL-Rethinker-7B)
  - Aim to enhance the slow-thinking capabilities of vision-language models using reinforcement learning (without relying on distillation) to advance the SOTA. | Task: Math
* 25.04 [SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement](https://arxiv.org/abs/2504.07934) | [Paper📑](https://arxiv.org/abs/2504.07934) [Code🖥️](https://github.com/si0wang/ThinkLite-VL)
  - Propose a novel way of repurposing Monte Carlo Tree Search (MCTS) to enable effective data filtering. | Task: Math reasoning
* 25.04 [GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning]() | [Paper📑](https://github.com/RyanLiu112/GenPRM/blob/main/static/paper.pdf) [Project🌐](https://ryanliu112.github.io/GenPRM/) [Code🖥️](https://github.com/RyanLiu112/GenPRM)
  - A generative process reward model that performs explicit COT reasoning with code verification before providing judgment for each reasoning step. | Task: Math
* 25.03 [OpenVLThinker: An Early Exploration to Vision-Language Reasoning via Iterative Self-Improvement](https://arxiv.org/abs/2503.17352) | [Paper📑](https://arxiv.org/abs/2503.17352) [Code🖥️](https://github.com/yihedeng9/OpenVLThinker) [Dataset🤗](https://huggingface.co/ydeng9/OpenVLThinker-7B)
  - Investigate whether R1-like reasoning capabilities can be successfully integrated into LVLMs and assesses their impact on challenging multimodal reasoning tasks. | Task: Math
* 25.03 [R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](https://arxiv.org/html/2503.12937v1) | [Paper📑](https://arxiv.org/html/2503.12937v1) [Code🖥️](https://github.com/jingyi0000/R1-VL) [Dataset🤗](https://github.com/jingyi0000/R1-VL#)
  - Design Step-wise Group Relative Policy Optimization (StepGRPO) that enables MLLMs to self-improve reasoning ability. | Task: Math 
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

* 26.02 [OCR-Agent: Agentic OCR with Capability and Memory Reflection](https://arxiv.org/abs/2602.21053) | [Paper📑](https://arxiv.org/abs/2602.21053) [Code🖥️](https://github.com/AIGeeksGroup/OCR-Agent)
  - Iterative self-correction framework using Capability Reflection (error diagnosis) and Memory Reflection (avoiding repeated attempts), achieving SOTA on OCRBench v2 without training. | Task: Document Reasoning
* 26.02 [OmniOCR: Generalist OCR for Ethnic Minority Languages](https://arxiv.org/abs/2602.21042) | [Paper📑](https://arxiv.org/abs/2602.21042) [Code🖥️](https://github.com/AIGeeksGroup/OmniOCR)
  - Universal OCR framework using Dynamic LoRA for low-resource ethnic minority scripts, achieving 39-66% accuracy improvements on Tibetan, Shui, and other scripts. | Task: Document Reasoning
* 26.02 [DODO: Discrete OCR Diffusion Models](https://arxiv.org/abs/2602.16872) | [Paper📑](https://arxiv.org/abs/2602.16872)
  - Adapts block discrete diffusion for OCR enabling parallel token processing, achieving up to 3× faster inference while maintaining near-SOTA accuracy. | Task: Document Reasoning
* 26.02 [PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing](https://arxiv.org/abs/2601.21957) | [Paper📑](https://arxiv.org/abs/2601.21957)
  - Compact 0.9B VLM for multi-task document parsing in diverse real-world conditions covering OCR, layout understanding, and chart comprehension. | Task: Document Reasoning
* 26.02 [MemOCR: Layout-Aware Visual Memory for Efficient Long-Horizon Reasoning](https://arxiv.org/abs/2601.21468) | [Paper📑](https://arxiv.org/abs/2601.21468)
  - Layout-aware visual memory mechanisms for MLLMs to improve long-horizon document and OCR reasoning efficiency. | Task: Document Reasoning
* 26.01 [ChartVerse: Scaling Chart Reasoning via Reliable Programmatic Synthesis from Scratch](https://arxiv.org/abs/2601.13606) | [Paper📑](https://arxiv.org/abs/2601.13606) [Code🖥️](https://github.com/starriver030515/ChartVerse) [Model🤗](https://huggingface.co/opendatalab/ChartVerse-8B) [Dataset🤗](https://huggingface.co/datasets/opendatalab/ChartVerse-SFT-1.8M)
  - Scalable chart reasoning framework using Rollout Posterior Entropy; ChartVerse-8B surpasses its teacher model Qwen3-VL-30B. | Task: Chart Reasoning
* 25.10 [From Charts to Code: A Hierarchical Benchmark for Multimodal Models](https://arxiv.org/abs/2510.17932) | [Paper📑](https://arxiv.org/abs/2510.17932)
  - Benchmark for chart understanding and code generation from charts. | Task: Chart Reasoning
* 25.09 [Visual-TableQA: Open-Domain Benchmark for Reasoning over Table Images](https://arxiv.org/abs/2509.07966) | [Paper📑](https://arxiv.org/abs/2509.07966)
  - Benchmark for visual question answering and reasoning over table images. | Task: Chart Reasoning
* 25.09 [MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing](https://arxiv.org/abs/2509.22186) | [Paper📑](https://arxiv.org/abs/2509.22186)
  - Efficient VLM for parsing and understanding high-resolution documents. | Task: Document Reasoning
* 25.09 [Visual Programmability: A Guide for Code-as-Thought in Chart Understanding](https://arxiv.org/abs/2509.09286) | [Paper📑](https://arxiv.org/abs/2509.09286) [Code🖥️](https://github.com/Aphelios-Tang/Code-as-Thought)
   - Introduce an adaptive framework that enables VLMs to dynamically choose between code-based and visual reasoning pathways for chart understanding. | Task: Chart Reasoning
* 25.07 [Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner](https://arxiv.org/abs/2507.15509) | [Paper📑](https://arxiv.org/abs/2507.15509) [Code🖥️](https://github.com/DocTron-hub/Chart-R1) [Model🤗](https://huggingface.co/collections/DocTron/chart-r1)
  - Combine chain-of-thought supervision with reinforcement learning, supported by programmatically synthesized step-by-step reasoning data. | Task: Chart Reasoning
* 25.06 [ChartReasoner: Code-Driven Modality Bridging for Long-Chain Reasoning in Chart Question Answering](https://arxiv.org/abs/2506.10116) | [Paper📑](https://arxiv.org/abs/2506.10116)
  - Combine chart code generation with long-chain reasoning LLMs to produce detailed reasoning processes. | Task: Chart Reasoning
* 25.05 [Point-RFT: Improving Multimodal Reasoning with Visually Grounded Reinforcement Finetuning](https://arxiv.org/abs/2505.19702) | [Paper📑](https://arxiv.org/abs/2505.19702)
  - Introduce a visually grounded chain-of-thought (CoT) paradigm, enabling the model to generate CoT reasoning aligned with visual elements. | Task: Chart Reasoning
* 25.04 [Bespoke-MiniChart-7B: Pushing The Frontiers Of Open VLMs For Chart Understanding](https://www.bespokelabs.ai/blog/bespoke-minichart-7b) | [Project🌐](https://www.bespokelabs.ai/blog/bespoke-minichart-7b) [Model🤗](https://huggingface.co/bespokelabs/Bespoke-MiniChart-7B)
  - Employ a three-stage training process, combining rejection sampling and DPO optimization to enhance out-of-distribution generalization. | Task: Chart Reasoning
* 25.03 [MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding](https://arxiv.org/pdf/2503.13964v1) | [Paper📑](https://arxiv.org/pdf/2503.13964v1) [Code🖥️](https://github.com/aiming-lab/MDocAgent)
  - Integrate text and image retrieval through various agents, enabling collaborative reasoning across modalities. | Task: Document Reasoning
* 24.11 [ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild](https://arxiv.org/abs/2407.04172) | [Paper📑](https://arxiv.org/abs/2407.04172) [Code🖥️](https://github.com/vis-nlp/ChartGemma) [Model🤗](https://huggingface.co/ahmed-masry/chartgemma) [Dataset🤗](https://huggingface.co/datasets/ahmed-masry/ChartGemma)
  - Generate multi-task instruction-tuning data from real chart images and integrating both COT and POT reasoning pathways. | Task: Chart Reasoning
* 24.09 (ICLR25 Oral) [ChartMoE: Mixture of Diversely Aligned Expert Connector for Chart Understanding](https://arxiv.org/abs/2409.03277) | [Paper📑](https://arxiv.org/abs/2409.03277) [Code🖥️](https://github.com/IDEA-FinAI/ChartMoE) [Model🤗](https://huggingface.co/IDEA-FinAI/chartmoe) [Dataset🤗](https://huggingface.co/datasets/Coobiw/ChartMoE-Data)
  - Utilize diverse chart-text aligned tasks (chart -> table/json/python-code) to augment chart understanding and reasoning. | Task: Chart Reasoning
* 24.09 [ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning](https://arxiv.org/abs/2402.12185) | [Project🌐](https://unimodal4reasoning.github.io/DocGenome_page/)  [Paper📑](https://arxiv.org/abs/2402.12185) [Code🖥️](https://github.com/Alpha-Innovator/ChartVLM)
  - Offer a new perspective on handling chart reasoning tasks that strongly depend on interpretable patterns. | Task: Chart Reasoning
* 24.07 (EMNLP24) [Multimodal Self-Instruct: Synthetic Abstract Image and Visual Reasoning Instruction Using Language Model](https://arxiv.org/abs/2407.07053) | [Paper📑](https://arxiv.org/abs/2407.07053)  [Project🌐](https://multi-modal-self-instruct.github.io/) [Code🖥️](https://github.com/zwq2018/Multi-modal-Self-instruct) [Dataset🤗](https://huggingface.co/datasets/zwq2018/Multi-modal-Self-instruct)
  - A multi-modal self-instruct, utilizing large language models and their code capabilities to synthesize massive abstract images and visual reasoning instructions across daily scenarios. | Task: Chart Reasoning
* 24.04 (EMNLP24) [TinyChart: Efficient Chart Understanding with Visual Token Merging and Program-of-Thoughts Learning](https://arxiv.org/abs/2404.16635) | [Paper📑](https://arxiv.org/abs/2404.16635) [Code🖥️](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart) [Model🤗](https://huggingface.co/mPLUG/TinyChart-3B-768-siglip) [Dataset🤗](https://huggingface.co/datasets/mPLUG/TinyChartData)
  - Employ PoT learning for numerical reasoning and Vision Token Merging to compress visual features from high-resolution images. | Task: Chart Reasoning
* 24.04 (MM24) [OneChart: Purify the Chart Structural Extraction via One Auxiliary Token](https://arxiv.org/abs/2404.09987) | [Paper📑](https://arxiv.org/abs/2404.09987) [Project🌐](https://onechartt.github.io/) [Code🖥️](https://github.com/LingyvKong/OneChart) [Model🤗](https://huggingface.co/kppkkp/OneChart) 
  - Introduce an auxiliary token and decoder combined with a customized L1 loss to enhance the reliability of structured and numerical information extraction. | Task: Chart Reasoning
* 24.04 (MM24) [NovaChart: A Large-scale Dataset towards Chart Understanding and Generation of Multimodal Large Language Models](https://dl.acm.org/doi/10.1145/3664647.3680790) | [Paper📑](https://dl.acm.org/doi/10.1145/3664647.3680790) [Code🖥️](https://github.com/Elucidator-V/NovaChart) [Dataset🤗](https://huggingface.co/datasets/ympan/novachart)
  - Construct a large-scale dataset for chart understanding and generation, covering 18 different chart types and 15 unique tasks. | Task: Chart Reasoning
* 24.02 (ACL24) [ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning](https://arxiv.org/abs/2401.02384) | [Paper📑](https://arxiv.org/abs/2401.02384) [Code🖥️](https://github.com/OpenGVLab/ChartAst) [Dataset🤗](https://huggingface.co/datasets/FanqingM/ChartAssistant)
  - Use large-scale chart data to align and instruction tuning | Task: Chart Reasoning
* 23.11 [ChartLlama: A Multimodal LLM for Chart Understanding and Generation](https://arxiv.org/abs/2311.16483) | [Paper📑](https://arxiv.org/abs/2311.16483) [Project🌐](https://tingxueronghua.github.io/ChartLlama/) [Code🖥️](https://github.com/tingxueronghua/ChartLlama-code) [Model🤗](https://huggingface.co/listen2you002/ChartLlama-13b) [Dataset🤗](https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset)
  - Generate a diverse and high-quality instruction-tuning dataset using GPT-4, and use LLaVA for unified multi-task training. | Task: Chart Reasoning
* 23.10 (EMNLP23) [UniChart: A Universal Vision-language Pretrained Model for
Chart Comprehension and Reasoning](https://arxiv.org/abs/2305.14761) | [Paper📑](https://arxiv.org/abs/2305.14761) [Code🖥️](https://github.com/vis-nlp/UniChart) [Model🤗](https://huggingface.co/ahmed-masry/unichart-base-960) [Dataset🤗](https://huggingface.co/datasets/ahmed-masry/unichart-pretrain-data)
  - Pretrains on a large and diverse chart dataset, explicitly modeling visual elements and structures. | Task: Chart Reasoning

#### Benchmark
* 25.11 (EMNLP25) [ChartM3: A Multi-Stage Code-Driven Pipeline for Constructing Multi-Dimensional and Multi-Step Visual Reasoning Data in Chart Comprehension](https://arxiv.org/abs/2511.02415) | [Paper📑](https://arxiv.org/abs/2511.02415)
  - Provide an evaluation set of 2,871 high-quality samples covering 62 chart types and 60 real-world scenarios, focusing on multi-dimensional and multi-step visual reasoning and complex business analysis. | Task: Chart Reasoning
* 25.05 [ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models](https://arxiv.org/abs/2505.13444) | [Paper📑](https://arxiv.org/abs/2505.13444) [Project🌐](https://chartmuseum-leaderboard.github.io/) [Code🖥️](https://github.com/Liyan06/ChartMuseum) [Dataset🤗](https://huggingface.co/datasets/lytang/ChartMuseum)
  - Feature real-world chart images and four distinct question types that assess textual, visual, combined, and synthesis reasoning abilities. | Task: Chart Reasoning
* 25.04 [CHARTQAPRO : A More Diverse and Challenging Benchmark for Chart Question Answering](https://arxiv.org/abs/2504.05506v2) | [Paper📑](https://arxiv.org/abs/2504.05506v2) [Code🖥️](https://github.com/vis-nlp/ChartQAPro) [Dataset🤗](https://huggingface.co/datasets/ahmed-masry/ChartQAPro)
  - Introduce a diverse benchmark with 1,341 charts and 1,948 questions covering various chart types and question formats, designed to rigorously evaluate the chart reasoning capabilities of large vision-language models in real-world scenarios. | Task: Chart Reasoning
* 25.01 (AAAI25) [EvoChart: A Benchmark and a Self-Training Approach Towards Real-World Chart Understanding](https://arxiv.org/abs/2409.01577) | [Paper📑](https://arxiv.org/abs/2409.01577) [Code🖥️](https://github.com/MuyeHuang/EvoChart) [Dataset🤗](https://huggingface.co/datasets/MuyeHuang/EvoChart-QA-Benchmark)
  - Feature 650 real-world charts, 1,250 expert-curated questions, and strict and flexible automatic evaluation metrics to assess chart comprehension abilities of VLMs in practical scenarios. | Task: Chart Reasoning
* 24.06 (NIPS24) [CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs](https://arxiv.org/abs/2406.18521) | [Paper📑](https://arxiv.org/abs/2406.18521) [Project🌐](https://charxiv.github.io/) [Code🖥️](https://github.com/princeton-nlp/CharXiv) [Dataset🤗](https://huggingface.co/datasets/princeton-nlp/CharXiv)
  - Focus on real and complex charts from arXiv papers, covering eight major domains. All content is expert-curated and verified, with evaluation using GPT-4o scoring and binary correctness metrics. | Task: Chart Reasoning
* 24.06 (VRISP25) [ChartBench: A Benchmark for Complex Visual Reasoning in Charts](https://arxiv.org/abs/2312.15915) | [Paper📑](https://arxiv.org/abs/2312.15915) [Project🌐](https://chartbench.github.io/) [Code🖥️](https://github.com/IDEA-FinAI/ChartBench) [Dataset🤗](https://huggingface.co/datasets/SincereX/ChartBench)
  - Cover 9 major categories and 42 subcategories of charts without data point annotations, emphasizing numerical extraction ability. | Task: Chart Reasoning
* 24.04 (NAACL24) [MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning](https://arxiv.org/abs/2311.10774) | [Paper📑](https://arxiv.org/abs/2311.10774) [Code🖥️](https://github.com/FuxiaoLiu/MMC) [Dataset🤗](https://huggingface.co/datasets/xywang1/MMC)
  - Propose a comprehensive human-annotated benchmark with nine distinct tasks evaluating reasoning capabilities over various charts, and support both GPT-4 scoring and multiple-choice exact matching. | Task: Chart Reasoning
* 22.05 (ACL22) [ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning](https://arxiv.org/abs/2203.10244) | [Paper📑](https://arxiv.org/abs/2203.10244) [Code🖥️](https://github.com/vis-nlp/ChartQA) [Dataset🤗](https://huggingface.co/datasets/ahmed-masry/ChartQA)
  - Use real-world charts and open-ended questions to evaluate chart understanding, reasoning, and data extraction, with relaxed accuracy as the metric. | Task: Chart Reasoning


<a name="VisualGeneration"></a>
### Visual-Audio Generation
#### Image MLLM
* 26.02 [DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing](https://arxiv.org/abs/2602.12205) | [Paper📑](https://arxiv.org/abs/2602.12205) [Code🖥️](https://github.com/DeepGenTeam/DeepGen) [Model🤗](https://huggingface.co/deepgenteam/DeepGen-1.0)
  - Lightweight 5B unified model for image generation and editing using hierarchical feature extraction, learnable think tokens, and MR-GRPO reinforcement learning, outperforming much larger models. | Task: Image Generation
* 26.02 [UniReason 1.0: A Unified Reasoning Framework for World Knowledge Aligned Image Generation and Editing](https://arxiv.org/abs/2602.02437) | [Paper📑](https://arxiv.org/abs/2602.02437) [Code🖥️](https://github.com/AlenjandroWang/UniReason) [Model🤗](https://huggingface.co/Alex11556666/UniReason)
  - Integrates text-to-image generation and editing through dual reasoning with world knowledge planning and visual refinement on reasoning-intensive benchmarks. | Task: Image Generation
* 26.02 [Generated Reality: Human-centric World Simulation using Interactive Video Generation](https://arxiv.org/abs/2602.18422) | [Paper📑](https://arxiv.org/abs/2602.18422) [Project🌐](https://codeysun.github.io/generated-reality/)
  - Human-centric video world model conditioned on tracked head and hand poses via bidirectional video diffusion for dexterous XR interactions. | Task: Image/Video Generation
* 26.01 [Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders](https://arxiv.org/abs/2601.10332) | [Paper📑](https://arxiv.org/abs/2601.10332) [Code🖥️](https://github.com/zhijie-group/Think-Then-Generate)
  - "Think-then-generate" paradigm where LLM encoders reason about prompts before image generation using Dual-GRPO reinforcement optimization. | Task: Image Generation
* 26.01 [Re-Align: Structured Reasoning-guided Alignment for In-Context Image Generation and Editing](https://arxiv.org/abs/2601.05124) | [Paper📑](https://arxiv.org/abs/2601.05124) [Code🖥️](https://github.com/hrz2000/realign)
  - Bridges multimodal understanding and image generation via In-Context Chain-of-Thought (IC-CoT) with RL-based training. | Task: Image Generation & Editing
* 26.01 [Unified Thinker: A General Reasoning Modular Core for Image Generation](https://arxiv.org/abs/2601.03127) | [Paper📑](https://arxiv.org/abs/2601.03127)
  - General reasoning modular core enhancing image generation models with chain-of-thought reasoning capabilities. | Task: Image Generation
* 25.12 [REASONEDIT: Towards Reasoning-Enhanced Image Editing Models](https://arxiv.org/abs/2511.22625) | [Paper📑](https://arxiv.org/abs/2511.22625)
  - Enhances image editing models with explicit reasoning capabilities. | Task: Image Editing
* 25.12 [EditThinker: Unlocking Iterative Reasoning for Any Image Editor](https://arxiv.org/abs/2512.05965) | [Paper📑](https://arxiv.org/abs/2512.05965)
  - Enables iterative reasoning in image editing through a reasoning-aware framework. | Task: Image Editing
* 25.11 [IE-Critic-R1: Advancing the Explanatory Measurement of Text-Driven Image Editing for Human Perception Alignment](https://arxiv.org/abs/2511.18055) | [Paper📑](https://arxiv.org/abs/2511.18055) [Code🖥️](https://github.com/Coobiw/IE-Critic-R1) [Model🤗](https://huggingface.co/Coobiw/IE-Critic-R1-7B) [Dataset🤗](https://huggingface.co/datasets/Coobiw/IE-Bench-4k) [ColdStart SFT🤗](https://huggingface.co/datasets/Coobiw/IE-Bench-CoT-mixed)
  - IE-Critic-R1 treats image editing quality assessment as a reasoning task and implement "R1 moment" (longer reasoning thoughts, better performance). It is a pointwise, generative reward model, leveraging Chain-of-Thought (CoT) reasoning SFT and RLVR to provide accurate, human-aligned evaluations of image editing. | Task: Image Editing Quality Asssessment
* 25.05 [T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT](https://arxiv.org/pdf/2505.00703) | [Paper📑](https://arxiv.org/pdf/2505.00703) [Code🖥️](https://github.com/CaraJ7/T2I-R1)
  - A novel reasoning-enhanced text-to-image generation model powered by RL with a bi-level CoT reasoning process | Task: Video Generation
* 25.03 [GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing](https://arxiv.org/pdf/2503.10639) | [Paper📑](https://arxiv.org/pdf/2503.10639) 
  - A paradigm that enables generation and editing through an explicit language reasoning process before outputting images   | Task: Image Generation
* 25.03  [Unified Reward Model for Multimodal Understanding and Generation](https://arxiv.org/abs/2503.05236) | [Paper📑](https://arxiv.org/abs/2503.05236) [Code🖥️](https://codegoat24.github.io/UnifiedReward/) [Dataset🤗](https://huggingface.co/collections/CodeGoat24/unifiedreward-training-data-67c300d4fd5eff00fa7f1ede)
  -  Improve MLLM's understanding and generation ability with DPO  | Task: VQA & Generation
* 25.01 (CVPR25) [Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step](https://arxiv.org/pdf/2501.13926) | [Paper📑](https://arxiv.org/pdf/2501.13926) [Code🖥️](https://github.com/ZiyuGuo99/Image-Generation-CoT) [Model🤗](https://huggingface.co/ZiyuG/Image-Generation-CoT)
  - The first comprehensive investigation of the potential of CoT reasoning to enhance autoregressive image generation. | Task: Image Generation
* 24.12 [EVLM: Self-Reflective Multimodal Reasoning for Cross-Dimensional Visual Editing](https://arxiv.org/pdf/2412.10566) | [Paper📑](https://arxiv.org/pdf/2412.10566)
  - A system designed to interpret such instructions in conjunction with reference visuals, producing precise and context-aware editing prompts.  | Task: Image Generation
#### Video MLLM
* 26.02 [SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model](https://arxiv.org/abs/2602.21818) | [Paper📑](https://arxiv.org/abs/2602.21818)
  - Unified multimodal video foundation model enabling simultaneous video+audio generation, editing, and inpainting via dual-stream architecture, supporting 1080p/32FPS/15s with synchronized audio. | Task: Video Generation
* 26.02 [Solaris: Building a Multiplayer Video World Model in Minecraft](https://arxiv.org/abs/2602.22208) | [Paper📑](https://arxiv.org/abs/2602.22208) [Code🖥️](https://github.com/solaris-wm/solaris) [Model🤗](https://huggingface.co/nyu-visionx/solaris)
  - Multiplayer video world model for consistent multi-view observations in coordinated multi-agent Minecraft environments using Checkpointed Self Forcing technique. | Task: Video Generation
* 26.02 [MOVA: Towards Scalable and Synchronized Video-Audio Generation](https://arxiv.org/abs/2602.08794) | [Paper📑](https://arxiv.org/abs/2602.08794) [Code🖥️](https://github.com/OpenMOSS/MOVA) [Model🤗](https://huggingface.co/collections/OpenMOSS-Team/mova)
  - Open-source 32B MoE model generating high-quality synchronized audio-visual content including lip-synced speech, environment sounds, and music from image-text inputs. | Task: Video-Audio Generation
* 25.11 [Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation](https://arxiv.org/abs/2511.14993) | [Paper📑](https://arxiv.org/abs/2511.14993)
  - Foundation model family for image and video generation. | Task: Video Generation
* 25.11 [Planning with Sketch-Guided Verification for Physics-Aware Video Generation](https://arxiv.org/abs/2511.17450) | [Paper📑](https://arxiv.org/abs/2511.17450)
  - Physics-aware video generation with sketch-based planning and verification. | Task: Video Generation
* 25.10 [PhysMaster: Mastering Physical Representation for Video Generation via RL](https://arxiv.org/abs/2510.13809) | [Paper📑](https://arxiv.org/abs/2510.13809)
  - Physical reasoning for video generation with reinforcement learning. | Task: Video Generation
* 25.02 [C-Drag:Chain-of-Thought Driven Motion Controller for Video Generation](https://arxiv.org/pdf/2502.19868) | [Paper📑](https://arxiv.org/pdf/2502.19868) [Code🖥️](https://github.com/WesLee88524/C-Drag-Official-Repo) [Dataset🤗](https://drive.google.com/file/d/1L2SYadeqZPObvSj9Mb6fK-KHtR0n-DKk/view)
  - A Chain-of-Thought-based motion controller for controllable video generation | Task: Video Generation
#### Audio MLLM
* 26.02 [AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization](https://arxiv.org/abs/2602.07054) | [Paper📑](https://arxiv.org/abs/2602.07054) [Dataset🤗](https://huggingface.co/datasets/chaubeyG/EmoReAlM)
  - AVEm-DPO preference optimization improves audiovisual emotion reasoning in MLLMs by aligning responses with audiovisual cues and reducing text-prior hallucinations. | Task: Audio-Visual Reasoning
* 26.02 [EgoAVU: Egocentric Audio-Visual Understanding](https://arxiv.org/abs/2602.06139) | [Paper📑](https://arxiv.org/abs/2602.06139) [Dataset🤗](https://huggingface.co/datasets/facebook/EgoAVU_data)
  - Scalable data engine and 3M-sample dataset for egocentric audio-visual understanding, enabling up to 113% performance improvement on joint audio-visual reasoning tasks. | Task: Audio-Visual Reasoning
* 26.01 [LTX-2: Efficient Joint Audio-Visual Foundation Model](https://arxiv.org/abs/2601.03233) | [Paper📑](https://arxiv.org/abs/2601.03233) [Code🖥️](https://github.com/Lightricks/LTX-2) [Model🤗](https://huggingface.co/Lightricks/LTX-2)
  - Open-source 14B+5B asymmetric dual-stream audiovisual diffusion model generating synchronized video and audio with bidirectional cross-attention. | Task: Audio-Visual Generation
* 25.11 [UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](https://arxiv.org/abs/2511.03334) | [Paper📑](https://arxiv.org/abs/2511.03334)
  - Unified audio-video generation using cross-modal interactions. | Task: Audio-Visual Generation
* 25.11 [Harmony: Harmonizing Audio and Video Generation through Cross-Task Synergy](https://arxiv.org/abs/2511.21579) | [Paper📑](https://arxiv.org/abs/2511.21579)
  - Harmonizes audio and video generation via cross-task synergy. | Task: Audio-Visual Generation
* 25.06 [ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](https://arxiv.org/abs/2506.21448)


<a name="reasoning-with-agent"></a>
### Reasoning with Agent/Tool
* 26.02 [Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents](https://arxiv.org/abs/2602.16855) | [Paper📑](https://arxiv.org/abs/2602.16855) [Code🖥️](https://github.com/X-PLUG/MobileAgent) [Model🤗](https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Think)
  - GUI-Owl-1.5 multi-platform GUI agent family achieving SOTA on GUI automation (56.5 OSWorld, 71.6 AndroidWorld) and grounding (80.3 ScreenSpotPro) via MRPO multi-platform RL. | Task: GUI Agent
* 26.02 [GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL](https://arxiv.org/abs/2602.22190) | [Paper📑](https://arxiv.org/abs/2602.22190) [Code🖥️](https://github.com/GUI-Libra/GUI-Libra) [Model🤗](https://huggingface.co/collections/Ray2333/gui-libra)
  - Trains open-source GUI agents using action-aware SFT (81K curated dataset) and conservative RL with KL regularization for web and mobile tasks. | Task: GUI Agent
* 26.02 [PyVision-RL: Forging Open Agentic Vision Models via RL](https://arxiv.org/abs/2602.20739) | [Paper📑](https://arxiv.org/abs/2602.20739) [Code🖥️](https://github.com/agents-x-project/PyVision-RL) [Model🤗](https://huggingface.co/Agents-X/PyVision-Image-7B-RL)
  - RL framework for open-weight multimodal agents using oversampling-filtering-ranking rollout; releases PyVision-Image-7B and PyVision-Video-7B for tool-augmented reasoning. | Task: Agent/Tool Use
* 26.02 [Computer-Using World Model](https://arxiv.org/abs/2602.17365) | [Paper📑](https://arxiv.org/abs/2602.17365)
  - World model for desktop software predicting UI state changes via two-stage factorization to help agents simulate candidate actions before execution. | Task: GUI Agent
* 26.02 [V-Retrver: Evidence-Driven Agentic Reasoning for Universal Multimodal Retrieval](https://arxiv.org/abs/2602.06034) | [Paper📑](https://arxiv.org/abs/2602.06034) [Code🖥️](https://github.com/chendy25/V-Retrver) [Model🤗](https://huggingface.co/V-Retrver/V-Retrver-SFT-7B)
  - Reformulates multimodal retrieval as an agentic reasoning process where an MLLM selectively acquires visual evidence via external tools, achieving 23% average improvement. | Task: Agent/Tool Use
* 26.02 [Reasoning-Augmented Representations for Multimodal Retrieval](https://arxiv.org/abs/2602.07125) | [Paper📑](https://arxiv.org/abs/2602.07125) [Code🖥️](https://github.com/AugmentedRetrieval/ReasoningAugmentedRetrieval)
  - Data-centric framework externalizing reasoning before retrieval by using VLMs to densely caption visual evidence and resolve ambiguous multimodal queries. | Task: Agent/Tool Use
* 26.02 [WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents](https://arxiv.org/abs/2601.21872) | [Paper📑](https://arxiv.org/abs/2601.21872) [Code🖥️](https://github.com/yaoz720/GroundedPRM)
  - Reasoning-first WebPRM formulating reward modeling as text generation to improve web navigation through structured justifications and preference verdicts (ICLR 2026). | Task: GUI Agent
* 26.02 [Sparse Video Generation Propels Real-World Beyond-the-View Vision-Language Navigation](https://arxiv.org/abs/2602.05827) | [Paper📑](https://arxiv.org/abs/2602.05827) [Code🖥️](https://github.com/opendrivelab/sparsevideonav)
  - SparseVideoNav uses video generation for sparse future planning in beyond-the-view VLN tasks, achieving 27× speed-up and 2.5× higher success rate over LLM baselines. | Task: Visual Reasoning Agent
* 26.01 [AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning](https://arxiv.org/abs/2601.18631) | [Paper📑](https://arxiv.org/abs/2601.18631) [Code🖥️](https://github.com/ssmisya/AdaReasoner) [Model🤗](https://huggingface.co/AdaReasoner/AdaReasoner-7B-Randomized)
  - Multimodal model family learning tool usage as a reasoning skill via Tool-GRPO, +24.9% improvement surpassing GPT-4 on visual reasoning benchmarks. | Task: Visual Reasoning with Tools
* 26.01 [SenseNova-MARS: Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning](https://arxiv.org/abs/2512.24330) | [Paper📑](https://arxiv.org/abs/2512.24330)
  - Multimodal agentic reasoning and search framework using RL to empower visual reasoning with agent capabilities. | Task: Multimodal Agentic Reasoning
* 26.01 [EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience](https://arxiv.org/abs/2601.15876) | [Paper📑](https://arxiv.org/abs/2601.15876) [Code🖥️](https://github.com/meituan/EvoCUA) [Model🤗](https://huggingface.co/meituan/EvoCUA-32B-20260105)
  - SOTA computer-use agent (56.7% OSWorld) using autonomous task generation and iterative evolving learning with self-correction. | Task: GUI Agent
* 26.01 [DocDancer: Towards Agentic Document-Grounded Information Seeking](https://arxiv.org/abs/2601.05163) | [Paper📑](https://arxiv.org/abs/2601.05163)
  - Agentic framework for document-grounded multimodal information seeking and reasoning. | Task: Document Reasoning Agent
* 26.01 [ShowUI-pi: Flow-based Generative Models as GUI Dexterous Hands](https://arxiv.org/abs/2512.24965) | [Paper📑](https://arxiv.org/abs/2512.24965)
  - Flow-based generative models applied as GUI interaction agents with visual reasoning capabilities. | Task: GUI Agent
* 26.01 [PersonalAlign: Hierarchical Implicit Intent Alignment for Personalized GUI Agent](https://arxiv.org/abs/2601.09636) | [Paper📑](https://arxiv.org/abs/2601.09636)
  - Personalized GUI agent aligning hierarchical implicit user intent with long-term user-centric records. | Task: GUI Agent
* 25.12 [Step-GUI Technical Report](https://arxiv.org/abs/2512.15431) | [Paper📑](https://arxiv.org/abs/2512.15431)
  - Step-by-step GUI agent with visual understanding. | Task: GUI Agent
* 25.12 [MAI-UI Technical Report: Real-World Centric Foundation GUI Agents](https://arxiv.org/abs/2512.22047) | [Paper📑](https://arxiv.org/abs/2512.22047)
  - Foundation model for real-world GUI agent interaction with visual grounding. | Task: GUI Agent
* 25.11 [Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything](https://arxiv.org/abs/2511.02834)
* 25.11 [DeepEyesV2: Toward Agentic Multimodal Model](https://arxiv.org/abs/2511.05271) | [Paper📑](https://arxiv.org/abs/2511.05271)
  - Agentic multimodal model with tool-use and reasoning capabilities. | Task: Multimodal Agent
* 25.11 [GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization](https://arxiv.org/abs/2511.15705) | [Paper📑](https://arxiv.org/abs/2511.15705)
  - Combines visual reasoning with web augmentation for agentic geolocalization. | Task: Visual Reasoning Agent
* 25.10 [AudioToolAgent: An Agentic Framework for Audio-Language Models](https://arxiv.org/abs/2510.02995v1) | [Paper📑](https://arxiv.org/abs/2510.02995v1)
* 25.10 [GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness](https://arxiv.org/abs/2510.00536) | [Paper📑](https://arxiv.org/abs/2510.00536)
  - Efficient GUI interaction agents using visual understanding with spatio-temporal KV cache. | Task: GUI Agent
* 25.09 [UItron: Foundational GUI Agent with Advanced Perception and Planning](https://arxiv.org/abs/2508.21767) | [Paper📑](https://arxiv.org/abs/2508.21767)
  - Multimodal agent for GUI understanding and interaction. | Task: GUI Agent
* 25.09 [BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](https://arxiv.org/abs/2509.15566) | [Paper📑](https://arxiv.org/abs/2509.15566)
  - Reasoning model for GUI agent visual understanding and interaction. | Task: GUI Agent
* 25.08 [Think Before You Segment: An Object-aware Reasoning Agent for Referring Audio-Visual Segmentation](https://arxiv.org/abs/2508.04418)
* 25.08 [OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](https://arxiv.org/abs/2508.04482) | [Paper📑](https://arxiv.org/abs/2508.04482)
  - Survey of MLLM-based agents that operate computing devices via visual understanding. | Task: GUI Agent
* 25.08 [InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization](https://arxiv.org/abs/2508.05731) | [Paper📑](https://arxiv.org/abs/2508.05731)
  - Multimodal agent for GUI understanding with visual grounding and adaptive exploration. | Task: GUI Agent
* 25.08 [CODA: Coordinating the Cerebrum and Cerebellum for a Dual-Brain Computer Use Agent](https://arxiv.org/abs/2508.20096) | [Paper📑](https://arxiv.org/abs/2508.20096)
  - Dual-brain architecture for multimodal computer-use agent with decoupled RL. | Task: GUI Agent
* 25.06 [Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning](https://arxiv.org/abs/2506.13654)|[Paper📑](https://arxiv.org/pdf/2506.13654) [Code🖥️](https://github.com/egolife-ai/Ego-R1) [Project🌐](https://egolife-ai.github.io/Ego-R1/)
* 25.05 [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470) | [Paper📑](https://arxiv.org/pdf/2503.19470) [Code🖥️](https://github.com/Agent-RL/ReCall) 
* 25.05 [Reinforcement Learning for Long-Horizon Interactive LLM Agents](https://arxiv.org/abs/2502.01600)|[Paper📑](https://arxiv.org/pdf/2502.01600) 
* 25.05 [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073) |[Paper📑](https://arxiv.org/pdf/2504.20073) [Code🖥️](https://github.com/RAGEN-AI/RAGEN) [Project🌐](https://ragen-ai.github.io/)
* 25.05 [Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models with Reinforced Reasoning](https://arxiv.org/abs/2505.00024) | [Paper📑](https://arxiv.org/pdf/2505.00024) [Code🖥️](https://github.com/NVlabs/Tool-N1) 
* 25.05 [Agent RL Scaling Law: Spontaneous Code Execution for Mathematical Problem Solving](https://arxiv.org/abs/2505.07773)| [Paper📑](https://arxiv.org/pdf/2505.07773) [Code🖥️](https://github.com/yyht/openrlhf_async_pipline) 
* 25.04 [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)|[Paper📑](https://arxiv.org/pdf/2504.13958) [Code🖥️](https://github.com/qiancheng0/ToolRL) 
* 25.04 [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516v4) | [Paper📑](https://arxiv.org/pdf/2503.09516v4) [Code🖥️](https://github.com/PeterGriffinJin/Search-R1) 
* 25.04 [Acting Less is Reasoning More! Teaching Model to Act Efficiently](https://arxiv.org/abs/2504.14870)
* 25.04 [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441) | [Paper📑](https://arxiv.org/abs/2505.01441) 
* 25.04 [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) |[Paper📑](https://arxiv.org/pdf/2504.03160) [Code🖥️](https://github.com/GAIR-NLP/DeepResearcher) 
* 25.03 [TORL: Scaling Tool-Integrated RL](https://arxiv.org/abs/2503.23383) | [Paper📑](https://arxiv.org/pdf/2503.23383) [Code🖥️](https://github.com/GAIR-NLP/ToRL) 
* 25.03 [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592) | [Paper📑](https://arxiv.org/pdf/2503.05592) 
* 25.02 (CVPR25)[Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](https://arxiv.org/abs/2412.01694) | [Paper📑](https://arxiv.org/pdf/2412.01694) 
* 24.12 (ECCV24) [VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](https://arxiv.org/abs/2403.11481) | [Paper📑](https://arxiv.org/abs/2403.11481) [Code🖥️](https://github.com/YueFan1014/VideoAgent) [Project🌐](https://videoagent.github.io/)
  - Explore how reconciling several foundation models with a novel unified memory mechanism could tackle the challenging video understanding problem  | Task: Video captioning & QA

<a name="medical-reasoning"></a>
### Medical Reasoning
#### Image MLLM
* 26.02 [MediX-R1: Open Ended Medical Reinforcement Learning](https://arxiv.org/abs/2602.23363) | [Paper📑](https://arxiv.org/abs/2602.23363) [Code🖥️](https://github.com/mbzuai-oryx/MediX-R1) [Model🤗](https://huggingface.co/MBZUAI/MediX-R1-8B) [Dataset🤗](https://huggingface.co/datasets/MBZUAI/medix-rl-data)
  - Open-ended RL framework for medical MLLMs enabling free-form clinical answers via Group-Based RL with composite rewards; 8B model outperforms 27B MedGemma with ~51K training samples. | Task: Medical Reasoning
* 26.02 [Baichuan-M3: Modeling Clinical Inquiry for Reliable Medical Decision-Making](https://arxiv.org/abs/2602.06570) | [Paper📑](https://arxiv.org/abs/2602.06570) [Code🖥️](https://github.com/baichuan-inc/Baichuan-M3-235B) [Model🤗](https://huggingface.co/baichuan-inc/Baichuan-M3-235B)
  - Medical LLM shifting from passive Q&A to active clinical-grade decision support via proactive information acquisition, long-horizon reasoning, and hallucination suppression, achieving SOTA on HealthBench. | Task: Medical Reasoning
* 26.02 [MedSAM-Agent: Empowering Interactive Medical Image Segmentation with Multi-turn Agentic Reinforcement Learning](https://arxiv.org/abs/2602.03320) | [Paper📑](https://arxiv.org/abs/2602.03320) [Code🖥️](https://github.com/CUHK-AIM-Group/MedSAM-Agent) [Model🤗](https://huggingface.co/Saint-lsy/MedSAM-Agent-Qwen3-VL-8B-MedSAM2)
  - Reformulates medical image segmentation as multi-step decision-making using hybrid prompting and two-stage training with process rewards for autonomous reasoning. | Task: Medical Reasoning
* 26.02 [Hepato-LLaVA: An Expert MLLM for Hepatocellular Pathology Analysis on Whole Slide Images](https://arxiv.org/abs/2602.19424) | [Paper📑](https://arxiv.org/abs/2602.19424) [Project🌐](https://pris-cv.github.io/Hepto-LLaVA/)
  - Specialized MLLM for hepatocellular carcinoma diagnosis with Sparse Topo-Pack Attention modeling tissue topology; includes HepatoPathoVQA (33K expert-validated Q&A pairs). | Task: Medical Reasoning
* 26.02 [MedCLIPSeg: Probabilistic Vision-Language Adaptation for Medical Image Segmentation](https://arxiv.org/abs/2602.20423) | [Paper📑](https://arxiv.org/abs/2602.20423) [Code🖥️](https://github.com/HealthX-Lab/MedCLIPSeg) [Model🤗](https://huggingface.co/TahaKoleilat/MedCLIPSeg)
  - Adapts CLIP for medical image segmentation via Probabilistic Vision-Language Adapter with uncertainty-aware attention, tested across 16 datasets spanning 5 modalities and 6 organs. | Task: Medical Reasoning
* 26.01 [UniX: Unifying Autoregression and Diffusion for Chest X-Ray Understanding and Generation](https://arxiv.org/abs/2601.11522) | [Paper📑](https://arxiv.org/abs/2601.11522) [Code🖥️](https://github.com/ZrH42/UniX) [Model🤗](https://huggingface.co/ZrH42/UniX)
  - Unified medical foundation model combining autoregressive understanding and diffusion generation for chest X-rays, +46.1% in understanding. | Task: Medical Image Understanding & Generation
* 25.12 [OralGPT-Omni: A Versatile Dental Multimodal Large Language Model](https://arxiv.org/abs/2511.22055) | [Paper📑](https://arxiv.org/abs/2511.22055)
  - Versatile dental MLLM for oral health diagnosis and reasoning across modalities. | Task: Medical Reasoning
* 25.12 [DentalGPT: Incentivizing Multimodal Complex Reasoning in Dentistry](https://arxiv.org/abs/2512.11558) | [Paper📑](https://arxiv.org/abs/2512.11558)
  - Incentivizes complex multimodal reasoning for dental diagnosis and treatment. | Task: Medical Reasoning
* 25.12 [Colon-X: Advancing Intelligent Colonoscopy from Multimodal Understanding to Clinical Reasoning](https://arxiv.org/abs/2512.03667) | [Paper📑](https://arxiv.org/abs/2512.03667)
  - Advances colonoscopy with multimodal understanding and clinical reasoning capabilities. | Task: Medical Reasoning
* 25.10 [M3Retrieve: Benchmarking Multimodal Retrieval for Medicine](https://arxiv.org/abs/2510.06888) | [Paper📑](https://arxiv.org/abs/2510.06888)
  - Multimodal retrieval benchmark for medical domain. | Task: Medical Reasoning
* 25.09 [MedVista3D: Vision-Language Modeling for Reducing Diagnostic Errors in 3D CT Disease Detection](https://arxiv.org/abs/2509.03800) | [Paper📑](https://arxiv.org/abs/2509.03800)
  - VLM for medical 3D CT analysis to reduce diagnostic errors. | Task: Medical Reasoning
* 25.08 [MedBLINK: Probing Basic Perception in Multimodal Language Models for Medicine](https://arxiv.org/abs/2508.02951) | [Paper📑](https://arxiv.org/abs/2508.02951)
  - Tests multimodal LLMs on basic medical visual perception tasks. | Task: Medical Reasoning
#### Audio MLLM
* 25.04 (ICASSP 2025) [AuscMLLM: Bridging Classification and Reasoning in Heart Sound Analysis with a Multimodal Large Language Model](https://ieeexplore.ieee.org/document/10889373) |
* 24.09 (JBHI 2024) [Multi-Task Learning for Audio-Based Infant Cry Detection and Reasoning](https://ieeexplore.ieee.org/document/10663705) | 
#### Omni MLLM
* 25.06 (ACL 2025) [MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration](https://aclanthology.org/2025.findings-acl.1298/) | [Paper📑](https://aclanthology.org/2025.findings-acl.1298/) [Code🖥️](https://github.com/yczhou001/MAM) 

<a name="embodied-reasoning"></a>
### Embodied Reasoning

* 26.02 [VLANeXt: Recipes for Building Strong VLA Models](https://arxiv.org/abs/2602.18532) | [Paper📑](https://arxiv.org/abs/2602.18532) [Code🖥️](https://github.com/DravenALG/VLANeXt) [Model🤗](https://huggingface.co/DravenALG/VLANeXt)
  - Systematically identifies 12 key design findings across foundational components for VLA models, yielding SOTA simulation and real-world benchmark performance (CVPR 2026). | Task: Robot Control

* 26.02 [SimVLA: A Simple VLA Baseline for Robotic Manipulation](https://arxiv.org/abs/2602.18224) | [Paper📑](https://arxiv.org/abs/2602.18224) [Code🖥️](https://github.com/LUOyk1999/SimVLA) [Model🤗](https://huggingface.co/YuankaiLuo/SimVLA-LIBERO)
  - Minimal VLA baseline strictly decoupling perception from control with standard VL backbone, achieving SOTA on simulation benchmarks with only 0.5B parameters. | Task: Robotic Manipulation

* 26.02 [GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning](https://arxiv.org/abs/2602.12099) | [Paper📑](https://arxiv.org/abs/2602.12099) [Code🖥️](https://github.com/open-gigaai/giga-brain-0) [Project🌐](https://gigabrain05m.github.io/)
  - VLA trained via world model-based RL (RAMP) on 10,000+ hours of robot data, achieving ~30% improvement on challenging tasks like laundry folding and espresso preparation. | Task: Robotic Manipulation

* 26.02 [Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning](https://arxiv.org/abs/2602.07845) | [Paper📑](https://arxiv.org/abs/2602.07845) [Code🖥️](https://github.com/rd-vla/rd-vla) [Project🌐](https://rd-vla.github.io/)
  - Recurrent VLA using latent iterative refinement instead of chain-of-thought tokens to adaptively scale compute at inference, achieving 0%→90%+ task success with 4 iterations. | Task: Robotic Manipulation

* 26.02 [VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model](https://arxiv.org/abs/2602.10098) | [Paper📑](https://arxiv.org/abs/2602.10098) [Code🖥️](https://github.com/ginwind/VLA-JEPA) [Model🤗](https://huggingface.co/ginwind/VLA-JEPA)
  - JEPA-style pretraining for VLA policies predicting future latent states from current observations, improving robustness to camera motion and irrelevant backgrounds. | Task: Robotic Manipulation

* 26.02 [DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos](https://arxiv.org/abs/2602.06949) | [Paper📑](https://arxiv.org/abs/2602.06949) [Model🤗](https://huggingface.co/nvidia/DreamDojo) [Project🌐](https://dreamdojo-world.github.io/)
  - Foundation world model trained on 44k hours of egocentric human video enabling teleoperation, policy evaluation, and model-based planning for dexterous robotics at 10.81 FPS. | Task: Robotic Manipulation

* 26.02 [ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation](https://arxiv.org/abs/2602.11598) | [Paper📑](https://arxiv.org/abs/2602.11598) [Code🖥️](https://github.com/amap-cvlab/ABot-Navigation) [Project🌐](https://amap-cvlab.github.io/ABot-Navigation/ABot-N0/)
  - Unified VLA navigation model with hierarchical Brain-Action architecture achieving SOTA on 7 benchmarks across 5 navigation task types, trained on 16.9M expert trajectories. | Task: Embodied Navigation

* 26.02 [TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments](https://arxiv.org/abs/2602.02459) | [Paper📑](https://arxiv.org/abs/2602.02459) [Code🖥️](https://github.com/ucla-mobility/TIC-VLA) [Project🌐](https://ucla-mobility.github.io/TIC-VLA/)
  - Latency-aware VLA framework modeling delayed semantic reasoning during action generation via delayed semantic-control interface for real-time navigation. | Task: Embodied Navigation

* 26.02 [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2602.20309) | [Paper📑](https://arxiv.org/abs/2602.20309) [Code🖥️](https://github.com/AIoT-MLSys-Lab/QuantVLA)
  - Training-free PTQ framework for VLA models combining selective quantization, attention temperature matching, and output head balancing, achieving ~70% memory savings (CVPR 2026). | Task: Robot Control

* 26.02 [FRAPPE: Infusing World Modeling into Generalist Policies via Multiple Future Representation Alignment](https://arxiv.org/abs/2602.17259) | [Paper📑](https://arxiv.org/abs/2602.17259) [Code🖥️](https://github.com/OpenHelix-Team/frappe) [Model🤗](https://huggingface.co/collections/hhhJB/frappe)
  - Improves world-awareness in robotic policies via parallel progressive latent alignment with visual foundation models, reducing error accumulation in multi-step prediction. | Task: Robotic Manipulation

* 26.02 [TactAlign: Human-to-Robot Policy Transfer via Tactile Alignment](https://arxiv.org/abs/2602.13579) | [Paper📑](https://arxiv.org/abs/2602.13579) [Project🌐](https://yswi.github.io/tactalign/)
  - Cross-embodiment tactile alignment using rectified flow for zero-shot transfer on contact-rich manipulation tasks including pivoting, insertion, and lid closing. | Task: Robotic Manipulation

* 26.02 [World Guidance: World Modeling in Condition Space for Action Generation](https://arxiv.org/abs/2602.22010) | [Paper📑](https://arxiv.org/abs/2602.22010) [Project🌐](https://selen-suyue.github.io/WoGNet/)
  - WoG maps predicted future observations into compact condition representations for fine-grained action generation, validated across simulation and real-world robot environments. | Task: Robot Control

* 26.02 [Green-VLA: Staged Vision-Language-Action Model for Generalist Robots](https://arxiv.org/abs/2602.00919) | [Paper📑](https://arxiv.org/abs/2602.00919) [Code🖥️](https://github.com/greenvla/GreenVLA)
  - Five-stage VLA framework for real-world robot deployment achieving generalization across embodiments via multimodal training and RL, reaching 69.5% success on ALOHA Table-Cleaning. | Task: Robotic Manipulation

* 26.02 [Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs](https://arxiv.org/abs/2602.21198) | [Paper📑](https://arxiv.org/abs/2602.21198) [Code🖥️](https://github.com/Reflective-Test-Time-Planning/Reflective-Test-Time-Planning)
  - Reflective Test-Time Planning with reflection-in-action and reflection-on-action enabling long-horizon credit assignment in robot decision-making. | Task: Embodied Reasoning

* 26.01 [ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models](https://arxiv.org/abs/2601.11404) | [Paper📑](https://arxiv.org/abs/2601.11404)
  - Action Chain-of-Thought paradigm for VLA models with Explicit and Implicit Action Reasoner components, achieving 98.5% on LIBERO. | Task: Robotic Manipulation

* 26.01 [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163) | [Paper📑](https://arxiv.org/abs/2601.16163) [Model🤗](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)
  - Adapts pretrained video models into robot policies through single-stage post-training, achieving 98.5% on LIBERO and SOTA on real-world bimanual manipulation. | Task: Robot Control

* 26.01 [DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation](https://arxiv.org/abs/2601.22153) | [Paper📑](https://arxiv.org/abs/2601.22153) [Code🖥️](https://github.com/hzxie/DynamicVLA) [Dataset🤗](https://huggingface.co/datasets/hzxie/DOM)
  - Compact 0.4B VLA model for dynamic object manipulation with continuous inference and latent-aware action streaming. | Task: Robotic Manipulation

* 26.01 [SOP: A Scalable Online Post-Training System for Vision-Language-Action Models](https://arxiv.org/abs/2601.03044) | [Paper📑](https://arxiv.org/abs/2601.03044) [Project🌐](https://agibot.com/research/sop_en)
  - Scalable online distributed post-training system for VLA models enabling real-world robot policy adaptation through fleet learning. | Task: Robot Control

* 26.01 [FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation](https://arxiv.org/abs/2601.13976) | [Paper📑](https://arxiv.org/abs/2601.13976) [Code🖥️](https://github.com/Fantasy-AMAP/fantasy-vln) [Model🤗](https://huggingface.co/acvlab/FantasyVLN)
  - Implicit reasoning framework for vision-language navigation encoding imagined visual tokens in latent space, reducing inference latency by an order of magnitude. | Task: Vision-Language Navigation

* 26.01 [RoboVIP: Multi-View Video Generation with Visual Identity Prompting Augments Robot Manipulation](https://arxiv.org/abs/2601.05241) | [Paper📑](https://arxiv.org/abs/2601.05241) [Code🖥️](https://github.com/RoboVIP/RoboVIP_VDM)
  - Visual identity prompting for multi-view video generation to augment robot manipulation data. | Task: Robotic Manipulation

* 26.01 [VLingNav: Embodied Navigation with Adaptive Reasoning and Visual-Assisted Linguistic Memory](https://arxiv.org/abs/2601.08665) | [Paper📑](https://arxiv.org/abs/2601.08665)
  - Embodied navigation agent with adaptive reasoning combining visual perception and linguistic memory. | Task: Embodied Navigation

* 25.12 [DualVLA: Building a Generalizable Embodied Agent via Partial Decoupling of Reasoning and Action](https://arxiv.org/abs/2511.22134) | [Paper📑](https://arxiv.org/abs/2511.22134)
  - Decouples reasoning and action for more generalizable embodied agents. | Task: Robotic Manipulation
* 25.12 [HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for VLA Models](https://arxiv.org/abs/2512.09928) | [Paper📑](https://arxiv.org/abs/2512.09928)
  - Enriches VLA models with hindsight, insight, and foresight via motion representations. | Task: Robotic Manipulation
* 25.12 [LEO-RobotAgent: A General-purpose Robotic Agent for Language-driven Embodied Operator](https://arxiv.org/abs/2512.10605) | [Paper📑](https://arxiv.org/abs/2512.10605)
  - General-purpose language-driven robotic agent for embodied task execution. | Task: Robotic Manipulation
* 25.12 [Steering VLA Models as Anti-Exploration: A Test-Time Scaling Approach](https://arxiv.org/abs/2512.02834) | [Paper📑](https://arxiv.org/abs/2512.02834)
  - Test-time scaling approach for steering VLA models for safe embodied behavior. | Task: Robot Control
* 25.11 [WMPO: World Model-based Policy Optimization for Vision-Language-Action Models](https://arxiv.org/abs/2511.09515) | [Paper📑](https://arxiv.org/abs/2511.09515)
  - World model-based policy optimization for VLA models in robotics. | Task: Robot Control
* 25.11 [RynnVLA-002: A Unified Vision-Language-Action and World Model](https://arxiv.org/abs/2511.17502) | [Paper📑](https://arxiv.org/abs/2511.17502)
  - Unified VLA and world model for robotic manipulation. | Task: Robot Control
* 25.11 [Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight](https://arxiv.org/abs/2511.16175) | [Paper📑](https://arxiv.org/abs/2511.16175)
  - VLA model with disentangled visual foresight for robotic control. | Task: Robot Control
* 25.11 [MobileVLA-R1: Reinforcing Vision-Language-Action for Mobile Robots](https://arxiv.org/abs/2511.17889) | [Paper📑](https://arxiv.org/abs/2511.17889)
  - Reinforcement-based VLA model for mobile robot tasks. | Task: Robot Control
* 25.10 [VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards](https://arxiv.org/abs/2510.00406) | [Paper📑](https://arxiv.org/abs/2510.00406)
  - Fine-tuning VLA models using RL with verified rewards in world simulators. | Task: Robot Control
* 25.10 [InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy](https://arxiv.org/abs/2510.13778) | [Paper📑](https://arxiv.org/abs/2510.13778)
  - VLA framework for robotic control with spatial grounding. | Task: Robot Control
* 25.10 [X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model](https://arxiv.org/abs/2510.10274) | [Paper📑](https://arxiv.org/abs/2510.10274)
  - Cross-embodiment VLA model for scalable robot learning. | Task: Robot Control
* 25.10 [GigaBrain-0: A World Model-Powered Vision-Language-Action Model](https://arxiv.org/abs/2510.19430) | [Paper📑](https://arxiv.org/abs/2510.19430)
  - VLA model integrating world models for robot reasoning. | Task: Robot Control
* 25.09 [Robix: A Unified Model for Robot Interaction, Reasoning and Planning](https://arxiv.org/abs/2509.01106) | [Paper📑](https://arxiv.org/abs/2509.01106)
  - Unified robotics model combining visual reasoning with interaction and planning. | Task: Robot Control
* 25.09 [FLOWER: Democratizing Generalist Robot Policies with Efficient VLA Flow Policies](https://arxiv.org/abs/2509.04996) | [Paper📑](https://arxiv.org/abs/2509.04996)
  - Vision-language-action model for generalist robot policies. | Task: Robot Control
* 25.08 [RynnEC: Bringing MLLMs into Embodied World](https://arxiv.org/abs/2508.14160) | [Paper📑](https://arxiv.org/abs/2508.14160)
  - Integrates multimodal LLMs into embodied AI settings for physical-world reasoning. | Task: Embodied Reasoning
* 25.08 [Do What? Teaching Vision-Language-Action Models to Reject the Impossible](https://arxiv.org/abs/2508.16292) | [Paper📑](https://arxiv.org/abs/2508.16292)
  - Trains VLA models to reason about task feasibility and reject impossible instructions. | Task: Robot Control
* 25.08 [Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in VLA Policies](https://arxiv.org/abs/2508.20072) | [Paper📑](https://arxiv.org/abs/2508.20072)
  - Uses discrete diffusion for action decoding in vision-language-action robotic policies. | Task: Robot Control

* 23.07 [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/) | [Paper📑](https://arxiv.org/pdf/2307.15818) [Project🌐](https://robotics-transformer2.github.io)
  - Co-finetunes a VLM on web and robot data, establishing the VLA paradigm by transferring internet-scale knowledge to robot control. | Task: General Robotic Manipulation

* 24.05 [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213) | [Paper📑](https://arxiv.org/pdf/2405.12213) [Code🖥️](https://github.com/octo-models/octo) [Project🌐](https://octo-models.github.io/) [Model🤗](https://huggingface.co/rail-berkeley/octo-base-1.5)
  - An open-source, generalist transformer policy pretrained on the large-scale Open X-Embodiment dataset, designed for efficient fine-tuning to new robots and tasks. | Task: Robotics

* 24.06 [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) | [Paper📑](https://arxiv.org/pdf/2406.09246) [Code🖥️](https://github.com/openvla/openvla) [Project🌐](https://openvla.github.io/) [Model🤗](https://huggingface.co/openvla)
    - A 7B-parameter open-source VLA model trained on the Open X-Embodiment dataset, achieving state-of-the-art performance for generalist manipulation. | Task: VLA

* 24.10 [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/blog/pi0) | [Paper📑](https://arxiv.org/abs/2410.24164) [Code🖥️](https://github.com/Physical-Intelligence/openpi)
  - A generalist policy using a novel flow matching architecture atop a pretrained VLM, enabling zero-shot generalization for dexterous manipulation. | Task: Robot Control

* 25.01 [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://www.physicalintelligence.company/research/fast) | [Paper📑](https://arxiv.org/pdf/2501.09747) [Code🖥️](https://github.com/Physical-Intelligence/openpi)
  - A compression-based action tokenization scheme that accelerates autoregressive VLA training by 5x with performance comparable to diffusion models. | Task: Robot Control

* 25.02 [Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](https://www.pi.website/research/hirobot) | [Paper📑](https://arxiv.org/pdf/2502.19417)
  - A hierarchical VLA model with a high-level VLM for reasoning and a low-level VLA for execution, enabling complex, open-ended instruction following. | Task: Robot Control

* 25.03 [Gemini Robotics: Bringing AI into the Physical World](https://arxiv.org/abs/2503.20020) | [Paper📑](https://arxiv.org/pdf/2503.20020) [Code🖥️](https://github.com/embodiedreasoning/ERQA) [Project🌐](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/) [Dataset🤗](https://github.com/embodiedreasoning/ERQA)
  - A VLA model built on the Gemini foundation model, demonstrating significant improvements in generality, interactivity, and dexterity for complex tasks. | Task: Advanced & Dexterous Manipulation

* 25.03 [COT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](https://arxiv.org/abs/2503.22020) | [Paper📑](https://arxiv.org/pdf/2503.22020) [Project🌐](https://cot-vla.github.io/)
  - A method that incorporates explicit visual CoT reasoning into VLAs by predicting future image frames autoregressively as visual goals before generating a short action sequence to achieve these goals. | Task: Robotics

* 25.03 [GR00T: A Foundation Model for General-Purpose Robotics](https://arxiv.org/abs/2503.14734) | [Paper📑](https://arxiv.org/pdf/2503.14734) [Code🖥️](https://github.com/NVIDIA/Isaac-GR00T) [Model🤗](https://huggingface.co/nvidia/GR00T-N1.5-3B) [Dataset🤗](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)
  - A general-purpose foundation model for robot learning that takes multimodal instructions and past observations to generate actions for the robot to execute. | Task: Robotics

* 25.04 [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://www.pi.website/blog/pi05) | [Paper📑](https://www.physicalintelligence.company/download/pi05.pdf)
  - An evolution of π₀ that uses co-training on diverse tasks to achieve long-horizon, dexterous manipulation in novel, unseen environments. | Task: Robot Control


* 25.06 [Chain-of-Action: Faithful and Deterministic Robot Policy via Language-guided State-Action Augmentation](https://chain-of-action.github.io/) | [Paper📑](https://arxiv.org/pdf/2506.09990) [Code🖥️](https://github.com/ByteDance-Seed/Chain-of-Action) [Project🌐](https://chain-of-action.github.io/) [Model🤗](https://huggingface.co/Solomonz/Chain-of-Action)
  - A novel robot policy, Chain-of-Action (CoA), that uses language as an intermediate representation to explicitly reason about the chain of actions for a given task, while being fully deterministic during inference. | Task: Robotics

* 25.07 [Vision-Language-Action Instruction Tuning: From Understanding to Manipulation](https://yangs03.github.io/InstructVLA_Home/) | [Paper📑](https://arxiv.org/pdf/2507.17520) [Code🖥️](https://github.com/InternRobotics/InstructVLA) [Project🌐](https://yangs03.github.io/InstructVLA_Home/) [Model🤗](https://huggingface.co/datasets/ShuaiYang03/VLA_Instruction_Tuning)
  - An end-to-end VLA model, InstructVLA, that introduces a novel training paradigm called Vision-Language-Action Instruction Tuning (VLA-IT) to preserve the flexible reasoning of VLMs while delivering high-performance robotic manipulation. | Task: Robotic

* 25.07 [MinD: Learning A Dual-System World Model for Real-Time Planning and Implicit Risk Analysis](https://manipulate-in-dream.github.io/) | [Paper📑](https://www.arxiv.org/pdf/2506.18897) [Code🖥️](https://github.com/manipulate-in-dream/MinD) [Project🌐](https://manipulate-in-dream.github.io/)
  - A dual-system world model, MinD, that enables real-time, risk-aware planning by conditioning a high-frequency action policy on single-step latent predictions from a low-frequency video generation model. | Task: Robotic


### Others

#### Image MLLM
* 26.02 [VISTA-Bench: Do Vision-Language Models Really Understand Visualized Text as Well as Pure Text?](https://arxiv.org/abs/2602.04802) | [Paper📑](https://arxiv.org/abs/2602.04802)
  - Benchmark testing whether VLMs truly understand text rendered visually in images as well as plain text, revealing a significant comprehension gap. | Task: Reasoning
* 26.02 [From Perception to Action: An Interactive Benchmark for Vision Reasoning](https://arxiv.org/abs/2602.21015) | [Paper📑](https://arxiv.org/abs/2602.21015) [Code🖥️](https://github.com/Social-AI-Studio/CHAIN)
  - CHAIN 3D physics-driven interactive benchmark evaluating whether VLMs understand causal constraints and execute structured action sequences in mechanical puzzles. | Task: Reasoning
* 26.01 [CoF-T2I: Video Models as Pure Visual Reasoners for Text-to-Image Generation](https://arxiv.org/abs/2601.10061) | [Paper📑](https://arxiv.org/abs/2601.10061)
  - Uses video generation models as visual reasoners for text-to-image generation, showing temporal modeling transfers to improved spatial reasoning. | Task: Image Generation
* 26.01 [OCRVerse: Towards Holistic OCR in End-to-End Vision-Language Models](https://arxiv.org/abs/2601.21639) | [Paper📑](https://arxiv.org/abs/2601.21639)
  - Holistic OCR framework within end-to-end vision-language models for comprehensive text understanding in images. | Task: OCR & Document Understanding
* 25.12 [GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation](https://arxiv.org/abs/2512.17495) | [Paper📑](https://arxiv.org/abs/2512.17495)
  - Exposes and evaluates visual grounding gaps in MLLMs across multiple dimensions. | Task: Visual Grounding
* 25.11 [Monet: Reasoning in Latent Visual Space Beyond Images and Language](https://arxiv.org/abs/2511.21395) | [Paper📑](https://arxiv.org/abs/2511.21395)
  - Enables vision-language reasoning in latent visual space, going beyond standard image-text paradigms. | Task: Reasoning
* 25.10 [SeeingEye: Agentic Information Flow Unlocks Multimodal Reasoning In Text-only LLMs](https://arxiv.org/abs/2510.25092) | [Paper📑](https://arxiv.org/abs/2510.25092)
  - Enables multimodal reasoning in text-only LLMs through agentic information flow. | Task: Reasoning
* 25.04 [InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners](https://arxiv.org/pdf/2504.14239) | [Paper📑](https://arxiv.org/pdf/2504.14239) [Code🖥️](https://github.com/Reallm-Labs/InfiGUI-R1)
  - an MLLM-based GUI agent designed to progressively evolve agents from Reactive Actors to Deliberative Reasoners. | task: UI
* 25.04 [GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents](https://arxiv.org/pdf/2504.10458) | [Paper📑](https://arxiv.org/pdf/2504.10458)
  - Enhances GUI agent through RL with unified action space modeling, achieving superior cross-platform performance using only 0.02% of the data required by previous methods. | Task: UI 
* 25.03 [UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning](https://arxiv.org/pdf/2503.21620) | [Paper📑](https://arxiv.org/pdf/2503.21620)
  - Introduce a unified rule-based action reward, enabling model optimization via policy-based algorithms like GRPO. | Task: UI 
* 25.03   [VLM-R1: A stable and generalizable R1-style Large Vision-Language Model](https://github.com/om-ai-lab/VLM-R1/tree/main?tab=readme-ov-file) [Code🖥️](https://github.com/om-ai-lab/VLM-R1/tree/main?tab=readme-ov-file) [Dataset🤗](https://huggingface.co/datasets/omlab/VLM-R1)  [Model🤗](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps/tree/main)
  - A reproduced R1-style VLM | Task: Referring Expression Comprehension
* 25.02 [MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning](https://arxiv.org/pdf/2502.19634)| [Paper📑](https://arxiv.org/pdf/2502.19634)
  - A MLLM trained with GRPO for medical image VQA.| Task: Medical Image VQA
#### Video MLLM
* 25.03 [R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning](https://arxiv.org/abs/2503.05379) | [Paper📑](https://arxiv.org/abs/2503.05379) [Code🖥️](https://github.com/HumanMLLM/R1-Omni) [Model🤗](https://huggingface.co/StarJiaxing/R1-Omni-0.5B/tree/main)
  - Impove reasoning capability, emotion recognition accuracy, and generalization ability with RLVR.  | Task: Emotion recognition

#### Audio MLLM
* 26.01 [The Sonar Moment: Benchmarking Audio-Language Models in Audio Geo-Localization](https://arxiv.org/abs/2601.03227) | [Paper📑](https://arxiv.org/abs/2601.03227)
  - Benchmark for audio-language models on spatial audio geo-localization reasoning tasks. | Task: Audio Reasoning
* 25.02 [ADIFF: Explaining audio difference using natural language](https://arxiv.org/abs/2502.04476)  [Code🖥️](https://github.com/soham97/ADIFF)  [Model](https://zenodo.org/records/14706090)
* 24.09 [What Are They Doing? Joint Audio-Speech Co-Reasoning](https://arxiv.org/abs/2409.14526)
* 24.09 [Chain-of-Thought Prompting for Speech Translation](https://arxiv.org/abs/2409.11538)

#### Omni LLM
* 26.01 [FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs](https://arxiv.org/abs/2601.13836) | [Paper📑](https://arxiv.org/abs/2601.13836)
  - Benchmark evaluating multimodal LLMs' ability to forecast future events from omni-modal context including temporal reasoning. | Task: Omni Reasoning
* 25.05 [AVCD: Mitigating Hallucinations in Audio-Visual Large Language Models through Contrastive Decoding](https://arxiv.org/abs/2505.20862)
* 25.03 [R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning](https://arxiv.org/abs/2503.05379)
* 23.11 [X-InstructBLIP: A Framework for Aligning Image, 3D, Audio, Video to LLMs and its Emergent Cross-Modal Reasoning](https://arxiv.org/abs/2311.18799)


<a name="benchmarks"></a>
## Benchmarks 📊

| Date  | Project                                                      | Task                                          | Links                                                        |
| ----- | ------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------ |
| 26.02 | A Very Big Video Reasoning Suite (VBVR): 1M+ video clips across 200 reasoning tasks | Video Reasoning | [[📑 Paper]](https://arxiv.org/abs/2602.20159) [[🤗 Model]](https://huggingface.co/Video-Reason/VBVR-Wan2.2) [[🤗 Data]](https://huggingface.co/datasets/Video-Reason/VBVR-Dataset) |
| 26.02 | OmniGAIA: Omni-Modal AI Agent Benchmark with hindsight-guided exploration | Omni-Modal Agent Reasoning | [[📑 Paper]](https://arxiv.org/abs/2602.22897) [[💻 Code]](https://github.com/RUC-NLPIR/OmniGAIA) [[🤗 Data]](https://huggingface.co/datasets/RUC-NLPIR/OmniGAIA) |
| 26.02 | SpatiaLab: Wild Spatial Reasoning benchmark across 6 VQA categories | Spatial Reasoning | [[📑 Paper]](https://arxiv.org/abs/2602.03916) [[💻 Code]](https://github.com/SpatiaLab-Reasoning/SpatiaLab) [[🤗 Data]](https://huggingface.co/datasets/ciol-research/SpatiaLab) |
| 26.02 | MuRGAt: Multimodal Fact-Level Attribution benchmark for verifiable reasoning | Multimodal Attribution | [[📑 Paper]](https://arxiv.org/abs/2602.11509) [[💻 Code]](https://github.com/meetdavidwan/murgat) |
| 26.02 | DeepVision-103K: Verifiable multimodal math dataset for RLVR training | Math Reasoning | [[📑 Paper]](https://arxiv.org/abs/2602.16742) [[💻 Code]](https://github.com/SKYLENAGE-AI/DeepVision-103K) [[🤗 Data]](https://huggingface.co/datasets/skylenage/DeepVision-103K) |
| 26.02 | UniVBench: Unified evaluation for video foundation models across understanding, generation, editing | Video Foundation Model Evaluation | [[📑 Paper]](https://arxiv.org/abs/2602.21835) [[💻 Code]](https://github.com/JianhuiWei7/UniVBench) |
| 26.02 | RISE-Video: Benchmark for video generators decoding implicit world rules | Video Generation Reasoning | [[📑 Paper]](https://arxiv.org/abs/2602.05986) [[💻 Code]](https://github.com/VisionXLab/Rise-Video) [[🤗 Data]](https://huggingface.co/datasets/VisionXLab/RISE-Video) |
| 26.01 | MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Methods | Multimodal Reasoning | [[📑 Paper]](https://arxiv.org/abs/2601.21821) [[🤗 Model]](https://huggingface.co/OpenDataArena/MMFineReason-8B) [[🤗 Data]](https://huggingface.co/datasets/OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking) |
| 26.01 | ChartVerse: Scaling Chart Reasoning via Reliable Programmatic Synthesis | Chart Reasoning | [[📑 Paper]](https://arxiv.org/abs/2601.13606) [[💻 Code]](https://github.com/starriver030515/ChartVerse) [[🤗 Model]](https://huggingface.co/opendatalab/ChartVerse-8B) [[🤗 Data]](https://huggingface.co/datasets/opendatalab/ChartVerse-SFT-1.8M) |
| 26.01 | VideoLoom: Joint Spatial-Temporal Understanding with LoomBench | Spatial-Temporal Reasoning | [[📑 Paper]](https://arxiv.org/abs/2601.07290) [[💻 Code]](https://github.com/JPShi/VideoLoom) [[🤗 Model]](https://huggingface.co/JPShi/VideoLoom-8B) |
| 26.01 | PROGRESSLM: Towards Progress Reasoning in Vision-Language Models | Task Progress Reasoning | [[📑 Paper]](https://arxiv.org/abs/2601.15224) [[💻 Code]](https://github.com/ProgressLM/ProgressLM) [[🤗 Data]](https://huggingface.co/datasets/Raymond-Qiancx/ProgressLM-Dataset) |
| 26.01 | FutureOmni: Evaluating Future Forecasting from Omni-Modal Context | Omni-Modal Temporal Reasoning | [[📑 Paper]](https://arxiv.org/abs/2601.13836) |
| 26.01 | Afri-MCQA: Multimodal Cultural Question Answering for African Languages | Multilingual Multimodal Reasoning | [[📑 Paper]](https://arxiv.org/abs/2601.05699) |
| 26.01 | AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark | Cultural Multimodal Reasoning | [[📑 Paper]](https://arxiv.org/abs/2601.17645) |
| 25.12 | HERBench: Multi-Evidence Integration in Video Question Answering | Video Reasoning | [[📑 Paper]](https://arxiv.org/abs/2512.14870) |
| 25.12 | SVBench: Evaluation of Video Generation Models on Social Reasoning | Video Social Reasoning | [[📑 Paper]](https://arxiv.org/abs/2512.21507) |
| 25.12 | IF-Bench: Benchmarking MLLMs for Infrared Images | Infrared Image Understanding | [[📑 Paper]](https://arxiv.org/abs/2512.09663) |
| 25.12 | VABench: Comprehensive Benchmark for Audio-Video Generation | Audio-Video Generation | [[📑 Paper]](https://arxiv.org/abs/2512.09299) |
| 25.11 | MME-CC: Challenging Multi-Modal Evaluation Benchmark of Cognitive Capacity | Cognitive Capacity | [[📑 Paper]](https://arxiv.org/abs/2511.03146) |
| 25.11 | GGBench: Geometric Generative Reasoning Benchmark for Unified Multimodal Models | Geometric Reasoning | [[📑 Paper]](https://arxiv.org/abs/2511.11134) |
| 25.11 | WEAVE: Benchmarking In-context Interleaved Comprehension and Generation | Multimodal Comprehension & Generation | [[📑 Paper]](https://arxiv.org/abs/2511.11434) |
| 25.10 | Uni-MMMU: Massive Multi-discipline Multimodal Unified Benchmark | Multimodal Multi-discipline Reasoning | [[📑 Paper]](https://arxiv.org/abs/2510.13759) |
| 25.10 | PhysToolBench: Benchmarking Physical Tool Understanding for MLLMs | Physical Tool Understanding | [[📑 Paper]](https://arxiv.org/abs/2510.09507) |
| 25.10 | BEAR: Benchmarking Multimodal Language Models for Atomic Embodied Capabilities | Embodied AI Capabilities | [[📑 Paper]](https://arxiv.org/abs/2510.08759) |
| 25.10 | OmniVideoBench: Towards Audio-Visual Understanding Evaluation for Omni MLLMs | Long-context, Video-Audio Unerstanding & Reasonin | [[📑 Paper]](https://arxiv.org/abs/2510.10689v1)  [[💻 Code]](https://github.com/NJU-LINK/OmniVideoBench) [[🌐 Project]](https://omnivideobench.github.io/omnivideobench_home/) [[🤗 Data]](https://huggingface.co/datasets/NJU-LINK/OmniVideoBench)|
| 25.10 | XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models | Capability Balancing among Different Modalities | [[📑 Paper]](https://arxiv.org/abs/2510.15148)  [[💻 Code]](https://github.com/XingruiWang/XModBench) [[🌐 Project]](https://xingruiwang.github.io/projects/XModBench/) |
| 25.10 | StreamingCoT: A Dataset for Temporal Dynamics and Multimodal Chain-of-Thought Reasoning in Streaming VideoQA | Termporal Reasoning | [[📑 Paper]](https://arxiv.org/abs/2510.25332) |
| 25.10 | Valor32k-AVQA v2.0: Open-Ended Audio-Visual Question Answering Dataset and Benchmark | Common Sense Omni Reasoning | [[📑 Paper]](https://dl.acm.org/doi/10.1145/3746027.3758261) |
| 25.09 | MARS2 2025 Challenge on Multimodal Reasoning | Multimodal Reasoning Challenge | [[📑 Paper]](https://arxiv.org/abs/2509.14142) |
| 25.09 | Visual-TableQA: Open-Domain Benchmark for Reasoning over Table Images | Table Reasoning | [[📑 Paper]](https://arxiv.org/abs/2509.07966) |
| 25.09 | AHELM: A Holistic Evaluation of Audio-Language Models | Audio-Language Understanding | [[📑 Paper]](https://arxiv.org/abs/2508.21376) |
| 25.09 | MDAR: A Multi-scene Dynamic Audio Reasoning Benchmark | Complex, Multi-scene, & Dynamically Evolving Speech & Audio Reasonin | [[📑 Paper]](https://arxiv.org/abs/2509.22461)  [[💻 Code]](https://github.com/luckyerr/MDAR) |
| 25.09 | MiMo-Audio-Eval Toolkit | Speech/Sound/Music Reasoning | [[💻 Code]](https://github.com/XiaomiMiMo/MiMo-Audio-Eval?tab=readme-ov-file)
| 25.08 | SpeechR: A Benchmark for Speech Reasoning in Large Audio-Language Models | Speech  Reasoning | [[📑 Paper]](https://www.arxiv.org/abs/2508.02018)  [[💻 Code]](https://github.com/Yanda95/SpeechR) [[Data]](https://drive.google.com/file/d/1BH2r2idILwUHX0NKsXz6GsSXdO0qWly8/view) |
| 25.08 | MMAU-Pro: A Challenging and Comprehensive Benchmark for Holistic Evaluation of Audio General Intelligence | Long-form, Spatial, and Multi-audio Reasoning on Speech/Music/Sound | [[📑 Paper]](https://arxiv.org/abs/2508.13992)  [[🤗 Data]](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro)  |
| 25.08 | R²-AVSBench: Think Before You Segment: An Object-aware Reasoning Agent for Referring Audio-Visual Segmentation | Segmentation Reasoning | [[📑 Paper]](https://arxiv.org/abs/2508.04418) [[🤗 Data]](https://drive.google.com/drive/folders/1Qz7MxBs7IpxgcTH8CaUsU3i9d366gRhM)|
| 25.07 | Towards Video Thinking Test: A Holistic Benchmark for Advanced Video  Reasoning and Understanding | Video Reasoning and Understanding             | [[📑 Paper]](https://arxiv.org/abs/2507.15028).  [[🌐 Project]](https://zhangyuanhan-ai.github.io/video-tt/) [[🤗 Data]](https://huggingface.co/datasets/lmms-lab/video-tt) |
| 25.06 | FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation | Financial Multi-Modal Reasoning Reasoning     | [[📑 Paper]](https://github.com/luo-junyu/FinMME). [[💻 Code]](https://github.com/luo-junyu/FinMME). [[🤗 Data]](https://huggingface.co/datasets/luojunyu/FinMME) |
| 25.06 | MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos | Video Reasoning                               | [[📑 Paper]](https://arxiv.org/abs/2506.04141). [[💻 Code]](https://github.com/GaryStack/MMR-V). [[🌐 Project]](https://mmr-v.github.io/home_page.html) [[🤗 Data]](https://huggingface.co/datasets/JokerJan/MMR-VBench) |
| 25.06 | OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models | Spatial Reasoning                             | [[📑 Paper]](https://arxiv.org/abs/2506.03135). [[💻 Code]](https://github.com/qizekun/OmniSpatial). [[🌐 Project]](https://qizekun.github.io/omnispatial/) [[🤗 Data]](https://huggingface.co/qizekun/datasets/OmniSpatial) |
| 25.06 | MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark | Phonatics, Prosody, Rhetoric, Syntactics, Semantics, and Paralinguistics in Speech Understanding & Reasoning | [[📑 Paper]](https://arxiv.org/abs/2506.04779)  [[💻 Code]](https://github.com/dingdongwang/mmsu_bench)  [[🤗 Data]](https://huggingface.co/datasets/ddwang2000/MMSU)
| 25.05 | Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities | Video&Audio Reasoning | [[📑 Paper]](https://arxiv.org/abs/2505.17862) [[💻 Code]](https://github.com/Lliar-liar/Daily-Omni)  [[🌐 Project]](https://lliar-liar.github.io/Daily-Omni/) [[🤗 Data]](https://huggingface.co/datasets/liarliar/Daily-Omni) |
| 25.05 | MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix | Multi-step Audio Reasoning                    | [[📑 Paper]](https://arxiv.org/abs/2505.13032). [[💻 Code]](https://github.com/ddlBoJack/MMAR). [[🎥 demo]](https://youtube.com/watch?v=Dab13opIGqU) [[🤗 Data]](https://huggingface.co/datasets/BoJack/MMAR) |
| 25.05 | On Path to Multimodal Generalist: General-Level and General-Bench | Multimodal Generation                         | [[🌐 Project]](https://generalist.top/) [[📑 Paper]](https://arxiv.org/abs/2505.04620) [[🤗 Data]](https://huggingface.co/General-Level) |
| 25.04 | VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models | Visual Reasoning                              | [[🌐 Project]](https://visulogic-benchmark.github.io/VisuLogic/) [[📑 Paper]](http://arxiv.org/abs/2504.15279) [[💻 Code]](https://github.com/VisuLogic-Benchmark/VisuLogic-Eval) [[🤗 Data]](https://huggingface.co/datasets/VisuLogic/VisuLogic) |
| 25.04 | IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs | Image-Grounded Video Perception and Reasoning | [[📑 Paper]](https://arxiv.org/pdf/2504.15415) [[💻 Code]](https://github.com/multimodal-art-projection/IV-Bench) |
| 25.04 | Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing | Reasoning-Informed viSual Editing             | [[📑 Paper]](https://arxiv.org/abs/2504.02826) [[💻 Code]](https://github.com/PhoenixZ810/RISEBench) |
| 25.04 | CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following | Music Information Retrieval & Knowledge       | [[📑 Paper]](https://arxiv.org/abs/2506.12285)  [[💻 Code]](https://github.com/nicolaus625/CMI-bench)    |
| 25.03 | MAVERIX: Multimodal Audio-Visual Evaluation Reasoning IndeX | Common Sense Omni Reasoning | [[📑 Paper]](https://arxiv.org/abs/2503.21699) [[🌐 Project]](https://maverix-benchmark.github.io/)| 
| 25.03 | V-STaR : Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning | Spatio-temporal Reasoning                     | [[🌐 Project]](https://v-star-bench.github.io/) [[📑 Paper]](https://arxiv.org/abs/2311.17982) [[🤗 Data]](https://huggingface.co/datasets/V-STaR-Bench/V-STaR) |
| 25.03 | MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs | Spatio-temporal Understanding                 | [[📑Paper]](https://arxiv.org/pdf/2503.13111)                 |
| 25.03 | Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning | 3D-CoT                                        | [[📑 Paper]](https://arxiv.org/pdf/2503.06232) [[🤗 Data]](https://huggingface.co/datasets/Battam/3D-CoT) |
| 25.02 | MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models | MM-IQ                                         | [[📑 Paper]](https://arxiv.org/pdf/2502.00698) [[💻 Code]](https://github.com/AceCHQ/MMIQ) |
| 25.02 | MM-RLHF: The Next Step Forward in Multimodal LLM Alignment   | MM-RLHF-RewardBench, MM-RLHF-SafetyBench      | [[📑 Paper]](https://arxiv.org/abs/2502.10391)                |
| 25.02 | ZeroBench: An Impossible* Visual Benchmark for Contemporary Large Multimodal Models | ZeroBench                                     | [[🌐 Project]](https://zerobench.github.io/) [[🤗 Dataset]](https://huggingface.co/datasets/jonathan-roberts1/zerobench) [[💻 Code]](https://github.com/jonathan-roberts1/zerobench/) |
| 25.02 | MME-CoT: Benchmarking Chain-of-Thought in LMMs for Reasoning Quality, Robustness, and Efficiency | MME-CoT                                       | [[📑 Paper]](https://arxiv.org/pdf/2502.09621) [[💻 Code]](https://github.com/CaraJ7/MME-CoT) |
| 25.02 | OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference | MM-AlignBench                                 | [[📑 Paper]](https://arxiv.org/abs/2502.18411) [[💻 Code]](https://github.com/PhoenixZ810/OmniAlign-V) |
|25.01 | AVTrustBench: Assessing and Enhancing Reliability and Robustness in Audio-Visual LLMs |  Adversarial attack, Compositional reasoning, and Modality-specific dependency in Visual&Audio | [[📑 Paper]](https://arxiv.org/abs/2501.02135)  |
| 25.01 | LlamaV-o1: Rethinking Step-By-Step Visual Reasoning in LLMs  | VRCBench                                      | [[📑 Paper]](https://arxiv.org/abs/2501.06186) [[💻 Code]](https://github.com/mbzuai-oryx/LlamaV-o1) |
| 24.12 | Online Video Understanding: A Comprehensive Benchmark and  Memory-Augmented Method | VideoChat-Online                              | [[Paper📑]](https://arxiv.org/abs/2501.00584) [[Code💻]](https://github.com/qirui-chen/MultiHop-EgoQA) |
| 24.11 | VLRewardBench: A Challenging Benchmark for Vision-Language Generative Reward Models | VLRewardBench                                 | [[📑 Paper]](https://arxiv.org/abs/2411.17451)                |
| 24.11 | Grounded Multi-Hop VideoQA in Long-Form Egocentric Videos    | MH-VidQA                                      | [[Paper📑]](https://arxiv.org/pdf/2408.14469) [[Code💻]](https://github.com/MCG-NJU/VideoChat-Online) |
| 24.10 | OmnixR: Evaluating Omni-modality Language Models on Reasoning across Modalities | Video&Audio Reasoning | [[📑 Paper]](https://arxiv.org/abs/2410.12219) | 
| 24.10 | MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark | Audio Understanding & Reasoning               | [[🌐 Project]](https://sakshi113.github.io/mmau_homepage/) [[📑 Paper]](https://arxiv.org/html/2410.19168v1) [[💻Code]](https://github.com/Sakshi113/mmau/tree/main) [[🤗 Data]](https://huggingface.co/datasets/apple/mmau) |
| 24.09 | MECD: Unlocking Multi-Event Causal Discovery in Video Reasoning | Video Causal Reasoning                        | [[📑 Paper]](https://arxiv.org/abs/2409.17647) [[💻Code]](https://github.com/tychen-SJTU/MECD-Benchmark) [[🤗 Data]](https://huggingface.co/datasets/tychen-sjtu/MECD) |
| 24.09 | OmniBench: Towards The Future of Universal Omni-Language Models | Reasoning with Image & Speech/Sound/Music | [[📑 Paper]](https://arxiv.org/abs/2409.15272) [[Code💻]](https://github.com/multimodal-art-projection/OmniBench) [[🌐 Project]](https://m-a-p.ai/OmniBench/)  [[🤗 Data]](https://huggingface.co/datasets/m-a-p/OmniBench) |
| 24.08 | MuChoMusic: Evaluating Music Understanding in Multimodal Audio-Language Models | Music Knowledge & Reasoning                   | [[🌐 Project]](https://mulab-mir.github.io/muchomusic/) [[📑 Paper]](https://zenodo.org/records/14877459) [[💻Code]](https://github.com/mulab-mir/muchomusic) [[ Data]](https://zenodo.org/records/12709974) |
| 24.07 | REXTIME: A Benchmark Suite for Reasoning-Across-Time in Videos | REXTIME                                       | [[Paper📑]](https://arxiv.org/abs/2406.19392) [[Code💻]](https://github.com/ReXTime/ReXTime) |
| 24.06 | AudioBench: A Universal Benchmark for Audio Large Language Models | Speech & Sound Understanding                  | [[Paper📑]](https://arxiv.org/pdf/2406.16020) [[Code🖥️]](https://github.com/AudioLLMs/AudioBench) |
| 24.06 | ChartMimic: Evaluating LMM’s Cross-Modal Reasoning Capability via Chart-to-Code Generation | ChartBench                                    | [[Project🌐]](https://chartmimic.github.io/) [[Paper📑]](https://arxiv.org/abs/2406.09961) [[Code🖥️]](https://github.com/ChartMimic/ChartMimic) |
| 24.05 | M3CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought | M3CoT                                         | [[📑 Paper]](https://arxiv.org/html/2405.16473v1)             |
| 24.02 | AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension | Speech & Sound Understanding                  | [[📑 Paper]](https://aclanthology.org/2024.acl-long.109.pdf)  [[Code💻]](https://github.com/OFA-Sys/AIR-Bench?tab=readme-ov-file) |
| 23.10 | CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models | Audio Reasoning (Attributes & Orders)         | [[Project🌐]](https://sreyan88.github.io/compa_iclr/) [[Paper📑]](https://openreview.net/forum?id=86NGO8qeWs) |

<a name="Open-sourceprojects"></a>

## Open-source Projects 
| Project | GitHub Stars | Links |
|---------|-------------|-------|
| **Reason-RFT** | ![Reason-RFT](https://img.shields.io/github/stars/tanhuajie/Reason-RFT) | [💻 GitHub](https://github.com/tanhuajie/Reason-RFT) [🤗 Dataset](https://huggingface.co/datasets/tanhuajie2001/Reason-RFT-CoT-Dataset) |
| **EasyR1** | ![EasyR1](https://img.shields.io/github/stars/hiyouga/EasyR1) | [💻 GitHub](https://github.com/hiyouga/EasyR1) |
| **Multimodal Open R1** | ![Multimodal Open R1](https://img.shields.io/github/stars/EvolvingLMMs-Lab/open-r1-multimodal) | [💻 GitHub](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) [🤗 Model](https://huggingface.co/lmms-lab/Qwen2-VL-2B-GRPO-8k) [🤗 Dataset](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) |
| **LMM-R1** | ![LMM-R1](https://img.shields.io/github/stars/TideDra/lmm-r1) | [💻 GitHub](https://github.com/TideDra/lmm-r1) |
| **MMR1** | ![MMR1](https://img.shields.io/github/stars/LengSicong/MMR1) | [💻 GitHub](https://github.com/LengSicong/MMR1) [🤗 Model](https://huggingface.co/MMR1/MMR1-Math-v0-7B) [🤗 Dataset](https://huggingface.co/datasets/MMR1/MMR1-Math-RL-Data-v0) |
| **R1-V** | ![R1-V](https://img.shields.io/github/stars/Deep-Agent/R1-V) | [💻 GitHub](https://github.com/Deep-Agent/R1-V) [🎯 Blog](https://deepagent.notion.site/rlvr-in-vlms) [🤗 Dataset](https://huggingface.co/collections/MMInstruction/r1-v-67aae24fa56af9d2e2755f82) |
| **R1-Multimodal-Journey** | ![R1-Multimodal-Journey](https://img.shields.io/github/stars/FanqingM/R1-Multimodal-Journey) | [💻 GitHub](https://github.com/FanqingM/R1-Multimodal-Journey) |
| **VLM-R1** | ![VLM-R1](https://img.shields.io/github/stars/om-ai-lab/VLM-R1) | [💻 GitHub](https://github.com/om-ai-lab/VLM-R1) [🤗 Model](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps) [🤗 Dataset](https://huggingface.co/datasets/omlab/VLM-R1) [🤗 Demo](https://huggingface.co/spaces/omlab/VLM-R1-Referral-Expression) |
| **R1-Vision** | ![R1-Vision](https://img.shields.io/github/stars/yuyq96/R1-Vision) | [💻 GitHub](https://github.com/yuyq96/R1-Vision) [🤗 Cold-Start Dataset](https://huggingface.co/collections/yuyq96/r1-vision-67a6fb7898423dca453efa83) |
| **R1-Onevision** | ![R1-Onevision](https://img.shields.io/github/stars/Fancy-MLLM/R1-Onevision) | [💻 GitHub](https://github.com/Fancy-MLLM/R1-Onevision) [🤗 Model](https://huggingface.co/Fancy-MLLM/R1-Onevision-7B) [🤗 Dataset](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision) [🤗 Demo](https://huggingface.co/spaces/Fancy-MLLM/R1-Onevision) [📝 Report](https://yangyi-vai.notion.site/r1-onevision) |
| **Open R1 Video** | ![Open R1 Video](https://img.shields.io/github/stars/Wang-Xiaodong1899/Open-R1-Video) | [💻 GitHub](https://github.com/Wang-Xiaodong1899/Open-R1-Video) [🤗 Model](https://huggingface.co/Xiaodong/Open-R1-Video-7B) [🤗 Dataset](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k) |
| **Video-R1** | ![Video-R1](https://img.shields.io/github/stars/tulerfeng/Video-R1) | [💻 GitHub](https://github.com/tulerfeng/Video-R1) [🤗 Dataset](https://huggingface.co/datasets/Video-R1/DVD-counting) |
| **Open-LLaVA-Video-R1** | ![Open-LLaVA-Video-R1](https://img.shields.io/github/stars/Hui-design/Open-LLaVA-Video-R1) | [💻 GitHub](https://github.com/Hui-design/Open-LLaVA-Video-R1) |
| **R1V-Free** | ![R1V-Free](https://img.shields.io/github/stars/Exgc/R1V-Free) | [💻 GitHub](https://github.com/Exgc/R1V-Free) |
| **SeekWorld** | ![SeekWorld](https://img.shields.io/github/stars/TheEighthDay/SeekWorld) | [💻 GitHub](https://github.com/TheEighthDay/SeekWorld) |
| **IE-Critic-R1** | ![SeekWorld](https://img.shields.io/github/stars/Coobiw/IE-Critic-R1) | [💻 GitHub](https://github.com/Coobiw/IE-Critic-R1) <br>[🤗 Model](https://huggingface.co/Coobiw/IE-Critic-R1-7B) <br>[🤗 Data](https://huggingface.co/datasets/Coobiw/IE-Bench-4k) <br>[🤗 ColdStart SFT](https://huggingface.co/datasets/Coobiw/IE-Bench-CoT-mixed) |

<a name="Contributiong"></a>
## Contributing 
If you are interested in contributing, please refer to [HERE](contribution.md) for instructions in contribution.
