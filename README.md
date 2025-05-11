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
    - [Reasoning Segmentation and Detection](#reasoning-segmentation-and-detection)
      - [Image MLLM](#image-mllm-1)
      - [Video MLLM](#video-mllm-1)
    - [Spatial and Temporal Grounding and Understanding](#spatial-and-temporal-grounding-and-understanding)
      - [Image MLLM](#image-mllm-2)
      - [Video MLLM](#video-mllm-2)
    - [Math Reasoning](#math-reasoning)
      - [Image MLLM](#image-mllm-3)
    - [Chart Rasoning](#chart-rasoning)
    - [Visual Generation](#visual-generation)
      - [Image MLLM](#image-mllm-4)
      - [Video MLLM](#video-mllm-3)
    - [Others](#others)
      - [Image MLLM](#image-mllm-5)
      - [Video MLLM](#video-mllm-4)
  - [Benchmarks 📊](#benchmarks-)
  - [Open-source Projects](#open-source-projects)
  - [Contributing](#contributing)


<a name="PapersandProjects"></a>
## Papers and Projects 📄

<a name="VQA"></a>
### Commonsense Reasoning
#### Image MLLM
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
* 24.08 (ECCV24)[VISA: Reasoning Video Object Segmentation via Large Language Model](http://arxiv.org/abs/2407.11325) | [Paper📑](http://arxiv.org/abs/2407.11325) [Code🖥️](https://github.com/cilinyan/VISA) [Dataset🤗](https://github.com/cilinyan/ReVOS-api)
   - Leverage the world knowledge reasoning capabilities of MLLMs while possessing the ability to segment and track objects in videos with a mask decoder | Task: Reasoning Segmentation
* 24.07 (NeruIPS24)[One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos](https://arxiv.org/abs/2409.19603) |  [Paper📑](https://arxiv.org/abs/2409.19603) [Code🖥️](https://github.com/dvlab-research/LISA) [Model🤗](https://huggingface.co/ZechenBai/VideoLISA-3.8B)
  - Integrating a Sparse Dense Sampling strategy into the video-LLM to balance temporal context and spatial detail within computational constraints |  Task: Reasoning Segmentation
* 24.01 (CVPR24) [OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding](https://arxiv.org/abs/2401.10229) | [Paper📑](https://arxiv.org/abs/2401.10229) [Code🖥️](https://github.com/lxtGH/OMG-Seg)
  - A transformer-based encoder-decoder architecture with task-specific queries and outputs for multiple tasks | Task: Reasoning Segmentation/Detection

<a name="Spatio-TemporalReasoning"></a>
### Spatial and Temporal Grounding and Understanding
#### Image MLLM
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
* 25.94 [VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning](https://arxiv.org/pdf/2504.06958) | [Paper📑](https://arxiv.org/pdf/2504.06958) [Code💻](https://github.com/OpenGVLab/VideoChat-R1)
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

<a name="MathReasoning"></a>

### Math Reasoning
#### Image MLLM
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
* 25.03 [MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding](https://arxiv.org/pdf/2503.13964v1) | [Paper📑](https://arxiv.org/pdf/2503.13964v1) [Code🖥️](https://github.com/aiming-lab/MDocAgent)
  - Integrates text and image retrieval through various agents, enabling collaborative reasoning across modalities. | Task: Document Reasoning
* 24.09 [ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning](https://arxiv.org/abs/2402.12185) | [Project🌐](https://unimodal4reasoning.github.io/DocGenome_page/)  [Paper📑](https://arxiv.org/abs/2402.12185) [Code🖥️](https://github.com/Alpha-Innovator/ChartVLM)
  - Offer a new perspective on handling chart reasoning tasks that strongly depend on interpretable patterns. | Task: Chart Reasoning
* 24.07 (EMNLP24) [Multimodal Self-Instruct: Synthetic Abstract Image and Visual Reasoning Instruction Using Language Model](https://arxiv.org/abs/2407.07053) | [Paper📑](https://arxiv.org/abs/2407.07053)  [Project🌐](https://multi-modal-self-instruct.github.io/) [Code🖥️](https://github.com/zwq2018/Multi-modal-Self-instruct) [Dataset🤗](https://huggingface.co/datasets/zwq2018/Multi-modal-Self-instruct)
  - A multi-modal self-instruct, utilizing large language models and their code capabilities to synthesize massive abstract images and visual reasoning instructions across daily scenarios. | Task: Chart Reasoning
* 24.02 (ACL24) [ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning](https://arxiv.org/abs/2401.02384) | [Paper📑](https://arxiv.org/abs/2401.02384) [Code🖥️](https://github.com/OpenGVLab/ChartAst) [Dataset🤗]
  - Use large-scale chart data to align and instruction tuning | Task: Chart Reasoning


<a name="VisualGeneration"></a>
### Visual Generation
#### Image MLLM
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
* 25.02 [C-Drag:Chain-of-Thought Driven Motion Controller for Video Generation](https://arxiv.org/pdf/2502.19868) | [Paper📑](https://arxiv.org/pdf/2502.19868) [Code🖥️](https://github.com/WesLee88524/C-Drag-Official-Repo) [Dataset🤗](https://drive.google.com/file/d/1L2SYadeqZPObvSj9Mb6fK-KHtR0n-DKk/view)
  - A Chain-of-Thought-based motion controller for controllable video generation | Task: Video Generation


<a name="others"></a>
### Others
#### Image MLLM
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

<a name="benchmarks"></a>
## Benchmarks 📊

| Date  | Project                                                      | Task                                     | Links                                                        |
| ----- | ------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------ |
| 25.04 | VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models | Visual Reasoning | [[🌐 Project]](https://visulogic-benchmark.github.io/VisuLogic/) [[📑 Paper]](http://arxiv.org/abs/2504.15279) [[💻 Code]](https://github.com/VisuLogic-Benchmark/VisuLogic-Eval) [[🤗 Data]](https://huggingface.co/datasets/VisuLogic/VisuLogic) |
| 25.04 | IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs | Image-Grounded Video Perception and Reasoning | [[📑 Paper]](https://arxiv.org/pdf/2504.15415) [[💻 Code]](https://github.com/multimodal-art-projection/IV-Bench) |
| 25.04 | Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing | Reasoning-Informed viSual Editing | [[📑 Paper]](https://arxiv.org/abs/2504.02826) [[💻 Code]](https://github.com/PhoenixZ810/RISEBench)
| 25.03 | V-STaR : Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning | Spatio-temporal Reasoning                | [[🌐 Project]](https://v-star-bench.github.io/) [[📑 Paper]](https://arxiv.org/abs/2311.17982) [[🤗 Data]](https://huggingface.co/datasets/V-STaR-Bench/V-STaR) |
| 25.03 | MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs | Spatio-temporal Understanding | [[📑Paper]](https://arxiv.org/pdf/2503.13111) |
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

<a name="Contributiong"></a>
## Contributing 
If you are interested in contributing, please refer to [HERE](contribution.md) for instructions in contribution.
