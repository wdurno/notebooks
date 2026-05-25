# Fine tuning on a budget

In this experiment, we assess a small language model (SLM) fine tuning strategy designed to run on consumer hardware and use small datasets. 
The motive: left-shift AI product design to empower the user to customize AI and run it on their own hardware, 
respecting typical consumer constraints. 
AI must be capable of learning on small datasets and small hardware to be useful to the diverse long tail of use cases where most applications exist. 
Example 1: Imagine a factory worker training an AI to work the line, 
where data must be few, processing fast, and knowledge specific to the one factory and role. 
Example 2: Imagine a consumer with endlessly-specific preferences training a household assistant AI. 
The key challenge: generalizing existing, powerful models fit with multi-million dollar budgets to new applications. 

The focus for this work:
1. We'll assess the ability to estimate meaningfully accurate Fisher Information matrices from small datasets applied to quantized low-rank adapters (QLoRA) and elastic weight consolidation (EWC) and higher-rank generalizations. 
2. Since right-sized experience replay (ER) buffers tend to outperform EWC but increase computational resource requirements, we'll also measure how much ER buffer size and EWC rank can trade-off for equivalent results. 
   Ideally, we observe a useful trade-off point for use on hardware-constrained contexts.

## Model, hardware, metrics, experimental cases, & data

Base model: `microsoft/Phi-4-mini-instruct`

Hardware: `NVIDIA GeForce RTX 4070`

Core metrics:
1. Learning on new generated tasks.
2. Retention of prior model behavior and prior synthetic tasks.
3. Stability of Fisher estimates as `EWC n0` varies.
4. Hardware cost: VRAM, wall-clock time, tokens per second, and disk footprint.

Experimental cases:
1. Sweep `EWC n0` to measure how much data initializes useful EWC.
2. Compare EWC against experience replay buffer size.
3. Sweep QLoRA rank and EWC rank, where EWC rank is the column count of `L` in `LL^T + Lambda`.
4. Sweep fine tuning dataset size.
5. Test multi-step continual learning.

Run table:

| Case | Train n | EWC n0 | ER buffer | QLoRA rank | EWC rank | EWC lambda | Target EM | Retention | VRAM | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Data:
1. Synthetic factory-manual QA.
2. Synthetic rule transformations.
3. Synthetic preference-following tasks.
