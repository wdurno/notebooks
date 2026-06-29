# KL Projection for Cost-Effective Speed-Ups in AI for Robotics 

Modern AI in robotics frequently utilizes a hierarchy of models to balance the trade-off between speed and cognition. 
With a large budget, each model in the hierarchy can be carefully curated, or even trained together on vast datasets. 
We're not doing this here. 
Instead, we focus on working from an existing small transformer model with available open source weights, and build the model hierarchy outward from there. 
The models below the transformer in the hierarchy will be fit with KL projection, so will be computationally cheap to fit. 
The key observation: optimizing the transformer's weights is the only step requiring data center-sized compute loads, and the rest is vastly cheaper. 
In this case, the transformer model will be a small visual language model (VLM) and it'll only have a single lower model, an LSTM. 
Only a single consumer-grade GPU is used. 
This experiment's hypothesis: an existing, high-quality transformer model can be used to cost-effectively seed a robotics model hierarchy. 

## Model Hierarchy 

The VLM will only fire every `K` game steps, whereas the LSTM will fire every step. 
`K` is a hyperparameter. 
Our VLM will control the LSTM by communicating through an additional head, so will only communicate to LSTM `K`-series once at series start. 
Specifically, the VLM will read the prior `K` observations, then produce a head output that is fed into the LSTM to guide its next `K` actions. 
The head output will be concatenated to the visual encodings given to the LSTM. 
The VLM's visual encoding stack will provide inputs to both the LSTM and VLM for every image obtained.
While it'd be nice to have the LSTM communicate back to the VLM with new custom tokens, that'd require a large computational budget so will not be attempted.
The VLM will continue owning language generation whereas the LSTM only ever owns action generation. 

## KL-Projection 

An initial dataset will be generated with the VLM alone, not involving the LSTM. 
It'll control the robot with agentic commands. 
After collecting a dataset, the agentic commands will be converted deterministically to low-dimensional vectors, then fed into the LSTM as prior actions and target predicted actions. 
The loss function: KL divergence between the LSTM & the VLM's predicted action vectors. 

## Experimental design 

The experiment is phased, locking-in foundational results for the next phase, and builds toward a unified system. 
Each phase has metrics, allowing us to evaluate success before moving onto the next phase. 
If successful, we should have the LSTM following the VLM's intent. 
The two recurring metrics are action latency and coherency. 
Action latency is measured with wall clock time. 
Raw latency statistics will be recorded, then aggregated during write-up. 
Coherency is the robot's ability to do as it says. 
In every phase of the experiment, the Robot is (1) conversing with the experimenter, and (2) tasked with chasing a red ball. 

The robot is a Picar-V running an onboard Raspberry Pi. 
The Raspberry Pi merely runs a Flask server, handling no AI computation. 
AI processing is done on a PC with a consumer-grade GPU. 

### Experiment phase 1: data collection

In this phase, we give the VLM complete control over the robot without involving the LSTM. 
It controls the robot with agentic Python commands. 
The purpose of this phase is data collection that can be used for KL-projection. 
Primary metrics: dataset size, action latency, and coherency. 

### Experiment phase 2: KL-projection

The data from phase 1 is used to fit the LSTM with KL-projection. 
After KL-projection, the robot will be run with its full model hierarchy. 
The LSTM will be given full control over robot movement, while the VLM will control language and high-level strategy via its additional head. 
Primary metrics:
1. Action latency: The LSTM should be capable of running much faster than the VLM, non-VLM iterations should be fast. 
2. Coherency: The VLM's stated intent should match the behaviour of the LSTM. Coherency doesn't need to be perfect, just good enough to enable fine tuning in phase 3. 

### Experiment phase 3: LSTM finalizing 

In this phase, we add a few fine tuning options to exploring finalization. 
We expect this phase to require exploration & creativity to get desired results, so each option will be given a hyperparameters adjusting degree of involvement.
Primary metrics: action latency and coherency. 
Fine tuning options:
1. The VLM will be given trainable QLoRA parameters. Hyperparameter: QLoRA rank. Setting rank to 0 disables QLoRA. 
2. All trainable parameters will share an EWC regularizer. EWC weights will be initialized from all prior data. Hyperparameters: EWC rank & regularizer weight $\lambda$. Setting $\lambda = 0$ disables EWC. 
3. The model will be given an experience replay buffer of fixed maximum length. When the buffer gets full, adding new data will eject old data. Since EWC is approximately equivalent to the log likelihood of all prior data, all old data ejected from the replay buffer will be integrated into the EWC buffer in an online learning fashion. Hyperparameter: experiment replay buffer maximum length. 
4. All trainable parameters will share a single RL loss. Hyperparameter: None. It is the baseline loss and cannot be disabled. 
