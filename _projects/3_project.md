---
layout: page
title: Low-Rank Adaptation Defense with Robustness
description: A LoRA-based defense pipeline for a ResNet-18 model
img: 
importance: 6
category: Normal
---

Developed and evaluated a LoRA-based defense pipeline for a ResNet-18 model to counter Feature Importance Attacks (FIA). Achieved a best validation adversarial accuracy of 98.60% within only 2 epochs, demonstrating rapid model robustness improvement.

## LoRADAR: Low-Rank Adaptation Defense with Robustness
### for ResNet18 base model in cifar-10 dataset.
I propose LoRADAR (Low-Rank Adaptation Defense with  Robustness), a novel method that integrates low-rank adaptation to defend against Rectangular Occlusion Attacks (ROA)[3,5] and Feature Importance-based Attacks (FIA)[2]. My approach leverages parameter-efficient fine-tuning and enhances robustness while maintaining accuracy on clean data.

**Introduction**

Contemporary deep learning models, though highly accurate, are often parameter-heavy and resource-intensive to adapt. Traditional finetuning requires updating a large fraction of model parameters, which can be computationally expensive and prone to overfitting. Low-Rank Adaptation (LoRA) addresses this challenge by introducing a small number of rank-decomposed parameters into the model, functioning as a hot-pluggable module that can be easily integrated or removed without permanently modifying the pre-trained model’s original weights. In doing so, LoRA:
1. Preserves the Integrity of Pre-trained Weights: The base model parameters remain fixed, ensuring the original pre-trained knowledge and capabilities are retained. This approach allows for immediate reversion to the original model state if desired.
2. Efficient Adaptation: Only a minor subset of parameters—specifically, the low-rank matrices—needs to be trained, significantly reducing both memory usage and computational overhead.
3. Rapid Experimentation and Deployment: Because LoRA can be “plugged in” without rewriting or altering core model weights, it enables efficient testing of different adaptation strategies, robust training protocols, and deployment scenarios without extensive model surgery.

My work unites the parameter-efficient adaptation offered by LoRA with robust adversarial training methods targeting Rectangular Occlusion Attacks (ROA) and Feature Importance-based Attacks (FIA). My approach aims to combine memory and computational efficiency with improved adversarial robustness, enabling a pre-trained ResNet-18, already tuned for CIFAR-10 classification, to become both resource-friendly and robust.

**Low-Rank Adaptation (LoRA)**

* Concept and Advantages:
Low-Rank Adaptation (LoRA) is a parameter-efficient finetuning technique introduced by Hu et al. (2021) in their paper[8] “LoRA: Low-Rank Adaptation of Large Language Models” (Hu et al., 2021). The central idea behind LoRA is to enable the adaptation of large, pre-trained models to new tasks or domains without fully updating all of their parameters. Instead, LoRA inserts a small number of additional trainable parameters in low-rank form, substantially reducing memory footprints and computational overhead.

* Key Concept and Advantages of LoRA:
    1. Low-Rank Decomposition of Weight Updates:
    Traditional finetuning updates full-rank parameter matrices, which can be prohibitively expensive for large models. LoRA assumes that the parameter updates necessary for adaptation lie in a low-dimensional subspace. To model these updates efficiently, LoRA decomposes the weight adjustments into the product of two low-rank matrices, A and B, where the rank r is significantly smaller than the original dimensionality. This ensures that the number of additional parameters is only a fraction of the full parameter count.
    2. Parameter Efficiency and Memory Savings:
    By focusing on a low-rank subspace, LoRA drastically cuts down the number of parameters that require gradients. For instance, if a layer’s parameter matrix is of size $W ∈ ℝ^{d×d}$, a low-rank decomposition with rank r adds only $O(d*r)$ parameters rather than $O(d^2)$. Consequently, this allows for adaptation with minimal extra storage and memory usage compared to full finetuning.
	3.	Plug-and-Play Adaptation:
    One of LoRA’s strengths is that it is “hot-pluggable.” The original pre-trained weights remain unaltered and are kept frozen, while LoRA parameters are learned and stored separately. This architecture allows for straightforward insertion or removal of LoRA modules without permanently changing the base model. As a result, multiple adapted versions can be toggled at inference time by simply swapping the LoRA parameters.
	4.	Minimal Impact on Base Representations:
    Since the core weights of the model are frozen, the model’s original, powerful representations are preserved. LoRA’s rank-limited updates provide a controlled adaptation mechanism that nudges the model’s decision boundaries or feature utilization patterns just enough to improve performance on the new target task. This ensures that the model can retain its general pre-trained knowledge while acquiring new, specialized behaviors.
    
* Role of the Rank Parameter ($r$):
	1.	Defining the Subspace for Adaptation:
        Instead of updating a full matrix of size $d × d$ (for simplicity, assume square shapes), LoRA posits that the meaningful updates lie within a lower-dimensional space. The rank $r$ specifies the size of this reduced dimension. When implementing LoRA:
        - Two low-rank parameter matrices lora_A and lora_B are introduced, with shapes typically $(r × d)$ and $(d × r)$, respectively.
        - Multiplying these together yields a rank-r approximation to the full $d × d$ update matrix, greatly shrinking the number of additional trainable parameters.
	2.	Balancing Model Flexibility and Efficiency:
	    - Higher Rank ($r$):
        Increasing the rank provides the model with a richer subspace in which to learn its adaptations, potentially capturing more complex patterns or nuanced adversarial defenses. However, a higher rank also means more parameters, which somewhat reduces the memory and computational savings that LoRA aims to achieve.
	    - Lower Rank ($r$):
        Reducing the rank decreases the number of parameters and preserves more computational and memory efficiency. This is beneficial if resources are limited or if you wish to keep overhead minimal. On the downside, too low a rank may overly constrain the model’s ability to represent necessary adjustments, resulting in limited performance gains.
* Implementation Details:
In the implementation, the core idea is to integrate LoRA modules directly into the architecture of a pre-trained ResNet-18 model without altering its original learned weights. To achieve this, each convolutional layer is replaced—or more specifically, wrapped—with a custom LoRAConv2d class. This class introduces and manages the low-rank adaptation parameters while leaving the base convolutional filters entirely intact.
Layer Wrapping Process:
    1. Preserving Original Weights:
    When a the first nn.Conv2d convolutional layer (conv1) in each block from ResNet-18 is encountered, it is not deleted or overridden. Instead, it is incorporated into a LoRAConv2d module. Inside this wrapper:
        - The original convolutional weights remain fixed and are never updated.
        - Two new parameter sets, lora_A and lora_B, are introduced with Zero-initialize them for a stable start. These matrices represent the low-rank approximation of the desired weight updates.
    2. Low-Rank Parameterization:
        - The lora_A and lora_B parameters are significantly smaller than the full convolutional weight matrix. By constraining the update to a low-rank subspace, the number of learnable parameters remains minimal.
        - During the forward pass, the output of the original convolution is computed first. Then, a low-rank approximation is formed by multiplying lora_A and lora_B to produce a rank-limited weight update (weight_update). This update is then added to the output of the original convolution, effectively shifting the layer’s behavior without altering the original kernel weights.
	3. Frozen Base Layers and Trainable LoRA:
        - All original parameters of the ResNet-18 backbone are frozen (i.e., requires_grad=False). As a result, no gradients propagate through these weights during training, preserving the original pre-trained representations.
        - Only the LoRA parameters (lora_A and lora_B) and the final classification layer (fc) remain trainable. This ensures that the adaptation is confined to a small, memory-efficient set of parameters.
	4. Backward Pass and Gradient Flow:
        - When backpropagation is performed, gradients flow into and update only the LoRA parameters and the fc layer’s weights. Since the primary convolutional weights stay constant, the bulk of the model’s complexity and memory footprint remains stable.
        - This focus on a limited parameter subset makes each training step more efficient. Memory usage is reduced because no gradients or optimizer states need to be maintained for the large pre-trained parameter sets.

* Practical Advantages:
    - Efficiency: By training only a handful of parameters (LoRA and fc), the training process becomes more lightweight and faster, especially compared to full finetuning approaches.
    - Stability and Control: Keeping the original weights frozen ensures that the pre-trained model’s general knowledge is preserved. The LoRA parameters subtly adjust the model’s decision boundaries to improve robustness against adversarial perturbations, rather than drastically altering its underlying feature extraction capabilities.
    - Modularity: The LoRA modules can be easily removed or replaced without impacting the integrity of the original network. This hot-pluggable feature allows rapid experimentation with different LoRA configurations and attack scenarios.

**Robustness Training**

Ensuring a model’s resilience against adversarial perturbations is integral to its reliability in real-world scenarios. In my implementation, the training process incorporates adversarial attacks—specifically, Rectangular Occlusion Attack (ROA) and Feature Importance-based Attack (FIA)—to foster robustness. By training with both clean and adversarially perturbed images, the model learns to maintain high accuracy even under non-ideal input conditions.

Adversarial Training Procedure:
1.	Adversarial Example Generation:
    At each training iteration, the first 500 clean images is sampled from the CIFAR-10 dataset. The chosen attack method (ROA or FIA) is then applied to these images to produce a corresponding set of adversarial examples.
    - ROA: Perturbs the image by occluding a portion of it, compelling the model to rely on unaffected regions for correct classification.
    - FIA: Targets critical features identified by the model, minimally perturbing these salient areas while causing maximum degradation if the model overly depends on them.
2.	Combining Clean and Adversarial Inputs:
    The training batch is composed of both the original clean images and their adversarial counterparts in 1:1 ratio. By doubling the input in this manner, the model simultaneously optimizes for performance on clean data and robustness to adversarial examples within the same gradient update.
3.	Parameter-Efficient Robust Finetuning with LoRA:
    Only the LoRA parameters and the final classification layer (fc) are trainable. The rest of the model, including all original convolutional weights, remains frozen. This ensures that robustness improvements do not come at the cost of excessive computational overhead or loss of previously learned representations.

Optimizer and Learning Rate Scheduler:
- Optimizer: An Adam optimizer is employed due to its adaptive learning rate and stable convergence properties when updating a small subset of parameters. The initial learning rate typically starts at 0.001, providing a balanced approach to convergence speed and stability.
- Learning Rate Scheduler: A StepLR scheduler is used to reduce the learning rate by a factor (e.g., gamma=0.1) after a certain number of epochs (step_size=10). While the training persent is only for 10 epochs, having the scheduler in place ensures that if extended training or additional experiments are conducted, the learning rate would properly adjust over longer durations.
- Parameter Settings:
    - LoRA Parameters: Low-rank matrices (lora_A and lora_B) inserted into convolutional layers form the main adaptation target. By optimizing these parameters, the model is gently guided towards more robust decision boundaries.
	- Final Classification Layer (fc): Also updated during training, ensuring that newly learned robust features are effectively translated into accurate classifications.
	- Frozen Layers: All original convolutional and batch normalization layers remain intact, preserving the foundational representational power acquired from pre-training.

Training Duration and Metrics:
- Number of Epochs: The robustness training is conducted for 10 epochs, a concise training period that still provides an opportunity for the model to integrate adversarial robustness without extensive computational cost.
- Monitoring Progress: After each epoch, the model is evaluated on precomputed adversarial test sets alongside the clean test set. By tracking performance metrics—both training and test clean accuracy, adversarial accuracy, and training loss—the incremental improvements in robustness can be observed. The best-performing model checkpoint (based on validation clean accuracy) can be identified within these 5 epochs.

**Mathematical Formulation and Detailed Approach**

* Base Model and Parameters:
    Let $\mathcal{M}(\theta)$ be a pre-trained neural network model with parameters $\theta$. For concreteness, consider a convolutional neural network (e.g., ResNet-18) trained on a classification task such as CIFAR-10. The original parameters $\theta_0$ are fixed after pre-training.

* LoRA: Low-Rank Adaptation:

    LoRA introduces a low-rank parameterization to adapt the model without changing $\theta_0$. For a given convolutional layer with weights $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}} \times k \times k}$ (assuming a kernel size of $k \times k$), LoRA focuses on adapting the effective weight matrix after flattening spatial and input dimensions. Consider the simplified form (e.g., after reshaping) $W \in \mathbb{R}^{D \times D}$ just for explanation, where $D = d_{\text{out}} \times (d_{\text{in}} \times k \times k)$.

    LoRA introduces two low-rank matrices $A \in \mathbb{R}^{D \times r}$ and $B \in \mathbb{R}^{r \times D}$ with rank $r \ll D$. Instead of updating W fully, LoRA defines a weight update:

    $\Delta W = A B$

    where A and B are the only new trainable parameters. The effective weight for that layer becomes:

    $W_{\text{eff}} = W_0 + A B$

    Here, $W_0$ is the original pre-trained weight that remains frozen, and A, B are learned to adapt the model.

    Because $r \ll D$, the number of additional parameters is greatly reduced. If the original layer has $D^2$ parameters, the LoRA addition has only $2Dr$, which is significantly smaller for a properly chosen $r$. By selecting a small $r$, the number of additional parameters is limited, ensuring parameter efficiency and maintaining the core feature extraction capability of the base model [Hu et al., 2021] paper[8]

* Robust Adversarial Training:

    During training, the model is exposed to adversarially perturbed inputs. Let $(x, y)$ denote a clean sample and its label. We generate an adversarial sample $\hat{x} = x + \delta$, where $\delta$ is crafted by an attack algorithm (e.g., ROA or FIA) to maximize the model’s loss[Madry et al., 2018; Goodfellow et al., 2015]paper[9]:

    $\hat{x} = \text{AttackAlg}(x, y, \theta_0, A, B)$

    Given this adversarial example, the training objective is to minimize the classification loss $\mathcal{L}$ over both clean and adversarial samples. Let $\mathcal{D}$ be the training dataset. For each batch $\{(x_i, y_i)\}$,
    we form $\hat{x}i$ using the chosen adversarial attack. The loss is:

    $\mathcal{L}{\text{total}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}\big[ \mathcal{L}(f(W_0 + A B; x), y) + \mathcal{L}(f(W_0 + A B; \hat{x}), y) \big]$


    Since $W_0$ is fixed, gradients do not flow into the base parameters. The only updates occur on A, B (the LoRA parameters) and on the final classification layer’s parameters (if those are allowed to update). Thus, the optimization problem is:

    $\min_{A,B,\theta_{\text{fc}}} \mathcal{L}{\text{total}}$

    where $\theta{\text{fc}}$ denotes the parameters of the final classification layer.

    By training on both $x$ and $\hat{x}$, the model learns robust decision boundaries that are harder to circumvent with adversarial perturbations. The low-rank structure ensures parameter efficiency and maintains the core representational power of $\theta_0$.

**Algorithm Diagram**

Below is a pseudo-code style algorithmic diagram. It illustrates the my main training loop for robust adaptation using LoRA. In practice, the LoRA A and B parameters are applied to the Pre-trained base model before training.
```
           Algorithm: Robust Training with LoRA
------------------------------------------------------
Inputs: Pre-trained model weights W0 (frozen),
        LoRA rank r,
        Training dataset D,
        Attack method (ROA or FIA),
        Number of epochs E, batch size B,
        Learning rate η

Initialize A, B, and fc parameters

for epoch = 1 to E do:
    Shuffle D
    for each batch { (x_i, y_i) } of size B in D:
        # Generate adversarial examples
        x_adv = AttackAlg(x, y, W0, A, B)

        # Combine clean and adversarial samples
        X_combined = [x; x_adv]
        Y_combined = [y; y]

        # Forward pass
        outputs = f(W0 + A B; X_combined)

        # Compute loss
        loss = L(outputs, Y_combined)

        # Backpropagation (only A, B, fc receive gradients)
        ZeroGrad(A, B, fc)
        loss.backward()
        Update(A, B, fc) with optimizer (e.g., Adam, lr=η)

    # Evaluate on validation set, monitor clean & adv accuracy

end for

Output: LoRA parameters A, B, updated fc layer
------------------------------------------------------
```

**Application-Wise Extension Using Multiple LoRA Configurations**

The methodology described can be extended into a practical defensive tool by preparing multiple sets of LoRA parameters, each trained to handle a specific type of adversarial attack. For example, one set might be specialized to counteract ROA, another to handle FIA, and yet others to address additional attack types or variations. Once these specialized LoRA parameter sets are obtained, they can be used in tandem through an ensemble approach.*(Not implemented in code, but the methodology is outlined.)*. Similarly, Tramèr et al. (2018) paper[10] demonstrate that ensembles of models, each trained with distinct adversarial strategies or on varied perturbations, can substantially improve robustness against attacks—an insight that applies to our approach by allowing multiple LoRA-adapted parameters to collectively defend against diverse adversarial threats.

Ensemble with Majority Voting:
1.	Multiple LoRA Configurations:
    Suppose we have a base model $\mathcal{M}(\theta_0)$ and several LoRA parameter sets $(A_1, B_1), (A_2, B_2), \ldots, (A_n, B_n)$, each tuned to a particular attack scenario. For instance:
	- $LoRA_{\text{ROA}}$: Primarily robust against ROA.
	- $LoRA_{\text{FIA}}$: Primarily robust against FIA.
	- $LoRA_{\text{Others}\ldots}$: Additional sets adapted to other known attacks.
2.	Forward Pass Through Each Adaptation:
    When a suspicious input x is presented, we can:
	- Apply $LoRA_{\text{ROA}}$ parameters to the base model and run inference:
    $\hat{y}{\text{ROA}} = f(W_0 + A_{\text{ROA}} B_{\text{ROA}}; x)$
	- Apply $LoRA_{\text{FIA}}$ parameters to the same base model (replacing the previous LoRA sets) and run inference:
    $\hat{y}{\text{FIA}} = f(W_0 + A_{\text{FIA}} B_{\text{FIA}}; x)$
	- Repeat for all available LoRA configurations $LoRA_{\text{Others}\ldots}$. Each adapted model provides a prediction $\hat{y}_i$.
3.	Voting or Confidence Aggregation:
    Once all predictions $\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_n$ are obtained—each from a differently adapted LoRA model—the ensemble decision-making comes into play:
	- Majority Vote: Count how many $LoRA_{\text{i}}$ models predict each class. The class with the highest vote wins, providing a more stable prediction than any single LoRA configuration alone.
	- Weighted Voting or Confidence-Based Fusion: Instead of a simple majority, we could use the predicted probabilities or confidence scores. Each LoRA configuration might have a reliability score (e.g., $LoRA_{\text{ROA}}$ is more trusted if we suspect ROA). A weighted aggregation of predictions can yield a final decision that is both robust and informative.
4.	Attack Identification and Defense:
    By inspecting how each LoRA configuration performs on the input, one can also identify the likely attack type:
	- If the input is best handled (highest accuracy or confidence) by $LoRA_{\text{ROA}}$, it suggests the input was crafted using ROA.
	- If $LoRA_{\text{FIA}}$ excels comparatively, then FIA is the prime suspect.
    By coupling these inferences with the ensemble approach, the system not only produces a final robust classification outcome but also pinpoints which adversarial strategy is likely at play. This identification capability can inform dynamic defenses—such as adjusting the system or applying pre-processing steps specific to certain attacks.
5.	Adaptive Security Measures:
    Knowing the most probable attack type can prompt adaptive response strategies:
	- If ROA is detected frequently, the system might deploy additional occlusion-handling filters or dynamic masks.
	- If FIA is common, emphasis might shift to smoothing feature importances or applying transformations that reduce the effectiveness of importance-based perturbations.

**Results, Analysis and Discussion**

The parameters for the attack algorithms are adapted from earlier experimental sections on ROA and FIA. All attack algorithms utilize the directly fine-tuned original ResNet18 model as their target for the attacks.

The training process utilizes the Adam optimizer, initialized with a learning rate of 0.001, and is configured to update only the parameters of the model that require gradients—specifically, the LoRA-adapted layers and the final classification (fc) layer. This selective optimization ensures efficient training by focusing computational resources on the adaptive components while keeping the base model parameters frozen. Additionally, a StepLR scheduler is employed, which systematically reduces the learning rate by a factor of 0.1 every 10 epochs. This learning rate decay strategy facilitates stable convergence by allowing larger updates in the initial phases of training for rapid adaptation, followed by finer adjustments in later stages to refine the model’s robustness against adversarial attacks like ROA and FIA. 

For my experiments, I utilized the CIFAR-10 dataset, encompassing the entire training set of 50,000 images and the first 500 samples from the testing set to expedite the training process. 

The LoRA rank parameter is set to 4, as recommended in the original LoRA paper [8], utilizing a small value.

The validation metics including both clean image accuracy and advserval image accuracy. 

Results Table of the best epoch
|metric| Under FIA attack|Under ROA attack|
|---|---|---|
|Validaiton clean accuracy|89.40%|92.40% |
|Validaiton advserval accuracy|98.60% |66.00%|

The training and validation metrics for the ROA attack defense illustrate a notable trend in the balance between clean image accuracy and adversarial robustness over the 10 training epochs. The model consistently achieves high clean accuracy, with the Best Validation Clean Accuracy reaching 92.40% in epoch 5. Concurrently, the adversarial accuracy demonstrates a steady improvement, peaking at 66.80% in epoch 10. This steady growth in adversarial accuracy, alongside the consistently high clean accuracy, indicates the model’s capability to learn robust features while maintaining strong performance on clean data.

From the training process, the clean accuracy remains above 95% throughout, reflecting the robustness of LoRA in preserving the original model’s capacity for clean image classification. For example, in the final epoch, the training clean accuracy is 96.12%, and the validation clean accuracy is 91.80%. This shows that LoRA effectively integrates robust training into the ResNet-18 architecture without compromising its clean image classification performance.

However, adversarial accuracy exhibits a more gradual improvement, emphasizing the inherent difficulty of defending against ROA attacks. The training adversarial accuracy increases from 43.55% in epoch 1 to 67.20% in epoch 10, while the validation adversarial accuracy follows a similar trajectory, starting at 57.00% in epoch 1 and peaking at 66.80% in the final epoch. This trend underscores the challenge of simultaneously optimizing for clean and adversarial performance, as adversarial training requires the model to generalize across perturbations, which often reduces its specialization on clean data.

The results from the FIA (Feature Importance Attack) defense training reveal remarkable findings, even with only 2 epochs of training. The model achieved a Best Validation Adversarial Accuracy of 98.60% in epoch 2, demonstrating exceptional robustness against FIA attacks. Notably, the adversarial accuracy surged from 60.09% to 98.10% during training, reflecting the model’s rapid adaptation to adversarial perturbations. While adversarial accuracy was consistently high, the Best Validation Clean Accuracy of 89.40% suggests a slight trade-off in clean image performance. Compared to ROA defense, the FIA defense converged significantly faster, achieving near-optimal performance within a shorter training period, likely due to the simpler nature of FIA perturbations. FIA primarily targets feature-level importance, resulting in structured, low-complexity,  and less obvious perturbations compared to the region-based sticker distortions in ROA. 

The success of LoRA in this results can be attributed to its low-rank nature, which allows the model to efficiently adapt to the structured, feature-level perturbations of FIA attacks. By introducing compact, low-rank parameter updates through LoRA, the model rapidly fine-tunes its behavior to counteract the specific distortions imposed by FIA. This low-rank adaptation is particularly well-suited for addressing the less complex, more systematic perturbations characteristic of FIA, enabling rapid convergence and high adversarial accuracy with minimal training effort.

These findings highlight the efficiency of LoRA-based robust training in mitigating specific attacks, making it suitable for time-sensitive or resource-limited scenarios, while also emphasizing the balance between clean accuracy and adversarial robustness.

**LoRA parameter efficitecy and Hot-Pluggable**

The LoRA parameters are highly efficient, with a file size of just 273KB, which is approximately 164 times smaller than the original pretrained model size of 44.8MB.

In my implementation, the hot-pluggable feature is showcased by saving specific parameters, such as those containing “lora” and “fc.” These parameters are then reloaded and seamlessly updated into the base model, demonstrating the flexibility and modularity of the approach.

An interesting observation emerged after applying LoRA to the base model and re-evaluating its performance on adversarial image accuracy. The ROA LoRA achieved an accuracy of 48.20%, which is significantly lower than the validation adversarial accuracy of 66.80%. In contrast, the FIA LoRA demonstrated exceptional performance with an accuracy of 99.40%, surpassing its validation adversarial accuracy of 98.60%. This discrepancy highlights the differing effectiveness of LoRA adjustments in adversarial scenarios, indicating that the parameters learned for ROA and FIA interact differently with the base model and adversarial robustness objectives.

The reduced accuracy for ROA LoRA can be attributed to the nature of adversarial attacks in ROA, where visible and significant perturbations are introduced to the images. Such perturbations likely require the LoRA parameters to adapt heavily to the altered image distribution, and insufficient parameter optimization or generalization may lead to reduced robustness. On the other hand, FIA involves only subtle changes to the image using the attack gradient. These adjustments align more closely with the base model’s learned representations than those in ROA, enabling LoRA to effectively mitigate the attack’s gradient impact and restore accuracy through its low-rank decomposition. Further investigations into gradient dynamics, feature alignment, and adversarial adaptation can provide more insights into these results and inform future improvements.

**Why This Method Is Not Performing Well in My Experimental Statistics**

The use of Low-Rank Adaptation (LoRA), while highly effective in large-scale models such as large language models (LLMs) and diffusion models, encounters challenges when applied to smaller architectures like ResNet-18. LoRA excels in scenarios with models that possess a vast hypothesis space, where the low-rank parameter updates can encapsulate meaningful adaptations without compromising the base model’s underlying representation. These updates efficiently utilize the large parameter space to encode diverse and expressive features, enabling robust performance across tasks.

However, in smaller architectures like ResNet-18, the hypothesis space is inherently constrained by the significantly reduced number of parameters. This limited capacity restricts LoRA’s ability to encode nuanced transformations necessary for robust defenses against adversarial perturbations while maintaining high clean image accuracy. The constrained hypothesis space in ResNet-18 likely lacks the flexibility to accommodate both adversarial robustness and task-specific generalization, resulting in a trade-off that diminishes LoRA’s overall efficacy in this context.

Furthermore, models like LLMs and diffusion models benefit from their expansive parameter spaces, which allow LoRA to fully leverage low-rank adaptations to encode complex, high-dimensional patterns. This ensures that the updates remain both expressive and computationally efficient. Conversely, ResNet-18’s smaller parameter scale limits LoRA’s capacity to adapt effectively to sophisticated adversarial attacks, such as ROA and FIA. These attacks exploit intricate vulnerabilities in the model’s decision boundary, requiring more flexible and robust parameter adjustments than ResNet-18’s compact hypothesis space can afford.

This discrepancy between the design intent of LoRA and the architectural constraints of smaller models highlights a fundamental limitation: while LoRA can maintain high clean accuracy, it struggles to simultaneously enhance adversarial robustness in architectures with limited capacity. Addressing this performance gap may require hybrid approaches that augment LoRA with additional mechanisms tailored to smaller models, such as dynamic re-parameterization or targeted feature augmentation, to better navigate the constrained hypothesis space and improve adversarial resilience.

**Why This Method Is Suitable for Defense**
- *Parameter Efficiency and Focused Adaptation*:
    By introducing only a small set of low-rank parameters (LoRA) on top of the fixed, pre-trained weights, the model avoids the complexity and resource cost of retraining or fully finetuning all parameters. This efficiency is not just computational—it also ensures that the model’s existing robust features and general representations are preserved. In adversarial settings, preserving established feature extraction capabilities is crucial for maintaining baseline performance while incorporating defensive adjustments.
- *Hot-Pluggable Defense Mechanisms*:
    Because LoRA modules do not overwrite or permanently alter the underlying model weights, they are effectively “hot-pluggable.” This feature means that the defense mechanism can be easily deployed, removed, or replaced without permanently committing to the new parameters. In practice, this allows rapid experimentation with different robustness strategies and quick responses to evolving attack types. Moreover, if the defense proves ineffective or needs further tuning, the original model is still intact and can be reverted to quickly.
- *Maintaining High Clean Accuracy*:
    Traditional adversarial training methods often sacrifice significant clean accuracy to gain robustness. The LoRA-based approach constrains updates to a low-dimensional subspace, which nudges the model toward more robust decision boundaries without substantially distorting the base representations learned for clean inputs. As a result, the model can sustain strong clean performance while still improving resilience to carefully crafted adversarial perturbations.

In summary, this LoRA-based robust training method is suitable for defense because it efficiently integrates with an large parameter model, preserves original performance, and offers flexible, resource-light ways to enhance adversarial robustness. Its ability to hot-swap defensive adaptations and maintain strong clean accuracy makes it a practical choice for real-world adversarial defense scenarios.




# Bibliography
[1] How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses. Joana C. Costa et al. https://arxiv.org/abs/2305.10862

[2] Feature Importance-aware Transferable Adversarial Attacks.Zhibo Wang et al. https://arxiv.org/abs/2107.14185

[3] Defending Against Physically Realizable Attacks on Image Classification. Tong Wu et al. https://arxiv.org/abs/1909.09552

[4] FIA implementation https://github.com/hcguoO0/FIA/blob/main/attack.py

[5] ROA implementation https://github.com/tongwu2020/phattacks/blob/master/ROA/ROA.py

[6] DOA implementation https://github.com/tongwu2020/phattacks/blob/master/cifar/cifar_roa.py

[7] Dataset Normalization Value https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py

[8] LoRA: Low-Rank Adaptation of Large Language Models https://arxiv.org/abs/2106.09685

[9] Explaining and Harnessing Adversarial Examples https://arxiv.org/abs/1412.6572

[10] Ensemble Adversarial Training: Attacks and Defenses https://arxiv.org/abs/1705.07204 