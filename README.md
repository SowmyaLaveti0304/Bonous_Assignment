# Bonous_Assignment– Summer 2025

**Student Name:** Sowmya Laveti  
**Student ID:** 700771347  
**University:** University of Central Missouri  
**Course:** CS5720 – Neural Networks and Deep Learning

## Assignment Overview

This assignment covers key topics in NLP and deep learning:

- **Q1:** build a simple question answering system using pre-trained models.  
- **Q2:** Digit-Class Controlled Image Generation with Conditional GAN

## Build a simple question answering system using pre-trained models
This assignment is divided into 3 parts
   - 1. Basic Pipeline Setup
     2. Use a Custom Pretrained Model
     3. Test on my own example
### Tasks completed:
- Basic Pipeline Setup
   - Imported pipeline from transformers.
   - Initialized a default QA pipeline.
   - Ran it on a Charles Babbage context to extract “Charles Babbage” 
- Use a Custom Pretrained Model
   - Re-initialized the QA pipeline with deepset/roberta-base-squad2.
   - Queried the same Babbage context to verify you still get “Charles Babbage” with a score > 0.70.
- Test on Your Own Example
   - Wrote a 2–3 sentence context (with Abdul Kalam).
   - Asked two distinct questions of that context.
### Output:
{'score': 0.98927903175354, 'start': 1, 'end': 16, 'answer': 'Charles Babbage'}
{'score': 0.8692716956138611, 'start': 1, 'end': 16, 'answer': 'Charles Babbage'}
{'question': 'What nickname did he earn?', 'answer': 'Missile Man of India', 'score': 0.5, 'start': 301, 'end': 321}
{'question': 'During which years did he serve as President of India?', 'answer': '2002 to 2007', 'score': 0.95, 'start': 166, 'end': 178}

## Q2: Digit-Class Controlled Image Generation with Conditional GAN
### Tasks completed:
- Data Preparation: Downloaded and preprocessed MNIST 
- Label Embedding: Added an nn.Embedding(10, embed_dim) in both Generator and Discriminator to turn class labels (0–9) into learnable vectors.
- Generator Modification: Concatenated the noise vector z with the corresponding label embedding before feeding it through the generator’s fully-connected network.
- Discriminator Modification: Flattened input image and concatenated it with the same label embedding.
- Adversarial Training Loop: Discriminator step: Trained on real MNIST images with true labels (target 1) and on fake images from G with those same labels (target 0).
- Generator step: Trained to fool D (making D(fake,label) → 1).
- Visualization: Plotted the loss curves for G and D over epochs.
### Output:
![Figure_1](https://github.com/user-attachments/assets/546d0d7b-d4db-4023-86c8-6287a31386c9)



### Short Answer
**How does a Conditional GAN differ from a vanilla GAN?**
- A vanilla GAN learns to generate samples from an unlabeled distribution by mapping random noise 𝑧→𝑥, with no notion of “what” it’s drawing. A Conditional GAN (cGAN) instead injects side-information 𝑦 (e.g. class labels, text captions, or attributes) into both Generator and Discriminator—so G(𝑧,𝑦) produces samples matching 𝑦, and D(𝑥,𝑦) judges both realism and correctness of the conditioning.
- Real-world application:
In medical imaging, cGANs can be conditioned on disease labels to generate or augment scans showing specific pathologies (e.g. MRI slices with a tumor), helping train diagnostic models when real data for rare conditions is scarce.
---
**What does the discriminator learn in an image-to-image GAN?**
- In an image-to-image GAN (e.g. Pix2Pix), the discriminator learns not just to tell “real vs. fake” images, but to judge pairs inputimage+outputimage as either truly corresponding (real pair) or mismatched/generated (fake pair). By seeing the input alongside the candidate output, it enforces that the generator’s output is both photorealistic and semantically aligned with that specific input.
- Why pairing matters:
Without paired examples, the discriminator can only assess overall realism and has no way to verify that the output matches the input’s structure or content. Paired (supervised) data lets it learn the correct mapping—pushing the generator to produce outputs that not only look real but also faithfully transform that exact input.


