# Can Protective Perturbation Safeguard Personal Data from Being Exploited by Stable Diffusion? [CVPR'24]

## Preparation

### Requirements
Install dependencies
```bash
git clone https://github.com/ZhengyueZhao/GrIDPure.git
cd GrIDPure
pip install -r requirements.txt
```
### Pre-trained models for Purification
We follow [**DiffPure**](https://github.com/NVlabs/DiffPure) to apply an unconditional diffusion model trained on ImageNet to our purification experiments:
- [Guided Diffusion](https://github.com/openai/guided-diffusion) for
  ImageNet: (`256x256 diffusion unconditional`: [download link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt))

### Pre-trained Stable Diffusion models
You can download any Stable Diffusion models you like from https://huggingface.co/.

We use [_stable diffusion v1.5_](https://huggingface.co/runwayml/stable-diffusion-v1-5) in our experiment.

### Dataset
You can choose any images you like to run the experiments. For instance, we put some paintings of _Picasso_ into the file [clean_images](https://github.com/ZhengyueZhao/GrIDPure/tree/main/clean_images) and each image is cropped into the resolution of $512\times512$.

## How to run

### Generate protected images
First of all, you should generate protected images (i.e. images with protected perturbation or poisoned images) from clean images. We provide two simple yet effective methods to protect images in this repository.

To protect images with adversarial examples ([AdvDM](https://arxiv.org/abs/2302.04578)), you can run

```bash
python poison_adv.py \
  --pretrained_model_name_or_path="your path to stable diffusion models"  \
  --instance_data_dir="./clean_images" \
  --output_dir="./poisoned_images_adv" \
  --instance_prompt="a painting in the style of PCS" \
  --resolution=512 \
  --train_batch_size=1 \
  --poison_scale=8 \
  --poison_step_num=100
```

To protect images with ASPL ([Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth)), you can run

```bash
accelerate launch poison_anti_db.py \
  --pretrained_model_name_or_path="your path to stable diffusion models"  \
  --instance_data_dir_for_train="./clean_images" \
  --instance_data_dir_for_adversarial="./clean_images" \
  --instance_prompt="a painting in the style of PCS" \
  --class_data_dir="./class_data" \
  --num_class_images=200 \
  --class_prompt="a painting" \
  --output_dir="./poisoned_images_anti_db" \
  --center_crop \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=1 \
  --max_train_steps=50 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=6 \
  --checkpointing_iterations=10 \
  --learning_rate=5e-7 \
  --pgd_alpha=5e-3 \
  --pgd_eps=5e-2 
```

You can also try other protection methods such as [_Mist_](https://link.zhihu.com/?target=https%3A//github.com/mist-project/mist) and [_Glaze_](https://glaze.cs.uchicago.edu/) following their official code or application.

### Fine-tune a Stable Diffusion
Now we can use the protected images to fine-tune a Stable Diffusion model to assess the effectiveness of these methods. To fine-tune a Stable Diffusion model with LoRA, you can run
```bash
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="your path to stable diffusion models"  \
  --instance_data_dir="./clean_images" \
  --output_dir="your path to saving LoRA" \
  --instance_prompt="a painting in the style of PCS" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=200 \
  --seed="0" \
  --train_text_encoder
```
You can find more fine-tuning methods from [**huggingface/diffuesrs**](https://github.com/huggingface/diffusers/blob/main/examples).

We provide a simple script for generating images from tuned models, you can run:

```bash
python generate.py \
  --model_id="your path to stable diffusion models" \
  --lora_dir="your path to trained LoRA" \
  --output_dir="./generated_images" \
  --prompt="a painting in the style of PCS" \
  --img_num=50 \
  --train_text_encoder=1
```

## Purification
To purify protected images from unlearnable images into learnable images, you can run purification scripts.

-Run **DiffPure**:
```bash
python diffpure.py \
    --input_dir="./poisoned_images_adv" \
    --output_dir="./purified_images_diffpure" \
    --pure_steps=100
```

-Run **GrIDPure**:
```bash
python gridpure.py \
    --input_dir="./poisoned_images_adv" \
    --output_dir="./purified_images_gridpure" \
    --pure_steps=10 \
    --pure_iter_num=20 \
    --gamma=0.1
```

## Citation
Cite our paper:
```
@inproceedings{zhao2024Can,
  title={Can Protective Perturbation Safeguard Personal Data from Being Exploited by Stable Diffusion?},
  author={Zhengyue Zhao, Jinhao Duan, Kaidi Xu, Chenan Wang, Rui Zhang, Zidong Du, Qi Guo and Xing Hu. },
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```












