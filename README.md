# Real time diffusion [BETA]
Collection of real-time diffusion tools and comfy wrappers.

# Installation
Simply clone this repository into your custom_nodes subfolder in your comfyUI installation.
```bash
git clone https://github.com/lunarring/rtd_comfy
```
Next, navigate into the lunar_tools_comfy folder and install the requirements.
```bash
cd ComfyUI/custom_nodes/rtd_comfy
pip install -r requirements.txt
```
After restarting comfyUI, you should be able to see "Lunar Ring" after a rightclick/Add Node, or find the below nodes directly.

# Workflow examples

We provide .json for all workflow examples in https://github.com/lunarring/rtd_comfy/tree/main/workflows

You can also drag and drop the below screenshots of the examples.

Generally, to run the workflows in a continuous way, you will have to select auto queuing (instant) in ComfyUI.
<img width="332" alt="image" src="https://github.com/user-attachments/assets/5c4ff6cb-97e4-4157-94c7-ac4eccdd73ee">


## Webcam live transformation
![webcam](https://github.com/user-attachments/assets/f2b7f30f-261c-4178-a105-b374abbdffb5)

With this workflow, the webcam image is transformed using real time feed from the cam.

## Infinite Zoom with automatic prompt cycling
![zoom](https://github.com/user-attachments/assets/c1c08b15-7289-4eb0-97e5-37e277f23b76)

Here we zoom infinitely into the last generated image. 

## Blending of decoder embeddings
![decoder](https://github.com/user-attachments/assets/1a47550c-0e08-44fc-a82e-31e2fbdef0c4)

We only modify the decoder embeddings selectively ("dirty" and "clean"), which leaves the overall structure of the image intact ("photo of a room"). 

