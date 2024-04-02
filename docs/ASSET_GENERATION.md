# How to generate Assets. 

**Note**: Due to complicated environment conflicts and long generation time, we have computed and provide an asset bank using the methods described below. The asset bank is diverse with 200+ vehicle meshes and 150+ animated pedestrian sequences, and it is highly recommended to use the asset bank to save time! 


## Vehicle Generation

We use [GET3D](https://github.com/nv-tlabs/GET3D) for our vehicle meshes generation. We have found the [provided Colab](https://colab.research.google.com/drive/1AAE4jp39rXhW2zmlNwpWkvDPULugIXfk?usp=sharing) to be particularly useful. Thank the author of GET3D  for such an amazing project! 

## Pedestrian Generation

We use [AvatarClip](https://github.com/hongfz16/AvatarCLIP) for our pedestrian generation and [Mixamo](https://www.mixamo.com/#/) to animate the meshes. Please refer to [scripts/anim_ped/README.md](../scripts/anim_ped/README.md) for more information on this process. Thank the author of AvatarClip for such an amazing projects.