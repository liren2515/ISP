# ISP: Multi-Layered Garment Draping with Implicit Sewing Patterns
<p align="center"><img src="figs/isp.png"></p>

This is the repo for [**ISP: Multi-Layered Garment Draping with Implicit Sewing Patterns**](https://liren2515.github.io/page/isp/isp.html).

Download checkpoints from [here](https://drive.google.com/file/d/1Zhr93ejWGobqDnJjE-P95ssNTDYSFNXS/view?usp=sharing), and put `*.pth` at `./checkpoints`.

For layering inference:
```
python infer_layering.py
```

For fitting garment in rest pose (no layering involved):
```
python fitting_3D_mesh.py --which skirt --save_path tmp --save_name skirt-fit --res 256
```
