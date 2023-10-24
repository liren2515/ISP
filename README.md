# ISP

Put checkpoints at `./checkpoints`.

For layering inference:
```
python infer_layering.py
```

For fitting garment in rest pose (no layering involved):
```
python fitting_3D_mesh.py --which skirt --save_path tmp --save_name skirt-fit --res 256
```
