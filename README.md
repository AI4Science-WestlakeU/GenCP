# GenCP: Towards Generative Modeling Paradigm of Coupled physics

[Paper](URL) | [arXiv](URL) 

Official repo for the paper **GenCP: Towards Generative Modeling Paradigm of Coupled physics**.

Tianrun Gao*, Haoren Zheng*, Wenhao Deng*, Haodong Feng, Tao Zhang, Ruiqi Feng, Qianyi Chen, Tailin Wu.

We introduce a novel framework for learning decoupled physics and generating coupled multi-physics systems using Conditional Flow Matching. Our method leverages Conditional Flow Matching (CFM) to learn joint distributions of coupled physical fields, enabling accurate and efficient generation of complex multi-physics phenomena.

Framework of paper:

<a href="url"><img src="assets/scheme.png" align="center" width="1000" ></a>

## Installation

Install dependencies:
```
conda create -n gencp python=3.10
conda activate gencp

pip install -r requirements.txt
pip install -e .
```

Alternatively, you can use the provided `environment.yml`:
```bash
conda env create -f environment.yml
```

## Dataset

All datasets can be downloaded from [this link](URL). 
<!-- Checkpoints are available at [this link](URL).  -->

- **Double Cylinder**
- **NTcouple**
- **Turek-Hron**

## Coupling inference

Use Double Cylinder as example.

1. **Set dataset path** in config file or environment variable:
   ```bash
   export DOUBLE_CYLINDER_DATA_ROOT=/path/to/double_cylinder/
   ```
3. **Run inference**:
   ```bash
   cd GenCP
   python infer_multi.py \
     --config configs/double_cylinder/fsi_cno.yaml \
     --fluid-checkpoint-path /path/to/fluid.pth \
     --structure-checkpoint-path /path/to/structure.pth \
     --num-sampling-steps 10 \
   ```
4. **View results** in `./visualization_results/` directory

### Training a Single Field

1. **Prepare dataset** and update `dataset_path` in config file
2. **Start training**:
   ```bash
   cd GenCP
   python train.py --config configs/double_cylinder/fluid_cno.yaml
   ```
3. **Monitor training**: Checkpoints saved in `./results/double_cylinder/fluid_CNO/`
4. **Evaluate**: Use `infer_single.py` with trained checkpoint

## Training and infer with bash scripts

**NTcouple Multi-Field Inference:**
```bash
bash scripts/ntcouple/our_cno/infer_ntcouple_multi.sh
```

**NTcouple Single-Field Inference:**
```bash
bash scripts/ntcouple/our_cno/infer_ntcouple_neutron.sh
bash scripts/ntcouple/our_cno/infer_ntcouple_solid.sh
bash scripts/ntcouple/our_cno/infer_ntcouple_fluid.sh
```

## Related Projects
  
* [M2PDE](https://github.com/AI4Science-WestlakeU/M2PDE): Diffusion-based approach for multi-physics modeling (baseline comparison, included in this repo)


## Citation

If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
    gencp2024,
    title={GenCP: Generative Coupled Physics for Fluid-Structure Interaction},
    author={Author names},
    booktitle={Conference name},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and issues, please contact: gaotianrun@westlake.edu.cn
